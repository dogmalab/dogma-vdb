//! Smart chunker — detects file type and selects the best strategy.
//!
//! A `.rs` file chunks by function/impl boundaries, a `.md` by headers,
//! a `.txt` by paragraphs, a `.py` by def/class, and a `.jsonl` line by line.
//!
//! ## Design
//!
//! * All regex patterns are compiled **once** in the constructor.
//! * Strategies are stored as concrete fields — **zero heap allocation** per call.
//! * No `Box<dyn>` — static dispatch via enum dispatch.
//!
//! ## Usage
//!
//! ```rust
//! use dogma_vdb::smart_chunker::{SmartChunker, FileType};
//!
//! let chunker = SmartChunker::default();
//! let chunks = chunker.chunk_file("main.rs", "fn main() {}");
//! ```

mod code;
mod jsonl;
mod markdown;
mod paragraph;
mod semantic;
#[cfg(feature = "chunker-syntax")]
mod syntax;

use crate::doc::Document;
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::Path;

pub use code::CodeChunker;
pub use jsonl::JsonLinesChunker;
pub use markdown::MarkdownChunker;
pub use paragraph::ParagraphChunker;
pub use semantic::SemanticChunker;
#[cfg(feature = "chunker-syntax")]
pub use syntax::SyntaxChunker;

// ---------------------------------------------------------------------------
// FileType detection
// ---------------------------------------------------------------------------

/// Supported file types for smart chunking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum FileType {
    Rust,
    Python,
    JavaScript,
    Go,
    Markdown,
    Text,
    JsonLines,
    /// Explicit semantic chunking (not auto-detected from extension).
    Semantic,
    Unknown,
}

impl FileType {
    /// Detect file type from a file path.
    pub fn from_path(path: impl AsRef<Path>) -> Self {
        path.as_ref()
            .extension()
            .and_then(|e| e.to_str())
            .map(Self::from_extension)
            .unwrap_or(FileType::Unknown)
    }

    /// Detect file type from an extension string (without leading dot).
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_lowercase().as_str() {
            "rs" => FileType::Rust,
            "py" => FileType::Python,
            "js" | "jsx" | "ts" | "tsx" | "mjs" | "cjs" => FileType::JavaScript,
            "go" => FileType::Go,
            "md" | "markdown" => FileType::Markdown,
            "txt" | "text" => FileType::Text,
            "jsonl" | "vdb" | "ndjson" => FileType::JsonLines,
            _ => FileType::Unknown,
        }
    }

    /// Human-readable name.
    pub fn name(self) -> &'static str {
        match self {
            FileType::Rust => "Rust",
            FileType::Python => "Python",
            FileType::JavaScript => "JavaScript",
            FileType::Go => "Go",
            FileType::Markdown => "Markdown",
            FileType::Text => "Text",
            FileType::JsonLines => "JSON Lines",
            FileType::Semantic => "Semantic",
            FileType::Unknown => "Unknown",
        }
    }
}

// ---------------------------------------------------------------------------
// SmartChunk
// ---------------------------------------------------------------------------

/// A single chunk with structural metadata.
#[derive(Debug, Clone)]
pub struct SmartChunk {
    pub text: String,
    /// Name of the enclosing structure (`"fn main"`, `"class Foo"`, `"# Intro"`).
    pub structure: Option<String>,
    /// Hierarchical level (0 = top-level).
    pub level: usize,
    /// 0-indexed start line in the original file.
    pub start_line: usize,
    /// 0-indexed end line (exclusive).
    pub end_line: usize,
}

/// A file to process in a batch chunking operation.
///
/// Carries the file path (for type detection and document naming) and
/// its raw text content.
#[derive(Debug, Clone)]
pub struct InputFile {
    /// File path (used for type detection and `base_id`).
    pub path: String,
    /// Raw file content.
    pub text: String,
    /// Optional extra metadata to attach to every chunk from this file.
    pub metadata: HashMap<String, String>,
}

impl InputFile {
    /// Create a new input file with the given path and content.
    pub fn new(path: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            text: text.into(),
            metadata: HashMap::new(),
        }
    }

    /// Attach metadata that will be inherited by every chunk.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

// ---------------------------------------------------------------------------
// SmartChunker
// ---------------------------------------------------------------------------

/// Configuration for [`SmartChunker`].
#[derive(Debug, Clone)]
pub struct SmartChunkerConfig {
    pub max_chunk_size: usize,
    /// Overlap for paragraph fallback (characters).
    pub overlap: usize,
}

impl Default for SmartChunkerConfig {
    fn default() -> Self {
        Self {
            max_chunk_size: 1024,
            overlap: 64,
        }
    }
}

/// File-type–aware chunker.  Strategies are pre-compiled in the constructor.
///
/// ```rust
/// use dogma_vdb::smart_chunker::{SmartChunker, FileType};
///
/// let chunker = SmartChunker::default();
/// let chunks = chunker.chunk_file("main.rs", "fn hello() {}");
/// assert_eq!(chunks[0].structure.as_deref(), Some("hello"));
/// ```
#[derive(Debug)]
pub struct SmartChunker {
    config: SmartChunkerConfig,
    rust: CodeChunker,
    python: CodeChunker,
    javascript: CodeChunker,
    go: CodeChunker,
    markdown: MarkdownChunker,
    text: ParagraphChunker,
    jsonl: JsonLinesChunker,
    /// AST‑based code chunker (only with `chunker-syntax` feature).
    #[cfg(feature = "chunker-syntax")]
    syntax: Option<SyntaxChunker>,
    /// Embedding‑based prose chunker (injected via [`SmartChunker::with_semantic`]).
    semantic: Option<SemanticChunker>,
}

impl SmartChunker {
    pub fn new(config: SmartChunkerConfig) -> Self {
        let overlap = config.overlap;
        Self {
            config,
            rust: CodeChunker::new(&[
                r"^\s*pub\s+(?:unsafe\s+)?fn\s+(\w+)",
                r"^\s*(?:pub\s+)?fn\s+(\w+)",
                r"^\s*pub\s+(?:unsafe\s+)?trait\s+(\w+)",
                r"^\s*(?:pub\s+)?(?:struct|enum|union|trait|mod|type)\s+(\w+)",
                r"^\s*(?:pub\s+)?impl\s*(?:<[^>]*>)?\s+\w+",
                r"^\s*#!\[",
            ]),
            python: CodeChunker::new(&[
                r"^\s*(?:async\s+)?def\s+(\w+)",
                r"^\s*class\s+(\w+)",
                r"^\s*@\w+",
            ]),
            javascript: CodeChunker::new(&[
                r"^\s*(?:export\s+)?(?:async\s+)?function\s+(?:\*\s+)?(\w+)",
                r"^\s*(?:export\s+)?class\s+(\w+)",
                r"^\s*(?:export\s+)?(?:const|let|var)\s+\w+\s*=\s*(?:async\s+)?(?:function|\(|=>)",
                r"^\s*interface\s+(\w+)",
                r"^\s*type\s+(\w+)",
            ]),
            go: CodeChunker::new(&[
                r"^\s*func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)",
                r"^\s*type\s+(\w+)\s+struct",
                r"^\s*type\s+(\w+)\s+interface",
            ]),
            markdown: MarkdownChunker,
            text: ParagraphChunker::new(overlap),
            jsonl: JsonLinesChunker,
            #[cfg(feature = "chunker-syntax")]
            syntax: SyntaxChunker::new().ok(),
            semantic: None,
        }
    }

    /// Enable semantic chunking with an [`Embedder`](crate::embedding::Embedder).
    ///
    /// When set, `FileType::Semantic` chunks text via embedding similarity
    /// instead of paragraph boundaries.
    pub fn with_semantic(mut self, embedder: Box<dyn crate::embedding::Embedder>) -> Self {
        self.semantic = Some(SemanticChunker::new(embedder, 0.35));
        self
    }

    /// Enable semantic chunking with full configuration.
    pub fn with_semantic_config(
        mut self,
        embedder: Box<dyn crate::embedding::Embedder>,
        threshold: f32,
    ) -> Self {
        self.semantic = Some(SemanticChunker::new(embedder, threshold));
        self
    }

    /// Chunk a file, auto-detecting the type from its path.
    pub fn chunk_file(&self, path: impl AsRef<Path>, text: &str) -> Vec<SmartChunk> {
        let file_type = FileType::from_path(path);
        self.chunk_text(text, file_type)
    }

    /// Chunk text with an explicit [`FileType`].
    pub fn chunk_text(&self, text: &str, file_type: FileType) -> Vec<SmartChunk> {
        let max = self.config.max_chunk_size;
        match file_type {
            FileType::Rust | FileType::Python | FileType::JavaScript | FileType::Go => {
                // Syntax (AST) chunker has priority when feature is enabled
                #[cfg(feature = "chunker-syntax")]
                if let Some(ref syn) = self.syntax {
                    let result = syn.chunk(text, file_type, max);
                    if !result.is_empty() {
                        return result;
                    }
                }
                // Fallback: regex-based code chunker
                match file_type {
                    FileType::Rust => self.rust.chunk(text, max),
                    FileType::Python => self.python.chunk(text, max),
                    FileType::JavaScript => self.javascript.chunk(text, max),
                    FileType::Go => self.go.chunk(text, max),
                    _ => unreachable!(),
                }
            }
            FileType::Markdown => self.markdown.chunk(text, max),
            FileType::Text => self.text.chunk(text, max),
            FileType::JsonLines => self.jsonl.chunk(text, max),
            FileType::Semantic => {
                if let Some(ref sem) = self.semantic {
                    sem.chunk(text, max)
                } else {
                    // No embedder configured → paragraph fallback
                    self.text.chunk(text, max)
                }
            }
            FileType::Unknown => self.text.chunk(text, max),
        }
    }

    /// Chunk and wrap into [`Document`]s with enriched metadata.
    pub fn chunk_to_docs(
        &self,
        path: impl AsRef<Path>,
        text: &str,
        base_id: &str,
        mut extra_meta: HashMap<String, String>,
    ) -> Vec<Document> {
        let file_type = FileType::from_path(&path);
        let chunks = self.chunk_text(text, file_type);

        extra_meta.insert("language".into(), file_type.name().into());

        chunks
            .into_iter()
            .enumerate()
            .map(|(i, chunk)| {
                let mut meta = extra_meta.clone();
                if let Some(ref s) = chunk.structure {
                    meta.insert("structure".into(), s.clone());
                }
                meta.insert("level".into(), chunk.level.to_string());
                meta.insert("start_line".into(), chunk.start_line.to_string());
                meta.insert("end_line".into(), chunk.end_line.to_string());

                Document::builder(format!("{}-{}", base_id, i), chunk.text)
                    .metadatas(meta)
                    .build()
            })
            .collect()
    }

    /// Process multiple files in parallel using the rayon thread pool.
    ///
    /// Each file is chunked independently on its own rayon worker thread.
    /// The internal chunkers (`CodeChunker`, `MarkdownChunker`, etc.)
    /// remain 100% sequential — concurrency is only at the batch level.
    ///
    /// # Example
    ///
    /// ```rust
    /// use dogma_vdb::smart_chunker::{InputFile, SmartChunker};
    ///
    /// let chunker = SmartChunker::default();
    /// let files = vec![
    ///     InputFile::new("hello.py", "def greet(): pass"),
    ///     InputFile::new("note.md", "# Title\n\nSome text."),
    /// ];
    /// let docs = chunker.chunk_batch(&files);
    /// assert_eq!(docs.len(), 2); // one Python chunk + one markdown chunk
    /// ```
    pub fn chunk_batch(&self, files: &[InputFile]) -> Vec<Document> {
        files
            .par_iter()
            .flat_map(|f| {
                let file_type = FileType::from_path(&f.path);
                let base_id = Path::new(&f.path)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("file");
                let mut extra_meta = f.metadata.clone();
                extra_meta.insert("source".into(), f.path.clone());
                let file_type_name = file_type.name().to_string();
                extra_meta.insert("language".into(), file_type_name);

                let chunks = self.chunk_text(&f.text, file_type);

                chunks
                    .into_iter()
                    .enumerate()
                    .map(move |(i, chunk)| {
                        let mut meta = extra_meta.clone();
                        if let Some(ref s) = chunk.structure {
                            meta.insert("structure".into(), s.clone());
                        }
                        meta.insert("level".into(), chunk.level.to_string());
                        meta.insert("start_line".into(), chunk.start_line.to_string());
                        meta.insert("end_line".into(), chunk.end_line.to_string());

                        Document::builder(format!("{}-{}", base_id, i), chunk.text)
                            .metadatas(meta)
                            .build()
                    })
                    .collect::<Vec<Document>>()
            })
            .collect()
    }
}

impl Default for SmartChunker {
    fn default() -> Self {
        Self::new(SmartChunkerConfig::default())
    }
}

// ---------------------------------------------------------------------------
// Subdivide helper
// ---------------------------------------------------------------------------

/// Subdivide an oversized chunk by lines.
fn subdivide_by_lines(chunk: &SmartChunk, max_size: usize) -> Vec<SmartChunk> {
    let lines: Vec<&str> = chunk.text.lines().collect();
    let total = lines.len();
    if total == 0 {
        return vec![chunk.clone()];
    }
    let avg = chunk.text.len() / total;
    let per = (max_size / avg.max(1)).max(1);

    let mut out = Vec::new();
    let mut i = 0;
    while i < total {
        let end = (i + per).min(total);
        out.push(SmartChunk {
            text: lines[i..end].join("\n"),
            structure: chunk.structure.clone(),
            level: chunk.level,
            start_line: chunk.start_line + i,
            end_line: chunk.start_line + end,
        });
        i = end;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Batch parallel execution --

    #[test]
    fn test_batch_parallel_execution() {
        let chunker = SmartChunker::default();
        let files = vec![
            InputFile::new("a.rs", "fn hello() {}\nfn world() {}"),
            InputFile::new(
                "b.md",
                "# Title\n\nParagraph one.\n\n## Sub\n\nSub text.",
            ),
            InputFile::new("c.py", "def a(): pass\n\ndef b(): pass\n\nclass C: pass"),
            InputFile::new("d.txt", "Just a short text."),
            InputFile::new("e.jsonl", "{\"a\":1}\n{\"b\":2}"),
        ];

        let docs = chunker.chunk_batch(&files);

        // Verify we got the expected chunk count:
        // a.rs: 2 functions = 2 chunks
        // b.md: 2 headers = 2 chunks
        // c.py: 3 functions/classes = 3 chunks
        // d.txt: 1 text = 1 chunk
        // e.jsonl: 2 lines = 2 chunks
        // Total: 10
        assert_eq!(
            docs.len(),
            10,
            "batch should produce exactly 10 chunks across 5 files"
        );

        // Verify each doc has a unique ID and the expected metadata
        let mut ids = std::collections::HashSet::new();
        for doc in &docs {
            // Every ID must be unique
            assert!(
                ids.insert(doc.id.clone()),
                "duplicate document ID: {}",
                doc.id
            );
            // Every doc must have language and source metadata
            assert!(
                doc.metadata.contains_key("language"),
                "doc {} missing 'language'",
                doc.id
            );
            assert!(
                doc.metadata.contains_key("source"),
                "doc {} missing 'source'",
                doc.id
            );
            // start_line and end_line must be present and consistent
            let start: usize = doc
                .metadata
                .get("start_line")
                .and_then(|s| s.parse().ok())
                .unwrap_or(usize::MAX);
            let end: usize = doc
                .metadata
                .get("end_line")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            assert!(
                start < end || (start == 0 && end == 0),
                "doc {} has invalid line range: {start}..{end}",
                doc.id
            );
        }

        // Verify source files are correctly tracked
        let sources: std::collections::HashSet<&str> = docs
            .iter()
            .map(|d| d.metadata.get("source").map(|s| s.as_str()))
            .collect::<Option<std::collections::HashSet<_>>>()
            .unwrap();
        assert!(sources.contains("a.rs"));
        assert!(sources.contains("b.md"));
        assert!(sources.contains("c.py"));
        assert!(sources.contains("d.txt"));
        assert!(sources.contains("e.jsonl"));
    }

    #[test]
    fn test_batch_parallel_empty() {
        let chunker = SmartChunker::default();
        let docs = chunker.chunk_batch(&[]);
        assert!(docs.is_empty());
    }

    #[test]
    fn test_batch_parallel_metadata_inheritance() {
        let chunker = SmartChunker::default();
        let files = vec![InputFile::new("hello.rs", "fn greet() {}")
            .with_metadata("project", "dogma")
            .with_metadata("author", "test")];

        let docs = chunker.chunk_batch(&files);
        assert_eq!(docs.len(), 1);
        assert_eq!(
            docs[0].metadata.get("project").map(|s| s.as_str()),
            Some("dogma")
        );
        assert_eq!(
            docs[0].metadata.get("author").map(|s| s.as_str()),
            Some("test")
        );
    }

    // -- FileType detection --

    #[test]
    fn test_file_type_from_extension() {
        assert_eq!(FileType::from_extension("rs"), FileType::Rust);
        assert_eq!(FileType::from_extension("py"), FileType::Python);
        assert_eq!(FileType::from_extension("js"), FileType::JavaScript);
        assert_eq!(FileType::from_extension("ts"), FileType::JavaScript);
        assert_eq!(FileType::from_extension("go"), FileType::Go);
        assert_eq!(FileType::from_extension("md"), FileType::Markdown);
        assert_eq!(FileType::from_extension("txt"), FileType::Text);
        assert_eq!(FileType::from_extension("jsonl"), FileType::JsonLines);
        assert_eq!(FileType::from_extension("vdb"), FileType::JsonLines);
        assert_eq!(FileType::from_extension("unknown"), FileType::Unknown);
    }

    #[test]
    fn test_file_type_from_path() {
        assert_eq!(FileType::from_path("src/main.rs"), FileType::Rust);
        assert_eq!(FileType::from_path("/path/to/file.py"), FileType::Python);
        assert_eq!(FileType::from_path("README.md"), FileType::Markdown);
        assert_eq!(FileType::from_path("data.jsonl"), FileType::JsonLines);
    }

    #[test]
    fn test_file_type_name() {
        assert_eq!(FileType::Rust.name(), "Rust");
        assert_eq!(FileType::Unknown.name(), "Unknown");
    }

    // -- Rust chunking --

    #[test]
    fn test_chunk_rust_by_functions() {
        let code = "fn hello() {}\n\nfn main() { hello(); }";
        let chunker = SmartChunker::default();
        let chunks = chunker.chunk_text(code, FileType::Rust);
        assert!(chunks.len() >= 2);
        assert_eq!(chunks[0].structure.as_deref(), Some("hello"));
    }

    #[test]
    fn test_chunk_rust_pub_fn() {
        let code = "pub fn a() {}\npub unsafe fn b() {}";
        let chunker = SmartChunker::default();
        let chunks = chunker.chunk_text(code, FileType::Rust);
        let names: Vec<_> = chunks.iter().map(|c| c.structure.as_deref()).collect();
        assert!(names.contains(&Some("a")));
        assert!(names.contains(&Some("b")));
    }

    #[test]
    fn test_chunk_rust_struct_impl() {
        let code = "struct S { x: i32 }\nimpl S { fn m(&self) {} }\nfn f() {}";
        let chunker = SmartChunker::default();
        let chunks = chunker.chunk_text(code, FileType::Rust);
        assert!(chunks.len() >= 3);
    }

    // -- Python --

    #[test]
    fn test_chunk_python_def() {
        let code = "def a(): pass\n\ndef b(): pass";
        let chunker = SmartChunker::default();
        let chunks = chunker.chunk_text(code, FileType::Python);
        assert!(chunks.len() >= 2);
        assert_eq!(chunks[0].structure.as_deref(), Some("a"));
    }

    #[test]
    fn test_chunk_python_async() {
        let code = "async def fetch(): pass\n\nasync def process(): pass";
        let chunker = SmartChunker::default();
        let chunks = chunker.chunk_text(code, FileType::Python);
        assert!(chunks.len() >= 2);
        assert_eq!(chunks[0].structure.as_deref(), Some("fetch"));
    }

    #[test]
    fn test_chunk_python_class() {
        let code = "class Foo:\n    pass\n\ndef top(): pass";
        let chunker = SmartChunker::default();
        let chunks = chunker.chunk_text(code, FileType::Python);
        assert!(chunks.len() >= 2);
        assert_eq!(chunks[0].structure.as_deref(), Some("Foo"));
    }

    // -- JavaScript --

    #[test]
    fn test_chunk_javascript() {
        let code = "function a() {}\nclass B {}\nconst c = () => 42;";
        let chunker = SmartChunker::default();
        let chunks = chunker.chunk_text(code, FileType::JavaScript);
        assert!(chunks.len() >= 2);
        assert_eq!(chunks[0].structure.as_deref(), Some("a"));
    }

    #[test]
    fn test_chunk_exported() {
        let code = "export function greet() {}\nexport class Svc {}";
        let chunker = SmartChunker::default();
        let chunks = chunker.chunk_text(code, FileType::JavaScript);
        assert!(chunks.len() >= 2);
    }

    // -- Go --

    #[test]
    fn test_chunk_go() {
        let code = "func a() {}\nfunc b() {}";
        let chunker = SmartChunker::default();
        let chunks = chunker.chunk_text(code, FileType::Go);
        assert!(chunks.len() >= 2);
        assert_eq!(chunks[0].structure.as_deref(), Some("a"));
    }

    // -- Markdown --

    #[test]
    fn test_chunk_markdown() {
        let md = "# T\n\nx\n\n## S1\n\ny\n\n### S2\n\nz";
        let chunker = SmartChunker::default();
        let chunks = chunker.chunk_text(md, FileType::Markdown);
        assert!(chunks.len() >= 3);
        assert_eq!(chunks[0].structure.as_deref(), Some("T"));
        assert_eq!(chunks[0].level, 1);
    }

    #[test]
    fn test_chunk_markdown_no_headings() {
        // Text longer than max_chunk_size → should be split by paragraphs
        let md = "a\n\nb\n\nc\n\nd\n\ne\n\nf\n\ng\n\nh\n\ni\n\nj";
        let chunker = SmartChunker::new(SmartChunkerConfig {
            max_chunk_size: 10,
            overlap: 0,
        });
        let chunks = chunker.chunk_text(md, FileType::Markdown);
        assert!(
            chunks.len() >= 2,
            "got {} chunks for long markdown",
            chunks.len()
        );
    }

    // -- JSONL --

    #[test]
    fn test_chunk_jsonl() {
        let data = "{\"id\":\"a\"}\n{\"id\":\"b\"}\n{\"id\":\"c\"}\n";
        let chunker = SmartChunker::default();
        let chunks = chunker.chunk_text(data, FileType::JsonLines);
        assert_eq!(chunks.len(), 3);
    }

    #[test]
    fn test_chunk_jsonl_skip_empty() {
        let data = "{\"id\":\"a\"}\n\n{\"id\":\"b\"}";
        let chunker = SmartChunker::default();
        let chunks = chunker.chunk_text(data, FileType::JsonLines);
        assert_eq!(chunks.len(), 2);
    }

    // -- Text / Unknown fallback --

    #[test]
    fn test_chunk_text_paragraphs() {
        let chunker = SmartChunker::default();
        assert_eq!(chunker.chunk_text("short", FileType::Text).len(), 1);
        let long = "A".repeat(5000);
        assert!(chunker.chunk_text(&long, FileType::Text).len() >= 4);
    }

    #[test]
    fn test_unknown_falls_back() {
        let chunker = SmartChunker::default();
        let chunks = chunker.chunk_text("a\n\nb\n\nc", FileType::Unknown);
        assert!(chunks.len() >= 1);
    }

    // -- chunke_file auto-detect --

    #[test]
    fn test_chunk_file_auto() {
        let chunker = SmartChunker::default();
        let chunks = chunker.chunk_file("lib.rs", "fn a() {}\nfn b() {}");
        assert!(chunks.len() >= 2);
    }

    // -- chunk_to_docs --

    #[test]
    fn test_chunk_to_docs_metadata() {
        let md = "# T\n\nx\n\n## S\n\ny";
        let chunker = SmartChunker::default();
        let mut extra = HashMap::new();
        extra.insert("src".into(), "test.md".into());

        let docs = chunker.chunk_to_docs("doc.md", md, "doc", extra);
        assert!(docs.len() >= 2);
        for d in &docs {
            assert!(d.id.starts_with("doc-"));
            assert_eq!(d.metadata_val("src"), Some("test.md"));
            assert_eq!(d.metadata_val("language"), Some("Markdown"));
        }
        assert_eq!(docs[0].metadata_val("structure"), Some("T"));
    }

    // -- Empty text --

    #[test]
    fn test_empty_text_all_types() {
        let chunker = SmartChunker::default();
        for &ft in &[
            FileType::Rust,
            FileType::Python,
            FileType::Markdown,
            FileType::Text,
            FileType::JsonLines,
            FileType::Unknown,
        ] {
            let chunks = chunker.chunk_text("", ft);
            assert_eq!(chunks.len(), 1, "empty for {:?}", ft);
            assert!(chunks[0].text.is_empty());
        }
    }

    // -- Subdivide --

    #[test]
    fn test_subdivide_large_chunks() {
        let code = format!("fn huge() {{\n{}\n}}", "    let x = 1;\n".repeat(200));
        let chunker = SmartChunker::new(SmartChunkerConfig {
            max_chunk_size: 200,
            overlap: 0,
        });
        let chunks = chunker.chunk_text(&code, FileType::Rust);
        assert!(chunks.len() > 1);
        for c in &chunks {
            assert!(c.text.len() <= 300, "oversized: {}", c.text.len());
        }
    }

    // -- Line tracking --

    #[test]
    fn test_line_numbers_monotonic() {
        let code = "fn a() {}\nfn b() {}\nfn c() {}";
        let chunker = SmartChunker::default();
        let chunks = chunker.chunk_text(code, FileType::Rust);
        for c in &chunks {
            assert!(
                c.start_line <= c.end_line,
                "{} > {}",
                c.start_line,
                c.end_line
            );
        }
        for pair in chunks.windows(2) {
            assert!(pair[0].end_line <= pair[1].start_line, "gap/overlap");
        }
    }

    // -- Semantic chunker (integration) --

    #[test]
    fn test_semantic_fallback_when_not_configured() {
        let chunker = SmartChunker::default();
        // Without embedder, Semantic falls back to paragraph chunking
        let chunks = chunker.chunk_text("One.\n\nTwo.", FileType::Semantic);
        assert!(!chunks.is_empty());
        assert!(chunks[0].structure.is_none());
    }

    #[test]
    fn test_semantic_file_type_name() {
        assert_eq!(FileType::Semantic.name(), "Semantic");
        // Semantic is never auto-detected from extension
        assert_eq!(FileType::from_extension("txt"), FileType::Text);
    }

    #[test]
    fn test_semantic_empty_text() {
        let chunker = SmartChunker::default();
        let chunks = chunker.chunk_text("", FileType::Semantic);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].text.is_empty());
    }

    // -- Syntax chunker fallback (works with or without feature flag) --

    #[test]
    fn test_syntax_fallback_regex() {
        // Without chunker-syntax, or if syntax returns empty, regex kicks in.
        let chunker = SmartChunker::default();
        let code = "fn a() {}\nfn b() {}";
        let chunks = chunker.chunk_text(code, FileType::Rust);
        assert!(chunks.len() >= 2);
        assert_eq!(chunks[0].structure.as_deref(), Some("a"));
    }
}
