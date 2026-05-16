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

use crate::doc::Document;
use std::collections::HashMap;
use std::path::Path;

pub use code::CodeChunker;
pub use jsonl::JsonLinesChunker;
pub use markdown::MarkdownChunker;
pub use paragraph::ParagraphChunker;

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
        }
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
            FileType::Rust => self.rust.chunk(text, max),
            FileType::Python => self.python.chunk(text, max),
            FileType::JavaScript => self.javascript.chunk(text, max),
            FileType::Go => self.go.chunk(text, max),
            FileType::Markdown => self.markdown.chunk(text, max),
            FileType::Text => self.text.chunk(text, max),
            FileType::JsonLines => self.jsonl.chunk(text, max),
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
}
