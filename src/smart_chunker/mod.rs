//! Smart chunker — detects file type and selects among 3 strategies.
//!
//! | Strategy | Best for |
//! |----------|----------|
//! | [`ChunkStrategy::Code`] | Source files (Rust, Python, JS, Go). Detects functions, classes, structs. |
//! | [`ChunkStrategy::Paragraph`] | Prose, essays, books. Splits by `\n\n` with overlap. Optional semantic (embedding) refinement. |
//! | [`ChunkStrategy::FixedWindow`] | Everything else. Fixed-size byte windows with UTF-8 safety. Replaces markdown, JSONL, plain text. |
//!
//! ## Design
//!
//! * All regex patterns compiled **once** in the constructor.
//! * Pure sequential dispatch — concurrency only at the batch level.
//! * Every string slice is guarded by `.is_char_boundary()` — no UTF-8 panics.
//! * No `Box<dyn>` — all strategies are concrete fields.

mod code;
mod fixed_window;
mod paragraph;

use crate::doc::Document;
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::Path;

pub use code::CodeChunker;
pub use fixed_window::FixedWindowChunker;
pub use paragraph::ParagraphChunker;

// ---------------------------------------------------------------------------
// ChunkStrategy — 3 variants only
// ---------------------------------------------------------------------------

/// The chunking strategy to apply to a file.
///
/// Auto-detected from the file extension (see [`ChunkStrategy::from_path`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChunkStrategy {
    /// Code files (`.rs`, `.py`, `.js`, `.ts`, `.go`, …).  
    /// Splits by top-level definitions via regex.
    Code,
    /// Prose / text.  Splits by `\n\n` with configurable overlap.
    /// When coupled with an `Embedder`, uses semantic similarity.
    Paragraph,
    /// Generic fixed-size byte windows with UTF-8 safety and overlap.
    /// Used for everything else (markdown, JSONL, unknown extensions).
    FixedWindow,
}

impl ChunkStrategy {
    /// Detect strategy from a file path.
    pub fn from_path(path: impl AsRef<Path>) -> Self {
        path.as_ref()
            .extension()
            .and_then(|e| e.to_str())
            .map(Self::from_extension)
            .unwrap_or(ChunkStrategy::FixedWindow)
    }

    /// Detect strategy from an extension string (without leading dot).
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_lowercase().as_str() {
            "rs" | "py" | "js" | "jsx" | "ts" | "tsx" | "mjs" | "cjs" | "go" => ChunkStrategy::Code,
            "txt" | "text" | "md" | "markdown" | "jsonl" | "vdb" | "ndjson" | "json" | "yaml"
            | "toml" | "sh" => ChunkStrategy::FixedWindow,
            _ => ChunkStrategy::FixedWindow,
        }
    }

    /// Human-readable name.
    pub fn name(self) -> &'static str {
        match self {
            ChunkStrategy::Code => "Code",
            ChunkStrategy::Paragraph => "Paragraph",
            ChunkStrategy::FixedWindow => "FixedWindow",
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
    /// Name of the enclosing structure (`"fn main"`, `"class Foo"`).
    pub structure: Option<String>,
    /// Hierarchical level (0 = top-level).
    pub level: usize,
    /// 0-indexed start line in the original file.
    pub start_line: usize,
    /// 0-indexed end line (exclusive).
    pub end_line: usize,
}

/// A file to process in a batch chunking operation.
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
    /// Overlap for paragraph and fixed-window chunking (characters).
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

/// File-type–aware chunker with 3 strategies.
///
/// ```rust
/// use dogma_vdb::smart_chunker::{SmartChunker, ChunkStrategy};
///
/// let chunker = SmartChunker::default();
/// let chunks = chunker.chunk_file("main.rs", "fn hello() {}");
/// assert_eq!(chunks[0].structure.as_deref(), Some("hello"));
/// ```
pub struct SmartChunker {
    config: SmartChunkerConfig,
    rust: CodeChunker,
    python: CodeChunker,
    javascript: CodeChunker,
    go: CodeChunker,
    text: ParagraphChunker,
    window: FixedWindowChunker,
    /// Embedding provider for semantic chunking.
    embedder: Option<Box<dyn crate::embedding::Embedder>>,
}

impl std::fmt::Debug for SmartChunker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SmartChunker")
            .field("config", &self.config)
            .field("rust", &self.rust)
            .field("python", &self.python)
            .field("javascript", &self.javascript)
            .field("go", &self.go)
            .field("text", &self.text)
            .field("window", &self.window)
            .field("embedder", &self.embedder.as_ref().map(|_| ".."))
            .finish()
    }
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
            text: ParagraphChunker::new(overlap),
            window: FixedWindowChunker::new(overlap),
            embedder: None,
        }
    }

    /// Enable semantic chunking with an [`Embedder`](crate::embedding::Embedder).
    ///
    /// When set, `ChunkStrategy::Paragraph` uses embedding similarity in
    /// addition to paragraph boundaries.
    pub fn with_semantic(mut self, embedder: Box<dyn crate::embedding::Embedder>) -> Self {
        self.embedder = Some(embedder);
        self
    }

    /// Chunk a file, auto-detecting the strategy from its path.
    pub fn chunk_file(&self, path: impl AsRef<Path>, text: &str) -> Vec<SmartChunk> {
        let strategy = ChunkStrategy::from_path(path);
        self.chunk_text(text, strategy)
    }

    /// Chunk text with an explicit [`ChunkStrategy`].
    pub fn chunk_text(&self, text: &str, strategy: ChunkStrategy) -> Vec<SmartChunk> {
        let max = self.config.max_chunk_size;
        match strategy {
            ChunkStrategy::Code => {
                // Try each code language by detecting from content heuristics.
                // We default to Rust-like if we can't determine.
                let lines: Vec<&str> = text.lines().collect();
                let first = lines.first().copied().unwrap_or("");
                if first.contains("def ") || first.contains("class ") || text.contains("def ") {
                    self.python.chunk(text, max)
                } else if first.contains("function")
                    || first.contains("=>")
                    || first.contains("interface ")
                {
                    self.javascript.chunk(text, max)
                } else if first.contains("func ") || text.contains("func ") {
                    self.go.chunk(text, max)
                } else {
                    self.rust.chunk(text, max)
                }
            }
            ChunkStrategy::Paragraph => {
                // Use semantic chunking if embedder is available
                if let Some(ref emb) = self.embedder {
                    self.text.chunk_semantic(text, max, emb.as_ref())
                } else {
                    self.text.chunk(text, max)
                }
            }
            ChunkStrategy::FixedWindow => self.window.chunk(text, max),
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
        let strategy = ChunkStrategy::from_path(&path);
        let chunks = self.chunk_text(text, strategy);

        extra_meta.insert("language".into(), strategy.name().into());

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
    pub fn chunk_batch(&self, files: &[InputFile]) -> Vec<Document> {
        let batch_size = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        files
            .par_chunks(batch_size)
            .flat_map(|chunk| {
                if let Err(e) = crate::memory::ensure_memory() {
                    log::warn!("MemoryGuard: chunk_batch skipped — {e}");
                    return Vec::new();
                }
                let mut batch_docs = Vec::with_capacity(chunk.len() * 4);
                for f in chunk {
                    let strategy = ChunkStrategy::from_path(&f.path);
                    let base_id = Path::new(&f.path)
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("file");
                    let mut extra_meta = f.metadata.clone();
                    extra_meta.insert("source".into(), f.path.clone());
                    extra_meta.insert("language".into(), strategy.name().into());

                    let chunks = self.chunk_text(&f.text, strategy);

                    for (i, chunk) in chunks.into_iter().enumerate() {
                        let mut meta = extra_meta.clone();
                        if let Some(ref s) = chunk.structure {
                            meta.insert("structure".into(), s.clone());
                        }
                        meta.insert("level".into(), chunk.level.to_string());
                        meta.insert("start_line".into(), chunk.start_line.to_string());
                        meta.insert("end_line".into(), chunk.end_line.to_string());

                        batch_docs.push(
                            Document::builder(format!("{}-{}", base_id, i), chunk.text)
                                .metadatas(meta)
                                .build(),
                        );
                    }
                }
                batch_docs
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Batch parallel execution --

    #[test]
    fn test_batch_parallel_execution() {
        let chunker = SmartChunker::default();
        let files = vec![
            InputFile::new("a.rs", "fn hello() {}\nfn world() {}"),
            InputFile::new("b.md", "# Title\n\nParagraph one.\n\n## Sub\n\nSub text."),
            InputFile::new("c.py", "def a(): pass\n\ndef b(): pass\n\nclass C: pass"),
            InputFile::new("d.txt", "Just a short text."),
            InputFile::new("e.jsonl", "{\"a\":1}\n{\"b\":2}"),
        ];

        let docs = chunker.chunk_batch(&files);

        // a.rs: 2 functions = 2 chunks
        // b.md: FixedWindow (no headings detected) → 1 or more chunks
        // c.py: 3 functions/classes = 3 chunks
        // d.txt: FixedWindow → 1 chunk
        // e.jsonl: FixedWindow → 1 chunk
        // Total: ~8+
        assert!(
            docs.len() >= 7,
            "batch should produce at least 7 chunks across 5 files, got {}",
            docs.len()
        );

        let mut ids = std::collections::HashSet::new();
        for doc in &docs {
            assert!(
                ids.insert(doc.id.clone()),
                "duplicate document ID: {}",
                doc.id
            );
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

    // -- ChunkStrategy detection --

    #[test]
    fn test_chunk_strategy_from_extension() {
        assert_eq!(ChunkStrategy::from_extension("rs"), ChunkStrategy::Code);
        assert_eq!(ChunkStrategy::from_extension("py"), ChunkStrategy::Code);
        assert_eq!(ChunkStrategy::from_extension("js"), ChunkStrategy::Code);
        assert_eq!(ChunkStrategy::from_extension("ts"), ChunkStrategy::Code);
        assert_eq!(ChunkStrategy::from_extension("go"), ChunkStrategy::Code);
        assert_eq!(
            ChunkStrategy::from_extension("md"),
            ChunkStrategy::FixedWindow
        );
        assert_eq!(
            ChunkStrategy::from_extension("txt"),
            ChunkStrategy::FixedWindow
        );
        assert_eq!(
            ChunkStrategy::from_extension("jsonl"),
            ChunkStrategy::FixedWindow
        );
        assert_eq!(
            ChunkStrategy::from_extension("unknown"),
            ChunkStrategy::FixedWindow
        );
    }

    #[test]
    fn test_chunk_strategy_from_path() {
        assert_eq!(ChunkStrategy::from_path("src/main.rs"), ChunkStrategy::Code);
        assert_eq!(
            ChunkStrategy::from_path("/path/to/file.py"),
            ChunkStrategy::Code
        );
        assert_eq!(
            ChunkStrategy::from_path("README.md"),
            ChunkStrategy::FixedWindow
        );
        assert_eq!(
            ChunkStrategy::from_path("data.jsonl"),
            ChunkStrategy::FixedWindow
        );
    }

    #[test]
    fn test_chunk_strategy_name() {
        assert_eq!(ChunkStrategy::Code.name(), "Code");
        assert_eq!(ChunkStrategy::FixedWindow.name(), "FixedWindow");
    }

    // -- Code chunking --

    #[test]
    fn test_chunk_rust_by_functions() {
        let code = "fn hello() {}\n\nfn main() { hello(); }";
        let chunker = SmartChunker::default();
        let chunks = chunker.chunk_text(code, ChunkStrategy::Code);
        assert!(chunks.len() >= 2);
        assert_eq!(chunks[0].structure.as_deref(), Some("hello"));
    }

    #[test]
    fn test_chunk_rust_pub_fn() {
        let code = "pub fn a() {}\npub unsafe fn b() {}";
        let chunker = SmartChunker::default();
        let chunks = chunker.chunk_text(code, ChunkStrategy::Code);
        let names: Vec<_> = chunks.iter().map(|c| c.structure.as_deref()).collect();
        assert!(names.contains(&Some("a")));
        assert!(names.contains(&Some("b")));
    }

    #[test]
    fn test_chunk_rust_struct_impl() {
        let code = "struct S { x: i32 }\nimpl S { fn m(&self) {} }\nfn f() {}";
        let chunker = SmartChunker::default();
        let chunks = chunker.chunk_text(code, ChunkStrategy::Code);
        assert!(chunks.len() >= 3);
    }

    #[test]
    fn test_chunk_python_def() {
        let code = "def a(): pass\n\ndef b(): pass";
        let chunker = SmartChunker::default();
        let chunks = chunker.chunk_text(code, ChunkStrategy::Code);
        assert!(chunks.len() >= 2);
        assert_eq!(chunks[0].structure.as_deref(), Some("a"));
    }

    #[test]
    fn test_chunk_python_async() {
        let code = "async def fetch(): pass\n\nasync def process(): pass";
        let chunker = SmartChunker::default();
        let chunks = chunker.chunk_text(code, ChunkStrategy::Code);
        assert!(chunks.len() >= 2);
        assert_eq!(chunks[0].structure.as_deref(), Some("fetch"));
    }

    #[test]
    fn test_chunk_python_class() {
        let code = "class Foo:\n    pass\n\ndef top(): pass";
        let chunker = SmartChunker::default();
        let chunks = chunker.chunk_text(code, ChunkStrategy::Code);
        assert!(chunks.len() >= 2);
        assert_eq!(chunks[0].structure.as_deref(), Some("Foo"));
    }

    #[test]
    fn test_chunk_javascript() {
        let code = "function a() {}\nclass B {}\nconst c = () => 42;";
        let chunker = SmartChunker::default();
        let chunks = chunker.chunk_text(code, ChunkStrategy::Code);
        assert!(chunks.len() >= 2);
        assert_eq!(chunks[0].structure.as_deref(), Some("a"));
    }

    #[test]
    fn test_chunk_go() {
        let code = "func a() {}\nfunc b() {}";
        let chunker = SmartChunker::default();
        let chunks = chunker.chunk_text(code, ChunkStrategy::Code);
        assert!(chunks.len() >= 2);
        assert_eq!(chunks[0].structure.as_deref(), Some("a"));
    }

    // -- Paragraph chunking --

    #[test]
    fn test_chunk_paragraphs() {
        let text = "para one\n\npara two\n\npara three\n\npara four";
        let chunker = SmartChunker::new(SmartChunkerConfig {
            max_chunk_size: 20,
            overlap: 0,
        });
        let chunks = chunker.chunk_text(text, ChunkStrategy::Paragraph);
        assert!(chunks.len() >= 2);
    }

    #[test]
    fn test_chunk_paragraph_empty() {
        let chunker = SmartChunker::default();
        let chunks = chunker.chunk_text("", ChunkStrategy::Paragraph);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].text.is_empty());
    }

    // -- FixedWindow chunking --

    #[test]
    fn test_chunk_fixed_window() {
        let text = "line A\nline B\nline C\nline D\nline E\nline F\nline G\nline H\nline I\nline J";
        let chunker = SmartChunker::new(SmartChunkerConfig {
            max_chunk_size: 20,
            overlap: 0,
        });
        let chunks = chunker.chunk_text(text, ChunkStrategy::FixedWindow);
        assert!(
            chunks.len() >= 2,
            "got {} chunks for long text",
            chunks.len()
        );
    }

    #[test]
    fn test_chunk_fixed_window_respects_char_boundaries() {
        // Multi-byte UTF-8: each "ñ" is 2 bytes
        let text = "ññññññññññññññññññññññññññññññññ"; // 32 × 2 = 64 bytes
        let chunker = SmartChunker::new(SmartChunkerConfig {
            max_chunk_size: 10,
            overlap: 0,
        });
        // Should not panic
        let chunks = chunker.chunk_text(text, ChunkStrategy::FixedWindow);
        assert!(!chunks.is_empty());
        // Every chunk should have valid UTF-8 text that doesn't start/end mid-char
        for chunk in &chunks {
            assert!(!chunk.text.is_empty());
            // All "ñ" characters: length must be even (each ñ = 2 bytes)
            assert_eq!(
                chunk.text.len() % 2,
                0,
                "chunk text has odd byte length = broken UTF-8"
            );
        }
    }

    #[test]
    fn test_chunk_fixed_window_subdivide() {
        let text = "word\n".repeat(500);
        let chunker = SmartChunker::new(SmartChunkerConfig {
            max_chunk_size: 50,
            overlap: 0,
        });
        let chunks = chunker.chunk_text(&text, ChunkStrategy::FixedWindow);
        assert!(chunks.len() > 1);
    }
}
