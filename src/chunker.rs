//! Text chunker for splitting long documents.

use crate::doc::Document;
use std::collections::HashMap;

/// Trait for text chunkers that split documents into smaller pieces.
///
/// Implement this trait to provide custom chunking logic for the RAG pipeline.
/// The built-in [`SmartChunker`] selects the best strategy automatically
/// based on file type (code, prose, or generic).
pub trait Chunker: Send + Sync {
    /// Chunk a file's content into individual documents.
    ///
    /// * `path` — file path (used for type detection in SmartChunker)
    /// * `text` — raw file content
    fn chunk(&self, path: &str, text: &str) -> Vec<Document>;
}

/// Configuration for the [`TextSplitter`].
#[derive(Debug, Clone)]
pub struct TextSplitterConfig {
    /// Maximum chunk length in characters.
    pub chunk_size: usize,
    /// Overlap between consecutive chunks (characters).
    pub overlap: usize,
    /// Natural separator to break at (e.g. `"\n\n"`, `". "`).
    pub separator: String,
}

impl Default for TextSplitterConfig {
    fn default() -> Self {
        Self {
            chunk_size: 512,
            overlap: 64,
            separator: "\n\n".into(),
        }
    }
}

/// Splits long texts into overlapping chunks.
///
/// # Example
///
/// ```
/// use dogma_vdb::chunker::{TextSplitter, TextSplitterConfig};
///
/// let chunker = TextSplitter::default();
/// let chunks = chunker.chunk("A very long text...");
/// ```
#[derive(Debug, Clone)]
pub struct TextSplitter {
    config: TextSplitterConfig,
}

impl TextSplitter {
    pub fn new(config: TextSplitterConfig) -> Self {
        Self { config }
    }

    /// Split `text` into chunks according to the configuration.
    ///
    /// The algorithm prefers breaking at `separator` boundaries near
    /// `chunk_size`, falling back to character-level splitting.
    pub fn chunk(&self, text: &str) -> Vec<String> {
        if text.is_empty() || text.len() <= self.config.chunk_size {
            return vec![text.to_string()];
        }

        let mut chunks = Vec::new();
        let mut start = 0;
        let text_len = text.len();
        let sep = &self.config.separator;

        while start < text_len {
            if start + self.config.chunk_size >= text_len {
                chunks.push(text[start..].to_string());
                break;
            }

            let ideal_end = start + self.config.chunk_size;
            // Look for the separator in the candidate chunk [start..ideal_end]
            let end = if let Some(sep_pos) = text[start..ideal_end].rfind(sep) {
                // Split right after the separator, but only if it's not
                // at the very beginning (which would mean no progress).
                let break_at = start + sep_pos + sep.len();
                // Ensure we make progress: break_at must be > start
                if break_at > start {
                    break_at
                } else {
                    ideal_end
                }
            } else {
                ideal_end
            };

            chunks.push(text[start..end].to_string());
            start = end.saturating_sub(self.config.overlap);

            // Prevent infinite loop if overlap >= chunk_size
            if start >= end {
                start = end;
            }
        }

        chunks
    }

    /// Split text and wrap each chunk into a [`Document`].
    ///
    /// Each document gets an id of the form `{base_id}-{index}`.
    pub fn chunk_to_docs(
        &self,
        text: &str,
        base_id: &str,
        metadata: HashMap<String, String>,
    ) -> Vec<Document> {
        let chunks = self.chunk(text);
        chunks
            .into_iter()
            .enumerate()
            .map(|(i, chunk)| {
                Document::builder(format!("{}-{}", base_id, i), chunk)
                    .metadatas(metadata.clone())
                    .build()
            })
            .collect()
    }
}

impl Default for TextSplitter {
    fn default() -> Self {
        Self::new(TextSplitterConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_short_text() {
        let chunker = TextSplitter::default();
        let chunks = chunker.chunk("corto");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "corto");
    }

    #[test]
    fn test_chunk_long_text() {
        let text = "A".repeat(2000);
        let chunker = TextSplitter::new(TextSplitterConfig {
            chunk_size: 500,
            overlap: 50,
            separator: "\n".into(),
        });
        let chunks = chunker.chunk(&text);
        assert!(chunks.len() >= 4);
        // Each chunk should be at most chunk_size
        for chunk in &chunks {
            assert!(chunk.len() <= 500);
        }
    }

    #[test]
    fn test_chunk_exactly_at_boundary() {
        let text = "hello\n\nworld\n\nfoo";
        let chunker = TextSplitter::new(TextSplitterConfig {
            chunk_size: 12, // "hello\n\nworld" is 12 chars
            overlap: 0,
            separator: "\n\n".into(),
        });
        let chunks = chunker.chunk(text);
        // With 12 chars, "hello\n\n" is 7 chars, then we look at next portion
        assert!(chunks.len() >= 2);
        // First chunk should be up to a separator boundary near the chunk_size
        let clean: String = chunks.join("");
        // Let's just check that content is preserved:
        assert!(clean.contains("hello"));
        assert!(clean.contains("world"));
        assert!(clean.contains("foo"));
    }

    #[test]
    fn test_chunk_empty_text() {
        let chunker = TextSplitter::default();
        let chunks = chunker.chunk("");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "");
    }

    #[test]
    fn test_chunk_with_overlap() {
        let text = "abcdefghijklmnopqrstuvwxyz";
        let chunker = TextSplitter::new(TextSplitterConfig {
            chunk_size: 10,
            overlap: 4,
            separator: "\n".into(),
        });
        let chunks = chunker.chunk(text);
        // With overlap, chunks should share characters
        assert!(chunks.len() >= 3);
        if chunks.len() >= 2 {
            // Check overlap: end of chunk0 should appear in chunk1
            let overlap_start = chunks[0].len() - 4;
            let overlap_part = &chunks[0][overlap_start..];
            assert!(
                chunks[1].contains(overlap_part),
                "expected overlap '{}' in '{}'",
                overlap_part,
                chunks[1]
            );
        }
    }

    #[test]
    fn test_chunk_separator_priority() {
        let text = "AAA\n\nBBB\n\nCCC";
        let chunker = TextSplitter::new(TextSplitterConfig {
            chunk_size: 10,
            overlap: 0,
            separator: "\n\n".into(),
        });
        let chunks = chunker.chunk(text);
        // Should split at \n\n boundaries — the result should be at least 2 chunks
        assert!(chunks.len() >= 2);
        // All content should be preserved across chunks
        let all_text: String = chunks.join("");
        assert!(all_text.contains("AAA"));
        assert!(all_text.contains("BBB"));
        assert!(all_text.contains("CCC"));
    }

    #[test]
    fn test_chunk_to_docs() {
        let chunker = TextSplitter::new(TextSplitterConfig {
            chunk_size: 10,
            overlap: 0,
            separator: "\n".into(),
        });
        let text = "hello world foo bar";
        let mut meta = HashMap::new();
        meta.insert("source".into(), "test".into());

        let docs = chunker.chunk_to_docs(text, "doc", meta);
        assert!(!docs.is_empty());
        for doc in &docs {
            assert!(doc.id.starts_with("doc-"));
            assert_eq!(doc.metadata_val("source"), Some("test"));
        }
    }

    #[test]
    fn test_chunk_to_docs_with_empty_text() {
        let chunker = TextSplitter::default();
        let docs = chunker.chunk_to_docs("", "empty", HashMap::new());
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].id, "empty-0");
        assert_eq!(docs[0].text, "");
    }
}
