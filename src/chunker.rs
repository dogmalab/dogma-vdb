//! Text chunker for splitting long documents.

use crate::doc::Document;
use std::collections::HashMap;

/// Configuration for the [`Chunker`].
#[derive(Debug, Clone)]
pub struct ChunkerConfig {
    /// Maximum chunk length in characters.
    pub chunk_size: usize,
    /// Overlap between consecutive chunks (characters).
    pub overlap: usize,
    /// Natural separator to break at (e.g. `"\n\n"`, `". "`).
    pub separator: String,
}

impl Default for ChunkerConfig {
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
/// use dogma_vdb::chunker::{Chunker, ChunkerConfig};
///
/// let chunker = Chunker::default();
/// let chunks = chunker.chunk("A very long text...");
/// ```
#[derive(Debug, Clone)]
pub struct Chunker {
    config: ChunkerConfig,
}

impl Chunker {
    pub fn new(config: ChunkerConfig) -> Self {
        todo!()
    }

    pub fn chunk(&self, text: &str) -> Vec<String> {
        let _ = text;
        todo!()
    }

    pub fn chunk_to_docs(
        &self,
        text: &str,
        base_id: &str,
        metadata: HashMap<String, String>,
    ) -> Vec<Document> {
        let _ = (text, base_id, metadata);
        todo!()
    }
}

impl Default for Chunker {
    fn default() -> Self {
        Self::new(ChunkerConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_short_text() {
        let chunker = Chunker::default();
        let chunks = chunker.chunk("corto");
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn test_chunk_long_text() {
        let text = "A".repeat(2000);
        let chunker = Chunker::new(ChunkerConfig {
            chunk_size: 500,
            overlap: 50,
            separator: "\n".into(),
        });
        let chunks = chunker.chunk(&text);
        assert!(chunks.len() >= 4);
    }
}
