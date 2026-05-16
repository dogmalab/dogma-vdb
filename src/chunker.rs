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

impl Default for ChunkerConfig;

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
    pub fn new(config: ChunkerConfig) -> Self;
    pub fn chunk(&self, text: &str) -> Vec<String>;
    pub fn chunk_to_docs(
        &self,
        text: &str,
        base_id: &str,
        metadata: HashMap<String, String>,
    ) -> Vec<Document>;
}

impl Default for Chunker;

#[cfg(test)]
mod tests;
