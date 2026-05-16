//! Vector index trait and brute‑force implementation.

use crate::distance::Metric;
use crate::doc::Document;
use crate::error::Result;

/// A scored search result.
#[derive(Debug, Clone)]
pub struct ScoredDocument {
    pub score: f32,
    pub document: Document,
}

/// Trait for vector indexes.
///
/// The default implementation is [`BruteForceIndex`], which performs
/// an O(n·d) linear scan.
pub trait Index: Send + Sync {
    /// Insert documents into the index.
    fn insert(&mut self, docs: &[Document]);

    /// Search for the `k` nearest neighbours of `query`.
    fn search(&self, query: &[f32], k: usize) -> Vec<ScoredDocument>;

    /// Number of indexed documents.
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Brute‑force (linear scan) index.
///
/// Correct, simple, and fast enough for up to ~10 000 documents.
#[derive(Debug, Clone)]
pub struct BruteForceIndex {
    documents: Vec<Document>,
    metric: Metric,
}

impl BruteForceIndex {
    pub fn new(metric: Metric) -> Self;
    pub fn with_documents(docs: Vec<Document>, metric: Metric) -> Self;
    pub fn metric(&self) -> Metric;
    pub fn documents(&self) -> &[Document];
}

impl Index for BruteForceIndex;

#[cfg(test)]
mod tests;
