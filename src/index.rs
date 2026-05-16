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
    pub fn new(metric: Metric) -> Self {
        Self {
            documents: Vec::new(),
            metric,
        }
    }

    pub fn with_documents(docs: Vec<Document>, metric: Metric) -> Self {
        Self {
            documents: docs,
            metric,
        }
    }

    pub fn metric(&self) -> Metric {
        todo!()
    }

    pub fn documents(&self) -> &[Document] {
        &self.documents
    }
}

impl Index for BruteForceIndex {
    fn insert(&mut self, docs: &[Document]) {
        let _ = docs;
        todo!()
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<ScoredDocument> {
        let _ = (query, k);
        todo!()
    }

    fn len(&self) -> usize {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bruteforce_basic() {
        let docs = vec![
            Document::new("a", "gato"),
            Document::new("b", "perro"),
        ];
        let index = BruteForceIndex::with_documents(docs, Metric::Cosine);
        let query = vec![1.0, 0.0, 0.0];
        let results = index.search(&query, 2);
        assert_eq!(results.len(), 2);
    }
}
