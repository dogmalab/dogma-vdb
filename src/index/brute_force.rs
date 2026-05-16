//! Brute‑force (linear scan) index.
//!
//! Correct, simple, and fast enough for up to ~10 000 documents.

use crate::distance::Metric;
use crate::doc::Document;
use crate::index::{Index, ScoredDocument};

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
        self.metric
    }

    pub fn documents(&self) -> &[Document] {
        &self.documents
    }
}

impl Index for BruteForceIndex {
    fn insert(&mut self, docs: &[Document]) {
        self.documents.extend_from_slice(docs);
    }

    fn documents(&self) -> &[Document] {
        &self.documents
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<ScoredDocument> {
        if self.documents.is_empty() || k == 0 {
            return Vec::new();
        }

        let mut results: Vec<ScoredDocument> = self
            .documents
            .iter()
            .filter(|d| !d.embedding.is_empty())
            .map(|d| {
                let score = crate::distance::score(&d.embedding, query, self.metric);
                ScoredDocument {
                    score,
                    document: d.clone(),
                }
            })
            .collect();

        // Sort by score descending (higher = more similar)
        results.sort_unstable_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(k);
        results
    }

    fn delete(&mut self, ids: &[&str]) -> usize {
        let before = self.documents.len();
        self.documents.retain(|d| !ids.contains(&d.id.as_str()));
        before - self.documents.len()
    }

    fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        filter: &dyn Fn(&Document) -> bool,
    ) -> Vec<ScoredDocument> {
        if self.documents.is_empty() || k == 0 {
            return Vec::new();
        }

        let mut results: Vec<ScoredDocument> = self
            .documents
            .iter()
            .filter(|d| !d.embedding.is_empty())
            .filter(|d| filter(d))
            .map(|d| {
                let score = crate::distance::score(&d.embedding, query, self.metric);
                ScoredDocument {
                    score,
                    document: d.clone(),
                }
            })
            .collect();

        results.sort_unstable_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(k);
        results
    }

    fn len(&self) -> usize {
        self.documents.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_doc(id: &str, embedding: Vec<f32>) -> Document {
        Document::builder(id, id).embedding(embedding).build()
    }

    #[test]
    fn test_bruteforce_basic() {
        let docs = vec![make_doc("a", vec![1.0, 0.0, 0.0])];
        let index = BruteForceIndex::with_documents(docs, Metric::Cosine);
        let query = vec![1.0, 0.0, 0.0];
        let results = index.search(&query, 1);
        assert_eq!(results.len(), 1);
        assert!((results[0].score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_search_returns_top_k() {
        let docs = vec![
            make_doc("a", vec![1.0, 0.0]),
            make_doc("b", vec![0.0, 1.0]),
            make_doc("c", vec![0.5, 0.5]),
        ];
        let index = BruteForceIndex::with_documents(docs, Metric::Cosine);
        let query = vec![0.9, 0.1];
        let results = index.search(&query, 2);
        assert_eq!(results.len(), 2);
        assert!(results[0].score >= results[1].score);
    }

    #[test]
    fn test_search_empty_index() {
        let index = BruteForceIndex::new(Metric::Cosine);
        let results = index.search(&[1.0, 2.0], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_k_zero() {
        let docs = vec![make_doc("a", vec![1.0, 0.0])];
        let index = BruteForceIndex::with_documents(docs, Metric::Cosine);
        let results = index.search(&[1.0, 0.0], 0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_k_larger_than_docs() {
        let docs = vec![make_doc("a", vec![1.0, 0.0]), make_doc("b", vec![0.0, 1.0])];
        let index = BruteForceIndex::with_documents(docs, Metric::Cosine);
        let results = index.search(&[1.0, 0.0], 10);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_search_documents_without_embedding_skipped() {
        let docs = vec![
            make_doc("a", vec![1.0, 0.0]),
            Document::new("b", "no embedding"),
        ];
        let index = BruteForceIndex::with_documents(docs, Metric::Cosine);
        let results = index.search(&[1.0, 0.0], 5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].document.id, "a");
    }

    #[test]
    fn test_insert() {
        let mut index = BruteForceIndex::new(Metric::Euclidean);
        assert!(index.is_empty());
        index.insert(&[make_doc("a", vec![0.0, 0.0])]);
        assert_eq!(index.len(), 1);
        index.insert(&[make_doc("b", vec![1.0, 1.0]), make_doc("c", vec![2.0, 2.0])]);
        assert_eq!(index.len(), 3);
    }

    #[test]
    fn test_metric() {
        let index = BruteForceIndex::new(Metric::Dot);
        assert_eq!(index.metric(), Metric::Dot);
    }

    #[test]
    fn test_documents_reflects_inserts() {
        let mut index = BruteForceIndex::new(Metric::Cosine);
        index.insert(&[make_doc("x", vec![0.1, 0.2])]);
        assert_eq!(index.documents().len(), 1);
        assert_eq!(index.documents()[0].id, "x");
    }

    #[test]
    fn test_search_euclidean() {
        let docs = vec![
            make_doc("near", vec![1.0, 1.0]),
            make_doc("far", vec![100.0, 100.0]),
        ];
        let index = BruteForceIndex::with_documents(docs, Metric::Euclidean);
        let query = vec![1.1, 0.9];
        let results = index.search(&query, 2);
        assert_eq!(results.len(), 2);
        assert!(results[0].score > results[1].score);
        assert_eq!(results[0].document.id, "near");
    }

    #[test]
    fn test_search_dot_product() {
        let docs = vec![
            make_doc("high", vec![10.0, 0.0]),
            make_doc("low", vec![1.0, 0.0]),
        ];
        let index = BruteForceIndex::with_documents(docs, Metric::Dot);
        let query = vec![1.0, 0.0];
        let results = index.search(&query, 2);
        assert_eq!(results.len(), 2);
        assert!(results[0].score > results[1].score);
        assert_eq!(results[0].document.id, "high");
    }
}
