//! Brute‑force (linear scan) index.
//!
//! Correct, simple, and fast enough for up to ~10 000 documents.
//!
//! Supports optional **Scalar Quantization (SQ)** — when `sq` is `true`,
//! embeddings are compressed from `f32` to `i8` for ~4× less memory and
//! ~2× faster distance computation.

use crate::distance::Metric;
use crate::doc::Document;
use crate::index::{self, Index, ScoredDocument};
use rayon::prelude::*;

/// Brute‑force (linear scan) index.
#[derive(Debug, Clone)]
pub struct BruteForceIndex {
    documents: Vec<Document>,
    metric: Metric,
    sq: bool,
    /// Quantised embeddings (only used when `sq=true`).
    embedding_i8: Vec<Vec<i8>>,
    scale: f32,
    bias: f32,
}

impl BruteForceIndex {
    pub fn new(metric: Metric) -> Self {
        Self {
            documents: Vec::new(),
            metric,
            sq: false,
            embedding_i8: Vec::new(),
            scale: 1.0,
            bias: 0.0,
        }
    }

    /// Create with an explicit SQ flag.
    pub fn new_with(metric: Metric, sq: bool) -> Self {
        Self {
            documents: Vec::new(),
            metric,
            sq,
            embedding_i8: Vec::new(),
            scale: 1.0,
            bias: 0.0,
        }
    }

    pub fn with_documents(docs: Vec<Document>, metric: Metric) -> Self {
        Self {
            documents: docs,
            metric,
            sq: false,
            embedding_i8: Vec::new(),
            scale: 1.0,
            bias: 0.0,
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
        if self.sq {
            // First batch: compute scale/bias from all existing + new docs
            // Subsequent batches: use existing scale/bias (clamp handles outliers)
            if self.embedding_i8.is_empty() {
                let all_docs: Vec<Document> = self
                    .documents
                    .iter()
                    .cloned()
                    .chain(docs.iter().cloned())
                    .collect();
                let (scale, bias) = index::compute_scale_bias(&all_docs);
                self.scale = scale;
                self.bias = bias;

                // Re-quantize existing + new
                self.embedding_i8 = all_docs
                    .iter()
                    .map(|d| {
                        if d.embedding.is_empty() {
                            Vec::new()
                        } else {
                            index::quantize(&d.embedding, self.scale, self.bias)
                        }
                    })
                    .collect();
            } else {
                // Quantize new docs
                for doc in docs {
                    let q = if doc.embedding.is_empty() {
                        Vec::new()
                    } else {
                        index::quantize(&doc.embedding, self.scale, self.bias)
                    };
                    self.embedding_i8.push(q);
                }
            }
        }
        self.documents.extend_from_slice(docs);
    }

    fn documents(&self) -> &[Document] {
        &self.documents
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<ScoredDocument> {
        if self.documents.is_empty() || k == 0 {
            return Vec::new();
        }

        if self.sq {
            let query_i8 = index::quantize_query(query, self.scale, self.bias);
            let mut results: Vec<ScoredDocument> = self
                .documents
                .par_iter()
                .enumerate()
                .filter(|(_, d)| !d.embedding.is_empty())
                .map(|(i, d)| {
                    let score = index::score_i8(
                        &query_i8,
                        &self.embedding_i8[i],
                        self.metric,
                        self.scale,
                        self.bias,
                    );
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
        } else {
            let mut results: Vec<ScoredDocument> = self
                .documents
                .par_iter()
                .filter(|d| !d.embedding.is_empty())
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
    }

    fn delete(&mut self, ids: &[&str]) -> usize {
        let before = self.documents.len();
        let mut i = 0;
        self.documents.retain(|d| {
            let keep = !ids.contains(&d.id.as_str());
            if !keep && self.sq {
                self.embedding_i8.remove(i);
            } else {
                i += 1;
            }
            keep
        });
        if self.sq {
            // Recompute from current documents
            let (scale, bias) = index::compute_scale_bias(&self.documents);
            self.scale = scale;
            self.bias = bias;
            self.embedding_i8 = self
                .documents
                .iter()
                .map(|d| {
                    if d.embedding.is_empty() {
                        Vec::new()
                    } else {
                        index::quantize(&d.embedding, self.scale, self.bias)
                    }
                })
                .collect();
        }
        before - self.documents.len()
    }

    fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        filter: &(dyn Fn(&Document) -> bool + Sync),
    ) -> Vec<ScoredDocument> {
        if self.documents.is_empty() || k == 0 {
            return Vec::new();
        }

        if self.sq {
            let query_i8 = index::quantize_query(query, self.scale, self.bias);
            let mut results: Vec<ScoredDocument> = self
                .documents
                .par_iter()
                .enumerate()
                .filter(|(_, d)| !d.embedding.is_empty())
                .filter(|(_, d)| filter(d))
                .map(|(i, d)| {
                    let score = index::score_i8(
                        &query_i8,
                        &self.embedding_i8[i],
                        self.metric,
                        self.scale,
                        self.bias,
                    );
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
        } else {
            let mut results: Vec<ScoredDocument> = self
                .documents
                .par_iter()
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
    fn test_insert_and_search() {
        let mut index = BruteForceIndex::new(Metric::Euclidean);
        assert!(index.is_empty());
        index.insert(&[make_doc("a", vec![0.0, 0.0])]);
        assert_eq!(index.len(), 1);
        let results = index.search(&[0.0, 0.0], 5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].document.id, "a");
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

    // ------------------------------------------------------------------
    // SQ tests
    // ------------------------------------------------------------------

    #[test]
    fn test_sq_basic() {
        let mut index = BruteForceIndex::new_with(Metric::Cosine, true);
        index.insert(&[make_doc("a", vec![1.0, 0.0])]);
        assert_eq!(index.len(), 1);
        assert!(!index.embedding_i8.is_empty());

        let results = index.search(&[1.0, 0.0], 5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].document.id, "a");
        // SQ scores are approximate — ranking matters, not absolute value
    }

    #[test]
    fn test_sq_multi_docs() {
        let mut index = BruteForceIndex::new_with(Metric::Euclidean, true);
        index.insert(&[
            make_doc("near", vec![1.0, 1.0]),
            make_doc("far", vec![100.0, 100.0]),
        ]);

        let results = index.search(&[1.0, 1.0], 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].document.id, "near");
    }

    #[test]
    fn test_sq_same_ranking() {
        let mut bf = BruteForceIndex::new(Metric::Cosine);
        let mut sq = BruteForceIndex::new_with(Metric::Cosine, true);

        let docs = vec![
            make_doc("close", vec![0.9, 0.1]),
            make_doc("far", vec![0.1, 0.9]),
            make_doc("mid", vec![0.5, 0.5]),
        ];
        bf.insert(&docs);
        sq.insert(&docs);

        let query = vec![1.0, 0.0];
        let r1 = bf.search(&query, 3);
        let r2 = sq.search(&query, 3);

        // Rankings should match (both have same ordering by id)
        for (a, b) in r1.iter().zip(r2.iter()) {
            assert_eq!(a.document.id, b.document.id);
        }
    }

    #[test]
    fn test_sq_delete() {
        let mut index = BruteForceIndex::new_with(Metric::Cosine, true);
        index.insert(&[make_doc("a", vec![1.0, 0.0]), make_doc("b", vec![0.0, 1.0])]);
        assert_eq!(index.embedding_i8.len(), 2);

        index.delete(&["a"]);
        assert_eq!(index.len(), 1);
        assert_eq!(index.embedding_i8.len(), 1);
    }

    #[test]
    fn test_sq_filtered() {
        let mut index = BruteForceIndex::new_with(Metric::Cosine, true);
        index.insert(&[
            Document::builder("en", "hello")
                .embedding(vec![1.0, 0.0])
                .metadata("lang", "en")
                .build(),
            Document::builder("es", "hola")
                .embedding(vec![0.0, 1.0])
                .metadata("lang", "es")
                .build(),
        ]);

        let results = index.search_filtered(&[1.0, 0.0], 5, &|d: &Document| {
            d.metadata_val("lang") == Some("en")
        });
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].document.id, "en");
    }
}
