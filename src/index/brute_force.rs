//! Brute‑force (linear scan) index.
//!
//! Supports optional **Scalar Quantization (SQ)** — when `sq` is `true`,
//! embeddings are compressed from `f32` to `i8` for ~4× less memory and
//! ~2× faster distance computation.  Quantisation is per‑dimension to
//! preserve ranking across all metrics.

use crate::distance::Metric;
use crate::doc::Document;
use crate::index::{self, Index, ScoredDocument};
use crate::storage::traits::VectorStorage;
use rayon::prelude::*;
use std::sync::Arc;

/// Brute‑force (linear scan) index.
#[derive(Clone)]
pub struct BruteForceIndex {
    documents: Vec<Document>,
    metric: Metric,
    sq: bool,
    sq_rescore: bool,
    /// Quantised embeddings (per‑dimension, only used when `sq=true`).
    embedding_i8: Vec<Vec<i8>>,
    /// Per‑dimension scale factors.
    scales: Vec<f32>,
    /// Per‑dimension biases.
    biases: Vec<f32>,
    /// Zero-copy embedding storage (optional).
    storage: Option<Arc<dyn VectorStorage>>,
    /// Embedding dimension (0 = unknown).
    dim: usize,
}

impl std::fmt::Debug for BruteForceIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BruteForceIndex")
            .field("documents", &self.documents)
            .field("metric", &self.metric)
            .field("sq", &self.sq)
            .field("sq_rescore", &self.sq_rescore)
            .field("embedding_i8", &self.embedding_i8)
            .field("scales", &self.scales)
            .field("biases", &self.biases)
            .field("storage", &self.storage.as_ref().map(|_| ".."))
            .field("dim", &self.dim)
            .finish()
    }
}

impl BruteForceIndex {
    pub fn new(metric: Metric) -> Self {
        Self {
            documents: Vec::new(),
            metric,
            sq: false,
            sq_rescore: false,
            embedding_i8: Vec::new(),
            scales: Vec::new(),
            biases: Vec::new(),
            storage: None,
            dim: 0,
        }
    }

    /// Create with explicit SQ and rescore flags.
    pub fn new_with(metric: Metric, sq: bool, sq_rescore: bool) -> Self {
        Self {
            documents: Vec::new(),
            metric,
            sq,
            sq_rescore,
            embedding_i8: Vec::new(),
            scales: Vec::new(),
            biases: Vec::new(),
            storage: None,
            dim: 0,
        }
    }

    pub fn with_documents(docs: Vec<Document>, metric: Metric) -> Self {
        Self {
            documents: docs,
            metric,
            sq: false,
            sq_rescore: false,
            embedding_i8: Vec::new(),
            scales: Vec::new(),
            biases: Vec::new(),
            storage: None,
            dim: 0,
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
    fn set_storage(&mut self, storage: Arc<dyn VectorStorage>) {
        self.storage = Some(storage);
    }

    fn insert(&mut self, docs: &[Document]) {
        // Set embedding dimension from the first document that has one
        if self.dim == 0 {
            if let Some(doc) = docs.iter().find(|d| !d.embedding.is_empty()) {
                self.dim = doc.embedding.len();
            } else if let Some(doc) = self.documents.iter().find(|d| !d.embedding.is_empty()) {
                self.dim = doc.embedding.len();
            }
        }

        if self.sq {
            if self.embedding_i8.is_empty() {
                // First batch: compute per‑dim scale/bias from all existing + new
                let all_docs: Vec<Document> = self
                    .documents
                    .iter()
                    .cloned()
                    .chain(docs.iter().cloned())
                    .collect();
                let (scales, biases) = index::compute_scale_bias_per_dim(&all_docs);
                self.scales = scales;
                self.biases = biases;

                // Re-quantize existing + new
                self.embedding_i8 = all_docs
                    .par_iter()
                    .map(|d| {
                        if d.embedding.is_empty() {
                            Vec::new()
                        } else {
                            index::quantize(&d.embedding, &self.scales, &self.biases)
                        }
                    })
                    .collect();
            } else {
                // Subsequent batches: use existing scale/bias
                for doc in docs {
                    self.embedding_i8.push(if doc.embedding.is_empty() {
                        Vec::new()
                    } else {
                        index::quantize(&doc.embedding, &self.scales, &self.biases)
                    });
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
            // Quantize query ONCE
            let query_i8 = index::quantize_query(query, &self.scales, &self.biases);

            let mut results: Vec<ScoredDocument> = self
                .documents
                .par_iter()
                .enumerate()
                .filter(|(_, d)| !d.embedding.is_empty())
                .map(|(i, d)| {
                    let score = index::score_i8(&query_i8, &self.embedding_i8[i], self.metric);
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

            if self.sq_rescore {
                let rescore_k = (k * 2).min(results.len());
                for r in &mut results[..rescore_k] {
                    r.score = crate::distance::score(&r.document.embedding, query, self.metric);
                }
                results[..rescore_k].sort_unstable_by(|a, b| {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }

            results.truncate(k);
            results
        } else {
            let mut results: Vec<ScoredDocument> = if let Some(ref storage) = self.storage {
                let emb_all = storage.as_embeddings();
                let dim = self.dim;
                self.documents
                    .par_iter()
                    .enumerate()
                    .filter(|(_, d)| !d.embedding.is_empty())
                    .map(|(i, d)| {
                        let start = i * dim;
                        let emb = if start + dim <= emb_all.len() {
                            &emb_all[start..start + dim]
                        } else {
                            &d.embedding
                        };
                        let score = crate::distance::score(emb, query, self.metric);
                        ScoredDocument {
                            score,
                            document: d.clone(),
                        }
                    })
                    .collect()
            } else {
                self.documents
                    .par_iter()
                    .filter(|d| !d.embedding.is_empty())
                    .map(|d| {
                        let score = crate::distance::score(&d.embedding, query, self.metric);
                        ScoredDocument {
                            score,
                            document: d.clone(),
                        }
                    })
                    .collect()
            };

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
            // Recompute per‑dim from current documents
            let (scales, biases) = index::compute_scale_bias_per_dim(&self.documents);
            self.scales = scales;
            self.biases = biases;
            self.embedding_i8 = self
                .documents
                .par_iter()
                .map(|d| {
                    if d.embedding.is_empty() {
                        Vec::new()
                    } else {
                        index::quantize(&d.embedding, &self.scales, &self.biases)
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
            let query_i8 = index::quantize_query(query, &self.scales, &self.biases);
            let mut results: Vec<ScoredDocument> = self
                .documents
                .par_iter()
                .enumerate()
                .filter(|(_, d)| !d.embedding.is_empty())
                .filter(|(_, d)| filter(d))
                .map(|(i, d)| {
                    let score = index::score_i8(&query_i8, &self.embedding_i8[i], self.metric);
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

            if self.sq_rescore {
                let rescore_k = (k * 2).min(results.len());
                for r in &mut results[..rescore_k] {
                    r.score = crate::distance::score(&r.document.embedding, query, self.metric);
                }
                results[..rescore_k].sort_unstable_by(|a, b| {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }

            results.truncate(k);
            results
        } else {
            let mut results: Vec<ScoredDocument> = if let Some(ref storage) = self.storage {
                let emb_all = storage.as_embeddings();
                let dim = self.dim;
                self.documents
                    .par_iter()
                    .enumerate()
                    .filter(|(_, d)| !d.embedding.is_empty())
                    .filter(|(_, d)| filter(d))
                    .map(|(i, d)| {
                        let start = i * dim;
                        let emb = if start + dim <= emb_all.len() {
                            &emb_all[start..start + dim]
                        } else {
                            &d.embedding
                        };
                        let score = crate::distance::score(emb, query, self.metric);
                        ScoredDocument {
                            score,
                            document: d.clone(),
                        }
                    })
                    .collect()
            } else {
                self.documents
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
                    .collect()
            };

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
        let results = index.search(&[1.0, 0.0, 0.0], 1);
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
        let results = index.search(&[0.9, 0.1], 2);
        assert_eq!(results.len(), 2);
        assert!(results[0].score >= results[1].score);
    }

    #[test]
    fn test_search_empty_index() {
        let index = BruteForceIndex::new(Metric::Cosine);
        assert!(index.search(&[1.0, 2.0], 5).is_empty());
    }

    #[test]
    fn test_insert_and_search() {
        let mut index = BruteForceIndex::new(Metric::Euclidean);
        index.insert(&[make_doc("a", vec![0.0, 0.0])]);
        let results = index.search(&[0.0, 0.0], 5);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_search_documents_without_embedding_skipped() {
        let docs = vec![
            make_doc("a", vec![1.0, 0.0]),
            Document::new("b", "no embedding"),
        ];
        let index = BruteForceIndex::with_documents(docs, Metric::Cosine);
        assert_eq!(index.search(&[1.0, 0.0], 5).len(), 1);
    }

    // ------------------------------------------------------------------
    // SQ tests
    // ------------------------------------------------------------------

    #[test]
    fn test_sq_basic() {
        let mut index = BruteForceIndex::new_with(Metric::Cosine, true, false);
        index.insert(&[make_doc("a", vec![1.0, 0.0])]);
        assert!(!index.embedding_i8.is_empty());
        let results = index.search(&[1.0, 0.0], 5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].document.id, "a");
    }

    #[test]
    fn test_sq_multi_docs() {
        let mut index = BruteForceIndex::new_with(Metric::Euclidean, true, false);
        index.insert(&[
            make_doc("near", vec![1.0, 1.0]),
            make_doc("far", vec![100.0, 100.0]),
        ]);
        let results = index.search(&[1.0, 1.0], 2);
        assert_eq!(results[0].document.id, "near");
    }

    #[test]
    fn test_sq_same_ranking() {
        let mut bf = BruteForceIndex::new(Metric::Cosine);
        let mut sq = BruteForceIndex::new_with(Metric::Cosine, true, false);
        let docs = vec![
            make_doc("close", vec![0.9, 0.1]),
            make_doc("far", vec![0.1, 0.9]),
            make_doc("mid", vec![0.5, 0.5]),
        ];
        bf.insert(&docs);
        sq.insert(&docs);

        for (a, b) in bf
            .search(&[1.0, 0.0], 3)
            .iter()
            .zip(sq.search(&[1.0, 0.0], 3).iter())
        {
            assert_eq!(a.document.id, b.document.id);
        }
    }

    #[test]
    fn test_sq_delete() {
        let mut index = BruteForceIndex::new_with(Metric::Cosine, true, false);
        index.insert(&[make_doc("a", vec![1.0, 0.0]), make_doc("b", vec![0.0, 1.0])]);
        assert_eq!(index.embedding_i8.len(), 2);
        index.delete(&["a"]);
        assert_eq!(index.embedding_i8.len(), 1);
    }

    #[test]
    fn test_sq_filtered() {
        let mut index = BruteForceIndex::new_with(Metric::Cosine, true, false);
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
        let results =
            index.search_filtered(&[1.0, 0.0], 5, &|d| d.metadata_val("lang") == Some("en"));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].document.id, "en");
    }

    #[test]
    fn test_sq_rescore_recovers_ranking() {
        let mut sq = BruteForceIndex::new_with(Metric::Cosine, true, true);
        let docs = vec![
            make_doc("close", vec![0.9, 0.1]),
            make_doc("far", vec![0.1, 0.9]),
        ];
        sq.insert(&docs);
        let results = sq.search(&[1.0, 0.0], 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].document.id, "close");
    }

    #[test]
    fn test_sq_wide_range_preserves_ranking() {
        // Dimensions with vastly different ranges — per‑dim scaling must
        // preserve ranking.  This is the case that broke global min/max.
        let mut sq = BruteForceIndex::new_with(Metric::Cosine, true, false);
        sq.insert(&[
            make_doc("close", vec![1.0, 1.0]),
            make_doc("far", vec![1000.0, 1000.0]),
        ]);
        let results = sq.search(&[1.0, 1.0], 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].document.id, "close");
    }
}
