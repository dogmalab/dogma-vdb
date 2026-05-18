//! # dogma-vdb-rerank
//!
//! Agnostic local Cross-Encoder reranking for Rust.
//!
//! This crate knows **nothing** about `dogma-vdb` internals.  It operates
//! on plain `&[String]` slices, making it usable from any vector store
//! (dogma-vdb, LanceDB, Qdrant, etc.) or standalone inference pipeline.
//!
//! ## Rerankers
//!
//! | Implementation | Description |
//! |---|---|
//! | [`StubReranker`] | Deterministic mock for dev/test (no model needed) |
//! | [`OnnxReranker`] | Real ONNX Cross-Encoder via `ort` + `tokenizers` |
//!
//! ## Example
//!
//! ```ignore
//! use dogma_vdb_rerank::{CrossEncoderReranker, StubReranker};
//!
//! let reranker = StubReranker;
//! let scores = reranker.compute_scores("rust vs python", &[
//!     "Rust is a systems language".into(),
//!     "Python is dynamically typed".into(),
//! ])?;
//! ```

pub mod onnx;

use thiserror::Error;

/// Typed error for reranking operations.
#[derive(Error, Debug)]
pub enum RerankError {
    /// Model file could not be loaded or inference failed.
    #[error("ONNX model error: {0}")]
    ModelError(String),

    /// Tokenisation failed for the given query/document pair.
    #[error("Tokenizer error: {0}")]
    TokenizerError(String),

    /// Empty input provided.
    #[error("No documents to rerank")]
    EmptyInput,
}

/// A Cross-Encoder reranker that scores query–document relevance pairs.
///
/// Implementations should be [`Send`] + [`Sync`] so they can be shared
/// across threads (e.g. in an MCP server).
///
/// # Agnostic contract
///
/// This trait only accepts plain Rust types (`&str`, `&[String]`), not
/// `dogma-vdb::Document` — any data source can feed into it.
pub trait CrossEncoderReranker: Send + Sync {
    /// Score every `(query, document)` pair and return a ranked list of
    /// `(original_index, score)` pairs sorted by relevance descending.
    ///
    /// Returns `Ok(vec![])` when `documents` is empty.
    fn compute_scores(
        &self,
        query: &str,
        documents: &[String],
    ) -> Result<Vec<(usize, f32)>, RerankError>;
}

/// A stub Cross-Encoder reranker that returns **simulated** scores.
///
/// Useful for testing, development, and as a placeholder until a real
/// ONNX model is wired in.  Documents are scored by a simple heuristic
/// (shorter text = higher relevance), which is **not** semantically
/// meaningful.
///
/// When you have an actual ONNX Cross-Encoder model (e.g.
/// `bge-reranker-base`), use [`OnnxReranker`] instead.
pub struct StubReranker;

impl CrossEncoderReranker for StubReranker {
    fn compute_scores(
        &self,
        _query: &str,
        documents: &[String],
    ) -> Result<Vec<(usize, f32)>, RerankError> {
        if documents.is_empty() {
            return Ok(vec![]);
        }

        // Deterministic "relevance" based on text length
        // (shorter texts score higher — just a stable mock).
        let max_len = documents.iter().map(|d| d.len()).max().unwrap_or(1).max(1) as f32;

        let mut results: Vec<(usize, f32)> = documents
            .iter()
            .enumerate()
            .map(|(idx, text)| {
                // Score between 0.3 and 1.0 (longer = lower)
                let ratio = text.len() as f32 / max_len;
                let score = 1.0 - 0.7 * ratio;
                (idx, score)
            })
            .collect();

        // Stable sort — keep original order for equal scores
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(results)
    }
}

/// Convenience re-export of the ONNX-backed reranker.
pub use onnx::OnnxReranker;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stub_reranker_empty() {
        let reranker = StubReranker;
        let scores = reranker.compute_scores("query", &[]).unwrap();
        assert!(scores.is_empty());
    }

    #[test]
    fn test_stub_reranker_single() {
        let reranker = StubReranker;
        let scores = reranker
            .compute_scores("query", &["hello world".into()])
            .unwrap();
        assert_eq!(scores.len(), 1);
        assert_eq!(scores[0].0, 0);
        assert!(scores[0].1 > 0.0);
    }

    #[test]
    fn test_stub_reranker_ranking() {
        let reranker = StubReranker;
        // Longer text → lower score in the stub heuristic
        let scores = reranker
            .compute_scores(
                "rust",
                &[
                    "a".into(),                                  // short → high score
                    "very long document about something".into(), // long → low score
                    "medium text".into(),                        // medium
                ],
            )
            .unwrap();

        assert_eq!(scores.len(), 3);
        // Index 0 ("a") should be first (highest score)
        assert_eq!(scores[0].0, 0);
        assert!(scores[0].1 > scores[1].1);
        assert!(scores[1].1 > scores[2].1);
    }

    #[test]
    fn test_stub_reranker_scores_in_range() {
        let reranker = StubReranker;
        let docs: Vec<String> = (0..10)
            .map(|i| format!("document number {i} with some padding text"))
            .collect();
        let scores = reranker.compute_scores("test", &docs).unwrap();

        assert_eq!(scores.len(), 10);
        for (_, score) in &scores {
            assert!(*score >= 0.3);
            assert!(*score <= 1.0);
        }
    }
}
