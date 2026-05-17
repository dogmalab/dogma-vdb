//! Reranking trait for two-stage retrieval.
//!
//! The [`Reranker`] trait defines a generic interface for reordering
//! search results after the initial vector-similarity pass.  A default
//! [`NoRerank`] implementation is provided that leaves the order intact
//! (pass-through).
//!
//! This trait is intentionally decoupled from any specific reranking
//! engine — concrete implementations (ONNX Cross-Encoder, LLM-based,
//! etc.) live in separate crates or behind feature flags.

use crate::doc::Document;
use crate::error::Result;

/// Rerank a mutable slice of documents based on a query.
///
/// Implementations receive the original query text and a `&mut` reference
/// to the document list so they can reorder it **in place**.
///
/// # Two-stage pipeline
///
/// 1. **Retrieval** — the index fetches `k` nearest neighbours (inflated
///    by a multiplier when reranking is enabled).
/// 2. **Reranking** — this trait is called to re-score and reorder the
///    candidates using the original query text.
pub trait Reranker: Send + Sync {
    /// Reorder the documents by semantic relevance to the query.
    fn rerank(&self, query: &str, documents: &mut Vec<Document>) -> Result<()>;
}

/// A no-op reranker that leaves the document order unchanged.
///
/// This is the default implementation used when reranking is disabled.
/// It satisfies the trait contract without allocating or blocking.
pub struct NoRerank;

impl Reranker for NoRerank {
    fn rerank(&self, _query: &str, _documents: &mut Vec<Document>) -> Result<()> {
        // Pass-through — keep the original index order
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_rerank_preserves_order() {
        let reranker = NoRerank;
        let mut docs = vec![
            Document::new("a", "first"),
            Document::new("b", "second"),
            Document::new("c", "third"),
        ];
        reranker.rerank("test", &mut docs).unwrap();
        assert_eq!(docs[0].id, "a");
        assert_eq!(docs[1].id, "b");
        assert_eq!(docs[2].id, "c");
    }

    #[test]
    fn test_no_rerank_empty() {
        let reranker = NoRerank;
        let mut docs = vec![];
        reranker.rerank("test", &mut docs).unwrap();
        assert!(docs.is_empty());
    }
}
