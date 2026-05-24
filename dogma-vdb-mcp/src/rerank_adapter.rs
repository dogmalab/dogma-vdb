//! Adapter that bridges the agnostic [`dogma_vdb_rerank::CrossEncoderReranker`]
//! to the core [`dogma_vdb::rerank::Reranker`] trait.
//!
//! This allows any [`CrossEncoderReranker`] implementation (ONNX, stub,
//! LLM-based, etc.) to be used transparently as a [`dogma_vdb::rerank::Reranker`]
//! inside dogma-vdb's collection pipeline.

use dogma_vdb::doc::Document;
use dogma_vdb::error::Result as DogmaResult;
use dogma_vdb::rerank::Reranker;

/// Wraps a [`CrossEncoderReranker`] so it can be used wherever a
/// [`dogma_vdb::rerank::Reranker`] is expected.
pub struct DogmaRerankerAdapter {
    inner: Box<dyn dogma_vdb_rerank::CrossEncoderReranker>,
}

impl DogmaRerankerAdapter {
    /// Create a new adapter from any [`CrossEncoderReranker`].
    pub fn new(inner: Box<dyn dogma_vdb_rerank::CrossEncoderReranker>) -> Self {
        Self { inner }
    }
}

impl Reranker for DogmaRerankerAdapter {
    fn rerank(&self, query: &str, documents: &mut Vec<Document>) -> DogmaResult<()> {
        if documents.is_empty() {
            return Ok(());
        }

        // 1. Extract plain text from dogma-vdb's Document struct
        let texts: Vec<String> = documents.iter().map(|d| d.text.clone()).collect();

        // 2. Score with the agnostic Cross-Encoder engine
        let ranked = self
            .inner
            .compute_scores(query, &texts)
            .map_err(|e| dogma_vdb::error::Error::Embedding(e.to_string()))?;

        // 3. Reorder the documents in place
        let mut ordered = Vec::with_capacity(documents.len());
        for (idx, _) in &ranked {
            ordered.push(documents[*idx].clone());
        }
        *documents = ordered;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dogma_vdb_rerank::StubReranker;

    fn make_doc(id: &str, text: &str) -> Document {
        let mut d = Document::new(id, text);
        d.embedding = vec![0.0; 4];
        d
    }

    #[test]
    fn test_adapter_empty() {
        let adapter = DogmaRerankerAdapter::new(Box::new(StubReranker));
        let mut docs = vec![];
        adapter.rerank("test", &mut docs).unwrap();
        assert!(docs.is_empty());
    }

    #[test]
    fn test_adapter_reorders() {
        let adapter = DogmaRerankerAdapter::new(Box::new(StubReranker));
        let mut docs = vec![
            make_doc(
                "long",
                "this is a very long document that should score lower",
            ),
            make_doc("short", "short text"),
            make_doc("medium", "medium length text here"),
        ];

        adapter.rerank("test query", &mut docs).unwrap();

        // After reranking with StubReranker (shorter = higher score),
        // the order should be: short, medium, long
        assert_eq!(docs[0].id, "short");
        assert_eq!(docs[1].id, "medium");
        assert_eq!(docs[2].id, "long");
    }

    #[test]
    fn test_adapter_preserves_all_docs() {
        let adapter = DogmaRerankerAdapter::new(Box::new(StubReranker));
        let mut docs = vec![
            make_doc("a", "alpha"),
            make_doc("b", "beta gamma delta"),
            make_doc("c", "chi"),
        ];

        let ids_before: Vec<String> = docs.iter().map(|d| d.id.clone()).collect();
        adapter.rerank("query", &mut docs).unwrap();
        let mut ids_after: Vec<String> = docs.iter().map(|d| d.id.clone()).collect();
        ids_after.sort();

        // Same set of documents
        assert_eq!(ids_before, ids_after);
    }
}
