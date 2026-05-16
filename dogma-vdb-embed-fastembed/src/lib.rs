//! Fastembed (ONNX) implementation of the `Embedder` trait.
//!
//! # Status
//! Skeleton — actual fastembed integration pending.
//! The `FastEmbedder::default()` constructor panics until a real
//! ONNX model is wired in.

use dogma_vdb_embed::Embedder;

/// Embedder backed by fastembed (ONNX runtime).
pub struct FastEmbedder;

impl FastEmbedder {
    /// Create a new fastembed embedder.
    ///
    /// # Panics
    /// Always — this is a skeleton stub.
    #[allow(unreachable_code)]
    pub fn new() -> Self {
        todo!("FastEmbedder::new — ONNX model loading not yet implemented")
    }
}

impl Default for FastEmbedder {
    fn default() -> Self {
        Self::new()
    }
}

impl Embedder for FastEmbedder {
    fn embed(&self, _text: &str) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
        todo!("FastEmbedder::embed — ONNX inference not yet implemented")
    }

    fn dimension(&self) -> usize {
        384 // Default MiniLM-L6-v2 dimension
    }

    fn embed_batch(
        &self,
        _texts: &[&str],
    ) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error + Send + Sync>> {
        todo!("FastEmbedder::embed_batch — ONNX inference not yet implemented")
    }
}
