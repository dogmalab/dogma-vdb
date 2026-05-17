//! Fastembed (ONNX) implementation of the `Embedder` trait.
//!
//! Uses the [`fastembed`] crate to run embedding models locally
//! via ONNX Runtime.  The default model is `all-MiniLM-L6-v2` (384-dim,
//! ~90 MB download on first use).
//!
//! # Example
//! ```no_run
//! use dogma_vdb_embed::Embedder;
//! use dogma_vdb_embed_fastembed::FastEmbedder;
//!
//! let embedder = FastEmbedder::new().unwrap();
//! let vec = embedder.embed("Hello, world!").unwrap();
//! assert_eq!(vec.len(), embedder.dimension());
//! ```

use dogma_vdb_embed::Embedder;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

/// Embedder backed by fastembed (ONNX runtime).
///
/// Default model: `all-MiniLM-L6-v2`, 384 dimensions.
/// The model is downloaded automatically on first use (~90 MB).
pub struct FastEmbedder {
    model: TextEmbedding,
    dim: usize,
}

impl FastEmbedder {
    /// Create a new fastembed embedder with the default model
    /// (`all-MiniLM-L6-v2`, 384 dimensions).
    pub fn new() -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        Self::with_model(EmbeddingModel::AllMiniLML6V2)
    }

    /// Create with an explicit model.
    pub fn with_model(
        model_name: EmbeddingModel,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let opts = InitOptions::new(model_name).with_show_download_progress(true);
        let model =
            TextEmbedding::try_new(opts).map_err(|e| format!("FastEmbedder init failed: {e}"))?;
        // Known dimension for MiniLM-L6-v2; other models may differ.
        let dim = 384;
        Ok(Self { model, dim })
    }

    /// The embedding dimension (currently hardcoded to 384 for MiniLM-L6-v2).
    pub fn dimension(&self) -> usize {
        self.dim
    }
}

impl Embedder for FastEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
        let mut results = self
            .model
            .embed(vec![text], Some(1))
            .map_err(|e| format!("FastEmbedder embed failed: {e}"))?;
        results.pop().ok_or_else(|| "no embedding returned".into())
    }

    fn dimension(&self) -> usize {
        self.dim
    }

    fn embed_batch(
        &self,
        texts: &[&str],
    ) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error + Send + Sync>> {
        let owned: Vec<&str> = texts.iter().map(|s| *s).collect();
        Ok(self
            .model
            .embed(owned, Some(texts.len()))
            .map_err(|e| format!("FastEmbedder embed_batch failed: {e}"))?)
    }
}
