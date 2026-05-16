//! Zero‑dependency `Embedder` trait for `dogma-vdb`.
//!
//! This crate contains only the trait definition.  Implementations
//! live in separate crates so the core never pulls in heavy ML
//! dependencies.

/// Generates embedding vectors from text.
pub trait Embedder: Send + Sync {
    /// Embed a single text string.
    fn embed(&self, text: &str) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>>;

    /// Dimensionality of the produced vectors.
    fn dimension(&self) -> usize;

    /// Embed several texts at once.
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error + Send + Sync>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }
}
