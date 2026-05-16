//! The `Embedder` trait — zero dependencies.
//!
//! Implementations live in separate crates (e.g.
//! `dogma-vdb-embed-fastembed`) so the core never has to pull in
//! heavy ML dependencies.

use crate::error::Result;

/// Generates embedding vectors from text.
///
/// This trait is deliberately minimal.  A type‑level dimension
/// constant is **not** required — the returned `Vec<f32>` length is
/// the authoritative dimensionality, and `dimension()` is a
/// convenience getter.
pub trait Embedder: Send + Sync {
    /// Embed a single text string.
    fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Dimensionality of the produced vectors.
    fn dimension(&self) -> usize;

    /// Embed several texts at once.
    ///
    /// The default implementation calls `embed` sequentially.
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }
}
