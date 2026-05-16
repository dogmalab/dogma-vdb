//! High‑level `Collection` API.
//!
//! A `Collection` ties together a `.vdb` file and an in‑memory index.
//! It is the main entry point for end‑users.

use crate::distance::Metric;
use crate::doc::Document;
use crate::embedding::Embedder;
use crate::error::Result;
use crate::index::{BruteForceIndex, Index, ScopedDocument};
use crate::storage::JsonlStorage;
use std::path::PathBuf;

/// A named collection backed by a `.vdb` file.
///
/// # Example
///
/// ```ignore
/// use dogma_vdb::prelude::*;
///
/// let mut col = Collection::open("my_data.vdb")?;
/// col.insert(Document::new("id-1", "Rust is fast"))?;
/// let results = col.search(&[0.1, 0.2, 0.3], 5, Metric::Cosine);
/// ```
#[derive(Debug)]
pub struct Collection {
    name: String,
    storage: JsonlStorage,
    index: BruteForceIndex,
}

impl Collection {
    /// Open (or create) a collection from a `.vdb` path.
    pub fn open(path: impl Into<PathBuf>) -> Result<Self>;

    /// The collection name (derived from the file stem).
    pub fn name(&self) -> &str;

    /// Number of documents in the collection.
    pub fn len(&self) -> usize;

    pub fn is_empty(&self) -> bool;

    /// Insert a single document and persist immediately.
    pub fn insert(&mut self, doc: Document) -> Result<()>;

    /// Insert many documents at once (single persist).
    pub fn insert_batch(&mut self, docs: &[Document]) -> Result<()>;

    /// Search with an embedder: embed the query text, then search.
    pub fn search_query(
        &self,
        embedder: &dyn Embedder,
        text: &str,
        k: usize,
        metric: Metric,
    ) -> Result<Vec<ScoredDocument>>;

    /// Search directly with a query vector.
    pub fn search(&self, query: &[f32], k: usize, metric: Metric) -> Vec<ScoredDocument>;

    /// Iterate over all documents.
    pub fn documents(&self) -> impl Iterator<Item = &Document>;
}

#[cfg(test)]
mod tests;
