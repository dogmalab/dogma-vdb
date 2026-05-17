//! Vector index trait and implementations.
//!
//! Three backends are provided:
//! - [`BruteForceIndex`] — precise, O(n·d) linear scan
//! - [`HnswIndex`] — approximate, O(log n) via hierarchical graph
//! - [`IvfPqIndex`] — approximate, via IVF + Product Quantisation

mod brute_force;
mod hnsw;
mod ivf_pq;
mod sq;

pub use brute_force::BruteForceIndex;
pub use hnsw::{HnswConfig, HnswIndex};
pub use ivf_pq::{IvfPqConfig, IvfPqIndex};
pub use sq::*;

use crate::doc::Document;
use crate::storage::traits::VectorStorage;
use std::sync::Arc;

/// A scored search result.
#[derive(Debug, Clone)]
pub struct ScoredDocument {
    pub score: f32,
    pub document: Document,
}

/// Trait for vector indexes.
///
/// The default implementation is [`BruteForceIndex`], which performs
/// an O(n·d) linear scan.  For larger collections, [`HnswIndex`] provides
/// approximate O(log n) search at the cost of some recall.
pub trait Index: Send + Sync {
    /// Insert documents into the index.
    fn insert(&mut self, docs: &[Document]);

    /// Return a reference to all stored documents.
    fn documents(&self) -> &[Document];

    /// Delete documents by their IDs.
    ///
    /// Returns the number of documents actually removed.
    fn delete(&mut self, ids: &[&str]) -> usize;

    /// Search for the `k` nearest neighbours of `query`.
    fn search(&self, query: &[f32], k: usize) -> Vec<ScoredDocument>;

    /// Search with a metadata / content filter.
    ///
    /// Only documents for which `filter` returns `true` are considered.
    /// The default implementation post-filters the regular search results;
    /// override for more efficient pre-filtering.
    fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        filter: &(dyn Fn(&Document) -> bool + Sync),
    ) -> Vec<ScoredDocument> {
        // Default: post-filter (safe approximation)
        self.search(query, k * 3)
            .into_iter()
            .filter(|r| filter(&r.document))
            .take(k)
            .collect()
    }

    /// Number of indexed documents.
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Inject a [`VectorStorage`] for zero-copy embedding access.
    ///
    /// When set, index backends should use this storage for distance
    /// computation instead of the per-document embeddings stored in
    /// [`Document`].  Default is a no-op.
    fn set_storage(&mut self, _storage: Arc<dyn VectorStorage>) {}
}
