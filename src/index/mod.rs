//! Vector index trait and implementations.
//!
//! Two backends are provided:
//! - [`BruteForceIndex`] — precise, O(n·d) linear scan
//! - [`HnswIndex`] — approximate, O(log n) via hierarchical graph

mod brute_force;
mod hnsw;

pub use brute_force::BruteForceIndex;
pub use hnsw::{HnswConfig, HnswIndex};

use crate::doc::Document;

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

    /// Delete documents by their IDs.
    ///
    /// Returns the number of documents actually removed.
    fn delete(&mut self, ids: &[&str]) -> usize;

    /// Search for the `k` nearest neighbours of `query`.
    fn search(&self, query: &[f32], k: usize) -> Vec<ScoredDocument>;

    /// Return a reference to all stored documents.
    fn documents(&self) -> &[Document];

    /// Search with a metadata / content filter.
    ///
    /// Only documents for which `filter` returns `true` are considered.
    /// The default implementation post-filters the regular search results;
    /// override for more efficient pre-filtering.
    fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        filter: &dyn Fn(&Document) -> bool,
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
}
