//! Reciprocal Rank Fusion (RRF) for merging ranked result lists.
//!
//! Combines two ranked lists (e.g. vector similarity + BM25) using the
//! standard RRF formula:
//!
//! ```text
//! RRF_score(d) = 1 / (k + rank_vec(d)) + 1 / (k + rank_bm25(d))
//! ```
//!
//! where `k = 60` (standard constant).
//!
//! # Example
//! ```
//! use dogma_vdb::index::rrf::fuse;
//!
//! let vec_results = vec![(0u64, 0.95), (1, 0.80)];
//! let bm25_results = vec![(1u64, 12.5), (2, 8.3)];
//! let fused = fuse(&vec_results, &bm25_results, 2);
//! assert_eq!(fused.len(), 2);
//! ```

use std::cmp::Ordering;

/// RRF constant — the `k` in `1 / (k + rank)`.
const RRF_K: usize = 60;

/// Fuse two ranked result lists using Reciprocal Rank Fusion.
///
/// * `list_a` — first ranked list, sorted descending by score.
/// * `list_b` — second ranked list, sorted descending by score.
/// * `top_k` — return at most this many fused results.
///
/// Returns `Vec<(DocId, f32)>` sorted descending by RRF score.
/// `DocId` is a generic integer; typically `usize` is used.
///
/// # Performance
///
/// Uses a flat `Vec` with linear scan instead of a `HashMap`.
/// For the typical case (< 200 candidates) this is faster — no hash
/// computations, no intermediary allocations beyond the result buffer.
pub fn fuse<DocId: Copy + Eq>(
    list_a: &[(DocId, f32)],
    list_b: &[(DocId, f32)],
    top_k: usize,
) -> Vec<(DocId, f32)> {
    if list_a.is_empty() && list_b.is_empty() {
        return Vec::new();
    }

    // Pre-size to avoid reallocations during accumulation
    let max_entries = list_a.len() + list_b.len();
    let mut results: Vec<(DocId, f32)> = Vec::with_capacity(max_entries);

    // Accumulate list_a
    for (rank, &(doc_id, _)) in list_a.iter().enumerate() {
        let score = 1.0 / (RRF_K + rank + 1) as f32;
        if let Some(pos) = results.iter().position(|(id, _)| *id == doc_id) {
            results[pos].1 += score;
        } else {
            results.push((doc_id, score));
        }
    }

    // Accumulate list_b
    for (rank, &(doc_id, _)) in list_b.iter().enumerate() {
        let score = 1.0 / (RRF_K + rank + 1) as f32;
        if let Some(pos) = results.iter().position(|(id, _)| *id == doc_id) {
            results[pos].1 += score;
        } else {
            results.push((doc_id, score));
        }
    }

    // Sort descending by RRF score, return top-k
    results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    results.truncate(top_k);
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rrf_basic() {
        let a = vec![(0u64, 10.0), (1, 9.0), (2, 8.0)];
        let b = vec![(1u64, 50.0), (3, 40.0)];
        let fused = fuse(&a, &b, 5);
        // doc 1 appears in both lists → highest RRF score
        assert_eq!(fused[0].0, 1);
        assert_eq!(fused.len(), 4);
    }

    #[test]
    fn test_rrf_empty() {
        assert!(fuse::<u64>(&[], &[], 5).is_empty());
    }

    #[test]
    fn test_rrf_single_list() {
        let a = vec![(0u64, 1.0), (1, 0.5)];
        let fused = fuse(&a, &[], 2);
        assert_eq!(fused.len(), 2);
        assert_eq!(fused[0].0, 0);
    }

    #[test]
    fn test_rrf_top_k_truncation() {
        let a = vec![(0u64, 1.0), (1, 0.9), (2, 0.8), (3, 0.7)];
        let b = vec![(4u64, 1.0)];
        let fused = fuse(&a, &b, 2);
        assert_eq!(fused.len(), 2);
    }

    #[test]
    fn test_rrf_duplicate_in_same_list() {
        let a = vec![(0u64, 1.0), (0, 0.9)];
        let fused = fuse(&a, &[], 5);
        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].0, 0);
    }
}
