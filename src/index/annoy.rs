//! Annoy — Random Projection Forest (Spotify Annoy).
//!
//! Builds `n_trees` binary trees by recursively dividing the vector
//! space with random hyperplanes.  At search time, each tree is
//! traversed from root to the nearest leaf; all collected candidates
//! are then scored exactly.
//!
//! ## Design
//! - **Batch build** — `insert` replaces the entire forest (no incremental).
//! - **Pure Rust** — uses `SplitMix64` for deterministic randomness.
//! - Implements [`Index`] for interchangeability with other backends.

use crate::distance::{self, Metric};
use crate::doc::Document;
use crate::index::{Index, ScoredDocument};
use std::cmp::Ordering;
use std::collections::HashSet;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Annoy index configuration.
#[derive(Debug, Clone)]
pub struct AnnoyConfig {
    /// Number of random projection trees (default: 10).
    pub n_trees: usize,
    /// Number of candidates to inspect during search.  `-1` means
    /// `n_trees * k` (auto).  With the simple tree-walk strategy the
    /// actual candidates collected is at most `n_trees * leaf_size`.
    pub search_k: i32,
    /// Distance metric.
    pub metric: Metric,
    /// Leaf size — stop splitting when a node has ≤ this many items.
    pub leaf_size: usize,
}

impl Default for AnnoyConfig {
    fn default() -> Self {
        Self {
            n_trees: 10,
            search_k: -1,
            metric: Metric::Cosine,
            leaf_size: 10,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

enum TreeNode {
    Leaf(Vec<usize>),
    Split {
        left: Box<TreeNode>,
        right: Box<TreeNode>,
        n: Vec<f32>,
        d: f32,
    },
}

// Manual debug — `n: Vec<f32>` derives fine but the recursive enum
// doesn't auto-derive in all contexts.
impl std::fmt::Debug for TreeNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TreeNode::Leaf(indices) => f.debug_tuple("Leaf").field(indices).finish(),
            TreeNode::Split { left, right, n, d } => f
                .debug_struct("Split")
                .field("n", n)
                .field("d", d)
                .field("left", left)
                .field("right", right)
                .finish(),
        }
    }
}

struct Tree {
    root: TreeNode,
}

impl std::fmt::Debug for Tree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tree").field("root", &self.root).finish()
    }
}

// ---------------------------------------------------------------------------
// AnnoyIndex
// ---------------------------------------------------------------------------

/// Random Projection Forest index.
///
/// # Example
/// ```
/// use dogma_vdb::index::{AnnoyIndex, AnnoyConfig};
/// use dogma_vdb::distance::Metric;
/// use dogma_vdb::doc::Document;
/// use dogma_vdb::index::Index;
///
/// let docs = vec![
///     Document::builder("a", "").embedding(vec![1.0, 0.0]).build(),
///     Document::builder("b", "").embedding(vec![0.0, 1.0]).build(),
/// ];
///
/// let mut index = AnnoyIndex::new(AnnoyConfig {
///     n_trees: 2,
///     search_k: -1,
///     metric: Metric::Cosine,
///     leaf_size: 1,
/// });
/// index.insert(&docs);
///
/// let results = index.search(&[1.0, 0.0], 5);
/// assert_eq!(results.len(), 2);
/// assert_eq!(results[0].document.id, "a");
/// ```
#[derive(Debug)]
pub struct AnnoyIndex {
    documents: Vec<Document>,
    trees: Vec<Tree>,
    config: AnnoyConfig,
}

// Manually implement Clone because TreeNode is not automatically Clone.
impl Clone for AnnoyIndex {
    fn clone(&self) -> Self {
        Self {
            documents: self.documents.clone(),
            trees: self.trees.clone(),
            config: self.config.clone(),
        }
    }
}

impl Clone for TreeNode {
    fn clone(&self) -> Self {
        match self {
            TreeNode::Leaf(indices) => TreeNode::Leaf(indices.clone()),
            TreeNode::Split { left, right, n, d } => TreeNode::Split {
                left: left.clone(),
                right: right.clone(),
                n: n.clone(),
                d: *d,
            },
        }
    }
}

impl Clone for Tree {
    fn clone(&self) -> Self {
        Tree {
            root: self.root.clone(),
        }
    }
}

impl AnnoyIndex {
    pub fn new(config: AnnoyConfig) -> Self {
        Self {
            documents: Vec::new(),
            trees: Vec::new(),
            config,
        }
    }

    /// Number of stored documents.
    pub fn len(&self) -> usize {
        self.documents.len()
    }

    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }

    /// Access the stored documents.
    pub fn documents(&self) -> &[Document] {
        &self.documents
    }

    /// The index configuration.
    pub fn config(&self) -> &AnnoyConfig {
        &self.config
    }

    /// Build the forest from scratch.
    fn build(&mut self) {
        if self.documents.is_empty() {
            self.trees.clear();
            return;
        }

        let n = self.documents.len();
        let indices: Vec<usize> = (0..n).collect();
        let mut trees = Vec::with_capacity(self.config.n_trees);

        for tree_idx in 0..self.config.n_trees {
            let seed = tree_idx as u64;
            let root = Self::build_tree(&indices, &self.documents, self.config.leaf_size, seed);
            trees.push(Tree { root });
        }

        self.trees = trees;
    }

    /// Recursively build a single tree.
    fn build_tree(indices: &[usize], docs: &[Document], leaf_size: usize, seed: u64) -> TreeNode {
        if indices.len() <= leaf_size {
            return TreeNode::Leaf(indices.to_vec());
        }

        // Pick two random points
        let a = Self::pick_random(indices, seed);
        let b = Self::pick_random(indices, seed.wrapping_add(1));

        if a == b {
            return TreeNode::Leaf(indices.to_vec());
        }

        let emb_a = &docs[a].embedding;
        let emb_b = &docs[b].embedding;

        // Normal vector = a - b
        let n: Vec<f32> = emb_a.iter().zip(emb_b.iter()).map(|(x, y)| x - y).collect();
        let d = distance::dot(&n, emb_a);

        // Partition
        let mut left = Vec::with_capacity(indices.len() / 2);
        let mut right = Vec::with_capacity(indices.len() / 2);

        for &idx in indices {
            let side = distance::dot(&n, &docs[idx].embedding) <= d;
            if side {
                left.push(idx);
            } else {
                right.push(idx);
            }
        }

        // Degenerate split — all on one side
        if left.is_empty() || right.is_empty() {
            return TreeNode::Leaf(indices.to_vec());
        }

        TreeNode::Split {
            left: Box::new(Self::build_tree(
                &left,
                docs,
                leaf_size,
                seed.wrapping_add(2),
            )),
            right: Box::new(Self::build_tree(
                &right,
                docs,
                leaf_size,
                seed.wrapping_add(3),
            )),
            n,
            d,
        }
    }

    /// Deterministic pseudo‑random index picker (SplitMix64).
    fn pick_random(indices: &[usize], seed: u64) -> usize {
        let mut z = seed.wrapping_mul(0x9E3779B97F4A7C15);
        z ^= z >> 30;
        z = z.wrapping_mul(0xBF58476D1CE4E5B9);
        z ^= z >> 27;
        z = z.wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        let idx = (z as usize) % indices.len();
        indices[idx]
    }

    /// Navigate to the nearest leaf for `query` in a single tree.
    fn walk_tree<'a>(root: &'a TreeNode, query: &[f32]) -> &'a [usize] {
        let mut node = root;
        loop {
            match node {
                TreeNode::Leaf(indices) => return indices,
                TreeNode::Split {
                    left, right, n, d, ..
                } => {
                    if distance::dot(query, n) <= *d {
                        node = left.as_ref();
                    } else {
                        node = right.as_ref();
                    }
                }
            }
        }
    }

    /// Search for the `k` approximate nearest neighbours.
    ///
    /// For each tree, walks from the root to the nearest leaf and
    /// collects all document indices in that leaf.  Candidates are
    /// then scored with exact `f32` distance.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<ScoredDocument> {
        if self.documents.is_empty() || k == 0 || self.trees.is_empty() {
            return Vec::new();
        }

        // Collect candidates from all trees
        let mut candidate_set: HashSet<usize> = HashSet::new();
        for tree in &self.trees {
            let leaf_indices = Self::walk_tree(&tree.root, query);
            candidate_set.extend(leaf_indices.iter().copied());
        }

        // Score all candidates
        let mut scored: Vec<ScoredDocument> = candidate_set
            .iter()
            .filter(|&&idx| !self.documents[idx].embedding.is_empty())
            .map(|&idx| {
                let score =
                    distance::score(query, &self.documents[idx].embedding, self.config.metric);
                ScoredDocument {
                    score,
                    document: self.documents[idx].clone(),
                }
            })
            .collect();

        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        scored.truncate(k);
        scored
    }
}

// ---------------------------------------------------------------------------
// Index trait impl
// ---------------------------------------------------------------------------

impl Index for AnnoyIndex {
    fn insert(&mut self, docs: &[Document]) {
        // Annoy is batch-build: accumulate docs, then rebuild
        self.documents.extend_from_slice(docs);
        self.build();
    }

    fn documents(&self) -> &[Document] {
        &self.documents
    }

    fn delete(&mut self, ids: &[&str]) -> usize {
        let before = self.documents.len();
        let id_set: HashSet<&str> = ids.iter().copied().collect();
        self.documents.retain(|d| !id_set.contains(d.id.as_str()));
        let deleted = before - self.documents.len();
        if deleted > 0 {
            self.build();
        }
        deleted
    }

    fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        filter: &(dyn Fn(&Document) -> bool + Sync),
    ) -> Vec<ScoredDocument> {
        let multiplier = k * 5;
        self.search(query, multiplier)
            .into_iter()
            .filter(|r| filter(&r.document))
            .take(k)
            .collect()
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<ScoredDocument> {
        self.search(query, k)
    }

    fn len(&self) -> usize {
        self.documents.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_doc(id: &str, embedding: Vec<f32>) -> Document {
        Document::builder(id, id).embedding(embedding).build()
    }

    fn default_config() -> AnnoyConfig {
        AnnoyConfig {
            n_trees: 10,
            search_k: -1,
            metric: Metric::Cosine,
            leaf_size: 1,
        }
    }

    #[test]
    fn test_empty_index() {
        let index = AnnoyIndex::new(default_config());
        assert!(index.is_empty());
        let r = index.search(&[1.0, 0.0], 5);
        assert!(r.is_empty());
    }

    #[test]
    fn test_basic_search() {
        let mut index = AnnoyIndex::new(default_config());
        index.insert(&[make_doc("a", vec![1.0, 0.0])]);
        assert_eq!(index.len(), 1);

        let results = index.search(&[1.0, 0.0], 5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].document.id, "a");
    }

    #[test]
    fn test_search_returns_closest() {
        let mut index = AnnoyIndex::new(default_config());
        index.insert(&[
            make_doc("a", vec![1.0, 0.0]),
            make_doc("b", vec![0.0, 1.0]),
            make_doc("c", vec![0.5, 0.5]),
        ]);

        let results = index.search(&[1.0, 0.0], 3);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].document.id, "a");
    }

    #[test]
    fn test_top_k_respected() {
        let mut index = AnnoyIndex::new(default_config());
        let docs: Vec<Document> = (0..50)
            .map(|i| make_doc(&format!("d{}", i), vec![i as f32 * 0.02, 0.0]))
            .collect();
        index.insert(&docs);

        let results = index.search(&[1.0, 0.0], 5);
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_rebuild_on_insert() {
        let mut index = AnnoyIndex::new(default_config());
        index.insert(&[make_doc("a", vec![1.0, 0.0])]);
        assert_eq!(index.len(), 1);

        index.insert(&[make_doc("b", vec![0.0, 1.0])]);
        assert_eq!(index.len(), 2);
        // After rebuild, both docs are searchable
        assert_eq!(index.search(&[1.0, 0.0], 5).len(), 2);
    }

    #[test]
    fn test_delete_rebuilds() {
        let mut index = AnnoyIndex::new(default_config());
        index.insert(&[make_doc("a", vec![1.0, 0.0]), make_doc("b", vec![0.0, 1.0])]);
        assert_eq!(index.len(), 2);

        let deleted = Index::delete(&mut index, &["a"]);
        assert_eq!(deleted, 1);
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn test_deterministic() {
        let docs = vec![
            make_doc("x", vec![0.2, 0.8]),
            make_doc("y", vec![0.8, 0.2]),
            make_doc("z", vec![0.5, 0.5]),
        ];

        let mut idx1 = AnnoyIndex::new(default_config());
        idx1.insert(&docs);

        let mut idx2 = AnnoyIndex::new(default_config());
        idx2.insert(&docs);

        let r1 = idx1.search(&[0.8, 0.2], 3);
        let r2 = idx2.search(&[0.8, 0.2], 3);
        for (a, b) in r1.iter().zip(r2.iter()) {
            assert_eq!(a.document.id, b.document.id);
        }
    }

    #[test]
    fn test_larger_dataset() {
        let mut index = AnnoyIndex::new(AnnoyConfig {
            n_trees: 20,
            search_k: -1,
            metric: Metric::Cosine,
            leaf_size: 5,
        });

        let mut docs = Vec::with_capacity(200);
        for i in 0..200 {
            let angle = i as f64 * 0.0314;
            docs.push(make_doc(
                &format!("d{i}"),
                vec![angle.cos() as f32, angle.sin() as f32],
            ));
        }
        index.insert(&docs);

        let results = index.search(&[1.0, 0.0], 10);
        assert_eq!(results.len(), 10);
        // Top result should be close to [1.0, 0.0]
        assert!(
            results[0].score > 0.9,
            "top score should be high, got: {}",
            results[0].score
        );
    }
}
