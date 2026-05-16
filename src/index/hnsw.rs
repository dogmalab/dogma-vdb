//! HNSW (Hierarchical Navigable Small World) index.
//!
//! Approximate nearest-neighbour search using a multi-layer graph.
//! Based on Malkov & Yashunin (2016), "Efficient and robust
//! approximate nearest neighbor search using Hierarchical Navigable
//! Small World graphs".
//!
//! ## Design
//! - Pure Rust, zero new dependencies.
//! - Deterministic - layer assignment derived from `node_id`.
//! - Implements [`Index`] so it can be swapped with [`BruteForceIndex`].
//!
//! ## Performance
//! - Insert: O(log n) expected
//! - Search: O(log n) x `ef_search`

use crate::distance::{self, Metric};
use crate::doc::Document;
use crate::index::{Index, ScoredDocument};
use std::cmp::Ordering;
use std::collections::HashSet;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// HNSW index configuration.
#[derive(Debug, Clone)]
pub struct HnswConfig {
    /// Max connections per node per layer (default: 16).
    pub m: usize,
    /// Candidate list size during construction (default: 200).
    pub ef_construction: usize,
    /// Candidate list size during search (default: 50).
    pub ef_search: usize,
    /// Distance metric.
    pub metric: Metric,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 200,
            ef_search: 50,
            metric: Metric::Cosine,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

/// A neighbour candidate: `(score, node_id)`.
/// Higher score = more similar.
#[derive(Debug, Clone)]
struct Candidate {
    score: f32,
    node: usize,
}

impl Eq for Candidate {}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(Ordering::Equal)
    }
}

// ---------------------------------------------------------------------------
// HnswIndex
// ---------------------------------------------------------------------------

/// Hierarchical Navigable Small World index.
///
/// # Example
/// ```
/// use dogma_vdb::index::{HnswIndex, HnswConfig};
/// use dogma_vdb::distance::Metric;
/// use dogma_vdb::doc::Document;
/// use dogma_vdb::index::Index;
///
/// let mut index = HnswIndex::new(HnswConfig {
///     m: 8,
///     ef_construction: 50,
///     ef_search: 20,
///     metric: Metric::Cosine,
/// });
///
/// let doc = Document::builder("a", "hello")
///     .embedding(vec![1.0, 0.0, 0.0])
///     .build();
/// index.insert(&[doc]);
///
/// let results = index.search(&[1.0, 0.0, 0.0], 5);
/// assert_eq!(results.len(), 1);
/// ```
#[derive(Debug, Clone)]
pub struct HnswIndex {
    documents: Vec<Document>,
    /// `graphs[layer][node_id]` -> neighbours (node IDs).
    graphs: Vec<Vec<Vec<usize>>>,
    /// Highest layer each node belongs to.
    node_layers: Vec<usize>,
    /// Current entry point (node with the highest layer).
    entry_point: Option<usize>,
    config: HnswConfig,
    /// `1.0 / ln(m)` -- layer multiplier.
    ml: f64,
}

impl HnswIndex {
    pub fn new(config: HnswConfig) -> Self {
        let ml = 1.0 / (config.m as f64).ln();
        Self {
            documents: Vec::new(),
            graphs: Vec::new(),
            node_layers: Vec::new(),
            entry_point: None,
            config,
            ml,
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
    pub fn config(&self) -> &HnswConfig {
        &self.config
    }

    // ------------------------------------------------------------------
    // Public API - insert & search
    // ------------------------------------------------------------------

    /// Insert a batch of documents.
    ///
    /// Each document **must** have a non-empty embedding.  Documents
    /// without an embedding are silently skipped.
    pub fn insert(&mut self, docs: &[Document]) {
        for doc in docs {
            if doc.embedding.is_empty() {
                continue;
            }
            self.insert_one(doc.clone());
        }
    }

    /// Search for the `k` approximate nearest neighbours.
    ///
    /// Uses the configured `ef_search` internally; at least `k` candidates
    /// are evaluated if the collection has enough documents.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<ScoredDocument> {
        if self.documents.is_empty() || k == 0 {
            return Vec::new();
        }

        let ef = self.config.ef_search.max(k);
        let ep = match self.entry_point {
            Some(e) => e,
            None => return Vec::new(),
        };

        // 1. Descend through upper layers with ef = 1
        let top_layer = self.graphs.len().saturating_sub(1);
        let mut ep = ep;
        for layer in (1..=top_layer).rev() {
            let r = self.search_layer(query, ep, 1, layer);
            if let Some(best) = r.last() {
                ep = best.node;
            }
        }

        // 2. Full search at layer 0
        let candidates = self.search_layer(query, ep, ef, 0);

        // 3. Take top-k and wrap into ScoredDocument
        let mut scored: Vec<ScoredDocument> = candidates
            .into_iter()
            .take(k)
            .map(|c| ScoredDocument {
                score: c.score,
                document: self.documents[c.node].clone(),
            })
            .collect();

        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        scored
    }

    // ------------------------------------------------------------------
    // Internal
    // ------------------------------------------------------------------

    /// Assign a random layer to a new node (deterministic from node_id).
    fn assign_level(node_id: usize, ml: f64) -> usize {
        // SplitMix64-style hash for a geometric-like distribution
        let h = node_id.wrapping_mul(0x9E3779B97F4A7C15);
        let h = h ^ (h >> 30);
        let h = h.wrapping_mul(0xBF58476D1CE4E5B9);
        let h = h ^ (h >> 27);
        let h = h.wrapping_mul(0x94D049BB133111EB);
        let h = h ^ (h >> 31);
        let u = (h >> 11) as f64 * (1.0 / 9007199254740992.0);
        let u = u.max(1e-30);
        (-u.ln() * ml).floor() as usize
    }

    /// Insert a single document.
    fn insert_one(&mut self, doc: Document) {
        let node_id = self.documents.len();
        let emb = doc.embedding.clone();
        let node_level = Self::assign_level(node_id, self.ml);

        // Ensure enough layers
        while self.graphs.len() <= node_level {
            self.graphs.push(Vec::new());
        }
        for g in &mut self.graphs {
            while g.len() <= node_id {
                g.push(Vec::new());
            }
        }

        self.documents.push(doc);
        self.node_layers.push(node_level);

        let ep = match self.entry_point {
            None => {
                self.entry_point = Some(node_id);
                return;
            }
            Some(e) => e,
        };

        let top_layer = self.graphs.len().saturating_sub(1);
        let mut ep = ep;

        // Descend to the level where this node lives
        for layer in (node_level + 1..=top_layer).rev() {
            let r = self.search_layer(&emb, ep, 1, layer);
            if let Some(best) = r.last() {
                ep = best.node;
            }
        }

        let max_conn = self.config.m;
        let max_conn_0 = self.config.m * 2;

        for layer in (0..=node_level).rev() {
            let ef = if layer == 0 {
                self.config.ef_construction.max(self.config.m)
            } else {
                self.config.ef_construction
            };

            let candidates = self.search_layer(&emb, ep, ef, layer);

            let limit = if layer == 0 { max_conn_0 } else { max_conn };
            let neighbours: Vec<usize> =
                candidates.into_iter().take(limit).map(|c| c.node).collect();

            for &nei in &neighbours {
                self.graphs[layer][node_id].push(nei);
                let back = &mut self.graphs[layer][nei];
                back.push(node_id);
                let m_max = if layer == 0 { max_conn_0 } else { max_conn };
                if back.len() > m_max {
                    self.shrink_connections(layer, nei, m_max, &emb);
                }
            }

            if let Some(&first) = neighbours.first() {
                ep = first;
            }
        }

        // Update entry point if this node has a higher layer
        if node_level > self.node_layers.get(ep).copied().unwrap_or(0) {
            self.entry_point = Some(node_id);
        }
    }

    /// Single-layer search: find the `ef` nearest candidates.
    fn search_layer(&self, query: &[f32], entry: usize, ef: usize, layer: usize) -> Vec<Candidate> {
        let mut visited = vec![false; self.documents.len()];
        visited[entry] = true;

        let mut candidates = std::collections::BinaryHeap::new();
        let entry_score =
            distance::score(query, &self.documents[entry].embedding, self.config.metric);
        candidates.push(Candidate {
            score: entry_score,
            node: entry,
        });

        let mut results = std::collections::BinaryHeap::new();
        results.push(Candidate {
            score: entry_score,
            node: entry,
        });

        while let Some(current) = candidates.pop() {
            let furthest = match results.peek() {
                Some(f) => f.score,
                None => break,
            };

            if current.score < furthest && results.len() >= ef {
                break;
            }

            for &nei in &self.graphs[layer][current.node] {
                if visited[nei] {
                    continue;
                }
                visited[nei] = true;

                let nei_score =
                    distance::score(query, &self.documents[nei].embedding, self.config.metric);

                candidates.push(Candidate {
                    score: nei_score,
                    node: nei,
                });

                if results.len() < ef {
                    results.push(Candidate {
                        score: nei_score,
                        node: nei,
                    });
                } else {
                    let worst = results.peek().unwrap().score;
                    if nei_score > worst {
                        results.pop();
                        results.push(Candidate {
                            score: nei_score,
                            node: nei,
                        });
                    }
                }
            }
        }

        let mut out: Vec<Candidate> = results.into_sorted_vec();
        out.reverse();
        out
    }

    /// Shrink connections at a node, keeping only the closest `m_max`.
    fn shrink_connections(&mut self, layer: usize, node: usize, m_max: usize, centre: &[f32]) {
        if self.graphs[layer][node].len() <= m_max {
            return;
        }

        let mut scored: Vec<(usize, f32)> = self.graphs[layer][node]
            .iter()
            .map(|&n| {
                let s = distance::score(centre, &self.documents[n].embedding, self.config.metric);
                (n, s)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        scored.truncate(m_max);

        self.graphs[layer][node] = scored.into_iter().map(|(n, _)| n).collect();
    }
}

// ---------------------------------------------------------------------------
// Index trait impl
// ---------------------------------------------------------------------------

impl Index for HnswIndex {
    fn insert(&mut self, docs: &[Document]) {
        self.insert(docs);
    }

    fn documents(&self) -> &[Document] {
        &self.documents
    }

    fn delete(&mut self, ids: &[&str]) -> usize {
        let before = self.documents.len();
        let id_set: HashSet<&str> = ids.iter().copied().collect();
        let remaining: Vec<Document> = self
            .documents
            .iter()
            .filter(|d| !id_set.contains(d.id.as_str()))
            .cloned()
            .collect();
        let deleted = before - remaining.len();
        if deleted > 0 {
            let config = self.config.clone();
            *self = Self::new(config);
            self.insert(&remaining);
        }
        deleted
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

    fn default_config() -> HnswConfig {
        HnswConfig {
            m: 8,
            ef_construction: 50,
            ef_search: 20,
            metric: Metric::Cosine,
        }
    }

    #[test]
    fn test_empty_index() {
        let index = HnswIndex::new(default_config());
        let results = index.search(&[1.0, 0.0], 5);
        assert!(results.is_empty());
        assert!(index.is_empty());
    }

    #[test]
    fn test_single_insert() {
        let mut index = HnswIndex::new(default_config());
        index.insert(&[make_doc("a", vec![1.0, 0.0, 0.0])]);
        assert_eq!(index.len(), 1);

        let results = index.search(&[1.0, 0.0, 0.0], 5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].document.id, "a");
        assert!((results[0].score - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_search_returns_closest() {
        let mut index = HnswIndex::new(default_config());
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
        let mut index = HnswIndex::new(default_config());
        let docs: Vec<Document> = (0..20)
            .map(|i| make_doc(&format!("d{}", i), vec![i as f32 * 0.1, 0.0]))
            .collect();
        index.insert(&docs);

        let results = index.search(&[1.0, 0.0], 5);
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_insert_batch() {
        let mut index = HnswIndex::new(default_config());
        let docs: Vec<Document> = (0..10)
            .map(|i| make_doc(&format!("b{}", i), vec![i as f32, 0.0]))
            .collect();
        index.insert(&docs);
        assert_eq!(index.len(), 10);
    }

    #[test]
    fn test_documents_without_embedding_skipped() {
        let mut index = HnswIndex::new(default_config());
        index.insert(&[
            make_doc("a", vec![1.0, 0.0]),
            Document::new("b", "no embedding"),
        ]);
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn test_deterministic() {
        let docs = vec![
            make_doc("x", vec![0.2, 0.8]),
            make_doc("y", vec![0.8, 0.2]),
            make_doc("z", vec![0.5, 0.5]),
        ];

        let mut index1 = HnswIndex::new(default_config());
        index1.insert(&docs);

        let mut index2 = HnswIndex::new(default_config());
        index2.insert(&docs);

        let r1 = index1.search(&[0.8, 0.2], 3);
        let r2 = index2.search(&[0.8, 0.2], 3);
        assert_eq!(r1.len(), r2.len());
        for (a, b) in r1.iter().zip(r2.iter()) {
            assert_eq!(a.document.id, b.document.id);
        }
    }

    #[test]
    fn test_approximate_recall() {
        let mut index = HnswIndex::new(HnswConfig {
            m: 16,
            ef_construction: 100,
            ef_search: 50,
            metric: Metric::Cosine,
        });

        let mut docs = Vec::with_capacity(500);
        for i in 0..500 {
            let angle = i as f64 * 0.01256;
            docs.push(make_doc(
                &format!("d{}", i),
                vec![angle.cos() as f32, angle.sin() as f32],
            ));
        }
        index.insert(&docs);

        let query = vec![1.0, 0.0];
        let results = index.search(&query, 10);

        let ids: Vec<&str> = results.iter().map(|r| r.document.id.as_str()).collect();
        assert!(
            ids.contains(&"d0"),
            "d0 should be in top-10 results, got: {:?}",
            ids
        );
        assert_eq!(results.len(), 10);
    }
}
