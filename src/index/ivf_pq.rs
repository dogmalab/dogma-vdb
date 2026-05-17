//! IVF-PQ (Inverted File + Product Quantization) index.
//!
//! A minimal implementation of the standard IVF-PQ approximate nearest
//! neighbour algorithm:
//!
//! - **IVF**: vectors are partitioned into `K` clusters via simple K-Means.
//!   Search only visits the `probe` closest clusters (inverted index).
//! - **PQ**: each vector is split into `M` sub‑vectors, and each sub‑vector
//!   is quantised to a single byte via a 256‑centroid codebook.  Total
//!   storage per vector: `M` bytes instead of `D × 4` bytes.
//! - **Asymmetric search**: the query (f32) is compared to PQ centroids
//!   once, building a `[M × 256]` lookup table.  Every candidate doc then
//!   costs only `M` table look‑ups — no per‑doc f32 distance computation.
//!
//! ## Design
//! - Pure Rust, zero new dependencies.
//! - K‑Means is intentionally simple (random init, fixed iterations).
//! - IVF uses the configured metric; PQ sub‑spaces always use Euclidean
//!   (standard convention — sub‑vectors are not normalised).
//! - `D` must be divisible by `M`.

use crate::distance::{self, Metric};
use crate::doc::Document;
use crate::index::{Index, ScoredDocument};
use crate::storage::traits::VectorStorage;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// IVF-PQ index configuration.
#[derive(Debug, Clone)]
pub struct IvfPqConfig {
    /// Number of IVF clusters (`K`). Default: 256.
    pub n_clusters: usize,
    /// Number of PQ sub‑vectors (`M`). `D` must be divisible by `M`. Default: 8.
    pub n_subvectors: usize,
    /// Clusters to probe per search. Default: 8.
    pub n_probe: usize,
    /// Distance metric for IVF and final scoring.
    pub metric: Metric,
}

impl Default for IvfPqConfig {
    fn default() -> Self {
        Self {
            n_clusters: 256,
            n_subvectors: 8,
            n_probe: 8,
            metric: Metric::Cosine,
        }
    }
}

// ---------------------------------------------------------------------------
// K‑Means helper
// ---------------------------------------------------------------------------

/// Simple K‑Means clustering.
///
/// Returns `k` centroids (each `D`-dimensional).  Initialisation picks `k`
/// random data points.  Runs up to `max_iter` iterations, early‑stopping
/// when no assignment changes.
fn kmeans(data: &[Vec<f32>], k: usize, max_iter: usize, metric: Metric) -> Vec<Vec<f32>> {
    if data.is_empty() || k == 0 {
        return Vec::new();
    }
    let dim = data[0].len();
    let n = data.len();

    // 1. Initialise: pick k distinct data points
    let mut rng = 42u64;
    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(k);
    let mut used = vec![false; n];
    for _ in 0..k.min(n) {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let idx = (rng >> 33) as usize % n;
        if !used[idx] {
            centroids.push(data[idx].clone());
            used[idx] = true;
        } else {
            // Fallback: sequential scan for unused
            for i in 0..n {
                if !used[i] {
                    centroids.push(data[i].clone());
                    used[i] = true;
                    break;
                }
            }
        }
    }
    while centroids.len() < k {
        centroids.push(vec![0.0; dim]);
    }

    // 2. Iterate assignment → update
    let mut assignments = vec![0usize; n];
    for _iter in 0..max_iter {
        let mut changed = false;

        // Assignment: each point → nearest centroid (higher score = closer)
        for (i, vec) in data.iter().enumerate() {
            let mut best = 0;
            let mut best_score = f32::NEG_INFINITY;
            for (j, c) in centroids.iter().enumerate() {
                let s = distance::score(vec, c, metric);
                if s > best_score {
                    best_score = s;
                    best = j;
                }
            }
            if assignments[i] != best {
                assignments[i] = best;
                changed = true;
            }
        }
        if !changed {
            break;
        }

        // Update: mean of assigned points
        let mut sums: Vec<Vec<f32>> = vec![vec![0.0f32; dim]; k];
        let mut counts = vec![0usize; k];
        for (i, vec) in data.iter().enumerate() {
            let c = assignments[i];
            for (d, &v) in vec.iter().enumerate() {
                sums[c][d] += v;
            }
            counts[c] += 1;
        }
        for (j, sum) in sums.iter().enumerate() {
            if counts[j] > 0 {
                let inv = 1.0 / counts[j] as f32;
                for d in 0..dim {
                    centroids[j][d] = sum[d] * inv;
                }
            }
        }
    }

    centroids
}

// ---------------------------------------------------------------------------
// IvfPqIndex
// ---------------------------------------------------------------------------

/// IVF-PQ approximate nearest neighbour index.
///
/// # Example
/// ```
/// use dogma_vdb::index::{IvfPqConfig, IvfPqIndex};
/// use dogma_vdb::distance::Metric;
/// use dogma_vdb::doc::Document;
/// use dogma_vdb::index::Index;
///
/// let mut idx = IvfPqIndex::new(IvfPqConfig {
///     n_clusters: 4,
///     n_subvectors: 2,
///     n_probe: 2,
///     metric: Metric::Cosine,
/// });
///
/// let doc = Document::builder("a", "hello")
///     .embedding(vec![1.0, 0.0, 0.0, 0.0])
///     .build();
/// idx.insert(&[doc]);
///
/// let results = idx.search(&[1.0, 0.0, 0.0, 0.0], 5);
/// assert_eq!(results.len(), 1);
/// ```
#[derive(Clone)]
pub struct IvfPqIndex {
    documents: Vec<Document>,
    config: IvfPqConfig,

    // IVF
    centroids: Vec<Vec<f32>>,  // [K][D]
    clusters: Vec<Vec<usize>>, // [K] — doc IDs per cluster

    // PQ
    codebooks: Vec<Vec<Vec<f32>>>, // [M][256][subdim]
    codes: Vec<Vec<u8>>,           // [N][M]

    /// Zero-copy embedding storage (optional).
    storage: Option<Arc<dyn VectorStorage>>,
}

impl std::fmt::Debug for IvfPqIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IvfPqIndex")
            .field("documents", &self.documents)
            .field("config", &self.config)
            .field("centroids", &self.centroids)
            .field("clusters", &self.clusters)
            .field("codebooks", &self.codebooks)
            .field("codes", &self.codes)
            .field("storage", &self.storage.as_ref().map(|_| ".."))
            .finish()
    }
}

impl IvfPqIndex {
    pub fn new(config: IvfPqConfig) -> Self {
        Self {
            documents: Vec::new(),
            config,
            centroids: Vec::new(),
            clusters: Vec::new(),
            codebooks: Vec::new(),
            codes: Vec::new(),
            storage: None,
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
    pub fn config(&self) -> &IvfPqConfig {
        &self.config
    }

    // ------------------------------------------------------------------
    // Insert
    // ------------------------------------------------------------------

    /// Insert documents and rebuild the index.
    ///
    /// Each document **must** have a non‑empty embedding.  Documents
    /// without an embedding are silently skipped.
    ///
    /// IVF-PQ rebuilds from scratch after every insert (the codebooks
    /// depend on the full dataset distribution).
    pub fn insert(&mut self, docs: &[Document]) {
        let valid: Vec<Document> = docs
            .iter()
            .filter(|d| !d.embedding.is_empty())
            .cloned()
            .collect();
        if valid.is_empty() {
            return;
        }
        self.documents.extend(valid);
        self.build_index();
    }

    /// (Re)build the entire IVF-PQ index from `self.documents`.
    fn build_index(&mut self) {
        let n = self.documents.len();
        if n == 0 {
            return;
        }

        let dim = self.documents[0].embedding.len();
        let m = self.config.n_subvectors.max(1);
        let subdim = dim / m;
        assert!(
            subdim > 0,
            "IVF-PQ: dimension ({dim}) must be >= n_subvectors ({m})"
        );

        let all_vecs: Vec<Vec<f32>> = if let Some(ref storage) = self.storage {
            let flat = storage.as_embeddings();
            (0..n)
                .map(|i| {
                    let start = i * dim;
                    flat[start..start + dim].to_vec()
                })
                .collect()
        } else {
            self.documents.iter().map(|d| d.embedding.clone()).collect()
        };

        // ---- IVF: K‑Means ----
        let k = self.config.n_clusters.min(n);
        self.centroids = kmeans(&all_vecs, k, 20, self.config.metric);

        // Assign docs to nearest centroid
        self.clusters = vec![Vec::new(); k];
        for (i, vec) in all_vecs.iter().enumerate() {
            let mut best = 0;
            let mut best_score = f32::NEG_INFINITY;
            for (j, c) in self.centroids.iter().enumerate() {
                let s = distance::score(vec, c, self.config.metric);
                if s > best_score {
                    best_score = s;
                    best = j;
                }
            }
            self.clusters[best].push(i);
        }

        // ---- PQ: train codebooks ----
        let cb_size = 256usize;
        self.codebooks = Vec::with_capacity(m);

        for sub_idx in 0..m {
            let start = sub_idx * subdim;
            let end = start + subdim;
            let subvecs: Vec<Vec<f32>> = all_vecs.iter().map(|v| v[start..end].to_vec()).collect();

            let cb = kmeans(&subvecs, cb_size.min(n), 20, Metric::Euclidean);
            self.codebooks.push(cb);
        }

        // ---- PQ: encode ----
        self.codes = Vec::with_capacity(n);
        for vec in &all_vecs {
            let mut code = Vec::with_capacity(m);
            for sub_idx in 0..m {
                let start = sub_idx * subdim;
                let end = start + subdim;
                let subvec = &vec[start..end];

                let mut best = 0u8;
                let mut best_dist = f32::NEG_INFINITY;
                for (c_idx, centroid) in self.codebooks[sub_idx].iter().enumerate() {
                    let s = distance::score(subvec, centroid, Metric::Euclidean);
                    if s > best_dist {
                        best_dist = s;
                        best = c_idx as u8;
                    }
                }
                code.push(best);
            }
            self.codes.push(code);
        }
    }

    // ------------------------------------------------------------------
    // Search
    // ------------------------------------------------------------------

    /// Search for the `k` approximate nearest neighbours.
    ///
    /// 1. Finds the `probe` closest IVF centroids.
    /// 2. Pre‑computes PQ lookup tables for the query.
    /// 3. Scans all candidates via table look‑ups.
    /// 4. Global sort → top‑k.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<ScoredDocument> {
        if self.documents.is_empty() || k == 0 {
            return Vec::new();
        }

        let m = self.config.n_subvectors.max(1);
        let subdim = query.len() / m;
        let probe = self.config.n_probe.min(self.centroids.len());

        // 1. Score all centroids, pick top `probe`
        let mut centroid_scores: Vec<(usize, f32)> = self
            .centroids
            .par_iter()
            .enumerate()
            .map(|(i, c)| {
                let s = distance::score(query, c, self.config.metric);
                (i, s)
            })
            .collect();
        centroid_scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        let active_clusters: Vec<usize> = centroid_scores
            .into_iter()
            .take(probe)
            .map(|(i, _)| i)
            .collect();

        // 2. Pre‑compute PQ lookup tables [M][256]
        let luts: Vec<Vec<f32>> = (0..m)
            .into_par_iter()
            .map(|sub_idx| {
                let start = sub_idx * subdim;
                let end = start + subdim;
                let q_sub = &query[start..end];
                let cb = &self.codebooks[sub_idx];
                let mut lut = Vec::with_capacity(256);
                for c in cb {
                    let s = distance::score(q_sub, c, self.config.metric);
                    lut.push(s);
                }
                lut
            })
            .collect();

        // 3. Scan docs in selected clusters
        let mut results: Vec<ScoredDocument> = active_clusters
            .par_iter()
            .flat_map(|&ci| {
                let cluster = &self.clusters[ci];
                let mut local: Vec<ScoredDocument> = cluster
                    .iter()
                    .map(|&doc_id| {
                        let code = &self.codes[doc_id];
                        let score: f32 = (0..m).map(|s| luts[s][code[s] as usize]).sum();
                        ScoredDocument {
                            score,
                            document: self.documents[doc_id].clone(),
                        }
                    })
                    .collect();
                local.sort_unstable_by(|a, b| {
                    b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal)
                });
                local
            })
            .collect();

        // 4. Global sort & truncate
        results.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        results.truncate(k);
        results
    }
}

// ---------------------------------------------------------------------------
// Index trait impl
// ---------------------------------------------------------------------------

impl Index for IvfPqIndex {
    fn set_storage(&mut self, storage: Arc<dyn VectorStorage>) {
        self.storage = Some(storage);
    }

    fn insert(&mut self, docs: &[Document]) {
        self.insert(docs);
    }

    fn documents(&self) -> &[Document] {
        &self.documents
    }

    fn delete(&mut self, ids: &[&str]) -> usize {
        let before = self.documents.len();
        let id_set: std::collections::HashSet<&str> = ids.iter().copied().collect();
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
            self.documents = remaining;
            self.build_index();
        }
        deleted
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<ScoredDocument> {
        self.search(query, k)
    }

    fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        filter: &(dyn Fn(&Document) -> bool + Sync),
    ) -> Vec<ScoredDocument> {
        let multiplier = (k * 3).max(self.config.n_probe * 2);
        self.search(query, multiplier)
            .into_iter()
            .filter(|r| filter(&r.document))
            .take(k)
            .collect()
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
    use crate::index::BruteForceIndex;

    fn make_doc(id: &str, embedding: Vec<f32>) -> Document {
        Document::builder(id, id).embedding(embedding).build()
    }

    fn small_config() -> IvfPqConfig {
        IvfPqConfig {
            n_clusters: 4,
            n_subvectors: 2,
            n_probe: 2,
            metric: Metric::Cosine,
        }
    }

    #[test]
    fn test_empty_index() {
        let idx = IvfPqIndex::new(small_config());
        assert!(idx.search(&[1.0, 0.0, 0.0, 0.0], 5).is_empty());
        assert!(idx.is_empty());
    }

    #[test]
    fn test_single_insert() {
        let mut idx = IvfPqIndex::new(small_config());
        idx.insert(&[make_doc("a", vec![1.0, 0.0, 0.0, 0.0])]);
        assert_eq!(idx.len(), 1);
        let results = idx.search(&[1.0, 0.0, 0.0, 0.0], 5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].document.id, "a");
    }

    #[test]
    fn test_insert_batch() {
        let mut idx = IvfPqIndex::new(small_config());
        let docs: Vec<Document> = (0..10)
            .map(|i| make_doc(&format!("d{}", i), vec![i as f32, 0.0, 0.0, 0.0]))
            .collect();
        idx.insert(&docs);
        assert_eq!(idx.len(), 10);
    }

    #[test]
    fn test_search_returns_closest() {
        let mut idx = IvfPqIndex::new(small_config());
        idx.insert(&[
            make_doc("a", vec![1.0, 0.0, 0.0, 0.0]),
            make_doc("b", vec![0.0, 1.0, 0.0, 0.0]),
        ]);
        let results = idx.search(&[1.0, 0.0, 0.0, 0.0], 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].document.id, "a");
    }

    #[test]
    fn test_documents_without_embedding_skipped() {
        let mut idx = IvfPqIndex::new(small_config());
        idx.insert(&[
            make_doc("a", vec![1.0, 0.0, 0.0, 0.0]),
            Document::new("b", "no embedding"),
        ]);
        assert_eq!(idx.len(), 1);
    }

    #[test]
    fn test_delete() {
        let mut idx = IvfPqIndex::new(small_config());
        idx.insert(&[
            make_doc("a", vec![1.0, 0.0, 0.0, 0.0]),
            make_doc("b", vec![0.0, 1.0, 0.0, 0.0]),
        ]);
        assert_eq!(idx.len(), 2);
        let deleted = Index::delete(&mut idx, &["a"]);
        assert_eq!(deleted, 1);
        assert_eq!(idx.len(), 1);
    }

    #[test]
    fn test_recall_against_bf() {
        let mut bf = BruteForceIndex::new(Metric::Cosine);
        let mut ivf = IvfPqIndex::new(IvfPqConfig {
            n_clusters: 5,
            n_subvectors: 2,
            n_probe: 3,
            metric: Metric::Cosine,
        });

        let mut docs = Vec::with_capacity(50);
        for i in 0..50 {
            let angle = i as f64 * 0.12566;
            docs.push(make_doc(
                &format!("d{}", i),
                vec![angle.cos() as f32, angle.sin() as f32, 0.0, 0.0],
            ));
        }
        bf.insert(&docs);
        ivf.insert(&docs);

        let query = vec![1.0, 0.0, 0.0, 0.0];
        let bf_res = bf.search(&query, 5);
        let ivf_res = ivf.search(&query, 5);

        let bf_ids: std::collections::HashSet<&str> =
            bf_res.iter().map(|r| r.document.id.as_str()).collect();
        let hits = ivf_res
            .iter()
            .filter(|r| bf_ids.contains(r.document.id.as_str()))
            .count();
        assert!(
            hits >= 2,
            "IVF-PQ recall too low: {}/5 (expected >= 2)",
            hits
        );
    }

    #[test]
    fn test_search_results_sorted() {
        let mut idx = IvfPqIndex::new(small_config());
        let docs: Vec<Document> = (0..20)
            .map(|i| {
                let angle = i as f64 * 0.314159;
                make_doc(
                    &format!("d{}", i),
                    vec![angle.cos() as f32, angle.sin() as f32, 0.0, 0.0],
                )
            })
            .collect();
        idx.insert(&docs);

        let results = idx.search(&[1.0, 0.0, 0.0, 0.0], 10);
        assert!(!results.is_empty(), "should return at least some results");
        for i in 0..results.len().saturating_sub(1) {
            assert!(
                results[i].score >= results[i + 1].score,
                "results should be sorted by score descending"
            );
        }
    }
}
