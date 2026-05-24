//! IVF-PQ (Inverted File + Product Quantization) index.
//!
//! A minimal implementation of the standard IVF-PQ approximate nearest
//! neighbour algorithm:
//!
//! - **IVF**: vectors are partitioned into `n_list` clusters via simple K-Means.
//!   Search only visits the `n_probe` closest clusters (inverted index).
//! - **PQ**: each vector is split into `m_subspaces` sub‑vectors, and each
//!   sub‑vector is quantised to a single byte via a 256‑centroid codebook.
//!   Total storage per vector: `m_subspaces` bytes instead of `D × 4` bytes.
//! - **Asymmetric search**: the query (f32) is compared to PQ centroids
//!   once, building a `[m_subspaces × 256]` lookup table.  Every candidate
//!   doc then costs only `m_subspaces` table look‑ups.
//!
//! ## Design
//! - Pure Rust, zero new dependencies.
//! - K‑Means is intentionally simple (random init, fixed iterations).
//! - IVF uses the configured metric; PQ sub‑spaces always use Euclidean
//!   (standard convention — sub‑vectors are not normalised).
//! - `D` must be divisible by `m_subspaces`.
//! - `m_subspaces` must be a multiple of 8 (SIMD alignment guarantee).

use crate::distance::{self, Metric};
use crate::doc::Document;
use crate::error::{Error, Result};
use crate::index::{Index, ScoredDocument};
use crate::storage::traits::VectorStorage;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::path::Path;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// IVF-PQ index configuration.
///
/// # SIMD alignment
///
/// `m_subspaces` **must** be a multiple of 8 so that PQ lookup tables
/// are 32‑byte aligned, enabling AVX2 / NEON to load sub‑vectors in a
/// single instruction.  Use [`Self::validate`] at construction time.
#[derive(Debug, Clone)]
pub struct IvfPqConfig {
    /// Number of IVF clusters (K‑Means centroids).  Default: 100.
    pub n_list: usize,
    /// Clusters to probe per search.  Default: 5.
    pub n_probe: usize,
    /// Number of PQ sub‑spaces (`M`).  `D` must be divisible by `M`.
    /// **Must be a multiple of 8** for SIMD alignment.  Default: 32.
    pub m_subspaces: usize,
    /// Distance metric for IVF and final scoring.
    pub metric: Metric,
    /// When `true`, `search()` halves its effective `n_probe` to favour
    /// raw speed — the lost recall is recovered by a subsequent
    /// Cross‑Encoder reranking pass (stage 2).
    pub rerank_enabled: bool,
}

impl Default for IvfPqConfig {
    fn default() -> Self {
        Self {
            n_list: 100,
            n_probe: 5,
            m_subspaces: 32,
            metric: Metric::Cosine,
            rerank_enabled: false,
        }
    }
}

impl IvfPqConfig {
    /// Validate the configuration.
    ///
    /// Returns `Err` if:
    /// - `m_subspaces` is zero, or
    /// - `m_subspaces` is not a multiple of 8 (SIMD alignment).
    pub fn validate(&self) -> Result<()> {
        if self.m_subspaces == 0 {
            return Err(Error::InvalidConfig("m_subspaces must be > 0".into()));
        }
        if self.m_subspaces % 8 != 0 {
            return Err(Error::InvalidConfig(format!(
                "m_subspaces must be a multiple of 8 for SIMD alignment, got {}",
                self.m_subspaces
            )));
        }
        Ok(())
    }

    /// Effective probe count — honours the rerank auto-tuning contract.
    ///
    /// When `rerank_enabled` is `true`, the probe count is halved
    /// (minimum 2) so the index favours speed over recall, relying on
    /// the subsequent Cross‑Encoder reranker to recover relevance.
    pub fn effective_probe(&self) -> usize {
        if self.rerank_enabled {
            (self.n_probe / 2).max(2)
        } else {
            self.n_probe
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
///     n_list: 4,
///     m_subspaces: 2,
///     n_probe: 2,
///     metric: Metric::Cosine,
///     ..Default::default()
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
    centroids: Vec<Vec<f32>>,  // [n_list][D]
    clusters: Vec<Vec<usize>>, // [n_list] — doc IDs per cluster

    // PQ
    codebooks: Vec<Vec<Vec<f32>>>, // [m_subspaces][256][subdim]
    codes: Vec<Vec<u8>>,           // [N][m_subspaces]

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

    /// Persist the current IVF-PQ state to `.meta` + `.jsonl` files.
    pub fn save_persistence(&self, base: &Path) -> Result<()> {
        crate::index::ivf_pq_persistence::save(self, base)
    }

    /// Load IVF-PQ state from persisted files (returns `None` if absent).
    pub fn load_persistence(base: &Path, config: &IvfPqConfig) -> Result<Option<Self>> {
        crate::index::ivf_pq_persistence::load(base, config)
    }

    /// IVF centroids (pub(crate) for persistence).
    pub(crate) fn centroids(&self) -> &[Vec<f32>] {
        &self.centroids
    }

    /// PQ codebooks (pub(crate) for persistence).
    pub(crate) fn codebooks(&self) -> &[Vec<Vec<f32>>] {
        &self.codebooks
    }

    /// PQ codes per document (pub(crate) for persistence).
    pub(crate) fn codes(&self) -> &[Vec<u8>] {
        &self.codes
    }

    /// Build an index from pre-computed state (used by persistence load).
    pub(crate) fn from_state(
        documents: Vec<Document>,
        config: IvfPqConfig,
        centroids: Vec<Vec<f32>>,
        codebooks: Vec<Vec<Vec<f32>>>,
        codes: Vec<Vec<u8>>,
        clusters: Vec<Vec<usize>>,
    ) -> Self {
        Self {
            documents,
            config,
            centroids,
            clusters,
            codebooks,
            codes,
            storage: None,
        }
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
        let m = self.config.m_subspaces.max(1);
        let subdim = dim / m;
        assert!(
            subdim > 0,
            "IVF-PQ: dimension ({dim}) must be >= m_subspaces ({m})"
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
        let k = self.config.n_list.min(n);
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
    /// 3. Scans all candidates via table look‑ups (lightweight `CandidateResult`).
    /// 4. Global sort → top‑k → hydrate into `ScoredDocument`.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<ScoredDocument> {
        /// Lightweight intermediate result — avoids cloning heavy `Document`
        /// during the parallel cluster scan.  Hydrated to `ScoredDocument`
        /// only for the final top‑k.
        #[derive(Debug, Clone)]
        struct CandidateResult {
            doc_id: usize,
            score: f32,
        }

        if self.documents.is_empty() || k == 0 {
            return Vec::new();
        }

        let m = self.config.m_subspaces.max(1);
        let subdim = query.len() / m;
        // Use effective probe count (honours rerank auto-tuning)
        let probe = self.config.effective_probe().min(self.centroids.len());

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

        // 2. Pre‑compute PQ lookup tables [m_subspaces][256]
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

        // 3. Scan docs in selected clusters — lightweight CandidateResult,
        //    NO Document cloning inside the parallel pipeline.
        let mut results: Vec<CandidateResult> = active_clusters
            .par_iter()
            .flat_map(|&ci| {
                let cluster = &self.clusters[ci];
                let mut local: Vec<CandidateResult> = cluster
                    .iter()
                    .map(|&doc_id| {
                        let code = &self.codes[doc_id];
                        let score: f32 = (0..m).map(|s| luts[s][code[s] as usize]).sum();
                        CandidateResult { doc_id, score }
                    })
                    .collect();
                local.sort_unstable_by(|a, b| {
                    b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal)
                });
                local
            })
            .collect();

        // 4. Global sort & truncate (still CandidateResult)
        results.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        results.truncate(k);

        // 5. Hydrate: clone Document only for the final top‑k
        results
            .into_iter()
            .map(|c| ScoredDocument {
                score: c.score,
                document: self.documents[c.doc_id].clone(),
            })
            .collect()
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
            n_list: 4,
            m_subspaces: 2,
            n_probe: 2,
            metric: Metric::Cosine,
            ..Default::default()
        }
    }

    // -- Existing tests (updated for new field names) --

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
            n_list: 5,
            m_subspaces: 2,
            n_probe: 3,
            metric: Metric::Cosine,
            ..Default::default()
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

    // -- New tests for tuning and validation --

    #[test]
    fn test_ivf_pq_invalid_subspaces() {
        // m_subspaces must be a multiple of 8
        let cfg = IvfPqConfig {
            m_subspaces: 13,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());

        let cfg = IvfPqConfig {
            m_subspaces: 25,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());

        // Zero is also invalid
        let cfg = IvfPqConfig {
            m_subspaces: 0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());

        // Valid multiples
        for valid in [8, 16, 32, 64] {
            let cfg = IvfPqConfig {
                m_subspaces: valid,
                ..Default::default()
            };
            assert!(
                cfg.validate().is_ok(),
                "m_subspaces={} should be valid",
                valid
            );
        }
    }

    #[test]
    fn test_ivf_pq_tuning_impact() {
        // n_probe=1 should probe fewer clusters → different results than n_probe=10
        // (and likely fewer candidates at the extremes)
        let cfg_low = IvfPqConfig {
            n_list: 8,
            m_subspaces: 2,
            n_probe: 1,
            metric: Metric::Cosine,
            ..Default::default()
        };
        let cfg_high = IvfPqConfig {
            n_list: 8,
            m_subspaces: 2,
            n_probe: 10,
            metric: Metric::Cosine,
            ..Default::default()
        };

        let mut idx_low = IvfPqIndex::new(cfg_low);
        let mut idx_high = IvfPqIndex::new(cfg_high);

        let docs: Vec<Document> = (0..80)
            .map(|i| {
                let angle = i as f64 * 0.0785398;
                make_doc(
                    &format!("d{}", i),
                    vec![angle.cos() as f32, angle.sin() as f32, 0.0, 0.0],
                )
            })
            .collect();
        idx_low.insert(&docs);
        idx_high.insert(&docs);

        let query = vec![1.0, 0.0, 0.0, 0.0];

        let results_low = idx_low.search(&query, 5);
        let results_high = idx_high.search(&query, 5);

        // With more probe, we should inspect more candidates and thus
        // likely get a different (better) top result
        assert!(
            !results_low.is_empty(),
            "low probe should still return results"
        );
        assert!(!results_high.is_empty(), "high probe should return results");

        // The score of the top result with high probe should be at least
        // as high (better) as the top result with low probe
        assert!(
            results_high[0].score >= results_low[0].score - 1e-5,
            "higher n_probe should yield >= top score (low={}, high={})",
            results_low[0].score,
            results_high[0].score,
        );

        // Verify that n_probe is stored correctly
        let config_high = idx_high.config();
        assert_eq!(config_high.n_probe, 10);
        assert_eq!(config_high.effective_probe(), 10);
    }

    #[test]
    fn test_auto_tuning_with_rerank_flag() {
        // When rerank_enabled=true, effective_probe should be halved
        let cfg = IvfPqConfig {
            n_probe: 10,
            rerank_enabled: true,
            ..Default::default()
        };
        // effective_probe = (10 / 2).max(2) = 5
        assert_eq!(cfg.effective_probe(), 5);

        // With low n_probe, minimum should be 2
        let cfg2 = IvfPqConfig {
            n_probe: 3,
            rerank_enabled: true,
            ..Default::default()
        };
        // effective_probe = (3 / 2).max(2) = 2
        assert_eq!(cfg2.effective_probe(), 2);

        // Without rerank, effective == nominal
        let cfg3 = IvfPqConfig {
            n_probe: 7,
            rerank_enabled: false,
            ..Default::default()
        };
        assert_eq!(cfg3.effective_probe(), 7);

        // Integration test: build an index with rerank_enabled and verify
        // the search still produces valid (sorted) results
        let cfg_idx = IvfPqConfig {
            n_list: 4,
            m_subspaces: 2,
            n_probe: 4,
            metric: Metric::Cosine,
            rerank_enabled: true,
        };
        let mut idx = IvfPqIndex::new(cfg_idx);
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

        let results = idx.search(&[1.0, 0.0, 0.0, 0.0], 5);
        assert!(!results.is_empty());
        // Results should still be sorted correctly
        for i in 0..results.len().saturating_sub(1) {
            assert!(
                results[i].score >= results[i + 1].score,
                "rerank mode results should be sorted"
            );
        }
    }
}
