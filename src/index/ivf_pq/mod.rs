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

use crate::distance::Metric;
use crate::doc::Document;
use crate::error::{Error, Result};
use crate::index::{Index, ScoredDocument};
use crate::storage::traits::VectorStorage;
use std::path::Path;
use std::sync::Arc;

mod insert;
mod search;

#[cfg(test)]
mod tests;

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
///
/// # Incremental inserts
///
/// When `rebuild_threshold > 0.0`, small batch inserts use an incremental
/// append path (frozen centroids + codebooks) and only trigger a full
/// K‑Means + PQ rebuild when the ratio of stale (incrementally-added)
/// docs exceeds the threshold.  Large batches (≥ `batch_rebuild_size`)
/// always rebuild immediately.
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
    /// Maximum ratio of stale (incrementally-added) docs vs total before
    /// a full K‑Means + PQ rebuild is triggered.  Default: 0.20 (20%).
    ///
    /// - `0.0` = always rebuild (batch-only mode, same as original behaviour).
    /// - `f64::MAX` = never auto-rebuild (purely incremental, recall degrades).
    pub rebuild_threshold: f64,
    /// Minimum batch size that triggers an immediate full rebuild instead
    /// of incremental append.  Default: 1000.
    pub batch_rebuild_size: usize,
}

impl Default for IvfPqConfig {
    fn default() -> Self {
        Self {
            n_list: 100,
            n_probe: 5,
            m_subspaces: 32,
            metric: Metric::Cosine,
            rerank_enabled: false,
            rebuild_threshold: 0.20,
            batch_rebuild_size: 1000,
        }
    }
}

impl IvfPqConfig {
    /// Validate the configuration.
    ///
    /// Returns `Err` if:
    /// - `m_subspaces` is zero, or
    /// - `m_subspaces` is not a multiple of 8 (SIMD alignment).
    /// - `rebuild_threshold` is negative.
    /// - `batch_rebuild_size` is zero.
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
        if self.rebuild_threshold < 0.0 {
            return Err(Error::InvalidConfig(format!(
                "rebuild_threshold must be >= 0.0, got {}",
                self.rebuild_threshold
            )));
        }
        if self.batch_rebuild_size == 0 {
            return Err(Error::InvalidConfig(
                "batch_rebuild_size must be > 0".into(),
            ));
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
///     m_subspaces: 8,
///     n_probe: 2,
///     metric: Metric::Cosine,
///     ..Default::default()
/// });
///
/// let doc = Document::builder("a", "hello")
///     .embedding(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
///     .build();
/// idx.insert(&[doc]);
///
/// let results = idx.search(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 5);
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

    /// Number of documents added via incremental append since last full rebuild.
    /// Reset to 0 after every `build_index()` call.
    stale_docs: usize,

    /// Tombstone flags — `true` means the document at that index is deleted.
    /// Search skips tombstoned docs; rebuild compacts them away.
    tombstones: Vec<bool>,

    /// Number of tombstoned docs.  Triggers rebuild when ratio exceeds threshold.
    tombstone_count: usize,
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
            .field("stale_docs", &self.stale_docs)
            .field("tombstone_count", &self.tombstone_count)
            .finish()
    }
}

impl IvfPqIndex {
    pub fn new(config: IvfPqConfig) -> Self {
        debug_assert!(
            config.validate().is_ok(),
            "IvfPqConfig::new called with invalid config: {:?}",
            config.validate()
        );
        Self {
            documents: Vec::new(),
            config,
            centroids: Vec::new(),
            clusters: Vec::new(),
            codebooks: Vec::new(),
            codes: Vec::new(),
            storage: None,
            stale_docs: 0,
            tombstones: Vec::new(),
            tombstone_count: 0,
        }
    }

    /// Number of live (non-tombstoned) documents.
    pub fn len(&self) -> usize {
        self.documents.len() - self.tombstone_count
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Ratio of tombstoned docs vs total (0.0–1.0).
    pub fn tombstone_ratio(&self) -> f64 {
        if self.documents.is_empty() {
            return 0.0;
        }
        self.tombstone_count as f64 / self.documents.len() as f64
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

    /// Number of incrementally-added documents since last full rebuild.
    pub fn stale_docs(&self) -> usize {
        self.stale_docs
    }

    /// Ratio of stale (incrementally-added) docs vs total (0.0–1.0).
    ///
    /// When this exceeds [`IvfPqConfig::rebuild_threshold`], the next
    /// `insert()` triggers a full rebuild.
    pub fn stale_ratio(&self) -> f64 {
        if self.documents.is_empty() {
            return 0.0;
        }
        self.stale_docs as f64 / self.documents.len() as f64
    }

    /// Build an index from pre-computed state (used by persistence load).
    pub(crate) fn from_state(
        documents: Vec<Document>,
        config: IvfPqConfig,
        centroids: Vec<Vec<f32>>,
        codebooks: Vec<Vec<Vec<f32>>>,
        codes: Vec<Vec<u8>>,
        clusters: Vec<Vec<usize>>,
        stale_docs: usize,
    ) -> Self {
        let n = documents.len();
        Self {
            documents,
            config,
            centroids,
            clusters,
            codebooks,
            codes,
            storage: None,
            stale_docs,
            tombstones: vec![false; n],
            tombstone_count: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Index trait impl
// ---------------------------------------------------------------------------

impl Index for IvfPqIndex {
    fn set_storage(&mut self, storage: Arc<dyn VectorStorage>) {
        self.storage = Some(storage.clone());
        // If documents exist but centroids are empty (mmap mode),
        // rebuild the index reading embeddings from storage.
        if !self.documents.is_empty() && self.centroids.is_empty() && !storage.is_empty() {
            self.build_index();
        }
    }

    fn insert(&mut self, docs: &[Document]) {
        self.insert(docs);
    }

    fn documents(&self) -> &[Document] {
        &self.documents
    }

    fn delete(&mut self, ids: &[&str]) -> usize {
        let id_set: std::collections::HashSet<&str> = ids.iter().copied().collect();
        let mut deleted = 0usize;
        for (i, doc) in self.documents.iter().enumerate() {
            if !self.tombstones[i] && id_set.contains(doc.id.as_str()) {
                self.tombstones[i] = true;
                self.tombstone_count += 1;
                deleted += 1;
            }
        }

        // Trigger rebuild if tombstone ratio exceeds threshold
        if deleted > 0 && self.tombstone_ratio() >= self.config.rebuild_threshold {
            log::warn!(
                "IVF-PQ tombstone ratio {:.1}% >= {:.0}% — triggering rebuild",
                self.tombstone_ratio() * 100.0,
                self.config.rebuild_threshold * 100.0,
            );
            self.rebuild_compact();
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
        self.documents.len() - self.tombstone_count
    }

    fn save(&self, base_path: &std::path::Path) -> crate::error::Result<()> {
        self.save_persistence(base_path)
    }

    fn load(base_path: &std::path::Path) -> crate::error::Result<Option<Self>>
    where
        Self: Sized,
    {
        // Config is embedded in the persistence file; use default for
        // the signature — load_persistence reads it from the .meta file.
        let default_config = IvfPqConfig::default();
        crate::index::ivf_pq_persistence::load(base_path, &default_config)
    }
}
