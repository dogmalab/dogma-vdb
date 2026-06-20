use crate::distance::{self, Metric};
use crate::doc::Document;
use crate::index::kmeans::kmeans;

use super::IvfPqIndex;

impl IvfPqIndex {
    // ------------------------------------------------------------------
    // Insert
    // ------------------------------------------------------------------

    /// Insert documents — uses incremental append for small batches
    /// and full rebuild when stale ratio exceeds the threshold.
    ///
    /// Each document **must** have a non‑empty embedding.  Documents
    /// without an embedding are silently skipped.
    ///
    /// **Insertion logic:**
    /// - **Empty index** or batch ≥ `batch_rebuild_size` → full rebuild.
    /// - **Otherwise** → incremental append (frozen centroids + codebooks).
    ///   When `stale_ratio` exceeds `rebuild_threshold`, a full rebuild
    ///   is triggered automatically.
    pub fn insert(&mut self, docs: &[Document]) {
        if docs.is_empty() {
            return;
        }

        // Memory guard
        if let Err(e) = crate::memory::ensure_memory() {
            log::error!("MemoryGuard blocked IvfPqIndex::insert: {e}");
            return;
        }

        // Count valid docs (non-empty embeddings)
        let valid_count = docs.iter().filter(|d| !d.embedding.is_empty()).count();
        if valid_count == 0 {
            return;
        }

        // Empty index or large batch → full rebuild
        if self.documents.is_empty() || valid_count >= self.config.batch_rebuild_size {
            let valid: Vec<Document> = docs
                .iter()
                .filter(|d| !d.embedding.is_empty())
                .cloned()
                .collect();
            self.documents.extend(valid);
            self.build_index();
            return;
        }

        // Incremental append
        let appended = self.append_incremental(docs);
        if appended == 0 {
            return;
        }
        self.stale_docs += appended;

        // Check if we need a full rebuild
        let total = self.documents.len();
        let stale_ratio = self.stale_docs as f64 / total as f64;

        if stale_ratio >= self.config.rebuild_threshold {
            log::warn!(
                "IVF-PQ stale ratio {:.1}% >= {:.0}% — triggering full rebuild",
                stale_ratio * 100.0,
                self.config.rebuild_threshold * 100.0,
            );
            // self.documents already contains the incremental additions,
            // so build_index() reconstructs everything from scratch
            self.build_index();
        }
    }

    /// (Re)build the entire IVF-PQ index from `self.documents`.
    pub(crate) fn build_index(&mut self) {
        let n = self.documents.len();
        if n == 0 {
            return;
        }

        // Memory guard: build_index allocates all_vecs (large allocation)
        if let Err(e) = crate::memory::ensure_memory() {
            log::error!("MemoryGuard blocked IvfPqIndex::build_index: {e}");
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
        self.centroids = kmeans(&all_vecs, k, 30, self.config.metric);

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

            let cb = kmeans(&subvecs, cb_size.min(n), 30, Metric::Euclidean);
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

        self.stale_docs = 0;
        self.tombstones = vec![false; self.documents.len()];
        self.tombstone_count = 0;
    }

    /// Compact away tombstoned documents and rebuild the index from scratch.
    ///
    /// This is triggered when `tombstone_ratio()` exceeds the configured
    /// `rebuild_threshold`.  All live documents are re-indexed with fresh
    /// centroids and codebooks.
    pub(crate) fn rebuild_compact(&mut self) {
        let live: Vec<Document> = self
            .documents
            .iter()
            .zip(self.tombstones.iter())
            .filter(|(_, &tomb)| !tomb)
            .map(|(d, _)| d.clone())
            .collect();

        let config = self.config.clone();
        *self = Self::new(config);
        self.documents = live;
        self.build_index();
    }

    /// Append documents incrementally — no centroids or codebooks are updated.
    ///
    /// Each new doc is:
    /// 1. Assigned to the nearest existing IVF centroid.
    /// 2. Encoded with the existing PQ codebooks.
    /// 3. Appended to `self.documents`, `self.codes`, `self.clusters[best]`.
    ///
    /// Returns the number of valid documents actually appended.
    ///
    /// # Panics
    /// Panics if called on an uninitialised index (no centroids exist).
    fn append_incremental(&mut self, docs: &[Document]) -> usize {
        if docs.is_empty() || self.centroids.is_empty() {
            return 0;
        }

        let valid: Vec<&Document> = docs.iter().filter(|d| !d.embedding.is_empty()).collect();
        if valid.is_empty() {
            return 0;
        }

        let dim = self.documents[0].embedding.len();
        let m = self.config.m_subspaces.max(1);
        let subdim = dim / m;

        for &doc in &valid {
            let emb = &doc.embedding;

            // 1. Assign to nearest existing centroid
            let mut best_ci = 0usize;
            let mut best_score = f32::NEG_INFINITY;
            for (j, c) in self.centroids.iter().enumerate() {
                let s = distance::score(emb, c, self.config.metric);
                if s > best_score {
                    best_score = s;
                    best_ci = j;
                }
            }

            // 2. PQ encode with existing codebooks
            let mut code = Vec::with_capacity(m);
            for sub_idx in 0..m {
                let start = sub_idx * subdim;
                let end = start + subdim;
                let subvec = &emb[start..end];
                let cb = &self.codebooks[sub_idx];

                let mut best_c = 0u8;
                let mut best_dist = f32::NEG_INFINITY;
                for (c_idx, centroid) in cb.iter().enumerate() {
                    let s = distance::score(subvec, centroid, self.config.metric);
                    if s > best_dist {
                        best_dist = s;
                        best_c = c_idx as u8;
                    }
                }
                code.push(best_c);
            }

            // 3. Append
            self.documents.push(doc.clone());
            self.codes.push(code);
            self.tombstones.push(false);
            self.clusters[best_ci].push(self.documents.len() - 1);
        }

        valid.len()
    }
}
