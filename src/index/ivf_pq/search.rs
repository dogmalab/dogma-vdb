use crate::distance;
use crate::index::ScoredDocument;
use rayon::prelude::*;
use std::cmp::Ordering;

use super::IvfPqIndex;

impl IvfPqIndex {
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

        // 3. Scan docs in selected clusters — skip tombstoned docs,
        //    lightweight CandidateResult, NO Document cloning.
        let mut results: Vec<CandidateResult> = active_clusters
            .par_iter()
            .flat_map(|&ci| {
                let cluster = &self.clusters[ci];
                let mut local: Vec<CandidateResult> = cluster
                    .iter()
                    .filter(|&&doc_id| !self.tombstones[doc_id])
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
