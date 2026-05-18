//! IVF-PQ persistence — .meta + vectors.jsonl + mmap zero-copy loading.
//!
//! # Format
//!
//! `*.ivf_pq.meta` — JSON with config, centroids, codebooks:
//! ```json
//! {"config":{"n_list":100,...},"centroids":[[...],...],"codebooks":[[[...]]]}
//! ```
//!
//! `*.ivf_pq.jsonl` — one JSON object per document:
//! ```jsonl
//! {"id":"doc-0","cluster":5,"code":[12,-4,127,0,...]}
//! ```
//!
//! # Zero-copy loading
//!
//! On load, `.meta` is read into memory.  The `.jsonl` is **memory-mapped**
//! via `memmap2` — the kernel pages it on demand, lines are parsed from
//! the mapped region without intermediate kernel‑buffer copies.

use crate::doc::Document;
use crate::error::{Error, Result};
use crate::index::ivf_pq::{IvfPqConfig, IvfPqIndex};
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Serializable structures
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
struct MetaFile {
    config: MetaConfig,
    centroids: Vec<Vec<f32>>,
    codebooks: Vec<Vec<Vec<f32>>>,
}

#[derive(Serialize, Deserialize)]
struct MetaConfig {
    n_list: usize,
    m_subspaces: usize,
    n_probe: usize,
    metric: String,
    rerank_enabled: bool,
}

#[derive(Serialize, Deserialize)]
struct CodeEntry {
    id: String,
    cluster: usize,
    code: Vec<u8>,
}

// ---------------------------------------------------------------------------
// Path helpers
// ---------------------------------------------------------------------------

fn meta_path(base: &Path) -> PathBuf {
    let stem = base.file_stem().and_then(|s| s.to_str()).unwrap_or("col");
    let mut p = base.to_path_buf();
    p.set_file_name(format!("{stem}.ivf_pq.meta"));
    p
}

fn jsonl_path(base: &Path) -> PathBuf {
    let stem = base.file_stem().and_then(|s| s.to_str()).unwrap_or("col");
    let mut p = base.to_path_buf();
    p.set_file_name(format!("{stem}.ivf_pq.jsonl"));
    p
}

/// Check whether persisted IVF-PQ state exists for `base`.
pub fn exists(base: &Path) -> bool {
    meta_path(base).exists()
}

// ---------------------------------------------------------------------------
// Save
// ---------------------------------------------------------------------------

/// Save IVF-PQ index state to `.meta` + `.jsonl`.
pub fn save(index: &IvfPqIndex, base: &Path) -> Result<()> {
    let metric_name = format!("{:?}", index.config().metric);

    let meta = MetaFile {
        config: MetaConfig {
            n_list: index.config().n_list,
            m_subspaces: index.config().m_subspaces,
            n_probe: index.config().n_probe,
            metric: metric_name,
            rerank_enabled: index.config().rerank_enabled,
        },
        centroids: index.centroids().to_vec(),
        codebooks: index.codebooks().to_vec(),
    };

    let meta_str = serde_json::to_string_pretty(&meta)
        .map_err(|e| Error::Internal(format!("IVF-PQ meta serialize: {e}")))?;
    std::fs::write(meta_path(base), &meta_str).map_err(|e| Error::Io {
        path: meta_path(base),
        source: e,
    })?;

    // Write each document's code + cluster assignment as a JSONL line
    let docs = index.documents();
    let centroids = index.centroids();
    let codes = index.codes();

    let mut f = std::fs::File::create(jsonl_path(base)).map_err(|e| Error::Io {
        path: jsonl_path(base),
        source: e,
    })?;

    for (i, doc) in docs.iter().enumerate() {
        // Find nearest centroid for this doc
        let cluster = centroids
            .iter()
            .enumerate()
            .map(|(ci, c)| (ci, crate::distance::cosine(&doc.embedding, c)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(ci, _)| ci)
            .unwrap_or(0);

        let entry = CodeEntry {
            id: doc.id.clone(),
            cluster,
            code: codes.get(i).cloned().unwrap_or_default(),
        };
        let line = serde_json::to_string(&entry)
            .map_err(|e| Error::Internal(format!("IVF-PQ JSONL encode: {e}")))?;
        writeln!(f, "{line}").map_err(|e| Error::Io {
            path: jsonl_path(base),
            source: e,
        })?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Load
// ---------------------------------------------------------------------------

/// Load IVF-PQ index state from `.meta` + mmap'd `.jsonl`.
///
/// Returns `None` when no persisted state exists (caller should build
/// from scratch).  The `.jsonl` is **memory‑mapped** for zero‑copy reads
/// — lines are parsed from the mapped byte slice without a String allocation.
///
/// # Mmap safety
///
/// The mapped file must not be truncated externally while the returned
/// index is alive, or the process receives SIGBUS.  No external writes
/// to the `.jsonl` are expected after save.
pub fn load(base: &Path, config: &IvfPqConfig) -> Result<Option<IvfPqIndex>> {
    if !exists(base) {
        return Ok(None);
    }

    // Read .meta (small, sequential — no mmap needed)
    let meta_str = std::fs::read_to_string(meta_path(base)).map_err(|e| Error::Io {
        path: meta_path(base),
        source: e,
    })?;
    let meta: MetaFile = serde_json::from_str(&meta_str)
        .map_err(|e| Error::Internal(format!("IVF-PQ meta parse: {e}")))?;

    // Sanity
    if meta.centroids.len() != meta.config.n_list {
        return Err(Error::Internal(format!(
            "IVF-PQ meta: expected {} centroids, got {}",
            meta.config.n_list,
            meta.centroids.len()
        )));
    }

    // mmap .jsonl — kernel pages on demand, no malloc for file content
    let file = std::fs::File::open(jsonl_path(base)).map_err(|e| Error::Io {
        path: jsonl_path(base),
        source: e,
    })?;
    // SAFETY: memmap2 0.9 requires unsafe for Mmap::map because the caller
    // must guarantee the mapped file is not resized/truncated externally
    // while the Mmap is alive.  We control both save and load — no other
    // process modifies `.jsonl` after save.
    let mmap = unsafe { Mmap::map(&file) }
        .map_err(|e| Error::Internal(format!("IVF-PQ mmap: {e}")))?;
    let mmap_slice: &[u8] = &mmap;

    // Parse lines from the mmap'd region
    let mut codes: Vec<Vec<u8>> = Vec::new();
    let mut assignments: Vec<usize> = Vec::new();
    let mut documents: Vec<Document> = Vec::new();

    for line_bytes in mmap_slice.split(|&b| b == b'\n') {
        let line = std::str::from_utf8(line_bytes).unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let entry: CodeEntry = serde_json::from_str(line)
            .map_err(|e| Error::Internal(format!("IVF-PQ code parse: {e}")))?;

        documents.push(Document::new(&entry.id, ""));
        assignments.push(entry.cluster);
        codes.push(entry.code);
    }

    // Build clusters from assignments
    let n_list = meta.config.n_list;
    let mut clusters: Vec<Vec<usize>> = vec![Vec::new(); n_list];
    for (doc_idx, &cluster) in assignments.iter().enumerate() {
        if cluster < n_list {
            clusters[cluster].push(doc_idx);
        }
    }

    let idx = IvfPqIndex::from_state(
        documents,
        config.clone(),
        meta.centroids,
        meta.codebooks,
        codes,
        clusters,
    );

    Ok(Some(idx))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::Metric;
    use crate::doc::Document;
    use tempfile::TempDir;

    fn make_test_docs(n: usize, dim: usize) -> Vec<Document> {
        let mut seed = 42u64;
        (0..n)
            .map(|i| {
                let emb: Vec<f32> = (0..dim)
                    .map(|_j| {
                        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                        ((seed >> 33) as f64 / 1e9) as f32
                    })
                    .collect();
                Document::builder(format!("d{i}"), format!("doc {i}"))
                    .embedding(emb)
                    .build()
            })
            .collect()
    }

    #[test]
    fn test_persistence_lifecycle() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path().join("test_collection.vdb");

        // 1. Create index with 10 docs (dim must be multiple of m_subspaces)
        let dim = 16;
        let config = IvfPqConfig {
            n_list: 4,
            m_subspaces: 8, // dim / 2, multiple of 8
            n_probe: 2,
            metric: Metric::Cosine,
            ..Default::default()
        };
        config.validate().unwrap();

        let mut index = IvfPqIndex::new(config.clone());
        let docs = make_test_docs(10, dim);
        index.insert(&docs);

        // Capture results before save
        let query: Vec<f32> = docs[0].embedding.clone();
        let before = index.search(&query, 5);
        assert_eq!(before.len(), 5);

        // 2. Save to disk
        index.save_persistence(&base).unwrap();

        // Verify .meta and .jsonl exist
        assert!(exists(&base));
        assert!(meta_path(&base).exists());
        assert!(jsonl_path(&base).exists());

        // 3. Destroy and reload
        drop(index);
        let reloaded = IvfPqIndex::load_persistence(&base, &config)
            .unwrap()
            .expect("persistence should return Some");

        // 4. Search with same query — same results + scores
        let after = reloaded.search(&query, 5);
        assert_eq!(after.len(), 5, "reloaded index must return same count");

        for (b, a) in before.iter().zip(after.iter()) {
            assert_eq!(
                b.document.id, a.document.id,
                "result IDs must match after reload"
            );
            assert!(
                (b.score - a.score).abs() < 1e-5,
                "scores must match: {:.6} vs {:.6}",
                b.score,
                a.score
            );
        }
    }

    #[test]
    fn test_persistence_nonexistent() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path().join("ghost.vdb");
        assert!(!exists(&base));
        let config = IvfPqConfig::default();
        let result = IvfPqIndex::load_persistence(&base, &config).unwrap();
        assert!(result.is_none(), "no files → None");
    }

    #[test]
    fn test_persistence_empty_index() {
        // Save an index with no documents
        let tmp = TempDir::new().unwrap();
        let base = tmp.path().join("empty.vdb");
        let config = IvfPqConfig {
            n_list: 4,
            m_subspaces: 8,
            n_probe: 2,
            metric: Metric::Cosine,
            ..Default::default()
        };
        let mut index = IvfPqIndex::new(config.clone());
        // Insert 0 docs (empty index shouldn't save, but shouldn't panic)
        index.insert(&[]);
        index.save_persistence(&base).unwrap();
        assert!(exists(&base));
    }
}
