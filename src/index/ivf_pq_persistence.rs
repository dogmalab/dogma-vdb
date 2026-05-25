//! IVF-PQ persistence — atomic saves, soft-delete, compaction.
//!
//! # Atomic write protocol
//!
//! Every save writes to a `.tmp` file first, calls `sync_all()`, then
//! atomically renames (POSIX `rename`) over the target file.  If the
//! process crashes mid-write, the original file remains intact.
//!
//! # Soft-delete format
//!
//! Each `.jsonl` line may include `"d":1` to mark a record as deleted.
//! On load, lines are scanned **in reverse** with a dedup set, so the
//! latest version of each doc wins and deleted entries are ignored.
//!
//! # Compact
//!
//! `compact()` rewrites the `.jsonl` keeping only the active revision
//! of each document, reclaiming space from deleted / overwritten lines.

use crate::doc::Document;
use crate::error::{Error, Result};
use crate::index::ivf_pq::{IvfPqConfig, IvfPqIndex};
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
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
    /// Soft-delete flag.  `1` means this record is dead.
    #[serde(default)]
    d: u8,
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

pub fn exists(base: &Path) -> bool {
    meta_path(base).exists()
}

// ---------------------------------------------------------------------------
// Atomic save helper
// ---------------------------------------------------------------------------

/// Write `data` to `path` atomically: write to `.tmp`, sync, rename.
fn atomic_write(path: &Path, data: &[u8]) -> Result<()> {
    let tmp = path.with_extension("tmp");
    {
        let mut f = std::fs::File::create(&tmp).map_err(|e| Error::Io {
            path: tmp.clone(),
            source: e,
        })?;
        f.write_all(data).map_err(|e| Error::Io {
            path: tmp.clone(),
            source: e,
        })?;
        f.sync_all().map_err(|e| Error::Io {
            path: tmp.clone(),
            source: e,
        })?;
    }
    std::fs::rename(&tmp, path).map_err(|e| Error::Io {
        path: path.to_path_buf(),
        source: e,
    })
}

/// Write each line to `path` atomically via temp file.
fn atomic_write_lines(path: &Path, lines: &[String]) -> Result<()> {
    let tmp = path.with_extension("tmp");
    {
        let mut f = std::fs::File::create(&tmp).map_err(|e| Error::Io {
            path: tmp.clone(),
            source: e,
        })?;
        for line in lines {
            writeln!(f, "{line}").map_err(|e| Error::Io {
                path: tmp.clone(),
                source: e,
            })?;
        }
        f.sync_all().map_err(|e| Error::Io {
            path: tmp.clone(),
            source: e,
        })?;
    }
    std::fs::rename(&tmp, path).map_err(|e| Error::Io {
        path: path.to_path_buf(),
        source: e,
    })
}

// ---------------------------------------------------------------------------
// Save
// ---------------------------------------------------------------------------

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
    atomic_write(&meta_path(base), meta_str.as_bytes())?;

    let docs = index.documents();
    let centroids = index.centroids();
    let codes = index.codes();

    let mut lines: Vec<String> = Vec::with_capacity(docs.len());
    for (i, doc) in docs.iter().enumerate() {
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
            d: 0,
        };
        let line = serde_json::to_string(&entry)
            .map_err(|e| Error::Internal(format!("IVF-PQ JSONL encode: {e}")))?;
        lines.push(line);
    }

    atomic_write_lines(&jsonl_path(base), &lines)
}

// ---------------------------------------------------------------------------
// Load (reverse scan with soft-delete + overwrite dedup)
// ---------------------------------------------------------------------------

/// Load IVF-PQ state.  Lines are scanned **backwards** so the last valid
/// occurrence of each `id` wins.  Entries with `"d": 1` are skipped.
pub fn load(base: &Path, config: &IvfPqConfig) -> Result<Option<IvfPqIndex>> {
    if !exists(base) {
        return Ok(None);
    }

    let meta_str = std::fs::read_to_string(meta_path(base)).map_err(|e| Error::Io {
        path: meta_path(base),
        source: e,
    })?;
    let meta: MetaFile = serde_json::from_str(&meta_str)
        .map_err(|e| Error::Internal(format!("IVF-PQ meta parse: {e}")))?;
    // Sanity — allow fewer centroids than n_list (small dataset)
    if meta.centroids.is_empty() && meta.config.n_list > 0 {
        return Err(Error::Internal(format!(
            "IVF-PQ meta: no centroids but n_list={}",
            meta.config.n_list
        )));
    }

    let file = std::fs::File::open(jsonl_path(base)).map_err(|e| Error::Io {
        path: jsonl_path(base),
        source: e,
    })?;
    // SAFETY: memmap2 0.9 — caller guarantees no external truncation.
    let mmap =
        unsafe { Mmap::map(&file) }.map_err(|e| Error::Internal(format!("IVF-PQ mmap: {e}")))?;
    let _ = mmap.advise(memmap2::Advice::Random);
    let mmap_slice: &[u8] = &mmap;

    // ── Pre-allocate from file-size heuristic (~128 bytes/line avg) ──
    let estimated: usize = file
        .metadata()
        .ok()
        .and_then(|m| {
            let len = m.len() as usize;
            if len == 0 {
                None
            } else {
                Some((len / 128).max(meta.config.n_list))
            }
        })
        .unwrap_or(meta.config.n_list.max(16));

    let mut codes: Vec<Vec<u8>> = Vec::with_capacity(estimated);
    let mut assignments: Vec<usize> = Vec::with_capacity(estimated);
    let mut documents: Vec<Document> = Vec::with_capacity(estimated);
    let mut seen: HashSet<String> = HashSet::with_capacity(estimated);
    let mut non_utf8_lines: u64 = 0;
    let mut corrupt_json_lines: u64 = 0;

    // ── Reverse scan over mmap — no intermediate Vec allocation ──
    let mut end = mmap_slice.len();
    while end > 0 {
        let start = mmap_slice[..end]
            .iter()
            .rposition(|&b| b == b'\n')
            .map(|i| i + 1)
            .unwrap_or(0);

        let line_bytes = &mmap_slice[start..end];
        end = start.saturating_sub(1);

        let line = match std::str::from_utf8(line_bytes) {
            Ok(s) => s.trim(),
            Err(_) => {
                non_utf8_lines += 1;
                continue;
            }
        };
        if line.is_empty() {
            continue;
        }

        let entry: CodeEntry = match serde_json::from_str(line) {
            Ok(e) => e,
            Err(_) => {
                corrupt_json_lines += 1;
                continue;
            }
        };

        // Soft-delete: consume id directly, no clone
        if entry.d == 1 {
            seen.insert(entry.id);
            continue;
        }

        // Dedup check via borrow — no clone for the check
        if seen.contains(&entry.id) {
            continue;
        }

        // Move id into set, consume the rest of entry fields
        let cluster = entry.cluster;
        let code = entry.code;
        let id = entry.id;

        documents.push(Document::new(&id, ""));
        assignments.push(cluster);
        codes.push(code);
        seen.insert(id); // zero-copy move into set
    }

    if non_utf8_lines > 0 || corrupt_json_lines > 0 {
        log::warn!(
            "IVF-PQ load: skipped {} non-UTF-8 lines, {} corrupt JSON lines",
            non_utf8_lines,
            corrupt_json_lines,
        );
    }

    // Restore chronological order (oldest-first)
    documents.reverse();
    assignments.reverse();
    codes.reverse();

    let n_list = meta.config.n_list;
    let mut clusters: Vec<Vec<usize>> = vec![Vec::new(); n_list];
    for (doc_idx, &cluster) in assignments.iter().enumerate() {
        if cluster < n_list {
            clusters[cluster].push(doc_idx);
        }
    }

    Ok(Some(IvfPqIndex::from_state(
        documents,
        config.clone(),
        meta.centroids,
        meta.codebooks,
        codes,
        clusters,
    )))
}

// ---------------------------------------------------------------------------
// Compaction
// ---------------------------------------------------------------------------

/// Compact the `.jsonl` file, removing deleted and duplicate entries.
/// Only keeps the latest active revision of each document.
///
/// Call this as a maintenance operation when the file has accumulated
/// soft-deleted or overwritten lines (e.g. via append-mode operations).
/// No-op when the waste ratio is below `threshold` (default 0.2 = 20%).
///
/// ## Invariante
/// Solo elimina líneas — nunca modifica `cluster` o `code`.
/// El archivo `.meta` permanece válido sin cambios porque los centroids
/// y codebooks son inmutables entre saves explícitos.
#[cfg_attr(not(test), allow(dead_code))]
pub fn compact(base: &Path, threshold: f32) -> Result<()> {
    let jp = jsonl_path(base);
    if !jp.exists() {
        return Ok(());
    }

    // Load current lines via mmap (no heap allocation for the data itself)
    let file = std::fs::File::open(&jp).map_err(|e| Error::Io {
        path: jp.clone(),
        source: e,
    })?;
    let mmap = unsafe { Mmap::map(&file) }
        .map_err(|e| Error::Internal(format!("IVF-PQ compact mmap: {e}")))?;
    let _ = mmap.advise(memmap2::Advice::Random);
    let slice: &[u8] = &mmap;

    // Count total non-empty lines — lazy iterator, no Vec allocation
    let total = slice.split(|&b| b == b'\n').filter(|l| !l.is_empty()).count();
    if total == 0 {
        return Ok(());
    }

    // ── Pre-allocate from exact total ──
    let mut keep: Vec<String> = Vec::with_capacity(total);
    let mut seen: HashSet<String> = HashSet::with_capacity(total);
    let mut non_utf8_lines: u64 = 0;
    let mut corrupt_json_lines: u64 = 0;

    // ── Reverse scan over mmap — no intermediate Vec allocation ──
    let mut end = slice.len();
    while end > 0 {
        let start = slice[..end]
            .iter()
            .rposition(|&b| b == b'\n')
            .map(|i| i + 1)
            .unwrap_or(0);

        let line_bytes = &slice[start..end];
        end = start.saturating_sub(1);

        let line = match std::str::from_utf8(line_bytes) {
            Ok(s) => s.trim(),
            Err(_) => {
                non_utf8_lines += 1;
                continue;
            }
        };
        if line.is_empty() {
            continue;
        }

        let entry: CodeEntry = match serde_json::from_str(line) {
            Ok(e) => e,
            Err(_) => {
                corrupt_json_lines += 1;
                continue;
            }
        };

        // Soft-delete: consume id directly, no clone
        if entry.d == 1 {
            seen.insert(entry.id);
            continue;
        }

        // Dedup check via borrow — no clone for the check
        if seen.contains(&entry.id) {
            continue;
        }

        // Keep the line text, consume entry.id into set (zero-copy move)
        keep.push(line.to_string());
        seen.insert(entry.id);
    }

    if non_utf8_lines > 0 || corrupt_json_lines > 0 {
        log::warn!(
            "IVF-PQ compact: skipped {} non-UTF-8 lines, {} corrupt JSON lines",
            non_utf8_lines,
            corrupt_json_lines,
        );
    }

    let active = keep.len();
    let waste = total.saturating_sub(active);
    let ratio = if total > 0 {
        waste as f32 / total as f32
    } else {
        0.0
    };

    if ratio < threshold {
        return Ok(()); // not worth compacting
    }

    // Reverse back to original order then write
    keep.reverse();
    atomic_write_lines(&jp, &keep)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::Metric;
    use crate::doc::Document;
    use crate::index::ivf_pq::IvfPqConfig;
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

    fn make_index(n: usize, dim: usize) -> IvfPqIndex {
        let config = IvfPqConfig {
            n_list: 4,
            m_subspaces: 8,
            n_probe: 2,
            metric: Metric::Cosine,
            ..Default::default()
        };
        config.validate().unwrap();
        let mut idx = IvfPqIndex::new(config);
        idx.insert(&make_test_docs(n, dim));
        idx
    }

    // ── Atomic save ──────────────────────────────────────────────────────

    #[test]
    fn test_atomic_save_failure_recovery() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path().join("test.vdb");
        let index = make_index(5, 16);
        index.save_persistence(&base).unwrap();

        // Save a second time (should use atomic tmp+rename)
        let index2 = make_index(10, 16);
        index2.save_persistence(&base).unwrap();

        // Load and verify integrity
        let config = IvfPqConfig {
            n_list: 4,
            m_subspaces: 8,
            n_probe: 2,
            metric: Metric::Cosine,
            ..Default::default()
        };
        let loaded = IvfPqIndex::load_persistence(&base, &config)
            .unwrap()
            .expect("should load after atomic save");

        // Should have 10 docs (second save), not 5
        assert_eq!(loaded.len(), 10, "atomic overwrite must preserve full data");
    }

    #[test]
    fn test_atomic_save_tmp_cleaned() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path().join("test.vdb");
        let index = make_index(3, 16);
        index.save_persistence(&base).unwrap();

        // No .tmp files should remain after save
        let has_tmp = std::fs::read_dir(tmp.path())
            .unwrap()
            .any(|e| e.unwrap().path().extension().is_some_and(|x| x == "tmp"));
        assert!(!has_tmp, ".tmp files must be removed after atomic save");
    }

    // ── Soft-delete ──────────────────────────────────────────────────────

    #[test]
    fn test_soft_delete_and_overwrite() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path().join("test.vdb");
        let config = IvfPqConfig {
            n_list: 4,
            m_subspaces: 8,
            n_probe: 2,
            metric: Metric::Cosine,
            ..Default::default()
        };
        config.validate().unwrap();

        // 1. Insert 3 docs, save
        let mut idx = IvfPqIndex::new(config.clone());
        idx.insert(&make_test_docs(3, 16));
        idx.save_persistence(&base).unwrap();

        // 2. Simulate soft-delete: manually append a deleted line to .jsonl
        let jp = jsonl_path(&base);
        {
            let mut f = std::fs::OpenOptions::new().append(true).open(&jp).unwrap();
            writeln!(f, r#"{{"id":"d0","cluster":0,"code":[],"d":1}}"#).unwrap();
            // Overwrite d1: append a new version
            let cluster = 1;
            let entry = CodeEntry {
                id: "d1".into(),
                cluster,
                code: vec![0u8; 8],
                d: 0,
            };
            writeln!(f, "{}", serde_json::to_string(&entry).unwrap()).unwrap();
            f.sync_all().unwrap();
        }

        // 3. Reload — d0 should be gone, d1 should be the new version
        let loaded = IvfPqIndex::load_persistence(&base, &config)
            .unwrap()
            .expect("should load");
        assert_eq!(
            loaded.len(),
            2,
            "2 active docs after soft-delete (d0 gone) + overwrite (d1 replaced)"
        );

        // The deleted doc "d0" should not appear
        let ids: Vec<&str> = loaded.documents().iter().map(|d| d.id.as_str()).collect();
        assert!(
            !ids.contains(&"d0"),
            "deleted doc must not appear in loaded index"
        );
        assert!(ids.contains(&"d1"), "overwritten doc must appear");
    }

    #[test]
    fn test_load_ignores_deleted_entries() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path().join("test.vdb");
        let config = IvfPqConfig {
            n_list: 4,
            m_subspaces: 8,
            n_probe: 2,
            metric: Metric::Cosine,
            ..Default::default()
        };

        // Manually write a .jsonl with deleted + active entries
        let jp = jsonl_path(&base);
        let raw = "{\"id\":\"a\",\"cluster\":0,\"code\":[],\"d\":0}\n{\"id\":\"b\",\"cluster\":0,\"code\":[],\"d\":1}\n{\"id\":\"c\",\"cluster\":0,\"code\":[],\"d\":0}\n";
        std::fs::write(&jp, raw).unwrap();

        // Also write minimal .meta
        let mp = meta_path(&base);
        let meta = MetaFile {
            config: MetaConfig {
                n_list: 4,
                m_subspaces: 8,
                n_probe: 2,
                metric: "Cosine".into(),
                rerank_enabled: false,
            },
            centroids: vec![vec![0.0; 16]; 4],
            codebooks: vec![vec![vec![0.0; 2]; 256]; 8],
        };
        std::fs::write(&mp, serde_json::to_string_pretty(&meta).unwrap()).unwrap();

        let loaded = IvfPqIndex::load_persistence(&base, &config)
            .unwrap()
            .expect("should load");
        assert_eq!(loaded.len(), 2, "should load only active entries (a, c)");
        let ids: Vec<&str> = loaded.documents().iter().map(|d| d.id.as_str()).collect();
        assert!(ids.contains(&"a"));
        assert!(!ids.contains(&"b"), "deleted entry 'b' must be excluded");
        assert!(ids.contains(&"c"));
    }

    // ── Compaction ───────────────────────────────────────────────────────

    #[test]
    fn test_compaction_lifecycle() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path().join("test.vdb");
        let config = IvfPqConfig {
            n_list: 4,
            m_subspaces: 8,
            n_probe: 2,
            metric: Metric::Cosine,
            ..Default::default()
        };
        config.validate().unwrap();

        // Build index with 5 docs, save
        let mut idx = IvfPqIndex::new(config.clone());
        idx.insert(&make_test_docs(5, 16));
        idx.save_persistence(&base).unwrap();

        let jp = jsonl_path(&base);
        let size_before = std::fs::metadata(&jp).map(|m| m.len()).unwrap_or(0);

        // Append soft-delete markers for 3 docs + 2 overwrites = 5 waste lines
        {
            let mut f = std::fs::OpenOptions::new().append(true).open(&jp).unwrap();
            for i in 0..3 {
                writeln!(f, r#"{{"id":"d{i}","cluster":0,"code":[],"d":1}}"#).unwrap();
            }
            // Overwrite two docs with new versions
            for i in 0..2 {
                let entry = CodeEntry {
                    id: format!("d{i}"),
                    cluster: 1,
                    code: vec![1u8; 8],
                    d: 0,
                };
                writeln!(f, "{}", serde_json::to_string(&entry).unwrap()).unwrap();
            }
            f.sync_all().unwrap();
        }

        let size_mid = std::fs::metadata(&jp).map(|m| m.len()).unwrap_or(0);
        assert!(
            size_mid > size_before,
            "file should have grown after appends"
        );

        // Compact with 0% threshold (force always)
        compact(&base, 0.0).unwrap();

        let size_after = std::fs::metadata(&jp).map(|m| m.len()).unwrap_or(0);
        assert!(
            size_after < size_mid,
            "compacted file should be smaller: before={size_mid} after={size_after}"
        );

        // Reload and verify data integrity
        let loaded = IvfPqIndex::load_persistence(&base, &config)
            .unwrap()
            .expect("should load after compaction");
        assert_eq!(
            loaded.len(),
            4,
            "must have 4 active docs after compaction (d2 was soft-deleted)"
        );

        // Search still works (may return fewer than 4 with tiny IVF-PQ index)
        let q = make_test_docs(1, 16)[0].embedding.clone();
        let results = loaded.search(&q, 5);
        assert!(!results.is_empty(), "search must work after compaction");
    }

    #[test]
    fn test_compaction_below_threshold_skips() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path().join("test.vdb");
        let index = make_index(10, 16);
        index.save_persistence(&base).unwrap();

        let jp = jsonl_path(&base);
        let size_before = std::fs::metadata(&jp).map(|m| m.len()).unwrap_or(0);

        // Compact with 50% threshold — should skip since no waste
        compact(&base, 0.5).unwrap();

        let size_after = std::fs::metadata(&jp).map(|m| m.len()).unwrap_or(0);
        assert_eq!(
            size_after, size_before,
            "file should not change when waste < threshold"
        );
    }
}
