//! Storage abstraction for contiguous vector data.
//!
//! Provides a unified interface over memory-backed and memory-mapped
//! storage, so index backends don't care where the bytes come from.

use crate::error::Result;

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// A contiguous region of `f32` embedding data.
///
/// # Safety
///
/// The [`as_embeddings`] method performs an `unsafe` `u8 → f32` reinterpret.
/// This is safe **only** when:
///
/// 1. The underlying bytes are valid `f32` little-endian data (written by
///    [`bytemuck::cast_slice`] or equivalent).
/// 2. The byte slice starts at an address aligned to `align_of::<f32>()`.
///
/// This is the same approach used by HuggingFace `safetensors` and is
/// isolated to a single, auditable method.
pub trait VectorStorage: Send + Sync {
    /// Return the raw byte slice for the entire embedding region.
    fn as_bytes(&self) -> &[u8];

    /// Reinterpret the bytes as a contiguous `f32` slice.
    ///
    /// # Panics
    ///
    /// Panics if the byte pointer is not 4-byte aligned.
    fn as_embeddings(&self) -> &[f32] {
        let bytes = self.as_bytes();
        if bytes.is_empty() {
            return &[];
        }
        let ptr = bytes.as_ptr();
        let align = std::mem::align_of::<f32>();
        assert_eq!(
            ptr.align_offset(align),
            0,
            "VectorStorage: byte slice is not aligned to {align} bytes \
             (ptr={ptr:p}, mod={})",
            ptr as usize % align
        );
        // Safety: bytes were written via bytemuck::cast_slice which
        // guarantees valid f32 LE representation.  The alignment check
        // above ensures the pointer meets f32 requirements.
        // This is the standard pattern used by safetensors (HF) and the
        // `bytemuck` crate.
        assert_eq!(
            bytes.len() % 4,
            0,
            "VectorStorage: byte length {} must be multiple of 4",
            bytes.len()
        );
        unsafe { std::slice::from_raw_parts(ptr as *const f32, bytes.len() / 4) }
    }

    /// Persist changes to disk (no-op for memory-backed storage).
    fn flush(&self) -> Result<()> {
        Ok(())
    }

    /// Number of `f32` elements.
    fn len(&self) -> usize {
        self.as_bytes().len() / 4
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ---------------------------------------------------------------------------
// Memory-backed — for tests, volatile mode, JSONL fallback
// ---------------------------------------------------------------------------

/// In-memory storage backed by a `Vec<u8>`.
#[derive(Debug, Clone)]
pub struct MemoryBackedStorage {
    data: Vec<u8>,
}

impl MemoryBackedStorage {
    pub fn new(data: Vec<u8>) -> Self {
        debug_assert_eq!(
            data.len() % 4,
            0,
            "MemoryBackedStorage data length must be multiple of 4, got {}",
            data.len()
        );
        Self { data }
    }

    /// Build from an `&[f32]` slice (copies the data).
    pub fn from_f32_slice(slice: &[f32]) -> Self {
        let bytes: &[u8] = bytemuck::cast_slice(slice);
        Self {
            data: bytes.to_vec(),
        }
    }
}

impl VectorStorage for MemoryBackedStorage {
    fn as_bytes(&self) -> &[u8] {
        &self.data
    }
}

// ---------------------------------------------------------------------------
// Memory-mapped (memmap2) — zero-copy production backend
// ---------------------------------------------------------------------------

/// Zero-copy storage backed by a memory-mapped file.
///
/// The file is mapped into virtual memory; the operating system loads
/// pages on demand.  No heap allocation — load time is effectively zero.
///
/// # ⚠️ SIGBUS Warning
///
/// If an external process **modifies or truncates** the underlying file
/// while this mapping exists, the kernel delivers a `SIGBUS` signal and
/// the process **crashes immediately**.  Rust cannot safely recover from
/// `SIGBUS`.
///
/// **Guarantees you must uphold:**
/// - The mapped file must not be modified by any external agent while
///   an `MmapBackedStorage` referencing it is alive.
/// - dogma-vdb itself only **writes** to the file after closing the
///   collection (via `BinStorage::store`), so internal writes are safe.
///
/// Future work could add OS-level file locking (`fs2` crate) to prevent
/// accidental concurrent modification.
#[derive(Debug)]
pub struct MmapBackedStorage {
    /// Keep the file handle alive for the lifetime of the mapping.
    _file: std::fs::File,
    /// The memory-mapped region.
    mmap: memmap2::Mmap,
}

impl MmapBackedStorage {
    /// Memory-map the **entire** file at `path`.
    pub fn new(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let file = std::fs::File::open(path.as_ref()).map_err(|e| crate::error::Error::Io {
            path: path.as_ref().to_path_buf(),
            source: e,
        })?;
        // SAFETY: read-only mapping, file not modified while mapped.
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| crate::error::Error::Io {
            path: path.as_ref().to_path_buf(),
            source: e,
        })?;
        // Hint: random access pattern — no readahead, no page-clustering
        let _ = mmap.advise(memmap2::Advice::Random);
        Ok(Self { _file: file, mmap })
    }

    /// Memory-map a **slice** of a file (offset + length).
    ///
    /// `offset` must be page-aligned (typically 4096).  If your data
    /// starts at a non-aligned offset, map a larger region and slice.
    pub fn new_with_offset(
        path: impl AsRef<std::path::Path>,
        offset: u64,
        len: usize,
    ) -> Result<Self> {
        let file = std::fs::File::open(path.as_ref()).map_err(|e| crate::error::Error::Io {
            path: path.as_ref().to_path_buf(),
            source: e,
        })?;
        // SAFETY: mmap with explicit offset/len requires offset to be
        // page-aligned.  The caller is responsible for this guarantee.
        let mmap = unsafe {
            memmap2::MmapOptions::new()
                .offset(offset)
                .len(len)
                .map(&file)
        }
        .map_err(|e| crate::error::Error::Io {
            path: path.as_ref().to_path_buf(),
            source: e,
        })?;
        // Hint: random access pattern — no readahead, no page-clustering
        let _ = mmap.advise(memmap2::Advice::Random);
        Ok(Self { _file: file, mmap })
    }

}

impl VectorStorage for MmapBackedStorage {
    fn as_bytes(&self) -> &[u8] {
        &self.mmap
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::doc::Document;
    use crate::storage::BinStorage;
    use tempfile::tempdir;

    #[test]
    fn test_memory_backed_empty() {
        let s = MemoryBackedStorage::new(vec![]);
        assert!(s.is_empty());
        assert_eq!(s.as_bytes().len(), 0);
        assert_eq!(s.as_embeddings().len(), 0);
    }

    #[test]
    fn test_memory_backed_from_f32() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let s = MemoryBackedStorage::from_f32_slice(&data);
        assert_eq!(s.len(), 4);
        assert_eq!(s.as_embeddings(), &data[..]);
    }

    #[test]
    fn test_mmap_backed_roundtrip_via_full_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_emb.vdb");

        // Write a small binary file via BinStorage
        let docs = vec![
            Document::builder("a", "hello")
                .embedding(vec![0.1, 0.2, 0.3])
                .build(),
            Document::builder("b", "world")
                .embedding(vec![0.4, 0.5, 0.6])
                .build(),
        ];
        let storage = BinStorage::new(&path);
        storage.store(&docs).unwrap();

        // Memory-map the full file
        let mmap = MmapBackedStorage::new(&path).unwrap();
        assert!(mmap.len() > 0);

        // The first 24 bytes are the header, so as_embeddings covers
        // the whole file.  In practice we'd slice to just the embedding
        // region, but this verifies the mmap works.
        let emb = mmap.as_embeddings();

        // The file has a 24-byte header + metadata + padding + 6 f32s.
        // So emb should contain at least 6 floats at the end.
        let total_floats = emb.len();
        assert!(
            total_floats >= 6,
            "expected at least 6 floats, got {total_floats}"
        );
    }

    #[test]
    fn test_mmap_backed_alignment() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("align_test.vdb");

        let docs = vec![Document::builder("a", "")
            .embedding(vec![1.0, 2.0, 3.0, 4.0])
            .build()];
        let storage = BinStorage::new(&path);
        storage.store(&docs).unwrap();

        let mmap = MmapBackedStorage::new(&path).unwrap();
        // Should not panic — alignment must pass
        let _emb = mmap.as_embeddings();
    }

    #[test]
    fn test_vector_storage_alignment_check() {
        // MemoryBackedStorage with a Vec<u8> must always be 4-aligned
        let raw = vec![0u8; 128];
        let s = MemoryBackedStorage::new(raw);
        let emb = s.as_embeddings();
        assert_eq!(emb.len(), 32); // 128 / 4
    }

    #[test]
    fn test_flush_noop() {
        let s = MemoryBackedStorage::new(vec![0u8; 16]);
        assert!(s.flush().is_ok());
    }
}
