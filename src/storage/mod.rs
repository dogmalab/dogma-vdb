//! Binary (native) file storage.
//!
//! Default format for dogma-vdb.  Much faster than JSONL because it
//! avoids text parsing for f32 embeddings.
//!
//! ## File format (`.vdb`)
//!
//! ```text
//! Offset  Size  Field
//! ------  ----  -----
//! 0       4     magic: b"DVDB\0"  (4 bytes)
//! 4       4     version: u32 LE  (1)
//! 8       4     dim: u32 LE      (embedding dimension, 0 if no embeddings)
//! 12      4     count: u32 LE    (number of documents)
//! 16      8     emb_offset: u64 LE  (byte offset where embeddings start)
//! 24      —     metadata section (one block per document)
//! emb_offset  —  embeddings: count × dim × 4 bytes raw f32 LE
//! ```
//!
//! Each metadata block:
//! ```text
//! [2 bytes LE] id_len
//! [id_len]     id (UTF-8)
//! [4 bytes LE] text_len
//! [text_len]   text (UTF-8)
//! [2 bytes LE] meta_count (number of key-value pairs)
//! for each pair:
//!   [2 bytes LE] key_len
//!   [key_len]    key (UTF-8)
//!   [2 bytes LE] val_len
//!   [val_len]    val (UTF-8)
//! ```

use crate::doc::Document;
use crate::error::{Error, Result};
use memmap2::Mmap;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 4] = b"DVDB";
const CURRENT_VERSION: u32 = 2;

/// Alignment boundary for the embedding section (32 bytes = 256-bit SIMD).
const EMB_ALIGN: usize = 32;

// ---------------------------------------------------------------------------
// Helper: memory-map a file with random-access advice
// ---------------------------------------------------------------------------

/// Memory-map the entire file at `path` with random-access advice.
///
/// # Safety
///
/// `Mmap::map` is `unsafe` because the kernel can deliver `SIGBUS` if an
/// external process truncates the mapped file.  The caller guarantees that
/// the file is **not modified by any external agent** while the returned
/// `Mmap` is alive.
///
/// This function exists solely to centralize the `unsafe` block and the
/// `advise(Random)` call — **not** to override the caller's safety
/// responsibility.
pub(crate) fn mmap_path(path: &Path) -> Result<Mmap> {
    let file = std::fs::File::open(path).map_err(|e| Error::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    mmap_file(&file)
}

/// Memory-map an already-open file with random-access advice.
///
/// Useful when the caller already has a `File` handle (e.g. for metadata
/// queries before or after the mapping).
///
/// # Safety
///
/// Same as [`mmap_path`] — caller guarantees no external modification.
pub(crate) fn mmap_file(file: &std::fs::File) -> Result<Mmap> {
    // SAFETY: read-only mapping; caller guarantees no external modification.
    let mmap = unsafe { Mmap::map(file) }.map_err(|e| Error::Internal(format!("mmap: {e}")))?;
    let _ = mmap.advise(memmap2::Advice::Random);
    Ok(mmap)
}

/// Memory-map a portion of `file` at `offset` for `len` bytes.
///
/// # Safety
///
/// Same as [`mmap_file`] — caller guarantees no external modification.
/// Additionally, `offset` must be page-aligned (typically 4096).
pub(crate) fn mmap_file_offset(file: &std::fs::File, offset: u64, len: usize) -> Result<Mmap> {
    // SAFETY: caller guarantees page-aligned offset and no external modification.
    let mmap = unsafe {
        memmap2::MmapOptions::new()
            .offset(offset)
            .len(len)
            .map(file)
    }
    .map_err(|e| Error::Internal(format!("mmap offset: {e}")))?;
    let _ = mmap.advise(memmap2::Advice::Random);
    Ok(mmap)
}

/// Binary (native) storage for a collection of [`Document`]s.
#[derive(Debug, Clone)]
pub struct BinStorage {
    path: PathBuf,
}

impl BinStorage {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Detect whether this file uses the binary format.
    pub fn is_binary(path: &Path) -> bool {
        let mut buf = [0u8; 4];
        if let Ok(mut f) = std::fs::File::open(path) {
            use std::io::Read;
            if f.read_exact(&mut buf).is_ok() {
                return &buf == MAGIC;
            }
        }
        false
    }

    /// Load all documents from the binary file.
    pub fn load(&self) -> Result<Vec<Document>> {
        let data = std::fs::read(&self.path).map_err(|source| Error::Io {
            path: self.path.clone(),
            source,
        })?;
        Self::decode(&data, &self.path)
    }

    /// Overwrite the file with the given documents.
    pub fn store(&self, docs: &[Document]) -> Result<()> {
        let bytes = self.encode(docs)?;
        std::fs::write(&self.path, &bytes).map_err(|source| Error::Io {
            path: self.path.clone(),
            source,
        })?;
        Ok(())
    }

    /// Whether the file already exists on disk.
    pub fn exists(&self) -> bool {
        self.path.exists()
    }

    /// Read only the file header and return the embedding region info.
    ///
    /// Returns `(emb_offset, emb_len, dim, count)` where:
    /// - `emb_offset`: byte offset where the embedding data starts
    /// - `emb_len`: total byte length of the embedding section
    /// - `dim`: embedding dimension (0 if no embeddings)
    /// - `count`: number of documents
    ///
    /// This is useful for memory-mapping just the embedding region
    /// without loading metadata into memory.
    pub fn embedding_region(path: &Path) -> Result<(u64, usize, usize, usize)> {
        let mut buf = [0u8; 24];
        use std::io::Read;
        let mut f = std::fs::File::open(path).map_err(|source| Error::Io {
            path: path.to_path_buf(),
            source,
        })?;
        f.read_exact(&mut buf).map_err(|source| Error::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let _magic = &buf[0..4];
        let _version = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
        let dim = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize;
        let count = u32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]) as usize;
        let emb_offset = u64::from_le_bytes([
            buf[16], buf[17], buf[18], buf[19], buf[20], buf[21], buf[22], buf[23],
        ]);
        let emb_len = count * dim * 4;
        Ok((emb_offset, emb_len, dim, count))
    }

    // ------------------------------------------------------------------
    // Encoding
    // ------------------------------------------------------------------

    fn encode(&self, docs: &[Document]) -> Result<Vec<u8>> {
        let dim = docs
            .iter()
            .find(|d| !d.embedding.is_empty())
            .map_or(0, |d| d.embedding.len());
        let count = docs.len();

        // Calculate sizes
        let meta_size: usize = docs
            .iter()
            .map(|d| {
                2 + d.id.len()          // id_len + id
            + 4 + d.text.len()      // text_len + text
            + 2                     // meta_count
            + d.metadata.iter().map(|(k, v)| 2 + k.len() + 2 + v.len()).sum::<usize>()
            })
            .sum();

        let emb_size = count * dim * 4;
        let header_size = 24; // magic(4) + ver(4) + dim(4) + count(4) + emb_offset(8)

        let mut buf = Vec::with_capacity(header_size + meta_size + emb_size);

        // Header
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&CURRENT_VERSION.to_le_bytes());
        buf.extend_from_slice(&(dim as u32).to_le_bytes());
        buf.extend_from_slice(&(count as u32).to_le_bytes());
        // Pad to alignment boundary (32 bytes for SIMD, ≥4 for f32)
        let pad = (EMB_ALIGN - (meta_size % EMB_ALIGN)) % EMB_ALIGN;
        let emb_offset = (header_size + meta_size + pad) as u64;
        buf.extend_from_slice(&emb_offset.to_le_bytes());

        // Metadata blocks
        for doc in docs {
            write_u16(&mut buf, doc.id.len() as u16);
            buf.extend_from_slice(doc.id.as_bytes());

            write_u32(&mut buf, doc.text.len() as u32);
            buf.extend_from_slice(doc.text.as_bytes());

            write_u16(&mut buf, doc.metadata.len() as u16);
            for (k, v) in &doc.metadata {
                write_u16(&mut buf, k.len() as u16);
                buf.extend_from_slice(k.as_bytes());
                write_u16(&mut buf, v.len() as u16);
                buf.extend_from_slice(v.as_bytes());
            }
        }

        // Padding for alignment
        let old_len = buf.len();
        buf.resize(old_len + pad, 0);

        // Embeddings (contiguous f32 — pad empty embeddings with zeros)
        for doc in docs {
            if !doc.embedding.is_empty() {
                let bytes: &[u8] = bytemuck::cast_slice(&doc.embedding);
                buf.extend_from_slice(bytes);
            } else if dim > 0 {
                // Pad with zeros for documents that have no embedding
                buf.extend(std::iter::repeat(0u8).take(dim * 4));
            }
        }

        Ok(buf)
    }

    // ------------------------------------------------------------------
    // Decoding
    // ------------------------------------------------------------------

    fn decode(data: &[u8], path: &Path) -> Result<Vec<Document>> {
        if data.len() < 24 {
            return Err(Error::Io {
                path: path.to_path_buf(),
                source: std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "file too short"),
            });
        }

        let _magic = &data[0..4];
        let _version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        let dim = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
        let count = u32::from_le_bytes([data[12], data[13], data[14], data[15]]) as usize;
        let emb_offset = u64::from_le_bytes([
            data[16], data[17], data[18], data[19], data[20], data[21], data[22], data[23],
        ]) as usize;

        let mut pos = 24;
        let mut docs = Vec::with_capacity(count);

        for _ in 0..count {
            // id
            let id_len = read_u16(data, &mut pos, path)? as usize;
            let id = read_str(data, &mut pos, id_len, path)?;

            // text
            let text_len = read_u32(data, &mut pos, path)? as usize;
            let text = read_str(data, &mut pos, text_len, path)?;

            // metadata
            let meta_count = read_u16(data, &mut pos, path)? as usize;
            let mut metadata = HashMap::with_capacity(meta_count);
            for _ in 0..meta_count {
                let k_len = read_u16(data, &mut pos, path)? as usize;
                let k = read_str(data, &mut pos, k_len, path)?;
                let v_len = read_u16(data, &mut pos, path)? as usize;
                let v = read_str(data, &mut pos, v_len, path)?;
                metadata.insert(k, v);
            }

            docs.push(Document {
                id,
                text,
                embedding: Vec::new(), // filled below
                metadata,
            });
        }

        // Embeddings
        if dim > 0 {
            let expected = count * dim * 4;
            let emb_start = emb_offset;
            if emb_start + expected <= data.len() {
                let emb_slice = &data[emb_start..emb_start + expected];
                let floats: &[f32] = bytemuck::cast_slice(emb_slice);
                for (i, doc) in docs.iter_mut().enumerate() {
                    let start = i * dim;
                    doc.embedding = floats[start..start + dim].to_vec();
                }
            }
        }

        Ok(docs)
    }

    /// Whether the file exists and has the binary magic.
    pub fn exists_with_magic(&self) -> bool {
        self.exists() && Self::is_binary(&self.path)
    }
}

// ---------------------------------------------------------------------------
// Binary read helpers
// ---------------------------------------------------------------------------

fn read_u16(data: &[u8], pos: &mut usize, path: &Path) -> Result<u16> {
    if *pos + 2 > data.len() {
        return Err(Error::Io {
            path: path.to_path_buf(),
            source: std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "truncated binary"),
        });
    }
    let val = u16::from_le_bytes([data[*pos], data[*pos + 1]]);
    *pos += 2;
    Ok(val)
}

fn read_u32(data: &[u8], pos: &mut usize, path: &Path) -> Result<u32> {
    if *pos + 4 > data.len() {
        return Err(Error::Io {
            path: path.to_path_buf(),
            source: std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "truncated binary"),
        });
    }
    let val = u32::from_le_bytes([data[*pos], data[*pos + 1], data[*pos + 2], data[*pos + 3]]);
    *pos += 4;
    Ok(val)
}

fn read_str(data: &[u8], pos: &mut usize, len: usize, path: &Path) -> Result<String> {
    if *pos + len > data.len() {
        return Err(Error::Io {
            path: path.to_path_buf(),
            source: std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "truncated binary"),
        });
    }
    let s = String::from_utf8_lossy(&data[*pos..*pos + len]).to_string();
    *pos += len;
    Ok(s)
}

fn write_u16(buf: &mut Vec<u8>, val: u16) {
    buf.extend_from_slice(&val.to_le_bytes());
}

fn write_u32(buf: &mut Vec<u8>, val: u32) {
    buf.extend_from_slice(&val.to_le_bytes());
}

// Need `bytemuck` for safe f32↔[u8] reinterpret
// But we already have `wide` which depends on bytemuck, so it's free.

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn make_docs() -> Vec<Document> {
        vec![
            Document::builder("a", "hello")
                .embedding(vec![0.1, 0.2, 0.3])
                .metadata("lang", "en")
                .build(),
            Document::builder("b", "world")
                .embedding(vec![0.4, 0.5, 0.6])
                .metadata("lang", "es")
                .metadata("source", "book")
                .build(),
            Document::new("c", "no emb"),
        ]
    }

    #[test]
    fn test_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.vdb");
        let storage = BinStorage::new(&path);

        let docs = make_docs();
        storage.store(&docs).unwrap();
        assert!(storage.exists_with_magic());

        let loaded = storage.load().unwrap();
        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded[0].id, "a");
        assert_eq!(loaded[0].embedding, vec![0.1, 0.2, 0.3]);
        assert_eq!(loaded[0].metadata_val("lang"), Some("en"));
        assert_eq!(loaded[1].id, "b");
        assert_eq!(loaded[1].embedding, vec![0.4, 0.5, 0.6]);
        assert_eq!(loaded[1].metadata_val("source"), Some("book"));
        assert_eq!(loaded[2].id, "c");
        // Binary format pads empty embeddings with zeros
        assert_eq!(loaded[2].embedding.len(), 3);
        assert!(loaded[2].embedding.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_is_binary() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("bin.vdb");
        assert!(!BinStorage::is_binary(&path)); // doesn't exist

        let storage = BinStorage::new(&path);
        storage.store(&make_docs()).unwrap();
        assert!(BinStorage::is_binary(&path));
    }

    #[test]
    fn test_not_binary_for_jsonl() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("json.vdb");
        std::fs::write(&path, b"{\"id\":\"x\"}\n").unwrap();
        assert!(!BinStorage::is_binary(&path));
    }

    #[test]
    fn test_store_overwrites() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("overwrite.vdb");
        let storage = BinStorage::new(&path);

        storage.store(&[Document::new("a", "first")]).unwrap();
        storage.store(&[Document::new("b", "second")]).unwrap();

        let loaded = storage.load().unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].id, "b");
    }

    #[test]
    fn test_store_empty() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("empty.vdb");
        let storage = BinStorage::new(&path);
        storage.store(&[]).unwrap();
        assert!(storage.exists_with_magic());
        let loaded = storage.load().unwrap();
        assert!(loaded.is_empty());
    }

    #[test]
    fn test_no_embedding_mixed() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("mixed.vdb");
        let storage = BinStorage::new(&path);

        storage
            .store(&[
                Document::new("a", "text"),
                Document::builder("b", "emb").embedding(vec![1.0]).build(),
            ])
            .unwrap();

        let loaded = storage.load().unwrap();
        // Binary format pads empty embeddings with zeros
        assert_eq!(loaded[0].embedding.len(), 1);
        assert_eq!(loaded[0].embedding[0], 0.0);
        assert_eq!(loaded[1].embedding, vec![1.0]);
    }
}

pub mod traits;
