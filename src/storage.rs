//! JSONL file storage.
//!
//! Every line in a `.vdb` file is a complete, self‑describing JSON
//! object that can be deserialised into a [`Document`](crate::doc::Document).
//!
//! ```jsonl
//! {"id":"doc-1","text":"hello","embedding":[0.1,0.2],"metadata":{"source":"x"}}
//! {"id":"doc-2","text":"world","embedding":[0.3,0.4],"metadata":{}}
//! ```
//!
//! ## Thread safety
//!
//! `JsonlStorage` is `Send + Sync`.  Append operations use
//! `OpenOptions::append(true)` and are safe under concurrent
//! single‑writer access.

use crate::doc::Document;
use crate::error::{Error, Result};
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

/// File‑backed JSONL storage for a collection of [`Document`]s.
#[derive(Debug, Clone)]
pub struct JsonlStorage {
    path: PathBuf,
}

impl JsonlStorage {
    /// Create a handle for the file at `path`.
    ///
    /// The file is **not** created or opened until `load` / `store` /
    /// `append` is called.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    /// The underlying file path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Load **all** documents from the file.
    ///
    /// Lines are streamed lazily via `BufReader`; peak memory is
    /// proportional to the file size because all documents are
    /// returned.
    pub fn load(&self) -> Result<Vec<Document>> {
        let file = std::fs::File::open(&self.path).map_err(|source| Error::Io {
            path: self.path.clone(),
            source,
        })?;
        let reader = BufReader::new(file);
        let mut docs = Vec::new();
        for (line_num, line) in reader.lines().enumerate() {
            let line = line.map_err(|source| Error::Io {
                path: self.path.clone(),
                source,
            })?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let doc: Document =
                serde_json::from_str(trimmed).map_err(|source| Error::ParseJson {
                    line: line_num + 1,
                    detail: format!("invalid JSON on line {}: {}", line_num + 1, trimmed),
                    source,
                })?;
            docs.push(doc);
        }
        Ok(docs)
    }

    /// Overwrite the file with the given documents.
    pub fn store(&self, docs: &[Document]) -> Result<()> {
        let file = std::fs::File::create(&self.path).map_err(|source| Error::Io {
            path: self.path.clone(),
            source,
        })?;
        let mut writer = std::io::BufWriter::new(file);
        for doc in docs {
            let line = serde_json::to_string(doc).map_err(|source| Error::ParseJson {
                line: 0,
                detail: "failed to serialize document".into(),
                source,
            })?;
            writeln!(writer, "{}", line).map_err(|source| Error::Io {
                path: self.path.clone(),
                source,
            })?;
        }
        writer.flush().map_err(|source| Error::Io {
            path: self.path.clone(),
            source,
        })?;
        Ok(())
    }

    /// Append a single document to the end of the file.
    pub fn append(&self, doc: &Document) -> Result<()> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(|source| Error::Io {
                path: self.path.clone(),
                source,
            })?;
        let mut writer = std::io::BufWriter::new(file);
        let line = serde_json::to_string(doc).map_err(|source| Error::ParseJson {
            line: 0,
            detail: "failed to serialize document".into(),
            source,
        })?;
        writeln!(writer, "{}", line).map_err(|source| Error::Io {
            path: self.path.clone(),
            source,
        })?;
        writer.flush().map_err(|source| Error::Io {
            path: self.path.clone(),
            source,
        })?;
        Ok(())
    }

    /// Whether the file already exists on disk.
    pub fn exists(&self) -> bool {
        self.path.exists()
    }

    /// Number of non‑empty lines (documents) in the file.
    pub fn count(&self) -> Result<usize> {
        let file = std::fs::File::open(&self.path).map_err(|source| Error::Io {
            path: self.path.clone(),
            source,
        })?;
        let reader = BufReader::new(file);
        let mut count = 0usize;
        for line in reader.lines() {
            let line = line.map_err(|source| Error::Io {
                path: self.path.clone(),
                source,
            })?;
            if !line.trim().is_empty() {
                count += 1;
            }
        }
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.vdb");
        let storage = JsonlStorage::new(&path);

        let docs = vec![Document::new("a", "texto a"), Document::new("b", "texto b")];
        storage.store(&docs).unwrap();
        assert!(storage.exists());

        let loaded = storage.load().unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].id, "a");
        assert_eq!(loaded[1].id, "b");
    }

    #[test]
    fn test_append() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("append.vdb");
        let storage = JsonlStorage::new(&path);

        storage.append(&Document::new("1", "first")).unwrap();
        assert_eq!(storage.count().unwrap(), 1);

        storage.append(&Document::new("2", "second")).unwrap();
        assert_eq!(storage.count().unwrap(), 2);

        let loaded = storage.load().unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].id, "1");
        assert_eq!(loaded[1].id, "2");
    }

    #[test]
    fn test_count_empty_missing_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("nonexistent.vdb");
        let storage = JsonlStorage::new(&path);
        assert!(storage.count().is_err());
    }

    #[test]
    fn test_load_empty_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("empty.vdb");
        let storage = JsonlStorage::new(&path);
        // Create empty file
        std::fs::File::create(&path).unwrap();
        let loaded = storage.load().unwrap();
        assert!(loaded.is_empty());
    }

    #[test]
    fn test_exists() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("exists.vdb");
        let storage = JsonlStorage::new(&path);
        assert!(!storage.exists());
        storage.store(&[]).unwrap();
        assert!(storage.exists());
    }

    #[test]
    fn test_path() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("mypath.vdb");
        let storage = JsonlStorage::new(path.clone());
        assert_eq!(storage.path(), path);
    }

    #[test]
    fn test_load_with_metadata_and_embeddings() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("full.vdb");
        let storage = JsonlStorage::new(&path);

        let docs = vec![
            Document::builder("x", "hello")
                .embedding(vec![0.1, 0.2])
                .metadata("lang", "en")
                .build(),
            Document::builder("y", "world")
                .embedding(vec![0.3, 0.4])
                .metadata("lang", "es")
                .build(),
        ];
        storage.store(&docs).unwrap();

        let loaded = storage.load().unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].embedding, vec![0.1, 0.2]);
        assert_eq!(loaded[1].metadata_val("lang"), Some("es"));
    }

    #[test]
    fn test_store_overwrites() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("overwrite.vdb");
        let storage = JsonlStorage::new(&path);

        storage.store(&[Document::new("a", "first")]).unwrap();
        storage.store(&[Document::new("b", "second")]).unwrap();

        let loaded = storage.load().unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].id, "b");
    }

    #[test]
    fn test_count_after_append() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("count.vdb");
        let storage = JsonlStorage::new(&path);

        assert!(storage.count().is_err()); // file doesn't exist yet

        storage.store(&[]).unwrap(); // creates empty file
        assert_eq!(storage.count().unwrap(), 0);

        storage.append(&Document::new("1", "uno")).unwrap();
        assert_eq!(storage.count().unwrap(), 1);
    }

    #[test]
    fn test_corrupt_json_returns_error() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("corrupt.vdb");
        std::fs::write(&path, b"not valid json\n").unwrap();
        let storage = JsonlStorage::new(&path);
        let result = storage.load();
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            Error::ParseJson { line, .. } => assert_eq!(line, 1),
            _ => panic!("expected ParseJson error, got: {:?}", err),
        }
    }

    #[test]
    fn test_file_not_found() {
        let storage = JsonlStorage::new("/nonexistent/path/test.vdb");
        let result = storage.load();
        assert!(result.is_err());
        match result.unwrap_err() {
            Error::Io { .. } => {} // expected
            other => panic!("expected Io error, got: {:?}", other),
        }
    }
}
