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
use crate::error::Result;
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
        todo!()
    }

    /// The underlying file path.
    pub fn path(&self) -> &Path {
        todo!()
    }

    /// Load **all** documents from the file.
    ///
    /// Lines are streamed lazily via `BufReader`; peak memory is
    /// proportional to the file size because all documents are
    /// returned.
    pub fn load(&self) -> Result<Vec<Document>> {
        todo!()
    }

    /// Overwrite the file with the given documents.
    pub fn store(&self, docs: &[Document]) -> Result<()> {
        let _ = docs;
        todo!()
    }

    /// Append a single document to the end of the file.
    pub fn append(&self, doc: &Document) -> Result<()> {
        let _ = doc;
        todo!()
    }

    /// Whether the file already exists on disk.
    pub fn exists(&self) -> bool {
        todo!()
    }

    /// Number of non‑empty lines (documents) in the file.
    pub fn count(&self) -> Result<usize> {
        todo!()
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

        let docs = vec![
            Document::new("a", "texto a"),
            Document::new("b", "texto b"),
        ];
        storage.store(&docs).unwrap();
        assert!(storage.exists());

        let loaded = storage.load().unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].id, "a");
    }
}
