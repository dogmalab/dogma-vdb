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
    pub fn new(path: impl Into<PathBuf>) -> Self;

    /// The underlying file path.
    pub fn path(&self) -> &Path;

    /// Load **all** documents from the file.
    ///
    /// Lines are streamed lazily via `BufReader`; peak memory is
    /// proportional to the file size because all documents are
    /// returned.
    pub fn load(&self) -> Result<Vec<Document>>;

    /// Overwrite the file with the given documents.
    pub fn store(&self, docs: &[Document]) -> Result<()>;

    /// Append a single document to the end of the file.
    pub fn append(&self, doc: &Document) -> Result<()>;

    /// Whether the file already exists on disk.
    pub fn exists(&self) -> bool;

    /// Number of non‑empty lines (documents) in the file.
    pub fn count(&self) -> Result<usize>;
}

#[cfg(test)]
mod tests;
