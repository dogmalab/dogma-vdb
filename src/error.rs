//! Typed errors for the entire crate.

use std::path::PathBuf;
use thiserror::Error;

/// Every error that can originate from dogma-vdb.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    #[error("File not found: {0}")]
    FileNotFound(PathBuf),

    #[error("I/O error on {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("JSON parse error on line {line}: {detail}")]
    ParseJson {
        line: usize,
        detail: String,
        #[source]
        source: serde_json::Error,
    },

    #[error("Embedding failed: {0}")]
    Embedding(String),

    #[error("Collection '{0}' not found")]
    CollectionNotFound(String),

    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("Empty index on collection '{0}'")]
    EmptyIndex(String),

    #[error("Feature not available: {0}")]
    FeatureNotAvailable(&'static str),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

/// Alias for `Result<T, dogma_vdb::Error>`.
pub type Result<T> = std::result::Result<T, Error>;
