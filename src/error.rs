//! Typed errors for the entire crate.

use std::path::PathBuf;
use thiserror::Error;

/// Every error that can originate from dogma-vdb.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
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

    #[error("Feature not available: {0}")]
    FeatureNotAvailable(&'static str),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("Out of memory: {0}")]
    OutOfMemory(String),
}

/// Alias for `Result<T, dogma_vdb::Error>`.
pub type Result<T> = std::result::Result<T, Error>;
