//! Optional file‑system watcher (feature = "watch").
//!
//! Watches source directories for file changes, re‑chunks the
//! modified files, and updates the `.vdb` collection automatically.

use crate::error::Result;
use std::path::PathBuf;

/// Events emitted by the background watcher.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum WatchEvent {
    Updated {
        collection: String,
        docs_added: usize,
    },
    Deleted {
        collection: String,
        doc_id: String,
    },
    Error {
        collection: String,
        message: String,
    },
    Stopped,
}

/// Configuration for [`start_watching`].
#[derive(Debug, Clone)]
pub struct WatchConfig {
    pub source_dirs: Vec<PathBuf>,
    pub extensions: Vec<String>,
    pub output: PathBuf,
    pub debounce_ms: u64,
    pub initial_scan: bool,
}

impl Default for WatchConfig {
    fn default() -> Self {
        Self {
            source_dirs: vec![],
            extensions: vec!["md".into(), "txt".into(), "rs".into()],
            output: PathBuf::from("default.vdb"),
            debounce_ms: 500,
            initial_scan: true,
        }
    }
}

/// Start the file watcher on a background thread.
///
/// Returns a `Receiver<WatchEvent>` that yields events as files are
/// created, modified, or deleted.
pub fn start_watching(_config: WatchConfig) -> Result<crossbeam_channel::Receiver<WatchEvent>> {
    todo!()
}
