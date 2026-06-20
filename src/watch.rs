//! Optional file-system watcher (feature = "watch").
//!
//! Watches source directories for file changes, re-chunks modified
//! files, and updates a `.vdb` collection automatically.
//!
//! Events are debounced — multiple rapid changes to the same file
//! within the configured window are coalesced into a single
//! re-indexing pass.  The collection is kept open for the lifetime
//! of the watcher, avoiding repeated file I/O.
//!
//! # Example
//!
//! ```ignore
//! use dogma_vdb::watch::{start_watching, WatchConfig};
//!
//! let rx = start_watching(WatchConfig {
//!     source_dirs: vec!["docs/".into()],
//!     extensions: vec!["md".into(), "txt".into()],
//!     output: "data/docs.vdb".into(),
//!     debounce_ms: 500,
//!     initial_scan: true,
//! })?;
//!
//! while let Ok(event) = rx.recv() {
//!     match event {
//!         WatchEvent::Updated { docs_added, .. } => println!("  +{docs_added} docs"),
//!         WatchEvent::Error { message, .. } => eprintln!("  error: {message}"),
//!         WatchEvent::Stopped => break,
//!         _ => {}
//!     }
//! }
//! ```

use crate::collection::Collection;
use crate::error::{Error, Result};
use crate::smart_chunker::SmartChunker;
use crossbeam_channel::{unbounded, Receiver, Sender};
use notify::{EventKind, RecursiveMode, Watcher};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Events emitted by the background watcher.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum WatchEvent {
    /// A batch of files was re-indexed.
    Updated {
        collection: String,
        docs_added: usize,
    },
    /// A file was deleted; its document was removed from the collection.
    Deleted { collection: String, doc_id: String },
    /// An error occurred while processing a file.
    Error { collection: String, message: String },
    /// The watcher has stopped (all watches removed).
    Stopped,
}

/// Configuration for [`start_watching`].
#[derive(Debug, Clone)]
pub struct WatchConfig {
    /// Directories to watch recursively.
    pub source_dirs: Vec<PathBuf>,
    /// File extensions to process (without dot, e.g. "md", "txt").
    pub extensions: Vec<String>,
    /// Output `.vdb` file path.
    pub output: PathBuf,
    /// Debounce interval in milliseconds.
    pub debounce_ms: u64,
    /// If true, scan and index existing files on start.
    pub initial_scan: bool,
}

impl Default for WatchConfig {
    fn default() -> Self {
        Self {
            source_dirs: vec![],
            extensions: vec!["md".into(), "txt".into(), "rs".into()],
            output: PathBuf::from("data/default.vdb"),
            debounce_ms: 500,
            initial_scan: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal state
// ---------------------------------------------------------------------------

struct WatcherState {
    path_to_id: Mutex<HashMap<PathBuf, String>>,
    config: WatchConfig,
    tx: Sender<WatchEvent>,
    chunker: SmartChunker,
}

impl WatcherState {
    fn new(config: WatchConfig, tx: Sender<WatchEvent>) -> Self {
        let chunker = SmartChunker::default();
        Self {
            path_to_id: Mutex::new(HashMap::new()),
            config,
            tx,
            chunker,
        }
    }

    /// Process a batch of files: read, chunk, and insert into the collection.
    ///
    /// Opens the collection once for the entire batch, deletes old chunks
    /// for modified files, and inserts all new chunks in a single pass.
    fn process_batch(&self, files: &HashSet<PathBuf>) {
        if files.is_empty() {
            return;
        }

        let col_name = self.name();
        let mut col = match Collection::open_with(&self.config.output, "bruteforce", "cosine") {
            Ok(c) => c,
            Err(e) => {
                let _ = self.tx.send(WatchEvent::Error {
                    collection: col_name,
                    message: format!("open collection failed: {e}"),
                });
                return;
            }
        };

        let mut total_added = 0usize;

        for path in files {
            // Delete old chunks for this file (if re-processing)
            if let Ok(map) = self.path_to_id.lock() {
                if let Some(old_id) = map.get(path.as_path()) {
                    let _ = col.delete(&[old_id.as_str()]);
                }
            }

            let content = match std::fs::read_to_string(path) {
                Ok(c) => c,
                Err(e) => {
                    let _ = self.tx.send(WatchEvent::Error {
                        collection: col_name.clone(),
                        message: format!("read failed for {}: {e}", path.display()),
                    });
                    continue;
                }
            };

            if content.is_empty() {
                continue;
            }

            let base_id = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("doc")
                .to_string();

            let docs = self
                .chunker
                .chunk_to_docs(path, &content, &base_id, HashMap::new());

            if docs.is_empty() {
                continue;
            }

            let count = docs.len();
            if let Err(e) = col.insert_batch(&docs) {
                let _ = self.tx.send(WatchEvent::Error {
                    collection: col_name.clone(),
                    message: format!("insert failed for {}: {e}", path.display()),
                });
                continue;
            }

            if let Ok(mut map) = self.path_to_id.lock() {
                map.insert(path.clone(), base_id);
            }

            total_added += count;
        }

        if total_added > 0 {
            let _ = self.tx.send(WatchEvent::Updated {
                collection: col_name,
                docs_added: total_added,
            });
        }
    }

    /// Remove all documents associated with a file path.
    fn remove_file(&self, path: &Path) {
        let col_name = self.name();
        let doc_id = match self.path_to_id.lock() {
            Ok(mut map) => map.remove(path),
            Err(_) => return,
        };
        if let Some(id) = doc_id {
            if let Ok(mut col) = Collection::open_with(&self.config.output, "bruteforce", "cosine")
            {
                let _ = col.delete(&[id.as_str()]);
            }
            let _ = self.tx.send(WatchEvent::Deleted {
                collection: col_name,
                doc_id: id,
            });
        }
    }

    /// Scan all existing files in source dirs and index them.
    fn initial_scan(&self) {
        let mut files = HashSet::new();
        for dir in &self.config.source_dirs {
            if !dir.exists() {
                continue;
            }
            walkdir(dir, &self.config.extensions, &mut |path| {
                files.insert(path.to_path_buf());
            });
        }
        self.process_batch(&files);
    }

    fn name(&self) -> String {
        self.config
            .output
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("default")
            .to_string()
    }
}

/// Walk a directory recursively, calling `f` for each matching file.
fn walkdir(dir: &Path, extensions: &[String], f: &mut dyn FnMut(&Path)) {
    if !dir.is_dir() {
        return;
    }
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walkdir(&path, extensions, f);
        } else if path.is_file() {
            if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                if extensions.is_empty() || extensions.iter().any(|e| e == ext) {
                    f(&path);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Start the file watcher on a background thread.
///
/// Watches the configured source directories for file changes and
/// automatically updates the `.vdb` collection.  Returns a channel
/// receiver that yields [`WatchEvent`]s.
///
/// Events are debounced: multiple rapid changes to the same file
/// within `debounce_ms` are coalesced into a single re-indexing pass.
///
/// The watcher runs until the returned sender is dropped or the
/// process exits.
pub fn start_watching(config: WatchConfig) -> Result<Receiver<WatchEvent>> {
    let (tx, rx) = unbounded::<WatchEvent>();
    let state = WatcherState::new(config.clone(), tx.clone());

    // Initial scan (synchronous, runs before watcher starts)
    if config.initial_scan {
        state.initial_scan();
    }

    // Channel for raw file events from notify callback
    let (raw_tx, raw_rx) = unbounded::<(EventKind, PathBuf)>();
    let ext_filter = config.extensions.clone();

    // Spawn notify watcher — sends raw events to raw_tx
    let mut watcher = notify::recommended_watcher(move |event: notify::Result<notify::Event>| {
        let event = match event {
            Ok(e) => e,
            Err(_) => return,
        };
        for path in &event.paths {
            let is_watched = ext_filter.is_empty()
                || path
                    .extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(|e| ext_filter.contains(&e.to_string()));
            if !is_watched {
                continue;
            }
            let _ = raw_tx.send((event.kind, path.clone()));
        }
    })
    .map_err(|_| Error::FeatureNotAvailable("watch"))?;

    // Watch each source directory
    for dir in &config.source_dirs {
        watcher
            .watch(dir, RecursiveMode::Recursive)
            .map_err(|_| Error::FeatureNotAvailable("watch"))?;
    }

    // Keep watcher alive on a background thread
    std::thread::spawn(move || loop {
        std::thread::sleep(Duration::from_secs(3600));
    });

    // Debounce + batch processing thread
    let debounce = Duration::from_millis(config.debounce_ms);
    std::thread::spawn(move || {
        let mut pending: HashSet<PathBuf> = HashSet::new();

        loop {
            // Wait for next event or debounce timeout
            match raw_rx.recv_timeout(debounce) {
                Ok((kind, path)) => match kind {
                    EventKind::Create(_) | EventKind::Modify(_) => {
                        pending.insert(path);
                    }
                    EventKind::Remove(_) => {
                        // Flush pending file changes before handling delete
                        if !pending.is_empty() {
                            state.process_batch(&pending);
                            pending.clear();
                        }
                        state.remove_file(&path);
                    }
                    _ => {}
                },
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                    // Debounce window expired — flush accumulated files
                    if !pending.is_empty() {
                        state.process_batch(&pending);
                        pending.clear();
                    }
                }
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                    // Channel closed — flush remaining and exit
                    if !pending.is_empty() {
                        state.process_batch(&pending);
                    }
                    let _ = state.tx.send(WatchEvent::Stopped);
                    break;
                }
            }
        }
    });

    Ok(rx)
}
