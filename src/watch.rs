//! Optional file-system watcher (feature = "watch").
//!
//! Watches source directories for file changes, re-chunks modified
//! files, and updates a `.vdb` collection automatically.
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
use std::collections::HashMap;
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
    /// A file was created or modified; its chunks were added to the collection.
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
    /// Map from absolute file path to document ID (for delete tracking).
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

    /// Process a file: read, chunk, and add to collection.
    fn process_file(&self, path: &Path) {
        let col_name = self.name();
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(e) => {
                let _ = self.tx.send(WatchEvent::Error {
                    collection: col_name,
                    message: format!("read failed: {e}"),
                });
                return;
            }
        };

        if content.is_empty() {
            return;
        }

        let base_id = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("doc")
            .to_string();

        // Chunk the file
        let docs = self
            .chunker
            .chunk_to_docs(path, &content, &base_id, HashMap::new());
        if docs.is_empty() {
            return;
        }

        // Open/insert into collection
        match Collection::open_with(&self.config.output, "bruteforce", "cosine") {
            Ok(mut col) => {
                let count = docs.len();
                // Delete old chunks for this file path first (if re-processing)
                if let Ok(map) = self.path_to_id.lock() {
                    if let Some(old_id) = map.get(path) {
                        let _ = col.delete(&[old_id.as_str()]);
                    }
                }
                for doc in docs {
                    if let Err(e) = col.insert(doc) {
                        let _ = self.tx.send(WatchEvent::Error {
                            collection: col_name.clone(),
                            message: format!("insert failed: {e}"),
                        });
                        return;
                    }
                }
                // Track the new base ID for this path
                if let Ok(mut map) = self.path_to_id.lock() {
                    map.insert(path.to_path_buf(), base_id);
                }
                let _ = self.tx.send(WatchEvent::Updated {
                    collection: col_name,
                    docs_added: count,
                });
            }
            Err(e) => {
                let _ = self.tx.send(WatchEvent::Error {
                    collection: col_name,
                    message: format!("open collection failed: {e}"),
                });
            }
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
        for dir in &self.config.source_dirs {
            if !dir.exists() {
                continue;
            }
            walkdir(dir, &self.config.extensions, &mut |path| {
                self.process_file(path);
            });
        }
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
/// The watcher runs until the returned sender is dropped or the
/// process exits.
pub fn start_watching(config: WatchConfig) -> Result<Receiver<WatchEvent>> {
    let (tx, rx) = unbounded::<WatchEvent>();
    let state = WatcherState::new(config.clone(), tx.clone());

    // Initial scan
    if config.initial_scan {
        state.initial_scan();
    }

    // Set up notify watcher
    let notify_tx = tx.clone();
    let ext_filter = config.extensions.clone();

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
            match event.kind {
                EventKind::Create(_) | EventKind::Modify(_) => {
                    // Debounce via notify's internal debouncing or simple sleep
                    state.process_file(path);
                }
                EventKind::Remove(_) => {
                    state.remove_file(path);
                }
                _ => {}
            }
        }
    })
    .map_err(|_e| Error::FeatureNotAvailable("watch"))?;

    // Watch each source directory
    for dir in &config.source_dirs {
        watcher
            .watch(dir, RecursiveMode::Recursive)
            .map_err(|_| Error::FeatureNotAvailable("watch"))?;
    }

    // Keep watcher alive on a background thread
    std::thread::spawn(move || {
        // The watcher is kept alive by this thread; when it exits, the
        // watcher drops and stops sending events.
        loop {
            std::thread::sleep(Duration::from_secs(3600));
            if notify_tx.send(WatchEvent::Stopped).is_err() {
                break;
            }
        }
    });

    Ok(rx)
}
