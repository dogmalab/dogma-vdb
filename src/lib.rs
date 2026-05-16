//! # dogma-vdb
//!
//! Portable vector database in JSONL format.
//!
//! ## Design principles
//!
//! * **1 dependency** (serde_json) for the core — zero‑cost, no bloat.
//! * **Portable format** — every `.vdb` file is plain JSONL, inspectable with
//!   `cat`, `grep`, `sed`, and versionable with `git`.
//! * **No server** — file‑based, zero config, no daemon.
//! * **MCP‑ready** — an optional MCP server lets Claude Desktop, Cursor,
//!   opencode, or any MCP‑compatible agent query your collections.
//! * **Watch mode** — an optional file watcher re‑indexes source files
//!   automatically on every change.
//!
//! ## Quick start
//!
//! ```ignore
//! use dogma_vdb::prelude::*;
//!
//! let mut col = Collection::open("my_data.vdb")?;
//! col.insert(Document::new("doc-1", "Rust is fast"))?;
//! let results = col.search(&[0.1, 0.2, 0.3], 5, Metric::Cosine);
//! ```

pub mod chunker;
pub mod collection;
pub mod distance;
pub mod doc;
pub mod embedding;
pub mod error;
pub mod index;
pub mod storage;

#[cfg(feature = "watch")]
pub mod watch;

#[cfg(feature = "mcp")]
pub mod mcp;

/// Convenience re‑exports of the most common types.
pub mod prelude {
    pub use crate::chunker::{Chunker, ChunkerConfig};
    pub use crate::collection::Collection;
    pub use crate::distance::Metric;
    pub use crate::doc::{Document, DocumentBuilder};
    pub use crate::embedding::Embedder;
    pub use crate::error::{Error, Result};
    pub use crate::index::{BruteForceIndex, Index, ScoredDocument};
    pub use crate::storage::JsonlStorage;
}
