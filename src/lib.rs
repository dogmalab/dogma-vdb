//! # dogma-vdb
//!
//! Portable vector database in JSONL / binary format.
//!
//! ## Design principles
//!
//! * **Minimal dependencies** — the core only pulls in `serde_json` + `serde` + `thiserror`.
//!   Everything else (rayon, notify) is optional behind feature flags.
//! * **Portable format** — every `.vdb` file is plain binary (or JSONL), debugeable with
//!   `cat`, `grep`, `sed`, and versionable with `git`.
//! * **No server** — file‑based, zero config, no daemon.
//! * **MCP‑ready** — an optional MCP server crate (`dogma-vdb-mcp`) lets Claude Desktop,
//!   Cursor, opencode, or any MCP‑compatible agent query your collections.
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
pub mod config;
pub mod distance;
pub mod doc;
pub mod embedding;
pub mod error;
pub mod filter;
pub mod index;
pub mod memory;
pub mod rerank;
pub mod smart_chunker;
pub mod storage;

#[cfg(feature = "sml")]
pub mod sml;

#[cfg(feature = "watch")]
pub mod watch;

// Re‑export for convenience
pub use config::{Config, CONFIG};

/// Convenience re‑exports of the most common types.
pub mod prelude {
    pub use crate::chunker::{Chunker, TextSplitter, TextSplitterConfig};
    pub use crate::collection::Collection;
    pub use crate::distance::Metric;
    pub use crate::doc::{Document, DocumentBuilder};
    pub use crate::embedding::Embedder;
    pub use crate::error::{Error, Result};
    pub use crate::index::{
        BruteForceIndex, HnswConfig, HnswIndex, Index, IvfPqConfig, IvfPqIndex, ScoredDocument,
    };
    #[cfg(feature = "sml")]
    pub use crate::sml::SmlCompiler;
}
