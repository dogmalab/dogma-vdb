//! SIMIL (System Intent Markup Language) — semantic metadata for dogma-vdb.
//!
//! This module compiles source code and plain text into SIMIL manifests
//! stored in `Document.metadata["sml"]`. It provides a unified semantic
//! layer across all document types.
//!
//! # Feature flag
//!
//! Requires `feature = "sml"` (default off).
//!
//! # Quick start
//!
//! ```ignore
//! use dogma_vdb::sml::{SmlCompiler, ingest};
//!
//! let compiler = SmlCompiler::new();
//! let metadata = ingest(source_code, "main.rs", &compiler);
//! // metadata["sml"] contains the SIMIL manifest
//! ```

pub mod ast;
pub mod compiler;
pub mod infer;
pub mod keywords;
pub mod serializer;

pub use ast::*;
pub use compiler::SmlCompiler;
pub use serializer::{serialize, serialize_batch};

use std::collections::HashMap;

use crate::smart_chunker::{ChunkStrategy, SmartChunker};

/// Full ingestion pipeline: source text → SmartChunks → SmlNodes → `HashMap["sml"]`.
///
/// Returns a `HashMap` ready to merge into `Document.metadata`.
pub fn ingest(text: &str, path: &str, compiler: &SmlCompiler) -> HashMap<String, String> {
    let mut meta = HashMap::new();

    if text.trim().is_empty() {
        return meta;
    }

    let chunker = SmartChunker::default();
    let strategy = ChunkStrategy::from_path(path);
    let chunks = chunker.chunk_text(text, strategy);

    if chunks.is_empty() {
        return meta;
    }

    let nodes = compiler.compile_batch(&chunks, text);
    let sml = serialize_batch(&nodes);

    if !sml.is_empty() {
        meta.insert("sml".to_string(), sml);
    }

    meta
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ingest_empty_text() {
        let compiler = SmlCompiler::new();
        let meta = ingest("", "test.txt", &compiler);
        assert!(meta.is_empty());
    }

    #[test]
    fn test_ingest_rust_code() {
        let compiler = SmlCompiler::new();
        let code = r#"
/// A user account entity.
pub struct UserAccount {
    pub username: String,
    pub email: String,
    pub role: String,
}
"#;
        let meta = ingest(code, "user.rs", &compiler);
        assert!(meta.contains_key("sml"));
        let sml = &meta["sml"];
        assert!(sml.contains("type UserAccount"));
    }

    #[test]
    fn test_ingest_plain_text() {
        let compiler = SmlCompiler::new();
        let text = "DeploymentPolicy: All deployments must be verified before production.";
        let meta = ingest(text, "policy.txt", &compiler);
        assert!(meta.contains_key("sml"));
    }

    #[test]
    fn test_ingest_python_code() {
        let compiler = SmlCompiler::new();
        let code = r#"
class DataProcessor:
    """Process incoming data streams."""
    def process(self, data):
        pass
"#;
        let meta = ingest(code, "processor.py", &compiler);
        assert!(meta.contains_key("sml"));
        let sml = &meta["sml"];
        assert!(sml.contains("DataProcessor"));
    }
}
