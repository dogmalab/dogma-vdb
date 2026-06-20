//! Document type with a fluent builder.
//!
//! A `Document` holds the raw text, its embedding vector, and arbitrary
//! key‑value metadata.  Every document is serialised as one JSON line
//! inside a `.vdb` file.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A single document with text, embedding vector, and metadata.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Document {
    pub id: String,
    pub text: String,
    #[serde(default)]
    pub embedding: Vec<f32>,
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

impl Document {
    /// Create a [`DocumentBuilder`] for a fluent construction.
    #[must_use]
    pub fn builder(id: impl Into<String>, text: impl Into<String>) -> DocumentBuilder {
        DocumentBuilder {
            id: id.into(),
            text: text.into(),
            embedding: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Quick constructor for a document without an embedding.
    #[must_use]
    pub fn new(id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
            embedding: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Dimensionality of the embedding vector.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.embedding.len()
    }

    /// Returns `true` if the embedding is non-empty.
    #[must_use]
    pub fn is_embedded(&self) -> bool {
        !self.embedding.is_empty()
    }

    /// Look up a metadata value by key.
    pub fn metadata_val(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(String::as_str)
    }
}

/// Fluent builder for [`Document`].
///
/// # Example
/// ```
/// use dogma_vdb::doc::Document;
///
/// let doc = Document::builder("id-1", "Hello, world!")
///     .embedding(vec![0.1, 0.2, 0.3])
///     .metadata("source", "book.pdf")
///     .metadata("page", "42")
///     .build();
/// ```
#[derive(Debug)]
pub struct DocumentBuilder {
    id: String,
    text: String,
    embedding: Vec<f32>,
    metadata: HashMap<String, String>,
}

impl DocumentBuilder {
    /// Set the embedding vector.
    pub fn embedding(mut self, v: Vec<f32>) -> Self {
        self.embedding = v;
        self
    }

    /// Add a single metadata key-value pair.
    pub fn metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Replace all metadata with the given map.
    pub fn metadatas(mut self, m: HashMap<String, String>) -> Self {
        self.metadata = m;
        self
    }

    /// Build the [`Document`].
    pub fn build(self) -> Document {
        Document {
            id: self.id,
            text: self.text,
            embedding: self.embedding,
            metadata: self.metadata,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_document_new() {
        let doc = Document::new("id-1", "hello");
        assert_eq!(doc.id, "id-1");
        assert_eq!(doc.text, "hello");
        assert!(doc.embedding.is_empty());
        assert!(doc.metadata.is_empty());
    }

    #[test]
    fn test_document_dimension() {
        let doc = Document::builder("a", "b")
            .embedding(vec![0.1, 0.2, 0.3])
            .build();
        assert_eq!(doc.dimension(), 3);
    }

    #[test]
    fn test_document_is_embedded() {
        let doc = Document::new("a", "b");
        assert!(!doc.is_embedded());
        let doc = Document::builder("a", "b").embedding(vec![0.1]).build();
        assert!(doc.is_embedded());
    }

    #[test]
    fn test_document_metadata_val() {
        let doc = Document::builder("a", "b").metadata("lang", "rust").build();
        assert_eq!(doc.metadata_val("lang"), Some("rust"));
        assert_eq!(doc.metadata_val("nonexistent"), None);
    }

    #[test]
    fn test_document_builder_full() {
        let doc = Document::builder("id-42", "Rust is safe")
            .embedding(vec![0.5, 0.6, 0.7, 0.8])
            .metadata("page", "10")
            .metadata("chapter", "3")
            .build();

        assert_eq!(doc.id, "id-42");
        assert_eq!(doc.text, "Rust is safe");
        assert_eq!(doc.embedding, vec![0.5, 0.6, 0.7, 0.8]);
        assert_eq!(doc.metadata_val("page"), Some("10"));
        assert_eq!(doc.metadata_val("chapter"), Some("3"));
        assert_eq!(doc.metadata.len(), 2);
    }

    #[test]
    fn test_document_builder_metadatas_replaces() {
        let mut meta = HashMap::new();
        meta.insert("source".into(), "test".into());

        let doc = Document::builder("id-42", "Rust is safe")
            .metadata("page", "10")
            .metadatas(meta)
            .build();

        // metadatas() replaces previous metadata
        assert_eq!(doc.metadata_val("source"), Some("test"));
        assert_eq!(doc.metadata_val("page"), None);
        assert_eq!(doc.metadata.len(), 1);
    }

    #[test]
    fn test_document_serde_roundtrip() {
        let doc = Document::builder("s-1", "serde test")
            .embedding(vec![0.1, 0.2])
            .metadata("key", "val")
            .build();

        let json = serde_json::to_string(&doc).unwrap();
        let deserialized: Document = serde_json::from_str(&json).unwrap();
        assert_eq!(doc, deserialized);
    }

    #[test]
    fn test_document_default_embedding_on_missing() {
        // JSONL lines may omit embedding field — should default to empty vec
        let json = r#"{"id":"x","text":"hello"}"#;
        let doc: Document = serde_json::from_str(json).unwrap();
        assert_eq!(doc.id, "x");
        assert!(doc.embedding.is_empty());
        assert!(doc.metadata.is_empty());
    }

    #[test]
    fn test_document_default_metadata_on_missing() {
        let json = r#"{"id":"y","text":"world","embedding":[0.5]}"#;
        let doc: Document = serde_json::from_str(json).unwrap();
        assert!(doc.metadata.is_empty());
        assert_eq!(doc.embedding, vec![0.5]);
    }
}
