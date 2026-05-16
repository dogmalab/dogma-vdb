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
    pub fn builder(id: impl Into<String>, text: impl Into<String>) -> DocumentBuilder {
        DocumentBuilder {
            id: id.into(),
            text: text.into(),
            embedding: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Quick constructor for a document without an embedding.
    pub fn new(id: impl Into<String>, text: impl Into<String>) -> Self {
        todo!()
    }

    /// Dimensionality of the embedding vector.
    pub fn dimension(&self) -> usize {
        todo!()
    }

    /// Returns `true` if the embedding is non‑empty.
    pub fn is_embedded(&self) -> bool {
        todo!()
    }

    /// Look up a metadata value by key.
    pub fn metadata_val(&self, key: &str) -> Option<&str> {
        todo!()
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
    pub fn embedding(mut self, v: Vec<f32>) -> Self {
        todo!()
    }

    pub fn metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        todo!()
    }

    pub fn metadatas(mut self, m: HashMap<String, String>) -> Self {
        todo!()
    }

    pub fn build(self) -> Document {
        todo!()
    }
}
