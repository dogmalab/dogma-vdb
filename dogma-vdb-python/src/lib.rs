//! Python bindings for dogma-vdb.
//!
//! Exposes `Collection`, `Document`, `ScoredDocument`, and `Metric`
//! to Python via PyO3.

use pyo3::prelude::*;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Metric
// ---------------------------------------------------------------------------

/// Distance metric for vector search.
#[pyclass(eq, eq_int, skip_from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyMetric {
    Cosine = 0,
    Dot = 1,
    Euclidean = 2,
}

impl From<PyMetric> for dogma_vdb::distance::Metric {
    fn from(m: PyMetric) -> Self {
        match m {
            PyMetric::Cosine => dogma_vdb::distance::Metric::Cosine,
            PyMetric::Dot => dogma_vdb::distance::Metric::Dot,
            PyMetric::Euclidean => dogma_vdb::distance::Metric::Euclidean,
        }
    }
}

// ---------------------------------------------------------------------------
// Document
// ---------------------------------------------------------------------------

/// A document with text, embedding, and metadata.
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct PyDocument {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub text: String,
    #[pyo3(get)]
    pub embedding: Vec<f32>,
    #[pyo3(get)]
    pub metadata: HashMap<String, String>,
}

#[pymethods]
impl PyDocument {
    #[new]
    #[pyo3(signature = (id, text, embedding=None, metadata=None))]
    fn new(
        id: &str,
        text: &str,
        embedding: Option<Vec<f32>>,
        metadata: Option<HashMap<String, String>>,
    ) -> Self {
        Self {
            id: id.to_string(),
            text: text.to_string(),
            embedding: embedding.unwrap_or_default(),
            metadata: metadata.unwrap_or_default(),
        }
    }

    /// Dimensionality of the embedding vector.
    fn dimension(&self) -> usize {
        self.embedding.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "Document(id='{}', text='{}...', dim={})",
            self.id,
            &self.text[..self.text.len().min(30)],
            self.embedding.len()
        )
    }
}

impl From<PyDocument> for dogma_vdb::doc::Document {
    fn from(d: PyDocument) -> Self {
        let mut builder = dogma_vdb::doc::Document::builder(d.id, d.text);
        if !d.embedding.is_empty() {
            builder = builder.embedding(d.embedding);
        }
        for (k, v) in d.metadata {
            builder = builder.metadata(k, v);
        }
        builder.build()
    }
}

impl From<dogma_vdb::doc::Document> for PyDocument {
    fn from(d: dogma_vdb::doc::Document) -> Self {
        Self {
            id: d.id,
            text: d.text,
            embedding: d.embedding,
            metadata: d.metadata,
        }
    }
}

// ---------------------------------------------------------------------------
// ScoredDocument
// ---------------------------------------------------------------------------

/// A search result: document + relevance score.
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct PyScoredDocument {
    #[pyo3(get)]
    pub score: f32,
    #[pyo3(get)]
    pub document: PyDocument,
}

impl From<dogma_vdb::index::ScoredDocument> for PyScoredDocument {
    fn from(s: dogma_vdb::index::ScoredDocument) -> Self {
        Self {
            score: s.score,
            document: s.document.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// Collection
// ---------------------------------------------------------------------------

/// A named vector collection backed by a `.vdb` file.
#[pyclass]
pub struct PyCollection {
    inner: dogma_vdb::collection::Collection,
}

#[pymethods]
impl PyCollection {
    /// Open or create a collection.
    #[new]
    #[pyo3(signature = (path, index_type="bruteforce", metric="cosine"))]
    fn new(path: &str, index_type: &str, metric: &str) -> PyResult<Self> {
        let inner = dogma_vdb::collection::Collection::open_with(path, index_type, metric)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Collection name (derived from file stem).
    #[getter]
    fn name(&self) -> &str {
        self.inner.name()
    }

    /// Number of documents.
    #[getter]
    fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether the collection is empty.
    #[getter]
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Insert a single document.
    fn insert(&mut self, doc: PyDocument) -> PyResult<()> {
        self.inner
            .insert(doc.into())
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    /// Insert multiple documents.
    fn insert_batch(&mut self, docs: Vec<PyDocument>) -> PyResult<()> {
        let rust_docs: Vec<dogma_vdb::doc::Document> = docs.into_iter().map(Into::into).collect();
        self.inner
            .insert_batch(&rust_docs)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    /// Search with a query vector.
    #[pyo3(signature = (query, k=10))]
    fn search(&self, query: Vec<f32>, k: usize) -> Vec<PyScoredDocument> {
        self.inner
            .search(&query, k)
            .into_iter()
            .map(Into::into)
            .collect()
    }

    /// Delete documents by their IDs.
    fn delete(&mut self, ids: Vec<String>) -> PyResult<usize> {
        let str_ids: Vec<&str> = ids.iter().map(|s| s.as_str()).collect();
        self.inner
            .delete(&str_ids)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    /// Replace a document by ID (delete + insert).
    fn update(&mut self, doc: PyDocument) -> PyResult<()> {
        self.inner
            .update(doc.into())
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    /// Iterate over all documents.
    fn documents(&self) -> Vec<PyDocument> {
        self.inner.documents().cloned().map(Into::into).collect()
    }

    /// Export to JSONL for debugging.
    fn export_jsonl(&self, path: &str) -> PyResult<()> {
        self.inner
            .export_jsonl(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        format!(
            "Collection(name='{}', len={}, path='{}')",
            self.inner.name(),
            self.inner.len(),
            self.inner.path().display()
        )
    }
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

/// Python bindings for dogma-vdb.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMetric>()?;
    m.add_class::<PyDocument>()?;
    m.add_class::<PyScoredDocument>()?;
    m.add_class::<PyCollection>()?;
    Ok(())
}
