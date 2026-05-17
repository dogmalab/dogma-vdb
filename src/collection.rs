//! High-level `Collection` API.
//!
//! A `Collection` ties together a `.vdb` file and an in-memory index.
//! It is the main entry point for end-users.
//!
//! The index backend is chosen from the global config (`config.toml`):
//! - `index_type = "bruteforce"` (default) — precise O(n·d) search
//! - `index_type = "hnsw"` — approximate O(log n) search

use crate::config::CONFIG;
use crate::distance::Metric;
use crate::doc::Document;
use crate::error::Result;
use crate::index::{
    BruteForceIndex, HnswConfig, HnswIndex, Index, IvfPqConfig, IvfPqIndex, ScoredDocument,
};
use crate::storage::BinStorage;
use std::fmt;
use std::path::PathBuf;

/// A named collection backed by a `.vdb` file.
///
/// # Example
/// ```ignore
/// use dogma_vdb::prelude::*;
///
/// let mut col = Collection::open("my_data.vdb")?;
/// col.insert(Document::new("id-1", "Rust is fast"))?;
/// let results = col.search(&[0.1, 0.2, 0.3], 5);
/// ```
pub struct Collection {
    name: String,
    storage: BinStorage,
    index: Box<dyn Index>,
}

impl Collection {
    /// Open (or create) a collection from a `.vdb` path.
    ///
    /// The index type and parameters are read from the global config
    /// (`config.toml` or env vars).
    pub fn open(path: impl Into<PathBuf>) -> Result<Self> {
        let cfg = &CONFIG.collection;
        let path: PathBuf = path.into();
        let metric = parse_metric(&cfg.index_metric);

        let index: Box<dyn Index> = match cfg.index_type.as_str() {
            "hnsw" => Box::new(HnswIndex::new(HnswConfig {
                m: cfg.hnsw_m,
                ef_construction: cfg.hnsw_ef_construction,
                ef_search: cfg.hnsw_ef_search,
                metric,
                flat_embeddings: cfg.hnsw_flat_embeddings,
                sq: cfg.sq,
                sq_rescore: cfg.sq_rescore,
            })),
            "ivf_pq" => Box::new(IvfPqIndex::new(IvfPqConfig {
                n_clusters: cfg.ivf_pq_n_clusters,
                n_subvectors: cfg.ivf_pq_n_subvectors,
                n_probe: cfg.ivf_pq_n_probe,
                metric,
            })),
            _ => Box::new(BruteForceIndex::new_with(metric, cfg.sq, cfg.sq_rescore)),
        };

        Self::build(path, index)
    }

    /// Open a collection with an explicit index type override.
    ///
    /// `index_type` can be `"bruteforce"`, `"hnsw"`, or `"ivf_pq"`.
    pub fn open_with(
        path: impl Into<PathBuf>,
        index_type: &str,
        index_metric: &str,
    ) -> Result<Self> {
        let cfg = &CONFIG.collection;
        let path: PathBuf = path.into();
        let metric = parse_metric(index_metric);

        let index: Box<dyn Index> = match index_type {
            "hnsw" => Box::new(HnswIndex::new(HnswConfig {
                m: cfg.hnsw_m,
                ef_construction: cfg.hnsw_ef_construction,
                ef_search: cfg.hnsw_ef_search,
                metric,
                flat_embeddings: cfg.hnsw_flat_embeddings,
                sq: cfg.sq,
                sq_rescore: cfg.sq_rescore,
            })),
            "ivf_pq" => Box::new(IvfPqIndex::new(IvfPqConfig {
                n_clusters: cfg.ivf_pq_n_clusters,
                n_subvectors: cfg.ivf_pq_n_subvectors,
                n_probe: cfg.ivf_pq_n_probe,
                metric,
            })),
            _ => Box::new(BruteForceIndex::new_with(metric, cfg.sq, cfg.sq_rescore)),
        };

        Self::build(path, index)
    }

    /// Internal: build a Collection from a path and a ready index.
    fn build(path: PathBuf, mut index: Box<dyn Index>) -> Result<Self> {
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("default")
            .to_string();

        // Auto-detect format: if the file exists and doesn't have the
        // binary magic, it's an old JSONL file — migrate transparently.
        if path.exists() && !BinStorage::is_binary(&path) {
            let jsonl = crate::storage::JsonlStorage::new(&path);
            let documents = jsonl.load()?;
            index.insert(&documents);
            let bin = BinStorage::new(&path);
            bin.store(&documents)?;
        }

        let storage = BinStorage::new(&path);
        if storage.exists_with_magic() {
            let documents = storage.load()?;
            index.insert(&documents);
        }

        Ok(Self {
            name,
            storage,
            index,
        })
    }

    /// The collection name (derived from the file stem).
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Number of documents in the collection.
    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Insert a single document and persist immediately.
    pub fn insert(&mut self, doc: Document) -> Result<()> {
        self.index.insert(&[doc]);
        self.storage.store(self.index.documents())?;
        Ok(())
    }

    /// Insert many documents at once.
    pub fn insert_batch(&mut self, docs: &[Document]) -> Result<()> {
        self.index.insert(docs);
        self.storage.store(self.index.documents())?;
        Ok(())
    }

    /// Delete documents by their IDs.
    ///
    /// Removes from both the index and the `.vdb` storage file.
    /// Returns the number of documents actually deleted.
    pub fn delete(&mut self, ids: &[&str]) -> Result<usize> {
        let deleted = self.index.delete(ids);
        if deleted > 0 {
            let remaining = self.index.documents().to_vec();
            self.storage.store(&remaining)?;
        }
        Ok(deleted)
    }

    /// Update (replace) a document by ID.
    ///
    /// Equivalent to `delete` followed by `insert`.  The old document
    /// is removed from both the index and storage, then the new
    /// document is appended.
    pub fn update(&mut self, doc: Document) -> Result<()> {
        self.delete(&[&doc.id])?;
        self.insert(doc)
    }

    /// Search with an embedder: embed the query text, then search.
    pub fn search_query(
        &self,
        embedder: &dyn crate::embedding::Embedder,
        text: &str,
        k: usize,
    ) -> Result<Vec<ScoredDocument>> {
        let query = embedder.embed(text)?;
        Ok(self.index.search(&query, k))
    }

    /// Search directly with a query vector.
    ///
    /// Uses the index's built-in metric (configured at construction time).
    pub fn search(&self, query: &[f32], k: usize) -> Vec<ScoredDocument> {
        self.index.search(query, k)
    }

    /// Search with a metadata / content filter.
    ///
    /// Only documents matching `filter` are considered.
    ///
    /// # Example
    /// ```
    /// use dogma_vdb::prelude::*;
    /// use dogma_vdb::filter;
    ///
    /// let dir = tempfile::tempdir().unwrap();
    /// let mut col = Collection::open(dir.path().join("filter.vdb")).unwrap();
    /// col.insert(
    ///     Document::builder("a", "rust is fast")
    ///         .embedding(vec![1.0, 0.0])
    ///         .metadata("lang", "en")
    ///         .build()
    /// ).unwrap();
    /// col.insert(
    ///     Document::builder("b", "rust es rapido")
    ///         .embedding(vec![0.0, 1.0])
    ///         .metadata("lang", "es")
    ///         .build()
    /// ).unwrap();
    ///
    /// let results = col.search_filtered(&[1.0, 0.0], 5, &filter::metadata_eq("lang", "en"));
    /// assert_eq!(results.len(), 1);
    /// assert_eq!(results[0].document.id, "a");
    /// ```
    pub fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        filter: &(dyn Fn(&Document) -> bool + Sync),
    ) -> Vec<ScoredDocument> {
        self.index.search_filtered(query, k, filter)
    }

    /// Iterate over all documents.
    pub fn documents(&self) -> impl Iterator<Item = &Document> {
        self.index.documents().iter()
    }

    /// The underlying file path.
    pub fn path(&self) -> &std::path::Path {
        self.storage.path()
    }

    /// Export the collection to a JSONL file for debugging / inspection.
    ///
    /// The exported file can be inspected with `cat`, `grep`, `sed`,
    /// and re-imported with a future `Collection::open()` call (the
    /// old JSONL format is auto-detected and migrated).
    pub fn export_jsonl(&self, path: impl Into<std::path::PathBuf>) -> Result<()> {
        let jsonl = crate::storage::JsonlStorage::new(path);
        jsonl.store(self.index.documents())
    }
}

// ---------------------------------------------------------------------------
// Debug impl (manual — Box<dyn Index> doesn't support derive)
// ---------------------------------------------------------------------------

impl fmt::Debug for Collection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Collection")
            .field("name", &self.name)
            .field("storage", &self.storage)
            .field(
                "index",
                &format_args!("Box<dyn Index>({} docs)", self.len()),
            )
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Parse a metric name from a config string.
fn parse_metric(s: &str) -> Metric {
    match s.to_lowercase().as_str() {
        "dot" | "dot_product" => Metric::Dot,
        "euclidean" | "l2" => Metric::Euclidean,
        _ => Metric::Cosine,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::doc::Document;

    fn make_doc(id: &str, embedding: Vec<f32>) -> Document {
        Document::builder(id, id).embedding(embedding).build()
    }

    #[test]
    fn test_open_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.vdb");
        let col = Collection::open(&path).unwrap();
        assert!(col.is_empty());
        assert_eq!(col.name(), "empty");
    }

    #[test]
    fn test_open_creates_file_on_insert() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("new.vdb");
        let mut col = Collection::open(&path).unwrap();
        assert!(!path.exists());
        col.insert(Document::new("a", "hello")).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn test_insert_and_search() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.vdb");
        let mut col = Collection::open(&path).unwrap();
        col.insert(make_doc("1", vec![1.0, 0.0])).unwrap();
        assert_eq!(col.len(), 1);
    }

    #[test]
    fn test_persistence_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("persist.vdb");
        {
            let mut col = Collection::open(&path).unwrap();
            col.insert(Document::new("a", "texto a")).unwrap();
            col.insert(Document::new("b", "texto b")).unwrap();
            assert_eq!(col.len(), 2);
        }
        let col = Collection::open(&path).unwrap();
        assert_eq!(col.len(), 2);
    }

    #[test]
    fn test_insert_batch() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("batch.vdb");
        let mut col = Collection::open(&path).unwrap();
        let docs = vec![make_doc("1", vec![1.0, 0.0]), make_doc("2", vec![0.0, 1.0])];
        col.insert_batch(&docs).unwrap();
        assert_eq!(col.len(), 2);
    }

    #[test]
    fn test_search_results() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("search.vdb");
        // Use cosine as default
        let mut col = Collection::open_with(&path, "bruteforce", "cosine").unwrap();

        col.insert(make_doc("a", vec![1.0, 0.0])).unwrap();
        col.insert(make_doc("b", vec![0.0, 1.0])).unwrap();

        let results = col.search(&[1.0, 0.0], 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].document.id, "a");
    }

    #[test]
    fn test_search_with_hnsw() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("hnsw_search.vdb");
        let mut col = Collection::open_with(&path, "hnsw", "cosine").unwrap();

        col.insert(make_doc("a", vec![1.0, 0.0])).unwrap();
        col.insert(make_doc("b", vec![0.0, 1.0])).unwrap();

        let results = col.search(&[1.0, 0.0], 2);
        assert_eq!(results.len(), 2);
        // HNSW is approximate — just check we get results
        assert_eq!(results[0].document.id, "a");
    }

    #[test]
    fn test_documents_iterator() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("iter.vdb");
        let mut col = Collection::open(&path).unwrap();

        col.insert(Document::new("a", "alpha")).unwrap();
        col.insert(Document::new("b", "beta")).unwrap();

        let ids: Vec<&str> = col.documents().map(|d| d.id.as_str()).collect();
        assert_eq!(ids, vec!["a", "b"]);
    }

    #[test]
    fn test_name_from_path() {
        let col = Collection::open("/some/path/my_collection.vdb").unwrap();
        assert_eq!(col.name(), "my_collection");
    }

    #[test]
    fn test_search_empty_collection() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty_search.vdb");
        let col = Collection::open(&path).unwrap();
        let results = col.search(&[1.0, 2.0], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_hnsw_empty_collection() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty_hnsw.vdb");
        let col = Collection::open_with(&path, "hnsw", "cosine").unwrap();
        let results = col.search(&[1.0, 2.0], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_delete_document() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("delete.vdb");
        let mut col = Collection::open(&path).unwrap();

        col.insert(Document::new("a", "alpha")).unwrap();
        col.insert(Document::new("b", "beta")).unwrap();
        col.insert(Document::new("c", "gamma")).unwrap();
        assert_eq!(col.len(), 3);

        let deleted = col.delete(&["a", "c"]).unwrap();
        assert_eq!(deleted, 2);
        assert_eq!(col.len(), 1);

        let ids: Vec<&str> = col.documents().map(|d| d.id.as_str()).collect();
        assert_eq!(ids, vec!["b"]);
    }

    #[test]
    fn test_delete_nonexistent() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("delete_none.vdb");
        let mut col = Collection::open(&path).unwrap();

        col.insert(Document::new("a", "alpha")).unwrap();
        let deleted = col.delete(&["x"]).unwrap();
        assert_eq!(deleted, 0);
        assert_eq!(col.len(), 1);
    }

    #[test]
    fn test_update_document() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("update.vdb");
        let mut col = Collection::open(&path).unwrap();

        col.insert(
            Document::builder("doc1", "old text")
                .embedding(vec![1.0, 0.0])
                .build(),
        )
        .unwrap();
        assert_eq!(col.len(), 1);

        col.update(
            Document::builder("doc1", "new text")
                .embedding(vec![0.0, 1.0])
                .build(),
        )
        .unwrap();

        assert_eq!(col.len(), 1);
        let doc = col.documents().next().unwrap();
        assert_eq!(doc.text, "new text");
        assert_eq!(doc.embedding, vec![0.0, 1.0]);
    }

    #[test]
    fn test_delete_persists_across_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("delete_persist.vdb");
        {
            let mut col = Collection::open(&path).unwrap();
            col.insert(Document::new("a", "keep")).unwrap();
            col.insert(Document::new("b", "remove")).unwrap();
            col.delete(&["b"]).unwrap();
        }
        let col = Collection::open(&path).unwrap();
        assert_eq!(col.len(), 1);
        assert_eq!(col.documents().next().unwrap().id, "a");
    }

    #[test]
    fn test_search_filtered_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("filter_test.vdb");
        let mut col = Collection::open_with(&path, "bruteforce", "cosine").unwrap();

        col.insert(
            Document::builder("en1", "hello")
                .embedding(vec![1.0, 0.0, 0.0])
                .metadata("lang", "en")
                .build(),
        )
        .unwrap();
        col.insert(
            Document::builder("en2", "world")
                .embedding(vec![0.0, 1.0, 0.0])
                .metadata("lang", "en")
                .build(),
        )
        .unwrap();
        col.insert(
            Document::builder("es1", "hola")
                .embedding(vec![0.0, 0.0, 1.0])
                .metadata("lang", "es")
                .build(),
        )
        .unwrap();

        // Filter for English only
        let results = col.search_filtered(&[1.0, 0.0, 0.0], 5, &|doc: &Document| {
            doc.metadata_val("lang") == Some("en")
        });
        assert_eq!(results.len(), 2);
        for r in &results {
            assert_eq!(r.document.metadata_val("lang"), Some("en"));
        }
    }

    #[test]
    fn test_search_filtered_no_match() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("filter_none.vdb");
        let mut col = Collection::open_with(&path, "bruteforce", "cosine").unwrap();

        col.insert(
            Document::builder("a", "text")
                .embedding(vec![1.0, 0.0])
                .metadata("lang", "en")
                .build(),
        )
        .unwrap();

        let results = col.search_filtered(&[1.0, 0.0], 5, &|doc: &Document| {
            doc.metadata_val("lang") == Some("es")
        });
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_filtered_no_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("filter_no_meta.vdb");
        let mut col = Collection::open_with(&path, "bruteforce", "cosine").unwrap();

        col.insert(
            Document::builder("a", "no metadata")
                .embedding(vec![1.0, 0.0])
                .build(),
        )
        .unwrap();

        // Filter for a key that doesn't exist
        let results = col.search_filtered(&[1.0, 0.0], 5, &|doc: &Document| {
            doc.metadata_val("lang").is_some()
        });
        assert!(results.is_empty());
    }
}
