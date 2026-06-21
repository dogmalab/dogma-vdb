//! High-level `Collection` API.
//!
//! A `Collection` ties together a `.vdb` file and an in-memory index.
//! It is the main entry point for end-users.
//!
//! The index backend is chosen from the global config (`config.toml`):
//! - `index_type = "bruteforce"` (default) — precise O(n·d) search
//! - `index_type = "hnsw"` — approximate O(log n) search

use crate::config::{CollectionConfig, CONFIG};
use crate::distance::Metric;
use crate::doc::Document;
use crate::error::Result;
use crate::index::{
    BruteForceIndex, HnswConfig, HnswIndex, Index, IvfPqConfig, IvfPqIndex, ScoredDocument,
};
use crate::storage::traits::VectorStorage;
use crate::storage::BinStorage;
use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

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
    /// Zero-copy embedding storage (mmap or memory-backed).
    emb_storage: Option<Arc<dyn VectorStorage>>,
}

impl Collection {
    /// Open (or create) a collection from a `.vdb` path.
    ///
    /// The index type and parameters are read from the global config
    /// (`config.toml` or env vars).
    pub fn open(path: impl Into<PathBuf>) -> Result<Self> {
        Self::open_with_config(path, &CONFIG.collection)
    }

    /// Open a collection with an explicit index type and metric override.
    ///
    /// `index_type` can be `"bruteforce"`, `"hnsw"`, or `"ivf_pq"`.
    /// HNSW/IVF-PQ parameters are still read from the global config.
    pub fn open_with(
        path: impl Into<PathBuf>,
        index_type: &str,
        index_metric: &str,
    ) -> Result<Self> {
        let mut cfg = CONFIG.collection.clone();
        cfg.index_type = index_type.to_string();
        cfg.index_metric = index_metric.to_string();
        Self::open_with_config(path, &cfg)
    }

    /// Open (or create) a collection with an explicit [`CollectionConfig`].
    ///
    /// This is the primary constructor — all other `open*` methods
    /// delegate to this one.  Use it when you need full control over
    /// index type, metric, and HNSW/IVF-PQ parameters without relying
    /// on the global config.
    pub fn open_with_config(path: impl Into<PathBuf>, cfg: &CollectionConfig) -> Result<Self> {
        let path: PathBuf = path.into();
        let index = build_index(cfg)?;
        Self::build(path, index)
    }

    /// Internal: build a Collection from a path and a ready index.
    fn build(path: PathBuf, mut index: Box<dyn Index>) -> Result<Self> {
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("default")
            .to_string();

        let storage = BinStorage::new(&path);

        // Load via mmap — zero-copy for embeddings
        let emb_storage: Option<Arc<dyn VectorStorage>> = if storage.exists_with_magic() {
            let (documents, mmap_store) = storage.load_mmap()?;
            let arc_store = mmap_store.map(|s| Arc::new(s) as Arc<dyn VectorStorage>);
            // Inject storage BEFORE insert so backends can read embeddings
            if let Some(ref s) = arc_store {
                index.set_storage(s.clone());
            }
            index.insert(&documents);
            arc_store
        } else {
            None
        };

        Ok(Self {
            name,
            storage,
            index,
            emb_storage,
        })
    }

    /// The collection name (derived from the file stem).
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Number of documents in the collection.
    #[must_use]
    pub fn len(&self) -> usize {
        self.index.len()
    }

    #[must_use]
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
    #[must_use]
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
    #[must_use]
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

    /// Access the zero-copy embedding storage, if available.
    ///
    /// Returns `None` when the collection is memory-backed (volatile)
    /// or the binary storage file doesn't exist yet.
    #[must_use]
    pub fn embedding_storage(&self) -> Option<&Arc<dyn VectorStorage>> {
        self.emb_storage.as_ref()
    }

    /// Export the collection to a JSONL file for debugging / inspection.
    ///
    /// Each line is a self-describing JSON object that can be inspected
    /// with `cat`, `grep`, `sed`.
    pub fn export_jsonl(&self, path: impl Into<std::path::PathBuf>) -> Result<()> {
        use std::io::Write;
        let path: std::path::PathBuf = path.into();
        let file = std::fs::File::create(&path).map_err(|source| crate::error::Error::Io {
            path: path.clone(),
            source,
        })?;
        let mut writer = std::io::BufWriter::new(file);
        for doc in self.index.documents() {
            let line =
                serde_json::to_string(doc).map_err(|source| crate::error::Error::ParseJson {
                    line: 0,
                    detail: "failed to serialize document".into(),
                    source,
                })?;
            writeln!(writer, "{}", line).map_err(|source| crate::error::Error::Io {
                path: path.clone(),
                source,
            })?;
        }
        writer
            .flush()
            .map_err(|source| crate::error::Error::Io { path, source })?;
        Ok(())
    }

    /// Execute a hybrid search combining vector similarity and BM25 text search.
    ///
    /// The pipeline follows the configured [`PerformanceProfile`]:
    ///
    /// 1. **Extract** — retrieve `3 * top_k` candidates from each active engine.
    /// 2. **Fuse** — if both engines are active, apply RRF and keep `2 * top_k`.
    /// 3. **Rerank** — if a reranker is provided and the profile enables it,
    ///    reorder the candidates.  Otherwise truncate directly to `top_k`.
    #[must_use]
    pub fn hybrid_search(
        &self,
        query_vec: &[f32],
        query_text: &str,
        bm25: Option<&crate::index::bm25::Bm25Index>,
        reranker: Option<&dyn crate::rerank::Reranker>,
        pipeline: &crate::config::QueryPipelineConfig,
    ) -> Vec<ScoredDocument> {
        let top_k = pipeline.top_k.max(1);
        let mul = pipeline.candidate_multiplier();

        let vec_results = self.index.search(query_vec, top_k * mul);
        let bm25_results: Vec<(usize, f32)> = if pipeline.use_bm25() {
            bm25.map(|b| b.search(query_text, top_k * mul))
                .unwrap_or_default()
        } else {
            Vec::new()
        };

        // Build a doc_id lookup: document index → position in documents slice.
        // ScoredDocument carries the document directly, so we resolve IDs here
        // for RRF fusion with BM25 results.
        let mut doc_index: HashMap<String, usize> = HashMap::new();
        for (i, doc) in self.index.documents().iter().enumerate() {
            doc_index.insert(doc.id.clone(), i);
        }

        // Fuse vector + BM25 results using document IDs as the common key
        let fused: Vec<(usize, f32)> = if pipeline.use_bm25() && !bm25_results.is_empty() {
            let vec_ids: Vec<(usize, f32)> = vec_results
                .iter()
                .filter_map(|r| doc_index.get(&r.document.id).map(|&idx| (idx, r.score)))
                .collect();
            crate::index::rrf::fuse(&vec_ids, &bm25_results, top_k * 2)
        } else {
            let mut v: Vec<(usize, f32)> = vec_results
                .iter()
                .filter_map(|r| doc_index.get(&r.document.id).map(|&idx| (idx, r.score)))
                .collect();
            v.truncate(top_k * 2);
            v
        };

        if pipeline.use_reranker() {
            if let Some(rank) = reranker {
                let mut docs: Vec<Document> = fused
                    .iter()
                    .map(|&(id, _)| self.index.documents()[id].clone())
                    .collect();
                if rank.rerank(query_text, &mut docs).is_ok() {
                    return docs
                        .into_iter()
                        .map(|d| ScoredDocument {
                            score: 0.0,
                            document: d,
                        })
                        .collect();
                }
            }
        }

        // Without reranker: sort fused, truncate, then hydrate only top-k
        let mut fused = fused;
        fused.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        fused.truncate(top_k);
        fused
            .into_iter()
            .map(|(id, score)| ScoredDocument {
                score,
                document: self.index.documents()[id].clone(),
            })
            .collect()
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

/// Build the appropriate index backend from a [`CollectionConfig`].
fn build_index(cfg: &CollectionConfig) -> Result<Box<dyn Index>> {
    let metric = parse_metric(&cfg.index_metric);

    match cfg.index_type.as_str() {
        "hnsw" => Ok(Box::new(HnswIndex::new(HnswConfig {
            m: cfg.hnsw_m,
            ef_construction: cfg.hnsw_ef_construction,
            ef_search: cfg.hnsw_ef_search,
            metric,
            flat_embeddings: cfg.hnsw_flat_embeddings,
            sq: cfg.sq,
            sq_rescore: cfg.sq_rescore,
        }))),
        "ivf_pq" => {
            let ivf_cfg = IvfPqConfig {
                n_list: cfg.ivf_pq_n_clusters,
                m_subspaces: cfg.ivf_pq_n_subvectors,
                n_probe: cfg.ivf_pq_n_probe,
                metric,
                rerank_enabled: std::env::var("DOGMA_RERANK").as_deref() == Ok("1"),
                ..IvfPqConfig::default()
            };
            ivf_cfg.validate()?;
            Ok(Box::new(IvfPqIndex::new(ivf_cfg)))
        }
        _ => Ok(Box::new(BruteForceIndex::new_with(
            metric,
            cfg.sq,
            cfg.sq_rescore,
        ))),
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

    // ── Hybrid search tests ──────────────────────────────────────────────

    #[test]
    fn test_hybrid_pipeline_precision_local() {
        use crate::config::{PerformanceProfile, QueryPipelineConfig};
        use crate::index::bm25::Bm25Index;

        let dir = tempfile::tempdir().unwrap();
        let mut col =
            Collection::open_with(dir.path().join("h.vdb"), "bruteforce", "cosine").unwrap();
        col.insert(
            Document::builder("a", "fn chunk_batch does parallel chunking")
                .embedding(vec![1.0, 0.0])
                .build(),
        )
        .unwrap();
        col.insert(
            Document::builder("b", "fn bake_cake makes a delicious cake")
                .embedding(vec![0.0, 1.0])
                .build(),
        )
        .unwrap();

        let mut bm25 = Bm25Index::new();
        for (i, d) in col.index.documents().iter().enumerate() {
            bm25.insert(i, &d.text);
        }

        let cfg = QueryPipelineConfig {
            profile: PerformanceProfile::PrecisionLocal,
            top_k: 5,
        };

        let results = col.hybrid_search(&[1.0, 0.0], "chunk_batch", Some(&bm25), None, &cfg);
        assert!(!results.is_empty(), "hybrid search should return results");
        assert_eq!(
            results[0].document.id, "a",
            "RRF fusion should rank 'chunk_batch' doc first when query matches text exactly"
        );
    }

    #[test]
    fn test_hybrid_search_with_phrase_query() {
        use crate::config::{PerformanceProfile, QueryPipelineConfig};
        use crate::index::bm25::Bm25Index;

        let dir = tempfile::tempdir().unwrap();
        let mut col =
            Collection::open_with(dir.path().join("phrase.vdb"), "bruteforce", "cosine").unwrap();
        col.insert(
            Document::builder("a", "part time job opening")
                .embedding(vec![1.0, 0.0])
                .build(),
        )
        .unwrap();
        col.insert(
            Document::builder("b", "time to find a part")
                .embedding(vec![0.0, 1.0])
                .build(),
        )
        .unwrap();

        let mut bm25 = Bm25Index::new();
        for (i, d) in col.index.documents().iter().enumerate() {
            bm25.insert(i, &d.text);
        }

        let cfg = QueryPipelineConfig {
            profile: PerformanceProfile::PrecisionLocal,
            top_k: 5,
        };

        // Phrase query "part time" should only match doc a
        let results = col.hybrid_search(&[1.0, 0.0], "\"part time\"", Some(&bm25), None, &cfg);
        assert!(!results.is_empty(), "phrase search should return results");
        assert_eq!(
            results[0].document.id, "a",
            "phrase 'part time' should match doc a (consecutive), not doc b"
        );
    }

    #[test]
    fn test_pipeline_profile_behavior() {
        use crate::config::{PerformanceProfile, QueryPipelineConfig};

        let dir = tempfile::tempdir().unwrap();
        let col = Collection::open_with(dir.path().join("p.vdb"), "bruteforce", "cosine").unwrap();
        // No documents — but we can still test that the profile dispatch works

        let cfg_fast = QueryPipelineConfig {
            profile: PerformanceProfile::MaxSpeed,
            top_k: 3,
        };

        // MaxSpeed: BM25 inactive → empty BM25 results
        let results = col.hybrid_search(&[1.0, 0.0], "anything", None, None, &cfg_fast);
        assert!(
            results.is_empty(),
            "empty collection with MaxSpeed returns empty"
        );

        // Verify profile flags
        assert!(!cfg_fast.use_bm25(), "MaxSpeed should not use BM25");
        assert!(!cfg_fast.use_reranker(), "MaxSpeed should not use reranker");

        let cfg_precision = QueryPipelineConfig {
            profile: PerformanceProfile::PrecisionLocal,
            top_k: 3,
        };
        assert!(cfg_precision.use_bm25(), "PrecisionLocal should use BM25");
        assert!(
            cfg_precision.use_reranker(),
            "PrecisionLocal should use reranker"
        );

        let cfg_hybrid = QueryPipelineConfig {
            profile: PerformanceProfile::HybridProduction,
            top_k: 3,
        };
        assert!(cfg_hybrid.use_bm25(), "HybridProduction should use BM25");
        assert!(
            cfg_hybrid.use_reranker(),
            "HybridProduction should use reranker"
        );
    }

    #[test]
    fn test_search_after_reopen_bruteforce() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("reopen_bf.vdb");

        // Insert + search in same session
        {
            let mut col = Collection::open_with(&path, "bruteforce", "cosine").unwrap();
            col.insert(
                Document::builder("a", "hello")
                    .embedding(vec![1.0, 0.0])
                    .build(),
            )
            .unwrap();
            col.insert(
                Document::builder("b", "world")
                    .embedding(vec![0.0, 1.0])
                    .build(),
            )
            .unwrap();
            let r = col.search(&[1.0, 0.0], 2);
            assert_eq!(r[0].document.id, "a");
        }

        // Reopen + search (uses mmap)
        {
            let col = Collection::open_with(&path, "bruteforce", "cosine").unwrap();
            assert_eq!(col.len(), 2);
            let r = col.search(&[1.0, 0.0], 2);
            assert_eq!(r.len(), 2, "search after reopen should return results");
            assert_eq!(r[0].document.id, "a");
        }
    }

    #[test]
    fn test_search_after_reopen_hnsw() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("reopen_hnsw.vdb");

        // Insert + search in same session
        {
            let mut col = Collection::open_with(&path, "hnsw", "cosine").unwrap();
            col.insert(
                Document::builder("a", "hello")
                    .embedding(vec![1.0, 0.0])
                    .build(),
            )
            .unwrap();
            col.insert(
                Document::builder("b", "world")
                    .embedding(vec![0.0, 1.0])
                    .build(),
            )
            .unwrap();
            let r = col.search(&[1.0, 0.0], 2);
            assert_eq!(r[0].document.id, "a");
        }

        // Reopen + search (uses mmap + graph rebuild)
        {
            let col = Collection::open_with(&path, "hnsw", "cosine").unwrap();
            assert_eq!(col.len(), 2);
            let r = col.search(&[1.0, 0.0], 2);
            assert_eq!(r.len(), 2, "search after reopen should return results");
            assert_eq!(r[0].document.id, "a");
        }
    }

    #[test]
    fn test_search_after_reopen_ivf_pq() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("reopen_ivf.vdb");

        // Insert + search in same session
        {
            let mut col = Collection::open_with(&path, "ivf_pq", "cosine").unwrap();
            for i in 0..20 {
                col.insert(
                    Document::builder(format!("d{i}"), format!("doc {i}"))
                        .embedding(vec![i as f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                        .build(),
                )
                .unwrap();
            }
            let r = col.search(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 3);
            assert!(!r.is_empty());
        }

        // Reopen + search (uses mmap + index rebuild)
        {
            let col = Collection::open_with(&path, "ivf_pq", "cosine").unwrap();
            assert_eq!(col.len(), 20);
            let r = col.search(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 3);
            assert!(!r.is_empty(), "search after reopen should return results");
        }
    }

    #[test]
    fn test_dim_from_storage_calculation() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dim_test.vdb");

        // Create collection with specific dim
        {
            let mut col = Collection::open_with(&path, "bruteforce", "cosine").unwrap();
            col.insert(
                Document::builder("a", "hello")
                    .embedding(vec![1.0, 2.0, 3.0, 4.0])
                    .build(),
            )
            .unwrap();
            col.insert(
                Document::builder("b", "world")
                    .embedding(vec![5.0, 6.0, 7.0, 8.0])
                    .build(),
            )
            .unwrap();
        }

        // Reopen — verify dim is calculated correctly from storage
        {
            let col = Collection::open_with(&path, "bruteforce", "cosine").unwrap();
            assert_eq!(col.len(), 2);

            // Search should work with correct dim
            let r = col.search(&[1.0, 2.0, 3.0, 4.0], 2);
            assert_eq!(r.len(), 2);
            assert_eq!(r[0].document.id, "a");

            // Verify the score is correct (cosine of identical vectors = 1.0)
            assert!(
                (r[0].score - 1.0).abs() < 1e-5,
                "score should be ~1.0 for identical vectors, got {}",
                r[0].score
            );
        }
    }

    #[test]
    fn test_dim_from_storage_21_docs() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dim_21.vdb");

        // Create 21 docs with dim=4 — each doc has a unique direction
        {
            let mut col = Collection::open_with(&path, "bruteforce", "cosine").unwrap();
            for i in 0..21 {
                let angle = i as f64 * 0.3;
                col.insert(
                    Document::builder(format!("d{i}"), format!("doc {i}"))
                        .embedding(vec![angle.cos() as f32, angle.sin() as f32, 0.0, 0.0])
                        .build(),
                )
                .unwrap();
            }
        }

        // Reopen — verify dim=4 (not dim=3), search works correctly
        {
            let col = Collection::open_with(&path, "bruteforce", "cosine").unwrap();
            assert_eq!(col.len(), 21);

            // Query close to d0 (angle=0, emb=[1,0,0,0])
            let r = col.search(&[1.0, 0.0, 0.0, 0.0], 3);
            assert_eq!(r.len(), 3, "should return 3 results");
            assert_eq!(r[0].document.id, "d0", "d0 should be closest to [1,0,0,0]");

            // Verify score is correct (cosine of [1,0,0,0] with itself = 1.0)
            assert!(
                (r[0].score - 1.0).abs() < 1e-5,
                "score should be ~1.0, got {}",
                r[0].score
            );

            // Verify dim is correct: score for d10 (angle=3.0, emb=[cos(3),sin(3),0,0])
            // should be different from d0
            let r10 = col.search(
                &[
                    (10.0_f64 * 0.3).cos() as f32,
                    (10.0_f64 * 0.3).sin() as f32,
                    0.0,
                    0.0,
                ],
                1,
            );
            assert_eq!(r10[0].document.id, "d10");
        }
    }

    #[test]
    fn test_triple_reopen_search() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("triple.vdb");

        // 1st open: create and insert
        {
            let mut col = Collection::open_with(&path, "bruteforce", "cosine").unwrap();
            for i in 1..=5 {
                col.insert(
                    Document::builder(format!("d{i}"), format!("doc {i}"))
                        .embedding(vec![i as f32, 0.0, 0.0, 0.0])
                        .build(),
                )
                .unwrap();
            }
            let r = col.search(&[1.0, 0.0, 0.0, 0.0], 1);
            assert_eq!(r.len(), 1);
            assert!(r[0].score > 0.9, "score should be ~1.0");
        }

        // 2nd open: reopen (simulates SessionManager::open)
        {
            let col = Collection::open_with(&path, "bruteforce", "cosine").unwrap();
            assert_eq!(col.len(), 5);
            let r = col.search(&[1.0, 0.0, 0.0, 0.0], 1);
            assert_eq!(r.len(), 1);
            assert!(r[0].score > 0.9, "score should be ~1.0 after reopen");
        }

        // 3rd open: reopen again (simulates debug block + SessionManager)
        {
            let col = Collection::open_with(&path, "bruteforce", "cosine").unwrap();
            assert_eq!(col.len(), 5);
            let r = col.search(&[1.0, 0.0, 0.0, 0.0], 1);
            assert_eq!(r.len(), 1);
            assert!(r[0].score > 0.9, "score should be ~1.0 after 2nd reopen");
        }
    }
}
