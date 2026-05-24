//! Integration tests for dogma-vdb.
//!
//! These tests use real temporary files and exercise the full
//! read-write-search cycle.

use dogma_vdb::prelude::*;
use dogma_vdb::storage::JsonlStorage;

fn make_embedded_doc(id: &str, embedding: Vec<f32>) -> Document {
    Document::builder(id, format!("doc {}", id))
        .embedding(embedding)
        .build()
}

#[test]
fn test_open_empty_collection() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.vdb");
    let col = Collection::open(&path).unwrap();
    assert!(col.is_empty());
}

#[test]
fn test_insert_and_search() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.vdb");
    let mut col = Collection::open(&path).unwrap();
    col.insert(Document::new("1", "documento uno")).unwrap();
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
    }
    let col = Collection::open(&path).unwrap();
    assert_eq!(col.len(), 2);
}

#[test]
fn test_insert_with_embedding_and_search() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("emb.vdb");
    let mut col = Collection::open_with(&path, "bruteforce", "cosine").unwrap();

    col.insert(make_embedded_doc("doc1", vec![1.0, 0.0, 0.0]))
        .unwrap();
    col.insert(make_embedded_doc("doc2", vec![0.0, 1.0, 0.0]))
        .unwrap();

    let results = col.search(&[0.9, 0.1, 0.0], 2);
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].document.id, "doc1");
    assert!((results[0].score - 1.0).abs() < 0.1);
}

#[test]
fn test_multiple_metrics() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("metrics.vdb");

    // Cosine
    {
        let mut col = Collection::open_with(&path, "bruteforce", "cosine").unwrap();
        col.insert(make_embedded_doc("a", vec![1.0, 0.0])).unwrap();
        col.insert(make_embedded_doc("b", vec![0.0, 1.0])).unwrap();
        let results = col.search(&[1.0, 0.0], 2);
        assert!((results[0].score - 1.0).abs() < 1e-6);
    }

    // Dot product
    {
        let mut col = Collection::open_with(&path, "bruteforce", "dot").unwrap();
        col.insert(make_embedded_doc("a", vec![1.0, 0.0])).unwrap();
        col.insert(make_embedded_doc("b", vec![0.0, 1.0])).unwrap();
        let results = col.search(&[1.0, 0.0], 2);
        assert!((results[0].score - 1.0).abs() < 1e-6);
    }

    // Euclidean (negated): identical = 0, negated = 0
    {
        let mut col = Collection::open_with(&path, "bruteforce", "euclidean").unwrap();
        col.insert(make_embedded_doc("a", vec![1.0, 0.0])).unwrap();
        col.insert(make_embedded_doc("b", vec![0.0, 1.0])).unwrap();
        let results = col.search(&[1.0, 0.0], 2);
        assert!((results[0].score - 0.0).abs() < 1e-6);
    }
}

#[test]
fn test_insert_batch() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("batch.vdb");
    let mut col = Collection::open(&path).unwrap();

    let docs: Vec<Document> = (0..10)
        .map(|i| {
            Document::builder(format!("id-{}", i), format!("doc {}", i))
                .embedding(vec![i as f32, 0.0])
                .build()
        })
        .collect();

    col.insert_batch(&docs).unwrap();
    assert_eq!(col.len(), 10);

    let col2 = Collection::open_with(&path, "bruteforce", "cosine").unwrap();
    assert_eq!(col2.len(), 10);
}

#[test]
fn test_chunker_integration() {
    let chunker = Chunker::new(ChunkerConfig {
        chunk_size: 100,
        overlap: 10,
        separator: "\n".into(),
    });

    let long_text = "line 1\nline 2\nline 3\nline 4\nline 5\n";
    let chunks = chunker.chunk(long_text);
    assert!(!chunks.is_empty());

    let docs = chunker.chunk_to_docs(long_text, "section", std::collections::HashMap::new());
    assert_eq!(docs.len(), chunks.len());
    for (i, doc) in docs.iter().enumerate() {
        assert_eq!(doc.id, format!("section-{}", i));
    }
}

#[test]
fn test_raw_storage_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("raw.vdb");
    let storage = JsonlStorage::new(&path);

    let docs: Vec<Document> = (0..5)
        .map(|i| {
            Document::builder(format!("k{}", i), format!("value {}", i))
                .embedding(vec![i as f32 * 0.1, 0.2])
                .metadata("idx", format!("{}", i))
                .build()
        })
        .collect();
    storage.store(&docs).unwrap();

    let loaded = storage.load().unwrap();
    assert_eq!(loaded.len(), 5);
    for (orig, loaded) in docs.iter().zip(loaded.iter()) {
        assert_eq!(orig, loaded);
    }
}

#[test]
fn test_document_builder_fluent() {
    let doc = Document::builder("my-id", "Hello, world!")
        .embedding(vec![0.1, 0.2, 0.3])
        .metadata("source", "test")
        .metadata("page", "42")
        .build();

    assert_eq!(doc.id, "my-id");
    assert_eq!(doc.text, "Hello, world!");
    assert_eq!(doc.embedding, vec![0.1, 0.2, 0.3]);
    assert_eq!(doc.metadata_val("source"), Some("test"));
    assert_eq!(doc.metadata_val("page"), Some("42"));
    assert!(doc.is_embedded());
    assert_eq!(doc.dimension(), 3);
}
