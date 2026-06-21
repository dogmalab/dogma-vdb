//! Integration tests for dogma-vdb.
//!
//! These tests use real temporary files and exercise the full
//! read-write-search cycle.

use dogma_vdb::prelude::*;

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
    let chunker = TextSplitter::new(TextSplitterConfig {
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

#[test]
fn test_dim_after_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("dim_test.vdb");

    // Insert docs with 4-dim embeddings
    {
        let mut col = Collection::open_with(&path, "bruteforce", "cosine").unwrap();
        for i in 0..5 {
            col.insert(make_embedded_doc(
                &format!("doc-{i}"),
                vec![1.0, 0.0, 0.0, 0.0],
            ))
            .unwrap();
        }
        eprintln!("After insert: col.len()={}", col.len());
    }

    // Reopen and check
    let col = Collection::open_with(&path, "bruteforce", "cosine").unwrap();
    eprintln!("After reopen: col.len()={}", col.len());

    if let Some(store) = col.embedding_storage() {
        let emb_all = store.as_embeddings();
        eprintln!("emb_all.len()={}", emb_all.len());
        let dim = emb_all.len() / col.len();
        eprintln!("dim = {dim}");
        assert_eq!(dim, 4, "dim should be 4 after reopen");
    } else {
        panic!("No embedding storage after reopen!");
    }

    // Search should work
    let results = col.search(&[1.0, 0.0, 0.0, 0.0], 3);
    eprintln!("search returned {} results", results.len());
    assert!(!results.is_empty(), "search should return results");
    assert_eq!(results[0].document.id, "doc-0");
}

#[test]
fn test_dim_with_default_config() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("dim_default.vdb");

    // Use Collection::open (default config) like seed_collection does
    {
        let mut col = Collection::open(&path).unwrap();
        for i in 0..5 {
            col.insert(make_embedded_doc(
                &format!("doc-{i}"),
                vec![1.0, 0.0, 0.0, 0.0],
            ))
            .unwrap();
        }
        eprintln!("After insert: col.len()={}", col.len());
    }

    // Reopen with default config
    let col = Collection::open(&path).unwrap();
    eprintln!("After reopen: col.len()={}", col.len());

    if let Some(store) = col.embedding_storage() {
        let emb_all = store.as_embeddings();
        eprintln!("emb_all.len()={}", emb_all.len());
        let dim = emb_all.len() / col.len();
        eprintln!("dim = {dim}");
        assert_eq!(dim, 4, "dim should be 4 after reopen");
    } else {
        panic!("No embedding storage after reopen!");
    }

    let results = col.search(&[1.0, 0.0, 0.0, 0.0], 3);
    eprintln!("search returned {} results", results.len());
    assert!(!results.is_empty(), "search should return results");
}

#[test]
fn test_dim_exact_memory_stress_scenario() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sessions.vdb");

    // Exact same setup as memory_stress test_needle_in_a_haystack
    let sid = "needle-test";
    let secret_port = "9091";

    let mut docs: Vec<Document> = (0..10)
        .map(|i| {
            Document::builder(format!("noise-pre-{i}"), format!("noise {i}"))
                .embedding(vec![0.0, 0.1, 0.0, 0.0])
                .metadata("node_type", "Message")
                .metadata("session_id", sid)
                .metadata("role", "user")
                .metadata("sequence", i.to_string())
                .metadata("edge_type", "NEXT")
                .metadata("created_at", "2026-01-01T00:00:00Z")
                .build()
        })
        .collect();

    docs.push(
        Document::builder("needle-1", format!("port {secret_port}"))
            .embedding(vec![0.9, 0.0, 0.0, 0.0])
            .metadata("node_type", "Message")
            .metadata("session_id", sid)
            .metadata("role", "user")
            .metadata("sequence", "10")
            .metadata("edge_type", "NEXT")
            .metadata("created_at", "2026-01-01T00:00:00Z")
            .build(),
    );

    for i in 0..10 {
        docs.push(
            Document::builder(format!("noise-post-{i}"), format!("noise post {i}"))
                .embedding(vec![0.0, 0.1, 0.0, 0.0])
                .metadata("node_type", "Message")
                .metadata("session_id", sid)
                .metadata("role", "user")
                .metadata("sequence", (11 + i).to_string())
                .metadata("edge_type", "NEXT")
                .metadata("created_at", "2026-01-01T00:00:00Z")
                .build(),
        );
    }

    // seed_collection: open and insert
    {
        let mut col = Collection::open(&path).unwrap();
        for doc in &docs {
            col.insert(doc.clone()).unwrap();
        }
        eprintln!("After seed: col.len()={}", col.len());
    }

    // Reopen
    let col = Collection::open(&path).unwrap();
    eprintln!("After reopen: col.len()={}", col.len());

    if let Some(store) = col.embedding_storage() {
        let emb_all = store.as_embeddings();
        eprintln!("emb_all.len()={}", emb_all.len());
        eprintln!("dim = {}", emb_all.len() / col.len());
    }

    // Search with filter (like search_similar does)
    let results = col.search_filtered(&[0.9, 0.0, 0.0, 0.0], 5, &|doc| {
        doc.metadata_val("session_id") == Some(sid)
            && doc.metadata_val("node_type") == Some("Message")
    });
    eprintln!("search_filtered returned {} results", results.len());
    if !results.is_empty() {
        eprintln!(
            "top result: id={}, score={}",
            results[0].document.id, results[0].score
        );
    }
}
