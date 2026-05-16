//! Integration tests for dogma-vdb.
//!
//! These tests use real temporary files and exercise the full
//! read‑write‑search cycle.

use dogma_vdb::prelude::*;

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
