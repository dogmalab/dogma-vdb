//! Integration tests for dogma-vdb.
//!
//! These tests use real temporary files and exercise the full
//! read‑write‑search cycle.

use dogma_vdb::prelude::*;

#[test]
fn test_open_empty_collection();

#[test]
fn test_insert_and_search();

#[test]
fn test_persistence_roundtrip();

#[test]
fn test_batch_insert();

#[test]
fn test_chunker_with_storage();

#[test]
fn test_metric_consistency();
