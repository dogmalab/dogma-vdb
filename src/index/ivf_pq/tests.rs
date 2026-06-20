use super::*;
use crate::index::BruteForceIndex;

fn make_doc(id: &str, embedding: Vec<f32>) -> Document {
    Document::builder(id, id).embedding(embedding).build()
}

fn small_config() -> IvfPqConfig {
    IvfPqConfig {
        n_list: 4,
        m_subspaces: 8,
        n_probe: 2,
        metric: Metric::Cosine,
        ..Default::default()
    }
}

// -- Existing tests (updated for new field names) --

#[test]
fn test_empty_index() {
    let idx = IvfPqIndex::new(small_config());
    assert!(idx
        .search(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 5)
        .is_empty());
    assert!(idx.is_empty());
}

#[test]
fn test_single_insert() {
    let mut idx = IvfPqIndex::new(small_config());
    idx.insert(&[make_doc("a", vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])]);
    assert_eq!(idx.len(), 1);
    let results = idx.search(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 5);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].document.id, "a");
}

#[test]
fn test_insert_batch() {
    let mut idx = IvfPqIndex::new(small_config());
    let docs: Vec<Document> = (0..10)
        .map(|i| {
            make_doc(
                &format!("d{}", i),
                vec![i as f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            )
        })
        .collect();
    idx.insert(&docs);
    assert_eq!(idx.len(), 10);
}

#[test]
fn test_search_returns_closest() {
    let mut idx = IvfPqIndex::new(small_config());
    idx.insert(&[
        make_doc("a", vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        make_doc("b", vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ]);
    let results = idx.search(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 2);
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].document.id, "a");
}

#[test]
fn test_documents_without_embedding_skipped() {
    let mut idx = IvfPqIndex::new(small_config());
    idx.insert(&[
        make_doc("a", vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        Document::new("b", "no embedding"),
    ]);
    assert_eq!(idx.len(), 1);
}

#[test]
fn test_delete() {
    let mut idx = IvfPqIndex::new(small_config());
    idx.insert(&[
        make_doc("a", vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        make_doc("b", vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ]);
    assert_eq!(idx.len(), 2);
    let deleted = Index::delete(&mut idx, &["a"]);
    assert_eq!(deleted, 1);
    assert_eq!(idx.len(), 1);
    // Tombstoned doc should not appear in search results
    let results = idx.search(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 10);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].document.id, "b");
}

#[test]
fn test_tombstone_does_not_rebuild() {
    let mut idx = IvfPqIndex::new(IvfPqConfig {
        n_list: 4,
        m_subspaces: 8,
        n_probe: 2,
        metric: Metric::Cosine,
        rebuild_threshold: 0.50, // only rebuild at 50%
        ..Default::default()
    });
    let docs: Vec<Document> = (0..20)
        .map(|i| {
            make_doc(
                &format!("d{}", i),
                vec![i as f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            )
        })
        .collect();
    idx.insert(&docs);
    assert_eq!(idx.len(), 20);

    // Delete 1 doc (5%) — should NOT trigger rebuild (< 50%)
    let deleted = Index::delete(&mut idx, &["d0"]);
    assert_eq!(deleted, 1);
    assert_eq!(idx.len(), 19);
    assert!(idx.tombstone_ratio() < 0.50);

    // Search should still work
    let results = idx.search(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 5);
    assert!(!results.is_empty());
    // d0 should not appear
    assert!(results.iter().all(|r| r.document.id != "d0"));
}

#[test]
fn test_tombstone_triggers_rebuild() {
    let mut idx = IvfPqIndex::new(IvfPqConfig {
        n_list: 4,
        m_subspaces: 8,
        n_probe: 2,
        metric: Metric::Cosine,
        rebuild_threshold: 0.20, // rebuild at 20%
        ..Default::default()
    });
    let docs: Vec<Document> = (0..10)
        .map(|i| {
            make_doc(
                &format!("d{}", i),
                vec![i as f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            )
        })
        .collect();
    idx.insert(&docs);

    // Delete 3 docs (30%) — should trigger rebuild at 20% threshold
    let deleted = Index::delete(&mut idx, &["d0", "d1", "d2"]);
    assert_eq!(deleted, 3);
    // After rebuild, tombstones should be cleared
    assert_eq!(idx.tombstone_count, 0);
    assert_eq!(idx.len(), 7);
}

#[test]
fn test_recall_against_bf() {
    let mut bf = BruteForceIndex::new(Metric::Cosine);
    let mut ivf = IvfPqIndex::new(IvfPqConfig {
        n_list: 5,
        m_subspaces: 8,
        n_probe: 3,
        metric: Metric::Cosine,
        ..Default::default()
    });

    let mut docs = Vec::with_capacity(50);
    for i in 0..50 {
        let angle = i as f64 * 0.12566;
        docs.push(make_doc(
            &format!("d{}", i),
            vec![
                angle.cos() as f32,
                angle.sin() as f32,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        ));
    }
    bf.insert(&docs);
    ivf.insert(&docs);

    let query = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let bf_res = bf.search(&query, 5);
    let ivf_res = ivf.search(&query, 5);

    let bf_ids: std::collections::HashSet<&str> =
        bf_res.iter().map(|r| r.document.id.as_str()).collect();
    let hits = ivf_res
        .iter()
        .filter(|r| bf_ids.contains(r.document.id.as_str()))
        .count();
    assert!(
        hits >= 2,
        "IVF-PQ recall too low: {}/5 (expected >= 2)",
        hits
    );
}

#[test]
fn test_search_results_sorted() {
    let mut idx = IvfPqIndex::new(small_config());
    let docs: Vec<Document> = (0..20)
        .map(|i| {
            let angle = i as f64 * 0.314159;
            make_doc(
                &format!("d{}", i),
                vec![
                    angle.cos() as f32,
                    angle.sin() as f32,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            )
        })
        .collect();
    idx.insert(&docs);

    let results = idx.search(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 10);
    assert!(!results.is_empty(), "should return at least some results");
    for i in 0..results.len().saturating_sub(1) {
        assert!(
            results[i].score >= results[i + 1].score,
            "results should be sorted by score descending"
        );
    }
}

// -- New tests for tuning and validation --

#[test]
fn test_ivf_pq_invalid_subspaces() {
    // m_subspaces must be a multiple of 8
    let cfg = IvfPqConfig {
        m_subspaces: 13,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());

    let cfg = IvfPqConfig {
        m_subspaces: 25,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());

    // Zero is also invalid
    let cfg = IvfPqConfig {
        m_subspaces: 0,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());

    // Valid multiples
    for valid in [8, 16, 32, 64] {
        let cfg = IvfPqConfig {
            m_subspaces: valid,
            ..Default::default()
        };
        assert!(
            cfg.validate().is_ok(),
            "m_subspaces={} should be valid",
            valid
        );
    }
}

#[test]
fn test_ivf_pq_tuning_impact() {
    // n_probe=1 should probe fewer clusters → different results than n_probe=10
    // (and likely fewer candidates at the extremes)
    let cfg_low = IvfPqConfig {
        n_list: 8,
        m_subspaces: 8,
        n_probe: 1,
        metric: Metric::Cosine,
        ..Default::default()
    };
    let cfg_high = IvfPqConfig {
        n_list: 8,
        m_subspaces: 8,
        n_probe: 10,
        metric: Metric::Cosine,
        ..Default::default()
    };

    let mut idx_low = IvfPqIndex::new(cfg_low);
    let mut idx_high = IvfPqIndex::new(cfg_high);

    let docs: Vec<Document> = (0..80)
        .map(|i| {
            let angle = i as f64 * 0.0785398;
            make_doc(
                &format!("d{}", i),
                vec![
                    angle.cos() as f32,
                    angle.sin() as f32,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            )
        })
        .collect();
    idx_low.insert(&docs);
    idx_high.insert(&docs);

    let query = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    let results_low = idx_low.search(&query, 5);
    let results_high = idx_high.search(&query, 5);

    // With more probe, we should inspect more candidates and thus
    // likely get a different (better) top result
    assert!(
        !results_low.is_empty(),
        "low probe should still return results"
    );
    assert!(!results_high.is_empty(), "high probe should return results");

    // The score of the top result with high probe should be at least
    // as high (better) as the top result with low probe
    assert!(
        results_high[0].score >= results_low[0].score - 1e-5,
        "higher n_probe should yield >= top score (low={}, high={})",
        results_low[0].score,
        results_high[0].score,
    );

    // Verify that n_probe is stored correctly
    let config_high = idx_high.config();
    assert_eq!(config_high.n_probe, 10);
    assert_eq!(config_high.effective_probe(), 10);
}

#[test]
fn test_auto_tuning_with_rerank_flag() {
    // When rerank_enabled=true, effective_probe should be halved
    let cfg = IvfPqConfig {
        n_probe: 10,
        rerank_enabled: true,
        ..Default::default()
    };
    // effective_probe = (10 / 2).max(2) = 5
    assert_eq!(cfg.effective_probe(), 5);

    // With low n_probe, minimum should be 2
    let cfg2 = IvfPqConfig {
        n_probe: 3,
        rerank_enabled: true,
        ..Default::default()
    };
    // effective_probe = (3 / 2).max(2) = 2
    assert_eq!(cfg2.effective_probe(), 2);

    // Without rerank, effective == nominal
    let cfg3 = IvfPqConfig {
        n_probe: 7,
        rerank_enabled: false,
        ..Default::default()
    };
    assert_eq!(cfg3.effective_probe(), 7);

    // Integration test: build an index with rerank_enabled and verify
    // the search still produces valid (sorted) results
    let cfg_idx = IvfPqConfig {
        n_list: 4,
        m_subspaces: 8,
        n_probe: 4,
        metric: Metric::Cosine,
        rerank_enabled: true,
        ..IvfPqConfig::default()
    };
    let mut idx = IvfPqIndex::new(cfg_idx);
    let docs: Vec<Document> = (0..20)
        .map(|i| {
            let angle = i as f64 * 0.314159;
            make_doc(
                &format!("d{}", i),
                vec![
                    angle.cos() as f32,
                    angle.sin() as f32,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            )
        })
        .collect();
    idx.insert(&docs);

    let results = idx.search(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 5);
    assert!(!results.is_empty());
    // Results should still be sorted correctly
    for i in 0..results.len().saturating_sub(1) {
        assert!(
            results[i].score >= results[i + 1].score,
            "rerank mode results should be sorted"
        );
    }
}
