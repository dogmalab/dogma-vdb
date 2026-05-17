//! Quick benchmark: all index backends, speed & recall.
//!
//! Run: cargo run --release --example bench
//!
//! Generates random 128-dim vectors and measures query latency for
//! BruteForce, HNSW, Annoy, and SQ variants.

use dogma_vdb::distance::Metric;
use dogma_vdb::doc::Document;
use dogma_vdb::index::{AnnoyConfig, AnnoyIndex, BruteForceIndex, HnswConfig, HnswIndex, Index};
use std::time::Instant;

fn random_vec(dim: usize) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seed = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    let h = hasher.finish();
    (0..dim)
        .map(|i| ((h.wrapping_add(i as u64 * 6364136223846793005) >> 16) as f64 * 0.0001) as f32)
        .collect()
}

fn make_docs(n: usize, dim: usize) -> Vec<Document> {
    (0..n)
        .map(|i| {
            Document::builder(format!("d{i}"), format!("doc {i}"))
                .embedding(random_vec(dim))
                .build()
        })
        .collect()
}

fn bench<F: Fn(&[f32], usize)>(label: &str, f: F, queries: usize) {
    let query = random_vec(128);
    let start = Instant::now();
    for _ in 0..queries {
        f(&query, 10);
    }
    let elapsed = start.elapsed();
    let per_query = elapsed.as_secs_f64() / queries as f64 * 1_000_000.0;
    println!("  {label:<22}  {elapsed:.3?} total  {per_query:.0} us/query");
}

fn main() {
    let dim = 128;
    let k = 10;

    for &n in &[100, 500, 1_000, 5_000] {
        println!("\n=== {n} documents, {dim}-dim vectors ===");
        let docs = make_docs(n, dim);

        // --- BruteForce ---
        let mut bf = BruteForceIndex::new(Metric::Cosine);
        bf.insert(&docs);
        bench(
            "BruteForce",
            |q, k| {
                bf.search(q, k);
            },
            100,
        );

        // --- BruteForce + SQ ---
        let mut bf_sq = BruteForceIndex::new_with(Metric::Cosine, true);
        bf_sq.insert(&docs);
        bench(
            "BF+SQ",
            |q, k| {
                bf_sq.search(q, k);
            },
            100,
        );

        // --- HNSW ---
        let mut hnsw = HnswIndex::new(HnswConfig {
            m: 16,
            ef_construction: 100,
            ef_search: 50,
            metric: Metric::Cosine,
            flat_embeddings: false,
            sq: false,
        });
        let start = Instant::now();
        hnsw.insert(&docs);
        let idx_time = start.elapsed();
        bench(
            "HNSW (ef=50)",
            |q, k| {
                hnsw.search(q, k);
            },
            100,
        );

        // --- HNSW + SQ ---
        let mut hnsw_sq = HnswIndex::new(HnswConfig {
            m: 16,
            ef_construction: 100,
            ef_search: 50,
            metric: Metric::Cosine,
            flat_embeddings: false,
            sq: true,
        });
        hnsw_sq.insert(&docs);
        bench(
            "HNSW+SQ (ef=50)",
            |q, k| {
                hnsw_sq.search(q, k);
            },
            100,
        );

        // --- HNSW + Flat ---
        let mut hnsw_f = HnswIndex::new(HnswConfig {
            m: 16,
            ef_construction: 100,
            ef_search: 50,
            metric: Metric::Cosine,
            flat_embeddings: true,
            sq: false,
        });
        hnsw_f.insert(&docs);
        bench(
            "HNSW+Flat (ef=50)",
            |q, k| {
                hnsw_f.search(q, k);
            },
            100,
        );

        // --- Annoy ---
        let mut annoy = AnnoyIndex::new(AnnoyConfig {
            n_trees: 10,
            search_k: -1,
            metric: Metric::Cosine,
            leaf_size: 10,
        });
        let start_a = Instant::now();
        annoy.insert(&docs);
        let annoy_build = start_a.elapsed();
        bench(
            "Annoy (10 trees)",
            |q, k| {
                annoy.search(q, k);
            },
            100,
        );

        println!("  Build time: HNSW={idx_time:.3?}  Annoy={annoy_build:.3?}");

        // --- Recall (approximate) ---
        let bf_results = bf.search(&docs[0].embedding, k);
        let bf_set: std::collections::HashSet<&str> =
            bf_results.iter().map(|r| r.document.id.as_str()).collect();

        for (label, results) in [
            ("HNSW", hnsw.search(&docs[0].embedding, k)),
            ("HNSW+SQ", hnsw_sq.search(&docs[0].embedding, k)),
            ("HNSW+Flat", hnsw_f.search(&docs[0].embedding, k)),
            ("Annoy", annoy.search(&docs[0].embedding, k)),
            ("BF+SQ", bf_sq.search(&docs[0].embedding, k)),
        ] {
            let set: std::collections::HashSet<&str> =
                results.iter().map(|r| r.document.id.as_str()).collect();
            let overlap = bf_set.intersection(&set).count();
            let recall = overlap as f64 / k as f64 * 100.0;
            println!("  Recall {label:<12}  {recall:.0}%");
        }
    }
}
