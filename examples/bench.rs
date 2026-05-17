//! Quick benchmark: BruteForce vs HNSW speed & recall.
//!
//! Run: cargo run --release --example bench
//!
//! Generates random 128-dim vectors and measures query latency.

use dogma_vdb::distance::Metric;
use dogma_vdb::doc::Document;
use dogma_vdb::index::{BruteForceIndex, HnswConfig, HnswIndex, Index};
use std::time::Instant;

fn random_vec(dim: usize) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    // Deterministic random
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
    let per_query = elapsed.as_secs_f64() / queries as f64 * 1_000_000.0; // microseconds
    println!("  {label:<20}  {elapsed:.3?} total  {per_query:.0} us/query");
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

        // --- HNSW ---
        let mut hnsw = HnswIndex::new(HnswConfig {
            m: 16,
            ef_construction: 100,
            ef_search: 50,
            metric: Metric::Cosine,
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

        // HNSW with higher ef
        let mut hnsw2 = HnswIndex::new(HnswConfig {
            m: 16,
            ef_construction: 200,
            ef_search: 200,
            metric: Metric::Cosine,
        });
        hnsw2.insert(&docs);
        bench(
            "HNSW (ef=200)",
            |q, k| {
                hnsw2.search(q, k);
            },
            100,
        );

        println!("  Build time: HNSW={idx_time:.3?}");

        // --- Recall (approximate) ---
        let bf_results = bf.search(&docs[0].embedding, k);
        let bf_result: std::collections::HashSet<&str> =
            bf_results.iter().map(|r| r.document.id.as_str()).collect();
        let hnsw_results = hnsw.search(&docs[0].embedding, k);
        let hnsw_result: std::collections::HashSet<&str> = hnsw_results
            .iter()
            .map(|r| r.document.id.as_str())
            .collect();
        let overlap = bf_result.intersection(&hnsw_result).count();
        let recall = overlap as f64 / k as f64 * 100.0;
        println!("  Recall (HNSW ef=50):  {recall:.0}%");
    }
}
