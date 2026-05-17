//! Quick benchmark: all index backends, speed & recall.
//!
//! Run: cargo run --release --example bench
//!
//! Generates random 128-dim vectors and measures query latency for
//! BruteForce, HNSW, IVF-PQ, and SQ variants.

use dogma_vdb::distance::Metric;
use dogma_vdb::doc::Document;
use dogma_vdb::index::{
    BruteForceIndex, HnswConfig, HnswIndex, Index, IvfPqConfig, IvfPqIndex,
};
use std::time::Instant;

fn random_vec(dim: usize) -> Vec<f32> {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    // Deterministic random via SplitMix64 (same algorithm as HNSW).
    // Produces well-distributed values in [-3, 3].
    let seed = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    (0..dim)
        .map(|i| {
            let id = seed.wrapping_mul(dim as u64).wrapping_add(i as u64);
            let mut z = id.wrapping_mul(0x9E3779B97F4A7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^= z >> 31;
            (z >> 10) as f64 * 6.0 / 9007199254740992.0 - 3.0
        })
        .map(|x| x as f32)
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
        let mut bf_sq = BruteForceIndex::new_with(Metric::Cosine, true, false);
        bf_sq.insert(&docs);
        bench(
            "BF+SQ",
            |q, k| {
                bf_sq.search(q, k);
            },
            100,
        );

        // --- BruteForce + SQ + Rescore ---
        let mut bf_sqr = BruteForceIndex::new_with(Metric::Cosine, true, true);
        bf_sqr.insert(&docs);
        bench(
            "BF+SQ+Rescore",
            |q, k| {
                bf_sqr.search(q, k);
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
            sq_rescore: false,
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
            sq_rescore: false,
        });
        hnsw_sq.insert(&docs);
        bench(
            "HNSW+SQ (ef=50)",
            |q, k| {
                hnsw_sq.search(q, k);
            },
            100,
        );

        // --- HNSW + SQ + Rescore ---
        let mut hnsw_sqr = HnswIndex::new(HnswConfig {
            m: 16,
            ef_construction: 100,
            ef_search: 50,
            metric: Metric::Cosine,
            flat_embeddings: false,
            sq: true,
            sq_rescore: true,
        });
        hnsw_sqr.insert(&docs);
        bench(
            "HNSW+SQ+Rescore",
            |q, k| {
                hnsw_sqr.search(q, k);
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
            sq_rescore: false,
        });
        hnsw_f.insert(&docs);
        bench(
            "HNSW+Flat (ef=50)",
            |q, k| {
                hnsw_f.search(q, k);
            },
            100,
        );

        // --- IVF-PQ ---
        let mut ivf_pq = IvfPqIndex::new(IvfPqConfig {
            n_clusters: 256.min(n),
            n_subvectors: 8,
            n_probe: 8,
            metric: Metric::Cosine,
        });
        let start_ivf = Instant::now();
        ivf_pq.insert(&docs);
        let ivf_build = start_ivf.elapsed();
        bench(
            "IVF-PQ",
            |q, k| {
                ivf_pq.search(q, k);
            },
            100,
        );

        println!("  Build time: HNSW={idx_time:.3?}  IVF-PQ={ivf_build:.3?}");

        // --- Recall (approximate) ---
        let bf_results = bf.search(&docs[0].embedding, k);
        let bf_set: std::collections::HashSet<&str> =
            bf_results.iter().map(|r| r.document.id.as_str()).collect();

        for (label, results) in [
            ("HNSW", hnsw.search(&docs[0].embedding, k)),
            ("HNSW+SQ", hnsw_sq.search(&docs[0].embedding, k)),
            ("HNSW+SQ+Rescore", hnsw_sqr.search(&docs[0].embedding, k)),
            ("HNSW+Flat", hnsw_f.search(&docs[0].embedding, k)),
            ("IVF-PQ", ivf_pq.search(&docs[0].embedding, k)),
            ("BF+SQ", bf_sq.search(&docs[0].embedding, k)),
            ("BF+SQ+Rescore", bf_sqr.search(&docs[0].embedding, k)),
        ] {
            let set: std::collections::HashSet<&str> =
                results.iter().map(|r| r.document.id.as_str()).collect();
            let overlap = bf_set.intersection(&set).count();
            let recall = overlap as f64 / k as f64 * 100.0;
            println!("  Recall {label:<12}  {recall:.0}%");
        }
    }
}
