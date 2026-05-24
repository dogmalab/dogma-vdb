//! Benchmark solo 100K vectores + tree-sitter chunking.
//! Correr: cargo run --release --example bench100k --features chunker-syntax

use dogma_vdb::distance::Metric;
use dogma_vdb::doc::Document;
use dogma_vdb::index::{BruteForceIndex, HnswConfig, HnswIndex, Index, IvfPqConfig, IvfPqIndex};
use std::time::Instant;

fn random_vec(dim: usize) -> Vec<f32> {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
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
            Document::builder(format!("d{}", i), format!("doc {}", i))
                .embedding(random_vec(dim))
                .build()
        })
        .collect()
}

fn main() {
    let dim = 128;
    let k = 10;
    let n = 100_000;

    println!("\n=== {} documents, {}-dim vectors, k={} ===", n, dim, k);
    let docs = make_docs(n, dim);
    let query = &docs[0].embedding;
    let iters = 50; // fewer iterations at 100K

    // --- BruteForce ---
    let mut bf = BruteForceIndex::new(Metric::Cosine);
    let t0 = Instant::now();
    bf.insert(&docs);
    println!("Build BF: {:?}", t0.elapsed());

    let t0 = Instant::now();
    for _ in 0..iters {
        bf.search(query, k);
    }
    println!(
        "BF: {:?} total, {} us/query",
        t0.elapsed(),
        t0.elapsed().as_secs_f64() / iters as f64 * 1_000_000.0
    );

    // --- BruteForce + SQ ---
    let mut bf_sq = BruteForceIndex::new_with(Metric::Cosine, true, false);
    let t0 = Instant::now();
    bf_sq.insert(&docs);
    println!("Build BF+SQ: {:?}", t0.elapsed());

    let t0 = Instant::now();
    for _ in 0..iters {
        bf_sq.search(query, k);
    }
    println!(
        "BF+SQ: {:?} total, {} us/query",
        t0.elapsed(),
        t0.elapsed().as_secs_f64() / iters as f64 * 1_000_000.0
    );

    // --- BruteForce + SQ + Rescore ---
    let mut bf_sqr = BruteForceIndex::new_with(Metric::Cosine, true, true);
    let t0 = Instant::now();
    bf_sqr.insert(&docs);
    println!("Build BF+SQ+Rescore: {:?}", t0.elapsed());

    let t0 = Instant::now();
    for _ in 0..iters {
        bf_sqr.search(query, k);
    }
    println!(
        "BF+SQ+Rescore: {:?} total, {} us/query",
        t0.elapsed(),
        t0.elapsed().as_secs_f64() / iters as f64 * 1_000_000.0
    );

    // --- HNSW ef=50 ---
    let mut hnsw = HnswIndex::new(HnswConfig {
        m: 16,
        ef_construction: 100,
        ef_search: 50,
        metric: Metric::Cosine,
        flat_embeddings: false,
        sq: false,
        sq_rescore: false,
    });
    let t0 = Instant::now();
    hnsw.insert(&docs);
    let hnsw_build = t0.elapsed();
    println!("Build HNSW (ef=50): {:?}", hnsw_build);

    let t0 = Instant::now();
    for _ in 0..iters {
        hnsw.search(query, k);
    }
    println!(
        "HNSW (ef=50): {:?} total, {} us/query",
        t0.elapsed(),
        t0.elapsed().as_secs_f64() / iters as f64 * 1_000_000.0
    );

    // --- HNSW ef=200 ---
    let mut hnsw_hr = HnswIndex::new(HnswConfig {
        m: 16,
        ef_construction: 200,
        ef_search: 200,
        metric: Metric::Cosine,
        flat_embeddings: false,
        sq: false,
        sq_rescore: false,
    });
    let t0 = Instant::now();
    hnsw_hr.insert(&docs);
    println!("Build HNSW (ef=200): {:?}", t0.elapsed());

    let t0 = Instant::now();
    for _ in 0..iters {
        hnsw_hr.search(query, k);
    }
    println!(
        "HNSW (ef=200): {:?} total, {} us/query",
        t0.elapsed(),
        t0.elapsed().as_secs_f64() / iters as f64 * 1_000_000.0
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
    let t0 = Instant::now();
    hnsw_sqr.insert(&docs);
    println!("Build HNSW+SQ+Rescore (ef=50): {:?}", t0.elapsed());

    let t0 = Instant::now();
    for _ in 0..iters {
        hnsw_sqr.search(query, k);
    }
    println!(
        "HNSW+SQ+Rescore (ef=50): {:?} total, {} us/query",
        t0.elapsed(),
        t0.elapsed().as_secs_f64() / iters as f64 * 1_000_000.0
    );

    // --- IVF-PQ ---
    let n_list = 256;
    let mut ivf = IvfPqIndex::new(IvfPqConfig {
        n_list,
        m_subspaces: 8,
        n_probe: 8,
        metric: Metric::Cosine,
        ..Default::default()
    });
    let t0 = Instant::now();
    ivf.insert(&docs);
    let ivf_build = t0.elapsed();
    println!(
        "Build IVF-PQ (n_list={}, n_probe=8): {:?}",
        n_list, ivf_build
    );

    let t0 = Instant::now();
    for _ in 0..iters {
        ivf.search(query, k);
    }
    println!(
        "IVF-PQ: {:?} total, {} us/query",
        t0.elapsed(),
        t0.elapsed().as_secs_f64() / iters as f64 * 1_000_000.0
    );

    // --- Recall ---
    let exact = bf.search(query, k);
    println!("\n--- Recall vs BruteForce ---");
    for (label, results) in [
        ("HNSW ef=50", hnsw.search(query, k)),
        ("HNSW ef=200", hnsw_hr.search(query, k)),
        ("HNSW+SQ+Rescore ef=50", hnsw_sqr.search(query, k)),
        ("IVF-PQ", ivf.search(query, k)),
        ("BF+SQ", bf_sq.search(query, k)),
        ("BF+SQ+Rescore", bf_sqr.search(query, k)),
    ] {
        let exact_set: std::collections::HashSet<&str> =
            exact.iter().map(|r| r.document.id.as_str()).collect();
        let overlap = results
            .iter()
            .filter(|r| exact_set.contains(r.document.id.as_str()))
            .count();
        let recall = overlap as f64 / k as f64 * 100.0;
        println!("  {}: {:.0}%", label, recall);
    }

    println!(
        "\nBuild times: BF={:.3?}  HNSW={:.3?}  IVF-PQ={:.3?}",
        t0.elapsed(),
        hnsw_build,
        ivf_build
    );
}
