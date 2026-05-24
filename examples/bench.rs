//! Full benchmark: all index backends at 100K scale + tree‑sitter chunking.
//!
//! Run:  cargo run --release --example bench
//!       cargo run --release --example bench --features chunker-syntax
//!
//! Generates random 128‑dim vectors and measures:
//!   - Insert/build time  (ingest speed)
//!   - Query latency      (us/query)
//!   - Recall vs BruteForce
//!   - Tree‑sitter chunking throughput (with feature chunker-syntax)

use dogma_vdb::distance::Metric;
use dogma_vdb::doc::Document;
use dogma_vdb::index::{
    BruteForceIndex, HnswConfig, HnswIndex, Index, IvfPqConfig, IvfPqIndex, ScoredDocument,
};
#[cfg(feature = "chunker-syntax")]
use dogma_vdb::smart_chunker::{FileType, SmartChunker};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Deterministic random vectors (SplitMix64, same as HNSW reference)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn bench_queries<F: Fn(&[f32], usize)>(label: &str, f: F, query: &[f32], k: usize, iters: usize) {
    let start = Instant::now();
    for _ in 0..iters {
        f(query, k);
    }
    let elapsed = start.elapsed();
    let per_query = elapsed.as_secs_f64() / iters as f64 * 1_000_000.0;
    println!(
        "  {:<28}  {:.3?} total  {:>8.0} us/query",
        label, elapsed, per_query
    );
}

fn compute_recall(approx: &[ScoredDocument], exact: &[ScoredDocument]) -> f64 {
    let exact_set: std::collections::HashSet<&str> =
        exact.iter().map(|r| r.document.id.as_str()).collect();
    let overlap = approx
        .iter()
        .filter(|r| exact_set.contains(r.document.id.as_str()))
        .count();
    overlap as f64 / exact.len() as f64 * 100.0
}

// ---------------------------------------------------------------------------
// Index benchmark for a single dataset size
// ---------------------------------------------------------------------------

fn bench_index(n: usize, dim: usize, k: usize) {
    println!();
    println!("═══════════════════════════════════════════════════");
    println!("  {} documents, {}-dim vectors, k={}", n, dim, k);
    println!("═══════════════════════════════════════════════════");

    let docs = make_docs(n, dim);
    let query = &docs[0].embedding;

    // -- BruteForce (ground truth) --
    let mut bf = BruteForceIndex::new(Metric::Cosine);
    let t0 = Instant::now();
    bf.insert(&docs);
    println!("  Build BF                              {:?}", t0.elapsed());
    bench_queries(
        "BF",
        |q, k| {
            bf.search(q, k);
        },
        query,
        k,
        100.min(n),
    );

    // -- BruteForce + SQ --
    let mut bf_sq = BruteForceIndex::new_with(Metric::Cosine, true, false);
    let t0 = Instant::now();
    bf_sq.insert(&docs);
    println!("  Build BF+SQ                           {:?}", t0.elapsed());
    bench_queries(
        "BF+SQ",
        |q, k| {
            bf_sq.search(q, k);
        },
        query,
        k,
        100.min(n),
    );

    // -- BruteForce + SQ + Rescore --
    let mut bf_sqr = BruteForceIndex::new_with(Metric::Cosine, true, true);
    let t0 = Instant::now();
    bf_sqr.insert(&docs);
    println!("  Build BF+SQ+Rescore                   {:?}", t0.elapsed());
    bench_queries(
        "BF+SQ+Rescore",
        |q, k| {
            bf_sqr.search(q, k);
        },
        query,
        k,
        100.min(n),
    );

    // -- HNSW (ef=50) --
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
    println!("  Build HNSW (ef_c=100, ef=50)          {:?}", hnsw_build);
    bench_queries(
        "HNSW (ef=50)",
        |q, k| {
            hnsw.search(q, k);
        },
        query,
        k,
        100.min(n),
    );

    // -- HNSW + Flat --
    let mut hnsw_f = HnswIndex::new(HnswConfig {
        m: 16,
        ef_construction: 100,
        ef_search: 50,
        metric: Metric::Cosine,
        flat_embeddings: true,
        sq: false,
        sq_rescore: false,
    });
    let t0 = Instant::now();
    hnsw_f.insert(&docs);
    println!("  Build HNSW+Flat (ef=50)               {:?}", t0.elapsed());
    bench_queries(
        "HNSW+Flat (ef=50)",
        |q, k| {
            hnsw_f.search(q, k);
        },
        query,
        k,
        100.min(n),
    );

    // -- HNSW high recall (ef=200) --
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
    println!("  Build HNSW (ef_c=200, ef=200)         {:?}", t0.elapsed());
    bench_queries(
        "HNSW (ef=200)",
        |q, k| {
            hnsw_hr.search(q, k);
        },
        query,
        k,
        100.min(n),
    );

    // -- HNSW + SQ + Rescore --
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
    println!("  Build HNSW+SQ+Rescore (ef=50)         {:?}", t0.elapsed());
    bench_queries(
        "HNSW+SQ+Rescore (ef=50)",
        |q, k| {
            hnsw_sqr.search(q, k);
        },
        query,
        k,
        100.min(n),
    );

    // -- IVF-PQ --
    let n_list = if n < 256 { n } else { 256.min(n / 4) };
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
        "  Build IVF-PQ (n_list={}, n_probe=8)    {:?}",
        n_list, ivf_build
    );
    bench_queries(
        "IVF-PQ",
        |q, k| {
            ivf.search(q, k);
        },
        query,
        k,
        100.min(n),
    );

    // -- IVF-PQ tuned (more probes) --
    let n_probe = (n_list / 8).max(4);
    let mut ivf_t = IvfPqIndex::new(IvfPqConfig {
        n_list,
        m_subspaces: 8,
        n_probe,
        metric: Metric::Cosine,
        ..Default::default()
    });
    let t0 = Instant::now();
    ivf_t.insert(&docs);
    println!(
        "  Build IVF-PQ (n_list={}, n_probe={})   {:?}",
        n_list,
        n_probe,
        t0.elapsed()
    );
    bench_queries(
        "IVF-PQ (tuned)",
        |q, k| {
            ivf_t.search(q, k);
        },
        query,
        k,
        100.min(n),
    );

    // -- Recall --
    let exact = bf.search(query, k);

    let recall_cases: Vec<(&str, Vec<ScoredDocument>)> = vec![
        ("HNSW ef=50", hnsw.search(query, k)),
        ("HNSW+Flat ef=50", hnsw_f.search(query, k)),
        ("HNSW ef=200", hnsw_hr.search(query, k)),
        ("HNSW+SQ+Rescore ef=50", hnsw_sqr.search(query, k)),
        ("IVF-PQ", ivf.search(query, k)),
        ("IVF-PQ tuned", ivf_t.search(query, k)),
        ("BF+SQ", bf_sq.search(query, k)),
        ("BF+SQ+Rescore", bf_sqr.search(query, k)),
    ];
    println!();
    for (label, results) in &recall_cases {
        let recall = compute_recall(results, &exact);
        println!("  Recall {:<26}  {:>5.0}%", label, recall);
    }

    println!(
        "  Build times:  BF={:.3?}  HNSW={:.3?}  IVF-PQ={:.3?}",
        t0.elapsed(),
        hnsw_build,
        ivf_build
    );
}

// ---------------------------------------------------------------------------
// Tree‑sitter chunking benchmark
// ---------------------------------------------------------------------------

#[cfg(feature = "chunker-syntax")]
fn bench_treesitter_chunking() {
    println!();
    println!("═══════════════════════════════════════════════════");
    println!("  Tree‑sitter chunking throughput");
    println!("═══════════════════════════════════════════════════");

    let src_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut all_code = String::new();
    let mut file_count = 0u64;
    let mut total_bytes = 0u64;

    fn collect_rs(dir: &std::path::Path, code: &mut String, files: &mut u64, bytes: &mut u64) {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    collect_rs(&path, code, files, bytes);
                } else if path.extension().map_or(false, |e| e == "rs") {
                    if let Ok(content) = std::fs::read_to_string(&path) {
                        *files += 1;
                        *bytes += content.len() as u64;
                        code.push_str(&content);
                        code.push('\n');
                    }
                }
            }
        }
    }

    collect_rs(&src_dir, &mut all_code, &mut file_count, &mut total_bytes);
    println!(
        "  Source: {} Rust files, {:.1} KB total",
        file_count,
        total_bytes as f64 / 1024.0
    );

    let chunker = SmartChunker::default();

    // Warmup
    let _ = chunker.chunk_text(&all_code, FileType::Rust);

    // Measure chunking
    const ITERS: usize = 20;
    let start = Instant::now();
    let mut total_chunks = 0usize;
    for _ in 0..ITERS {
        total_chunks += chunker.chunk_text(&all_code, FileType::Rust).len();
    }
    let elapsed = start.elapsed();
    let per_iter = elapsed / ITERS as u32;
    let mb_per_sec = (total_bytes as f64 * ITERS as f64) / elapsed.as_secs_f64() / 1024.0 / 1024.0;
    let chunks_per_sec = total_chunks as f64 / elapsed.as_secs_f64();

    println!("  Chunks produced: {}/iter (avg)", total_chunks / ITERS);
    println!("  Time per iteration: {:.3?}", per_iter);
    println!(
        "  Throughput: {:.1} MB/sec, {:.0} chunks/sec",
        mb_per_sec, chunks_per_sec
    );
}

#[cfg(not(feature = "chunker-syntax"))]
fn bench_treesitter_chunking() {
    println!();
    println!("  (tree‑sitter chunking: feature `chunker-syntax` not enabled)");
    println!("  Run with: cargo run --release --example bench --features chunker-syntax");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let dim = 128;
    let k = 10;

    // Sizes to test (add 100_000 to run the large scale)
    for &n in &[100, 1_000, 10_000, 100_000] {
        bench_index(n, dim, k);
    }

    // Tree‑sitter chunking benchmark
    bench_treesitter_chunking();
}
