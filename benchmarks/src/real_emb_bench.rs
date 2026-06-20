//! Real‑world IVF‑PQ recall benchmark with **FastEmbed ONNX embeddings**.
//!
//! Loads source files from the hermes‑agent repository, generates real
//! 384‑dim dense embeddings via `all-MiniLM-L6-v2` (ONNX), then builds
//! BruteForce and IVF‑PQ indices and reports Recall / Latency.
//!
//! First run downloads the model (~90 MB).  Subsequent runs use cache.
//!
//! Usage:
//!   cargo run --release -p dogma-vdb-benchmarks --bin real-emb-bench
//!   cargo run --release -p dogma-vdb-benchmarks --bin real-emb-bench -- --seed 42 --output json

use dogma_vdb::distance::Metric;
use dogma_vdb::doc::Document;
use dogma_vdb::index::{BruteForceIndex, Index, IvfPqConfig, IvfPqIndex, ScoredDocument};
use dogma_vdb_embed::Embedder;
use dogma_vdb_embed_fastembed::FastEmbedder;
use serde::Serialize;
use std::path::Path;
use std::time::Instant;

// ============================================================================
// Config
// ============================================================================

const DIM: usize = 384;
const N_LIST: usize = 256;
const N_PROBE: usize = 64;
const M_SUB: usize = 32; // 384/32 = 12 dims/sub-vector
const QUERIES: usize = 100; // number of held-out queries
const DOC_SIZE_LIMIT: usize = 200_000; // skip files larger than this (bytes)
const EMBED_BATCH: usize = 32; // embed N texts at a time to limit peak RAM

const DEFAULT_SEED: u64 = 42;

// ============================================================================
// Argument parsing
// ============================================================================

struct Args {
    seed: u64,
    output_json: bool,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    let mut seed = DEFAULT_SEED;
    let mut output_json = false;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--seed" => {
                i += 1;
                seed = args
                    .get(i)
                    .and_then(|s| s.parse().ok())
                    .expect("--seed requires a u64 value");
            }
            "--output" => {
                i += 1;
                if args.get(i).is_some_and(|v| v == "json") {
                    output_json = true;
                }
            }
            _ => {}
        }
        i += 1;
    }
    Args { seed, output_json }
}

// ============================================================================
// File walker
// ============================================================================

fn collect_source_texts(root: &Path) -> Vec<(String, String)> {
    let mut docs = Vec::new();
    let extensions = [
        "rs", "md", "py", "toml", "json", "yaml", "sh", "txt", "js", "ts",
    ];

    let mut entries: Vec<_> = Vec::new();
    collect_entries(root, root, &extensions, &mut entries);
    entries.sort(); // deterministic order

    for path in &entries {
        match std::fs::read_to_string(path) {
            Ok(content) if !content.is_empty() && content.len() <= DOC_SIZE_LIMIT => {
                let rel = path.strip_prefix(root).unwrap_or(path).to_string_lossy();
                let id = rel.replace(['/', '.'], "-");
                docs.push((id, content));
            }
            _ => {}
        }
    }
    docs
}

#[allow(clippy::only_used_in_recursion)]
fn collect_entries(base: &Path, dir: &Path, exts: &[&str], out: &mut Vec<std::path::PathBuf>) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let name = path.file_name().unwrap_or_default();
                if !is_skip_dir(name) {
                    collect_entries(base, &path, exts, out);
                }
            } else if path.is_file() {
                if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                    if exts.contains(&ext) {
                        out.push(path);
                    }
                }
            }
        }
    }
}

fn is_skip_dir(name: &std::ffi::OsStr) -> bool {
    name == "target"
        || name == ".git"
        || name == "node_modules"
        || name == ".venv"
        || name == "__pycache__"
        || name == ".bench-env"
}

// ============================================================================
// Recall helper
// ============================================================================

fn recall_at_k(actual: &[ScoredDocument], expected: &[ScoredDocument], k: usize) -> f64 {
    let expected_ids: std::collections::HashSet<&str> = expected
        .iter()
        .take(k)
        .map(|r| r.document.id.as_str())
        .collect();
    if expected_ids.is_empty() {
        return 1.0;
    }
    let hits = actual
        .iter()
        .take(k)
        .filter(|r| expected_ids.contains(r.document.id.as_str()))
        .count();
    hits as f64 / k.min(expected_ids.len()) as f64
}

// ============================================================================
// Main
// ============================================================================

fn seeded_shuffle<T>(data: &mut [T], seed: u64) {
    // Fisher-Yates shuffle with SplitMix64 RNG
    let mut next = seed;
    let mut i = data.len();
    while i > 1 {
        i -= 1;
        next = next.wrapping_mul(0x9E3779B97F4A7C15);
        let j = (next as usize) % (i + 1);
        data.swap(i, j);
    }
}

#[derive(Serialize)]
struct BenchOutput {
    seed: u64,
    backend: String,
    config: serde_json::Value,
    num_queries: usize,
    dimension: usize,
    recall_at_1_pct: f64,
    recall_at_10_pct: f64,
    latency_mean_us: f64,
    latency_p50_us: f64,
    latency_p95_us: f64,
    latency_p99_us: f64,
    qps: f64,
    index_docs: usize,
    build_time_s: f64,
    embed_time_s: f64,
}

fn main() {
    let args = parse_args();
    println!("=== IVF-PQ REAL-EMBEDDING BENCHMARK ===");
    println!("Model: all-MiniLM-L6-v2 (384-dim ONNX via fastembed)");
    println!("Seed: {}", args.seed);
    println!();

    // ---- 1. Load text sources ----
    let hermes_path = Path::new("/home/arggil/Documents/DEV-WORKSPACE/hermes-agent");
    if !hermes_path.exists() {
        eprintln!("ERROR: hermes-agent not found at {:?}", hermes_path);
        std::process::exit(1);
    }

    println!("[1/5] Scanning hermes-agent source files...");
    let t0 = Instant::now();
    let sources = collect_source_texts(hermes_path);
    let elapsed = t0.elapsed();
    println!(
        "      {} files found in {:.2}s",
        sources.len(),
        elapsed.as_secs_f64()
    );

    // ---- 2. Initialise FastEmbedder (downloads model on first run) ----
    println!("[2/5] Initialising FastEmbedder (model download if needed)...");
    let t0 = Instant::now();
    let embedder = match FastEmbedder::new() {
        Ok(e) => e,
        Err(e) => {
            eprintln!("ERROR: FastEmbedder init failed: {e}");
            eprintln!("      This requires network access to download the ONNX model.");
            eprintln!(
                "      Try: cargo run --release -p dogma-vdb-benchmarks --bin real-emb-bench"
            );
            std::process::exit(1);
        }
    };
    println!("      Model ready in {:.2}s", t0.elapsed().as_secs_f64());
    assert_eq!(embedder.dimension(), DIM, "embedder dimension mismatch");

    // ---- 3. Generate embeddings in small batches ----
    println!(
        "[3/5] Generating {} real embeddings (batch size {})...",
        sources.len(),
        EMBED_BATCH
    );
    let t0 = Instant::now();
    // We'll embed in batches of EMBED_BATCH to keep peak RAM low
    let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(sources.len());
    let texts: Vec<&str> = sources.iter().map(|(_, t)| t.as_str()).collect();
    for chunk in texts.chunks(EMBED_BATCH) {
        match embedder.embed_batch(chunk) {
            Ok(mut batch) => embeddings.append(&mut batch),
            Err(e) => {
                eprintln!("ERROR: embedding batch failed: {e}");
                std::process::exit(1);
            }
        }
        if embeddings.len().is_multiple_of(EMBED_BATCH * 10) {
            print!(".");
            use std::io::Write;
            std::io::stdout().flush().ok();
        }
    }
    println!(); // newline after progress dots
    let embed_time = t0.elapsed();
    assert_eq!(embeddings.len(), sources.len());

    let total_chars: usize = sources.iter().map(|(_, t)| t.len()).sum();
    println!(
        "      {} embeddings in {:.2}s ({:.0} chars/s, {:.0} docs/s)",
        embeddings.len(),
        embed_time.as_secs_f64(),
        total_chars as f64 / embed_time.as_secs_f64().max(0.001),
        embeddings.len() as f64 / embed_time.as_secs_f64().max(0.001),
    );

    // Build Documents
    let mut docs: Vec<Document> = sources
        .into_iter()
        .zip(embeddings)
        .map(|((id, text), emb)| Document::builder(id, text).embedding(emb).build())
        .collect();

    // ---- Shuffle with seed, then split ----
    seeded_shuffle(&mut docs, args.seed);
    let n = docs.len();
    let split = n.saturating_sub(QUERIES);
    if split < 10 {
        eprintln!("ERROR: too few documents ({})", n);
        std::process::exit(1);
    }
    let (index_docs, query_docs) = docs.split_at(split);
    println!(
        "      Split: {} index / {} query docs (seed={})",
        index_docs.len(),
        query_docs.len(),
        args.seed
    );
    println!();

    // ---- 4. Build BruteForce (ground truth) ----
    println!("[4/5] Building BruteForce (ground truth)...");
    let t0 = Instant::now();
    let mut bf = BruteForceIndex::new(Metric::Cosine);
    bf.insert(index_docs);
    let bf_build = t0.elapsed();
    println!(
        "      BF built in {:.3}s ({} docs)",
        bf_build.as_secs_f64(),
        bf.len()
    );

    // Warmup + ground-truth queries
    let q_embeddings: Vec<Vec<f32>> = query_docs.iter().map(|d| d.embedding.clone()).collect();
    let _query_texts: Vec<&str> = query_docs.iter().map(|d| d.text.as_str()).collect();

    let t0 = Instant::now();
    let mut ground_truth: Vec<Vec<ScoredDocument>> = Vec::with_capacity(q_embeddings.len());
    for q in &q_embeddings {
        ground_truth.push(bf.search(q, 10));
    }
    let bf_query_time = t0.elapsed();
    println!(
        "      {} BF queries in {:.2}s ({:.1} μs/q)",
        q_embeddings.len(),
        bf_query_time.as_secs_f64(),
        bf_query_time.as_secs_f64() * 1_000_000.0 / q_embeddings.len() as f64
    );
    println!();

    // ---- 5. Build IVF-PQ and query ----
    println!(
        "[5/5] Building IVF-PQ (K-Means++, n_list={}, M={}, n_probe={})...",
        N_LIST, M_SUB, N_PROBE
    );
    let t0 = Instant::now();
    let mut ivf = IvfPqIndex::new(IvfPqConfig {
        n_list: N_LIST,
        n_probe: N_PROBE,
        m_subspaces: M_SUB,
        metric: Metric::Cosine,
        rerank_enabled: false,
        ..IvfPqConfig::default()
    });
    ivf.insert(index_docs);
    let ivf_build = t0.elapsed();
    println!("      IVF-PQ built in {:.2}s", ivf_build.as_secs_f64());

    // Query
    let t0 = Instant::now();
    let mut total_recall_1 = 0.0f64;
    let mut total_recall_10 = 0.0f64;
    let mut latencies = Vec::with_capacity(q_embeddings.len());

    for (i, q) in q_embeddings.iter().enumerate() {
        let tq = Instant::now();
        let results = ivf.search(q, 10);
        let elapsed = tq.elapsed();
        latencies.push(elapsed.as_secs_f64() * 1_000_000.0);
        total_recall_1 += recall_at_k(&results, &ground_truth[i], 1);
        total_recall_10 += recall_at_k(&results, &ground_truth[i], 10);
    }
    let _ivf_query_time = t0.elapsed();
    let nq = q_embeddings.len() as f64;

    // Stats
    latencies.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let mean_lat = latencies.iter().sum::<f64>() / nq;
    let p50 = latencies[(latencies.len() as f64 * 0.50) as usize];
    let p95 = latencies[(latencies.len() as f64 * 0.95) as usize];
    let p99 = latencies[(latencies.len() as f64 * 0.99) as usize];

    let recall1_pct = total_recall_1 / nq * 100.0;
    let recall10_pct = total_recall_10 / nq * 100.0;
    let qps = 1_000_000.0 / mean_lat;

    // ====================================================================
    // REPORT
    // ====================================================================
    println!();
    println!("================================================================");
    println!("  FINAL REPORT — IVF-PQ with Real ONNX Embeddings");
    println!("================================================================");
    println!(
        "  Dataset:      {} docs (hermes-agent), {} held-out queries",
        index_docs.len(),
        q_embeddings.len()
    );
    println!("  Dimension:    {}", DIM);
    println!("  Model:        all-MiniLM-L6-v2 (ONNX / fastembed)");
    println!("  Seed:         {}", args.seed);
    println!();
    println!("  ── Embedding Pipeline ──");
    println!(
        "  Embed time:   {:.2}s ({:.0} docs/s)",
        embed_time.as_secs_f64(),
        index_docs.len() as f64 / embed_time.as_secs_f64().max(0.001)
    );
    println!();
    println!("  ── Index Build ──");
    println!("  IVF-PQ build: {:.2}s", ivf_build.as_secs_f64());
    println!(
        "  Config:       n_list={}, M={}, n_probe={}, K-Means++",
        N_LIST, M_SUB, N_PROBE
    );
    println!();
    println!("  ── Recall ──");
    println!("  Recall@1:     {:.1}%", recall1_pct);
    println!("  Recall@10:    {:.1}%", recall10_pct);
    println!();
    println!("  ── Latency ({} queries) ──", q_embeddings.len());
    println!("  Mean:         {:.1} μs", mean_lat);
    println!("  p50:          {:.1} μs", p50);
    println!("  p95:          {:.1} μs", p95);
    println!("  p99:          {:.1} μs", p99);
    println!("  QPS:          {:.0}", qps);
    println!("================================================================");

    // ---- JSON output ----
    if args.output_json {
        let output = BenchOutput {
            seed: args.seed,
            backend: "IVF-PQ".to_string(),
            config: serde_json::json!({
                "n_list": N_LIST,
                "m_subspaces": M_SUB,
                "n_probe": N_PROBE,
                "metric": "Cosine",
                "model": "all-MiniLM-L6-v2",
            }),
            num_queries: q_embeddings.len(),
            dimension: DIM,
            recall_at_1_pct: recall1_pct,
            recall_at_10_pct: recall10_pct,
            latency_mean_us: mean_lat,
            latency_p50_us: p50,
            latency_p95_us: p95,
            latency_p99_us: p99,
            qps,
            index_docs: index_docs.len(),
            build_time_s: ivf_build.as_secs_f64(),
            embed_time_s: embed_time.as_secs_f64(),
        };
        println!("{}", serde_json::to_string_pretty(&output).unwrap());
    }

    // Success criteria
    let recall10 = total_recall_10 / nq;
    if recall10 >= 0.70 {
        println!("✅ SUCCESS: Recall@10 >= 70% ({:.1}%)", recall10 * 100.0);
    } else {
        println!(
            "⚠️  Recall@10 = {:.1}% (< 70% target). IVF-PQ + PQ ceiling.",
            recall10 * 100.0
        );
        println!("   Suggested improvements: OpQ rotation, PQ residual, f32 rescore.");
    }
    println!();
}
