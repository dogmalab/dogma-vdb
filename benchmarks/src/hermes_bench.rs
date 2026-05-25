//! Real‑world IVF‑PQ recall validation using source code from hermes-agent.
//!
//! Reads actual source files, creates content‑derived embeddings (not random),
//! builds BruteForce and IVF‑PQ indices, and reports Recall@1 / Recall@10.
//!
//! Run:
//!   cargo run --release --bin hermes-bench

use dogma_vdb::distance::Metric;
use dogma_vdb::doc::Document;
use dogma_vdb::index::{BruteForceIndex, Index, IvfPqConfig, IvfPqIndex, ScoredDocument};
use std::path::Path;
use std::time::Instant;

// ============================================================================
// Config
// ============================================================================

const DIM: usize = 384;
// Use a larger n_probe to give IVF-PQ a fair chance at good recall
const N_PROBE: usize = 64;
const N_LIST: usize = 256;
const M_SUB: usize = 32; // 384/32 = 12 dims/sub-vector (≤16 ✓)

/// Make a content‑derived embedding from source text.
/// Uses **word n‑gram feature hashing** so that similar content
/// (shared keywords, idioms) produces similar vectors.
///
/// Algorithm:
/// 1. Tokenise into alphanumeric words + bigrams.
/// 2. Hash each token to `d` dimensions (multi‑hash).
/// 3. Accumulate, normalise, and output a dense `[f32; dim]`.
fn embed_from_text(text: &str, dim: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; dim];

    // Tokenise: split on non-alphanumeric, lowercase
    let tokens: Vec<String> = text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect();

    // Process each token and its bigrams
    for win in tokens.windows(2) {
        // Unigram
        hash_add(&mut v, &win[0], dim, 1.0);
        // Bigram (shared vocabulary → similar embeddings)
        let bigram = format!("{}_{}", win[0], win[1]);
        hash_add(&mut v, &bigram, dim, 0.8);
    }
    // Last unigram if there were tokens
    if let Some(last) = tokens.last() {
        hash_add(&mut v, last, dim, 1.0);
    }

    // L2 normalise
    let mag: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag > 0.0 {
        for x in &mut v {
            *x /= mag;
        }
    }
    v
}

/// Multi‑hash: add `weight` to `k` different pseudo‑random dimensions
/// derived from the token's hash.  Smooths out hash collisions.
fn hash_add(v: &mut [f32], token: &str, dim: usize, weight: f32) {
    let mut h: u64 = 0xcbf29ce484222325; // FNV-1a offset basis
    for b in token.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    // 3 hash dimensions from the same seed (avoids sparse single‑dim hits)
    let idx1 = (h as usize) % dim;
    let idx2 = ((h >> 16) as usize) % dim;
    let idx3 = ((h >> 32) as usize) % dim;
    v[idx1] += weight;
    v[idx2] += weight * 0.5;
    v[idx3] += weight * 0.25;
}

/// Walk a directory tree and collect all documents.
fn collect_documents(root: &Path, dim: usize) -> Vec<Document> {
    let mut docs = Vec::new();
    let extensions = ["rs", "md", "py", "toml", "json", "yaml", "sh", "txt"];
    collect_recursive(root, root, &extensions, dim, &mut docs);
    docs
}

fn collect_recursive(
    base: &Path,
    dir: &Path,
    extensions: &[&str],
    dim: usize,
    docs: &mut Vec<Document>,
) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let name = path.file_name().unwrap_or_default();
                if name != "target" && name != ".git" && name != "node_modules" && name != ".venv"
                {
                    collect_recursive(base, &path, extensions, dim, docs);
                }
            } else if path.is_file() {
                if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                    if extensions.contains(&ext) {
                        if let Ok(content) = std::fs::read_to_string(&path) {
                            if !content.is_empty() && content.len() <= 500_000 {
                                let rel = path
                                    .strip_prefix(base)
                                    .unwrap_or(&path)
                                    .to_string_lossy();
                                let id = rel.replace('/', "--").replace('.', "-");
                                let emb = embed_from_text(&content, dim);
                                if emb.iter().any(|&x| x != 0.0) {
                                    docs.push(
                                        Document::builder(id, content)
                                            .embedding(emb)
                                            .build(),
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

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

fn main() {
    println!("=== IVF-PQ Recall Validation with Real Data (hermes-agent) ===\n");

    // ---- Load data ----
    let hermes_path = Path::new("/home/arggil/Documents/DEV-WORKSPACE/hermes-agent");
    if !hermes_path.exists() {
        eprintln!("ERROR: hermes-agent not found at {:?}", hermes_path);
        std::process::exit(1);
    }

    println!("Loading documents from: {:?}", hermes_path);
    let start = Instant::now();
    let docs = collect_documents(hermes_path, DIM);
    let load_time = start.elapsed();

    println!(
        "  {} documents loaded in {:.2}s",
        docs.len(),
        load_time.as_secs_f64()
    );
    println!(
        "  Embedding dimension: {}",
        if docs.is_empty() {
            0
        } else {
            docs[0].embedding.len()
        }
    );

    if docs.len() < 10 {
        eprintln!("ERROR: too few documents ({})", docs.len());
        std::process::exit(1);
    }

    // Split: 80% index, 20% queries
    let split = (docs.len() as f64 * 0.8) as usize;
    let index_docs = &docs[..split];
    let query_texts: Vec<&str> = docs[split..].iter().map(|d| d.text.as_str()).collect();

    println!(
        "  Index docs: {}, Query docs: {}\n",
        index_docs.len(),
        query_texts.len()
    );

    // ---- Build BruteForce (ground truth) ----
    println!("Building BruteForce index (ground truth)...");
    let mut bf = BruteForceIndex::new(Metric::Cosine);
    let t0 = Instant::now();
    bf.insert(index_docs);
    println!(
        "  BF built in {:.2}s ({} docs)\n",
        t0.elapsed().as_secs_f64(),
        bf.len()
    );

    // Collect ground truth queries
    println!("Running {} ground-truth queries...", query_texts.len());
    let t0 = Instant::now();
    let mut bf_results: Vec<Vec<ScoredDocument>> = Vec::with_capacity(query_texts.len());
    for text in &query_texts {
        let q = embed_from_text(text, DIM);
        bf_results.push(bf.search(&q, 10));
    }
    println!(
        "  Ground truth computed in {:.2}s\n",
        t0.elapsed().as_secs_f64()
    );

    // ---- Build IVF-PQ (with K-Means++) ----
    println!(
        "Building IVF-PQ (K-Means++) — n_list={}, M={}, n_probe={}...",
        N_LIST, M_SUB, N_PROBE
    );
    let mut ivf = IvfPqIndex::new(IvfPqConfig {
        n_list: N_LIST,
        n_probe: N_PROBE,
        m_subspaces: M_SUB,
        metric: Metric::Cosine,
        rerank_enabled: false,
    });
    let t0 = Instant::now();
    ivf.insert(index_docs);
    println!(
        "  IVF-PQ built in {:.2}s\n",
        t0.elapsed().as_secs_f64()
    );

    // ---- Query IVF-PQ ----
    println!("Running {} IVF-PQ queries...", query_texts.len());
    let t0 = Instant::now();
    let mut total_recall_1 = 0.0;
    let mut total_recall_10 = 0.0;
    let mut latencies = Vec::with_capacity(query_texts.len());

    for (i, text) in query_texts.iter().enumerate() {
        let q = embed_from_text(text, DIM);
        let tq = Instant::now();
        let results = ivf.search(&q, 10);
        let elapsed = tq.elapsed();

        latencies.push(elapsed.as_secs_f64() * 1_000_000.0); // μs
        total_recall_1 += recall_at_k(&results, &bf_results[i], 1);
        total_recall_10 += recall_at_k(&results, &bf_results[i], 10);
    }

    let _total_time = t0.elapsed();
    let nq = query_texts.len() as f64;

    // Stats
    latencies.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let mean_lat = latencies.iter().sum::<f64>() / nq;
    let p50 = latencies[(latencies.len() as f64 * 0.50) as usize];
    let p95 = latencies[(latencies.len() as f64 * 0.95) as usize];
    let p99 = latencies[(latencies.len() as f64 * 0.99) as usize];

    println!("\n============================================================");
    println!("  RESULTS — IVF-PQ with K-Means++ on hermes-agent data");
    println!("============================================================");
    println!("  {:25} {:>8} / {:.0} queries", "Recall@1:", total_recall_1 / nq * 100.0, nq);
    println!("  {:25} {:>8}", "", format!("{:.1}%", total_recall_1 / nq * 100.0));
    println!("  {:25} {:>8} / {:.0} queries", "Recall@10:", total_recall_10 / nq * 100.0, nq);
    println!("  {:25} {:>8}", "", format!("{:.1}%", total_recall_10 / nq * 100.0));
    println!("  ───────────────────────────────────────────");
    println!("  {:25} {:>8.1} μs", "Latency mean:", mean_lat);
    println!("  {:25} {:>8.1} μs", "Latency p50:", p50);
    println!("  {:25} {:>8.1} μs", "Latency p95:", p95);
    println!("  {:25} {:>8.1} μs", "Latency p99:", p99);
    println!("============================================================\n");

    // ---- Comparison: old random K-Means (simulated by reducing quality) ----
    // We can't run the old code anymore, but we can report the improvement
    // by noting the test_recall_against_bf was passing at just 2/5 (40%).
    // The new code should do significantly better.

    println!("---");
    println!("Baseline comparison:");
    println!("  Old test_recall_against_bf threshold: 2/5 hits (40%)");
    println!("  New real-data Recall@10: {:.1}%", total_recall_10 / nq * 100.0);
    println!();

    // Check minimum success threshold: Recall@10 ≥ 70%
    let pass = (total_recall_10 / nq) >= 0.70;
    if pass {
        println!("✅ SUCCESS: IVF-PQ Recall@10 >= 70% threshold ({:.1}%)",
                 total_recall_10 / nq * 100.0);
    } else {
        println!("⚠️  IVF-PQ Recall@10 = {:.1}% (below 70% target). Try increasing n_probe or n_list.",
                 total_recall_10 / nq * 100.0);
    }
}
