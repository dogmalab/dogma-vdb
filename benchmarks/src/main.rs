//! Comprehensive grid benchmark for dogma-vdb.
//!
//! Measures (Grid Testing):
//!   - Sizes: 100K, 1M vectors (configurable)
//!   - Dimensions: 384, 1536
//!   - Metrics: Cosine, L2
//!   - Indices: BF, HNSW (M, ef variants), IVF-PQ (nlist, M variants)
//!
//! Output: raw JSON + formatted BENCHMARK.md.
//!
//! Usage:
//!   cargo run --release --bin dogma-vdb-grid-bench --features chunker-syntax
//!   cargo run --release --bin dogma-vdb-grid-bench --features chunker-syntax -- --quick
//!
//! No external dependencies beyond dogma-vdb.

use dogma_vdb::distance::Metric;
use dogma_vdb::doc::Document;
use dogma_vdb::index::{BruteForceIndex, HnswConfig, HnswIndex, Index, IvfPqConfig, IvfPqIndex};
use serde::Deserialize;
use std::fs;
use std::path::Path;
use std::time::Instant;

// ============================================================================
// Grid Configuration
// ============================================================================

/// Tweak these constants to control the benchmark run.
const SIZES: &[usize] = &[10_000];
const DIMS: &[usize] = &[128];
const METRICS: &[Metric] = &[Metric::Cosine];

/// HNSW — vary M (connections) and ef (candidates)
const HNSW_M_VALS: &[usize] = &[16, 32];
const HNSW_EF_VALS: &[usize] = &[50, 100, 150, 200];

/// IVF-PQ — fixed nlist=256, M_sub=8, vary n_probe
const IVF_NLIST_VALS: &[usize] = &[256];
const IVF_M_SUB_VALS: &[usize] = &[8];
const IVF_NPROBE_VALS: &[usize] = &[1, 2, 4, 8, 16, 32];

/// Recall threshold for a configuration to be considered valid
const RECALL_THRESHOLD: f64 = 0.90;

const QUERY_ITERS: usize = 50; // queries per config variation
const WARMUP: usize = 3; // warmup iterations before measurement

const DEFAULT_SEED: u64 = 42;

// ============================================================================
// Argument parsing (simple, no clap dep needed)
// ============================================================================

struct Args {
    seed: u64,
    output_json: bool,
    quick: bool,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    let mut seed = DEFAULT_SEED;
    let mut output_json = false;
    let mut quick = false;

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
                let val = args.get(i).expect("--output requires a format (json)");
                if val == "json" {
                    output_json = true;
                } else {
                    eprintln!("WARNING: unknown output format '{}', ignoring", val);
                }
            }
            "--quick" => quick = true,
            _ => {}
        }
        i += 1;
    }
    Args {
        seed,
        output_json,
        quick,
    }
}

// ============================================================================
// Generacion de datos deterministicos (SplitMix64)
// ============================================================================

#[derive(Clone)]
struct TestData {
    docs: Vec<Document>,
    queries: Vec<Vec<f32>>,
    dim: usize,
    n: usize,
}

fn seed_from_id(id: u64, dim: u64, global_seed: u64) -> u64 {
    global_seed
        .wrapping_mul(id)
        .wrapping_mul(dim)
        .wrapping_mul(0x9E3779B97F4A7C15)
}

fn random_vec(seed: u64, dim: usize) -> Vec<f32> {
    (0..dim)
        .map(|i| {
            let mut z = seed.wrapping_add(i as u64);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^= z >> 31;
            (z >> 10) as f64 * 6.0 / 9007199254740992.0 - 3.0
        })
        .map(|x| x as f32)
        .collect()
}

fn make_test_data(n: usize, dim: usize, global_seed: u64) -> TestData {
    let docs: Vec<Document> = (0..n)
        .map(|i| {
            let seed = seed_from_id(i as u64, dim as u64, global_seed);
            Document::builder(format!("d{i}"), format!("doc {i}"))
                .embedding(random_vec(seed, dim))
                .build()
        })
        .collect();

    // 50 fixed queries (first 50 docs, different seed)
    let queries: Vec<Vec<f32>> = (0..50)
        .map(|i| {
            let seed = seed_from_id(i as u64 + 999_999, dim as u64, global_seed);
            random_vec(seed, dim)
        })
        .collect();

    TestData {
        docs,
        queries,
        dim,
        n,
    }
}

// ============================================================================
// Medicion de RAM (Linux /proc/self/status)
// ============================================================================

fn read_vmrss_kb() -> u64 {
    let status = fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            if let Ok(kb) = rest.trim().trim_end_matches(" kB").parse::<u64>() {
                return kb;
            }
        }
    }
    0
}

/// Measures the RSS delta during execution of `f`.
/// Uses VmRSS (Resident Set Size) to capture physical memory
/// allocated at the exact moment, not the historical peak.
fn measure_ram_delta<F: FnOnce()>(f: F) -> u64 {
    let before = read_vmrss_kb();
    f();
    let after = read_vmrss_kb();
    after.saturating_sub(before)
}

// ============================================================================
// Metricas de latencia (percentiles)
// ============================================================================

#[derive(Debug, Clone)]
struct LatencyStats {
    p50_us: f64,
    p95_us: f64,
    p99_us: f64,
    mean_us: f64,
    min_us: f64,
    max_us: f64,
}

fn compute_latency_stats(latencies_us: &[f64]) -> LatencyStats {
    let mut sorted = latencies_us.to_vec();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let len = sorted.len();
    let mean_us = sorted.iter().sum::<f64>() / len as f64;

    let idx = |pct: f64| -> usize { ((len as f64 * pct / 100.0) as usize).min(len - 1) };

    LatencyStats {
        p50_us: sorted[idx(50.0)],
        p95_us: sorted[idx(95.0)],
        p99_us: sorted[idx(99.0)],
        mean_us,
        min_us: sorted[0],
        max_us: sorted[len - 1],
    }
}

// ============================================================================
// Per-configuration result
// ============================================================================

#[derive(Debug, Clone)]
struct IndexResult {
    label: String,
    build_time_s: f64,
    index_throughput: f64, // vectors/sec
    peak_ram_mb: f64,
    recall_k1: f64,
    recall_k10: f64,
    recall_k100: f64,
    latency: LatencyStats,
    qps: f64,
}

// ============================================================================
// Logica de Recall
// ============================================================================

/// Recall@K:  |{true_topK} ∩ {approx_topK}| / K
/// Donde true_topK son los IDs de los K vecinos mas cercanos exactos (BruteForce).
fn recall_at_k(
    approx: &[dogma_vdb::index::ScoredDocument],
    exact: &[dogma_vdb::index::ScoredDocument],
    k: usize,
) -> f64 {
    let exact_ids: std::collections::HashSet<&str> = exact
        .iter()
        .take(k)
        .map(|r| r.document.id.as_str())
        .collect();
    let matches = approx
        .iter()
        .take(k)
        .filter(|r| exact_ids.contains(r.document.id.as_str()))
        .count();
    matches as f64 / k as f64
}

// ============================================================================
// Benchmarks individuales
// ============================================================================

struct BenchContext {
    metric: Metric,
    queries: Vec<Vec<f32>>,
    _dim: usize,
    exact_results: Vec<Vec<dogma_vdb::index::ScoredDocument>>,
}

fn bench_bf(data: &TestData, ctx: &BenchContext) -> IndexResult {
    let mut bf = BruteForceIndex::new(ctx.metric);

    // Single insert — build time + RAM measured together
    let t0 = Instant::now();
    let ram = measure_ram_delta(|| bf.insert(&data.docs));
    let build_time = t0.elapsed();
    let throughput = data.n as f64 / build_time.as_secs_f64();

    // Warmup
    for _ in 0..WARMUP {
        let _ = bf.search(&ctx.queries[0], 10);
    }

    // Latency (200 queries sobre 50 queries distintas)
    let mut latencies = Vec::with_capacity(QUERY_ITERS);
    let t0 = Instant::now();
    for i in 0..QUERY_ITERS {
        let q = &ctx.queries[i % ctx.queries.len()];
        let start = Instant::now();
        let _ = bf.search(q, 100);
        latencies.push(start.elapsed().as_secs_f64() * 1_000_000.0);
    }
    let total_s = t0.elapsed().as_secs_f64();
    let stats = compute_latency_stats(&latencies);

    // Recall (BF = ground truth, siempre 100%)
    let _exact_k1 = ctx.exact_results[0].clone();
    let _exact_k10: Vec<_> = ctx.exact_results[0].iter().take(10).cloned().collect();
    let _exact_k100: Vec<_> = ctx.exact_results[0].iter().take(100).cloned().collect();
    let _res = bf.search(&ctx.queries[0], 100);

    IndexResult {
        label: "BF".to_string(),
        build_time_s: build_time.as_secs_f64(),
        index_throughput: throughput,
        peak_ram_mb: ram as f64 / 1024.0,
        recall_k1: 1.0,
        recall_k10: 1.0,
        recall_k100: 1.0,
        qps: QUERY_ITERS as f64 / total_s,
        latency: stats,
    }
}

fn bench_hnsw(data: &TestData, ctx: &BenchContext, m: usize, ef: usize) -> IndexResult {
    let label = format!("HNSW M={} ef={}", m, ef);

    let mut hnsw = HnswIndex::new(HnswConfig {
        m,
        ef_construction: ef.max(100),
        ef_search: ef,
        metric: ctx.metric,
        flat_embeddings: false,
        sq: false,
        sq_rescore: false,
    });

    // Single insert — build time + RAM measured together
    let t0 = Instant::now();
    let ram = measure_ram_delta(|| hnsw.insert(&data.docs));
    let build_time = t0.elapsed();
    let throughput = data.n as f64 / build_time.as_secs_f64();

    // Warmup
    for _ in 0..WARMUP {
        hnsw.search(&ctx.queries[0], 10);
    }

    // Latency
    let mut latencies = Vec::with_capacity(QUERY_ITERS);
    let t0 = Instant::now();
    for i in 0..QUERY_ITERS {
        let q = &ctx.queries[i % ctx.queries.len()];
        let start = Instant::now();
        hnsw.search(q, 100);
        latencies.push(start.elapsed().as_secs_f64() * 1_000_000.0);
    }
    let total_s = t0.elapsed().as_secs_f64();
    let stats = compute_latency_stats(&latencies);

    // Recall@K (vs BruteForce ground truth)
    let res = hnsw.search(&ctx.queries[0], 100);
    let recall1 = recall_at_k(&res, &ctx.exact_results[0], 1);
    let recall10 = recall_at_k(&res, &ctx.exact_results[0], 10);
    let recall100 = recall_at_k(&res, &ctx.exact_results[0], 100);

    IndexResult {
        label,
        build_time_s: build_time.as_secs_f64(),
        index_throughput: throughput,
        peak_ram_mb: ram as f64 / 1024.0,
        recall_k1: recall1,
        recall_k10: recall10,
        recall_k100: recall100,
        qps: QUERY_ITERS as f64 / total_s,
        latency: stats,
    }
}

fn bench_ivfpq(
    data: &TestData,
    ctx: &BenchContext,
    n_list: usize,
    m_sub: usize,
    n_probe: usize,
) -> IndexResult {
    let label = format!("IVF-PQ nlist={} M={} probe={}", n_list, m_sub, n_probe);

    let mut ivf = IvfPqIndex::new(IvfPqConfig {
        n_list,
        m_subspaces: m_sub,
        n_probe,
        metric: ctx.metric,
        ..Default::default()
    });

    // Single insert — build time + RAM measured together
    let t0 = Instant::now();
    let ram = measure_ram_delta(|| ivf.insert(&data.docs));
    let build_time = t0.elapsed();
    let throughput = data.n as f64 / build_time.as_secs_f64();

    for _ in 0..WARMUP {
        ivf.search(&ctx.queries[0], 10);
    }

    let mut latencies = Vec::with_capacity(QUERY_ITERS);
    let t0 = Instant::now();
    for i in 0..QUERY_ITERS {
        let q = &ctx.queries[i % ctx.queries.len()];
        let start = Instant::now();
        ivf.search(q, 100);
        latencies.push(start.elapsed().as_secs_f64() * 1_000_000.0);
    }
    let total_s = t0.elapsed().as_secs_f64();
    let stats = compute_latency_stats(&latencies);

    let res = ivf.search(&ctx.queries[0], 100);
    let recall1 = recall_at_k(&res, &ctx.exact_results[0], 1);
    let recall10 = recall_at_k(&res, &ctx.exact_results[0], 10);
    let recall100 = recall_at_k(&res, &ctx.exact_results[0], 100);

    IndexResult {
        label,
        build_time_s: build_time.as_secs_f64(),
        index_throughput: throughput,
        peak_ram_mb: ram as f64 / 1024.0,
        recall_k1: recall1,
        recall_k10: recall10,
        recall_k100: recall100,
        qps: QUERY_ITERS as f64 / total_s,
        latency: stats,
    }
}

// ============================================================================
// Formateo Markdown
// ============================================================================

fn fmt_metric(m: Metric) -> &'static str {
    match m {
        Metric::Cosine => "Cosine",
        Metric::Dot => "Dot",
        Metric::Euclidean => "L2",
        _ => "Unknown",
    }
}

fn fmt_mb(v: f64) -> String {
    format!("{:.1}", v)
}

fn fmt_us(v: f64) -> String {
    if v > 1_000.0 {
        format!("{:.1} ms", v / 1000.0)
    } else {
        format!("{:.0} us", v)
    }
}

fn fmt_recall(v: f64) -> String {
    format!("{:.0}%", v * 100.0)
}

fn fmt_qps(v: f64) -> String {
    if v > 1_000_000.0 {
        format!("{:.0}M", v / 1_000_000.0)
    } else if v > 1_000.0 {
        format!("{:.0}K", v / 1000.0)
    } else {
        format!("{:.0}", v)
    }
}

fn fmt_build(v: f64) -> String {
    if v > 60.0 {
        format!("{:.1} min", v / 60.0)
    } else if v > 1.0 {
        format!("{:.1}s", v)
    } else if v > 0.001 {
        format!("{:.0} ms", v * 1000.0)
    } else {
        format!("{:.1} us", v * 1_000_000.0)
    }
}

fn write_markdown_table(w: &mut String, title: &str, headers: &[&str], rows: &[Vec<String>]) {
    w.push_str(&format!("\n### {}\n\n", title));
    w.push_str("| ");
    w.push_str(&headers.join(" | "));
    w.push_str(" |\n");
    w.push('|');
    for h in headers {
        w.push_str(&"-".repeat(h.len() + 2));
        w.push('|');
    }
    w.push('\n');
    for row in rows {
        w.push_str("| ");
        w.push_str(&row.join(" | "));
        w.push_str(" |\n");
    }
    w.push('\n');
}

fn format_results(
    n: usize,
    dim: usize,
    metric: Metric,
    bf_result: &IndexResult,
    hnsw_results: &[IndexResult],
    ivf_results: &[IndexResult],
    all_results: &[IndexResult],
) -> String {
    let mut md = String::new();

    md.push_str(&format!(
        "\n---\n## {} docs, {} dim, {}\n",
        n,
        dim,
        fmt_metric(metric)
    ));
    md.push('\n');

    // Tabla 1: Build Time + Throughput + RAM
    {
        let mut rows = Vec::new();
        // BF row
        rows.push(vec![
            bf_result.label.clone(),
            fmt_build(bf_result.build_time_s),
            fmt_qps(bf_result.index_throughput),
            fmt_mb(bf_result.peak_ram_mb),
        ]);
        for r in hnsw_results {
            rows.push(vec![
                r.label.clone(),
                fmt_build(r.build_time_s),
                fmt_qps(r.index_throughput),
                fmt_mb(r.peak_ram_mb),
            ]);
        }
        for r in ivf_results {
            rows.push(vec![
                r.label.clone(),
                fmt_build(r.build_time_s),
                fmt_qps(r.index_throughput),
                fmt_mb(r.peak_ram_mb),
            ]);
        }
        write_markdown_table(
            &mut md,
            "Construccion: Build Time / Throughput / RAM",
            &["Index", "Build", "vec/s", "RAM (MB)"],
            &rows,
        );
    }

    // Tabla 2: Recall@K
    {
        let mut rows = Vec::new();
        rows.push(vec![
            bf_result.label.clone(),
            fmt_recall(bf_result.recall_k1),
            fmt_recall(bf_result.recall_k10),
            fmt_recall(bf_result.recall_k100),
        ]);
        for r in hnsw_results {
            rows.push(vec![
                r.label.clone(),
                fmt_recall(r.recall_k1),
                fmt_recall(r.recall_k10),
                fmt_recall(r.recall_k100),
            ]);
        }
        for r in ivf_results {
            rows.push(vec![
                r.label.clone(),
                fmt_recall(r.recall_k1),
                fmt_recall(r.recall_k10),
                fmt_recall(r.recall_k100),
            ]);
        }
        write_markdown_table(
            &mut md,
            "Precision: Recall@K (vs BruteForce)",
            &["Index", "Recall@1", "Recall@10", "Recall@100"],
            &rows,
        );
    }

    // Tabla 3: Latencia
    {
        let mut rows = Vec::new();
        rows.push(vec![
            bf_result.label.clone(),
            fmt_us(bf_result.latency.mean_us),
            fmt_us(bf_result.latency.p50_us),
            fmt_us(bf_result.latency.p95_us),
            fmt_us(bf_result.latency.p99_us),
        ]);
        for r in hnsw_results {
            rows.push(vec![
                r.label.clone(),
                fmt_us(r.latency.mean_us),
                fmt_us(r.latency.p50_us),
                fmt_us(r.latency.p95_us),
                fmt_us(r.latency.p99_us),
            ]);
        }
        for r in ivf_results {
            rows.push(vec![
                r.label.clone(),
                fmt_us(r.latency.mean_us),
                fmt_us(r.latency.p50_us),
                fmt_us(r.latency.p95_us),
                fmt_us(r.latency.p99_us),
            ]);
        }
        write_markdown_table(
            &mut md,
            "Rendimiento: Latencia de Consulta",
            &["Index", "Mean", "p50", "p95", "p99"],
            &rows,
        );
    }

    // Tabla 4: Recall vs QPS vs RAM (Sweet Spot)
    {
        let mut rows = Vec::new();
        for r in hnsw_results.iter().chain(ivf_results.iter()) {
            let speedup = if bf_result.latency.mean_us > 0.0 {
                bf_result.latency.mean_us / r.latency.mean_us
            } else {
                1.0
            };
            rows.push(vec![
                r.label.clone(),
                fmt_recall(r.recall_k10),
                fmt_qps(r.qps),
                format!("{:.0}x", speedup),
                fmt_mb(r.peak_ram_mb),
            ]);
        }
        write_markdown_table(
            &mut md,
            "Sweet Spot: Recall@10 vs QPS vs RAM",
            &["Index", "Recall@10", "QPS", "xBF", "RAM (MB)"],
            &rows,
        );
    }

    // Resaltar sweet spot
    md.push_str("#### Sweet Spot\n\n");
    if let Some(best) = hnsw_results
        .iter()
        .chain(ivf_results.iter())
        .filter(|r| r.recall_k10 >= 0.85)
        .max_by(|a, b| {
            (a.qps * 1000.0 + (1.0 - a.peak_ram_mb / 1000.0) * 100.0)
                .partial_cmp(&(b.qps * 1000.0 + (1.0 - b.peak_ram_mb / 1000.0) * 100.0))
                .unwrap()
        })
    {
        md.push_str(&format!(
            "- Mejor configuracion (Recall≥85%): **{}** — QPS={}, Latencia={}, RAM={} MB\n",
            best.label,
            fmt_qps(best.qps),
            fmt_us(best.latency.mean_us),
            fmt_mb(best.peak_ram_mb),
        ));
    }
    if let Some(fastest) = all_results
        .iter()
        .filter(|r| r.recall_k10 >= 0.50)
        .min_by(|a, b| a.latency.mean_us.partial_cmp(&b.latency.mean_us).unwrap())
    {
        md.push_str(&format!(
            "- Mas rapido (Recall≥50%): **{}** — {} us, Recall@10={}\n",
            fastest.label,
            fmt_us(fastest.latency.mean_us),
            fmt_recall(fastest.recall_k10),
        ));
    }
    if let Some(min_ram) = all_results
        .iter()
        .filter(|r| r.recall_k10 >= 0.50)
        .min_by(|a, b| a.peak_ram_mb.partial_cmp(&b.peak_ram_mb).unwrap())
    {
        md.push_str(&format!(
            "- Menor RAM (Recall≥50%): **{}** — {} MB, Recall@10={}\n",
            min_ram.label,
            fmt_mb(min_ram.peak_ram_mb),
            fmt_recall(min_ram.recall_k10),
        ));
    }

    md
}

// ============================================================================
// Output JSON
// ============================================================================

fn append_json(json_path: &Path, entry: &serde_json::Value) {
    let mut data: Vec<serde_json::Value> = if json_path.exists() {
        let s = fs::read_to_string(json_path).unwrap_or_else(|_| "[]".into());
        serde_json::from_str(&s).unwrap_or_default()
    } else {
        Vec::new()
    };
    data.push(entry.clone());
    fs::write(json_path, serde_json::to_string_pretty(&data).unwrap()).ok();
}

fn result_to_json(
    r: &IndexResult,
    n: usize,
    dim: usize,
    metric: Metric,
    seed: u64,
) -> serde_json::Value {
    serde_json::json!({
        "seed": seed,
        "label": r.label,
        "n": n,
        "dim": dim,
        "metric": fmt_metric(metric),
        "build_time_s": r.build_time_s,
        "index_throughput": r.index_throughput,
        "peak_ram_mb": r.peak_ram_mb,
        "recall_k1": r.recall_k1,
        "recall_k10": r.recall_k10,
        "recall_k100": r.recall_k100,
        "latency_mean_us": r.latency.mean_us,
        "latency_p50_us": r.latency.p50_us,
        "latency_p95_us": r.latency.p95_us,
        "latency_p99_us": r.latency.p99_us,
        "latency_min_us": r.latency.min_us,
        "latency_max_us": r.latency.max_us,
        "qps": r.qps,
    })
}

// ============================================================================
// Tree-Sitter Chunking Benchmark
// ============================================================================

#[cfg(feature = "chunker-syntax")]
fn bench_chunking() -> (f64, f64, u64) {
    use dogma_vdb::smart_chunker::{ChunkStrategy, SmartChunker};
    use std::path::Path;

    let src_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut all_code = String::new();
    let mut total_bytes = 0u64;

    fn collect(dir: &Path, code: &mut String, bytes: &mut u64) {
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    collect(&path, code, bytes);
                } else if path.extension().is_some_and(|e| e == "rs") {
                    if let Ok(content) = fs::read_to_string(&path) {
                        *bytes += content.len() as u64;
                        code.push_str(&content);
                        code.push('\n');
                    }
                }
            }
        }
    }
    collect(&src_dir, &mut all_code, &mut total_bytes);

    let chunker = SmartChunker::default();
    let _ = chunker.chunk_text(&all_code, ChunkStrategy::Code);

    let iters = 20;
    let start = Instant::now();
    let mut total_chunks = 0usize;
    for _ in 0..iters {
        total_chunks += chunker.chunk_text(&all_code, ChunkStrategy::Code).len();
    }
    let elapsed = start.elapsed();
    let mb_per_sec = (total_bytes as f64 * iters as f64) / elapsed.as_secs_f64() / 1024.0 / 1024.0;
    let chunks_per_sec = total_chunks as f64 / elapsed.as_secs_f64();
    (mb_per_sec, chunks_per_sec, total_bytes)
}

#[cfg(not(feature = "chunker-syntax"))]
fn bench_chunking() -> (f64, f64, u64) {
    (0.0, 0.0, 0)
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    let args = parse_args();
    let out_dir = Path::new("benchmarks");
    let _ = fs::create_dir_all(out_dir);
    let json_path = out_dir.join("bench_results.json");

    let quick = args.quick;
    let sizes: &[usize] = if quick { &[100_000] } else { SIZES };

    eprintln!("dogma-vdb Grid Benchmark");
    eprintln!("  Seed: {}", args.seed);
    eprintln!("  Sizes: {:?}", sizes);
    eprintln!("  Dims: {:?}", DIMS);
    eprintln!(
        "  Metrics: {:?}",
        METRICS.iter().map(|m| fmt_metric(*m)).collect::<Vec<_>>()
    );
    eprintln!("  HNSW grid: M={:?}, ef={:?}", HNSW_M_VALS, HNSW_EF_VALS);
    eprintln!(
        "  IVF grid: nlist={:?}, M_sub={:?}",
        IVF_NLIST_VALS, IVF_M_SUB_VALS
    );
    eprintln!("  Queries/config: {}", QUERY_ITERS);
    eprintln!();

    // -- Chunking benchmark --
    let (mb_per_sec, chunks_per_sec, chunk_bytes) = bench_chunking();
    if mb_per_sec > 0.0 {
        eprintln!(
            "Tree-sitter chunking: {:.1} MB/sec, {:.0} chunks/sec",
            mb_per_sec, chunks_per_sec
        );
    }

    let mut master_md = String::new();
    master_md.push_str("# dogma-vdb — Benchmark Grid Results\n\n");
    master_md.push_str("> Generado automaticamente | ");
    master_md.push_str(&format!(
        "Vectores 128-dim | Cosine | k=10 | {} queries/config\n\n",
        QUERY_ITERS
    ));
    master_md.push_str("## Parametros del Grid\n\n");
    master_md.push_str(&format!("- Tamaños: {:?}\n", sizes));
    master_md.push_str(&format!("- Dimensiones: {:?}\n", DIMS));
    master_md.push_str(&format!(
        "- Metricas: {:?}\n",
        METRICS.iter().map(|m| fmt_metric(*m)).collect::<Vec<_>>()
    ));
    master_md.push_str(&format!(
        "- HNSW grid: M∈{:?}, ef∈{:?}\n",
        HNSW_M_VALS, HNSW_EF_VALS
    ));
    master_md.push_str(&format!(
        "- IVF-PQ grid: nlist∈{:?}, M_sub∈{:?}\n",
        IVF_NLIST_VALS, IVF_M_SUB_VALS
    ));
    master_md.push_str(&format!("- Queries por configuracion: {}\n", QUERY_ITERS));

    if mb_per_sec > 0.0 {
        master_md.push_str(&format!("\n## Tree-Sitter Chunking\n\n- Throughput: **{:.1} MB/s**\n- Chunks: **{:.0}/s**\n- Source: {} KB\n",
            mb_per_sec, chunks_per_sec, chunk_bytes as f64 / 1024.0));
    }

    for &n in sizes {
        for &dim in DIMS {
            for &metric in METRICS {
                eprintln!("\n=== {} docs, {} dim, {:?} ===", n, dim, metric);
                eprintln!("Generating data...");
                let data = make_test_data(n, dim, args.seed);
                eprintln!("  Done. {} vectors, {} dim", data.n, data.dim);

                // Ground truth: BruteForce (para recall)
                eprintln!("Building BruteForce (ground truth)...");
                let mut bf = BruteForceIndex::new(metric);
                bf.insert(&data.docs);
                let exact_results: Vec<_> =
                    data.queries.iter().map(|q| bf.search(q, 100)).collect();
                eprintln!("  Done.");

                let ctx = BenchContext {
                    metric,
                    queries: data.queries.clone(),
                    _dim: dim,
                    exact_results,
                };

                // BF itself
                eprintln!("Benchmarking BF...");
                let bf_result = bench_bf(&data, &ctx);
                append_json(
                    &json_path,
                    &result_to_json(&bf_result, n, dim, metric, args.seed),
                );

                // HNSW grid
                let mut hnsw_results = Vec::new();
                for &m in HNSW_M_VALS {
                    for &ef in HNSW_EF_VALS {
                        eprintln!("Benchmarking HNSW M={} ef={}...", m, ef);
                        let r = bench_hnsw(&data, &ctx, m, ef);
                        append_json(&json_path, &result_to_json(&r, n, dim, metric, args.seed));
                        hnsw_results.push(r);
                    }
                }

                // IVF-PQ grid
                let mut ivf_results = Vec::new();
                for &nl in IVF_NLIST_VALS {
                    if nl > n {
                        continue;
                    }
                    for &ms in IVF_M_SUB_VALS {
                        if ms > dim || dim % ms != 0 {
                            continue;
                        }
                        for &np in IVF_NPROBE_VALS {
                            eprintln!("Benchmarking IVF-PQ nlist={} M={} probe={}...", nl, ms, np);
                            let r = bench_ivfpq(&data, &ctx, nl, ms, np);
                            append_json(&json_path, &result_to_json(&r, n, dim, metric, args.seed));
                            ivf_results.push(r);
                        }
                    }
                }

                // Formatear markdown
                let mut all: Vec<IndexResult> = Vec::new();
                all.push(bf_result.clone());
                all.extend(hnsw_results.iter().cloned());
                all.extend(ivf_results.iter().cloned());
                let section = format_results(
                    n,
                    dim,
                    metric,
                    &bf_result,
                    &hnsw_results,
                    &ivf_results,
                    &all,
                );
                master_md.push_str(&section);

                // Free memory
                drop(data);
                drop(bf_result);
                drop(hnsw_results);
                drop(ivf_results);
            }
        }
    }

    // Write final BENCHMARK.md
    master_md.push_str("\n---\n*Benchmark generado con dogma-vdb grid benchmark*\n");
    let bench_md_path = out_dir.join("BENCHMARK.md");
    fs::write(&bench_md_path, &master_md).ok();

    // ─── FASE 3 & 4: Scoring + TUNING_REPORT.md ───────────────────────────
    eprintln!("\nAnalizando resultados y generando TUNING_REPORT.md...");

    // Read results from JSON for independent analysis
    let tuning_md = generate_tuning_report(&json_path);

    let tuning_path = out_dir.join("TUNING_REPORT.md");
    fs::write(&tuning_path, &tuning_md).ok();

    eprintln!(
        "\nDone! Results written to {}, {} and {}",
        bench_md_path.display(),
        json_path.display(),
        tuning_path.display()
    );

    // ─── --output json: dump results to stdout ──────────────────────
    if args.output_json {
        if let Ok(json_str) = fs::read_to_string(&json_path) {
            println!("{}", json_str);
        }
    }
}

// ============================================================================
// PHASE 3: Scoring function and report generation
// ============================================================================

/// Generates the calibration report with top-3 configurations for
/// HNSW and IVF-PQ, computing Score = QPS / RAM_MB (only if Recall@10 >= 90%).
fn generate_tuning_report(json_path: &Path) -> String {
    let mut report = String::new();
    report.push_str("# Tuning Report — Dogma-VDB Autonomous Calibration\n\n");
    report.push_str(&format!(
        "> Auto-generated | Recall@10 >= {}% threshold\n\n",
        (RECALL_THRESHOLD * 100.0) as u8
    ));
    report.push_str("## Metodologia\n\n");
    report.push_str("Para cada configuracion del grid:\n");
    report.push_str("- Si Recall@10 < 90% → descartada\n");
    report.push_str("- Si Recall@10 >= 90% → **Score = QPS / RAM_MB**\n");
    report.push_str("- Mayor score = mejor eficiencia (maximos QPS por MB de RAM)\n\n");

    // Load results from JSON
    let json_str = match fs::read_to_string(json_path) {
        Ok(s) => s,
        Err(_) => {
            report.push_str("**Error**: No se pudo leer el archivo de resultados.\n");
            return report;
        }
    };

    #[derive(Deserialize, Debug, Clone)]
    struct JsonResult {
        label: String,
        recall_k10: f64,
        qps: f64,
        peak_ram_mb: f64,
        latency_mean_us: f64,
        build_time_s: f64,
        #[serde(rename = "n")]
        _n: usize,
        #[serde(rename = "dim")]
        _dim: usize,
    }

    let all_results: Vec<JsonResult> = match serde_json::from_str(&json_str) {
        Ok(r) => r,
        Err(_) => {
            report.push_str("**Error**: No se pudieron parsear los resultados JSON.\n");
            return report;
        }
    };

    // Separar HNSW e IVF-PQ
    let mut hnsw_results: Vec<&JsonResult> = all_results
        .iter()
        .filter(|r| r.label.starts_with("HNSW"))
        .collect();
    let mut ivf_results: Vec<&JsonResult> = all_results
        .iter()
        .filter(|r| r.label.starts_with("IVF-PQ"))
        .collect();

    // Ordenar por score (QPS / RAM_MB) descendente, solo si recall >= threshold
    let score_fn = |r: &JsonResult| -> f64 {
        if r.recall_k10 >= RECALL_THRESHOLD && r.peak_ram_mb > 0.0 {
            r.qps / r.peak_ram_mb
        } else if r.recall_k10 >= RECALL_THRESHOLD {
            // Si RAM_MB es 0 (no medido), usar QPS como score
            r.qps
        } else {
            0.0 // descartado
        }
    };

    hnsw_results.sort_by(|a, b| {
        score_fn(b)
            .partial_cmp(&score_fn(a))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    ivf_results.sort_by(|a, b| {
        score_fn(b)
            .partial_cmp(&score_fn(a))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // ─── HNSW Top 3 ──────────────────────────────────────────────────────
    report.push_str("---\n## HNSW — Top 3 Sweet Spots\n\n");
    report.push_str(
        "| # | Config | Build | Latencia | QPS | RAM (MB) | Recall@10 | Score (QPS/MB) |\n",
    );
    report.push_str(
        "|---|--------|-------|----------|-----|----------|-----------|----------------|\n",
    );

    let valid_hnsw: Vec<&&JsonResult> = hnsw_results
        .iter()
        .filter(|r| score_fn(r) > 0.0)
        .take(3)
        .collect();
    if valid_hnsw.is_empty() {
        report.push_str(
            "| — | Ninguna configuracion supera el umbral del 90% | — | — | — | — | — | — |\n",
        );
    }
    for (i, r) in valid_hnsw.iter().enumerate() {
        let s = score_fn(r);
        let build_s = r.build_time_s;
        let build_str = if build_s > 60.0 {
            format!("{:.1} min", build_s / 60.0)
        } else {
            format!("{:.1}s", build_s)
        };
        report.push_str(&format!(
            "| {} | {} | {} | {:.0} us | {:.0} | {:.1} | {:.0}% | {:.1} |\n",
            i + 1,
            r.label,
            build_str,
            r.latency_mean_us,
            r.qps,
            r.peak_ram_mb,
            r.recall_k10 * 100.0,
            s
        ));
    }

    // ─── IVF-PQ Top 3 ─────────────────────────────────────────────────────
    report.push_str("\n---\n## IVF-PQ — Top 3 Sweet Spots\n\n");
    report.push_str(
        "| # | Config | Build | Latencia | QPS | RAM (MB) | Recall@10 | Score (QPS/MB) |\n",
    );
    report.push_str(
        "|---|--------|-------|----------|-----|----------|-----------|----------------|\n",
    );

    let valid_ivf: Vec<&&JsonResult> = ivf_results
        .iter()
        .filter(|r| score_fn(r) > 0.0)
        .take(3)
        .collect();
    if valid_ivf.is_empty() {
        report.push_str(
            "| — | Ninguna configuracion supera el umbral del 90% | — | — | — | — | — | — |\n",
        );
    }
    for (i, r) in valid_ivf.iter().enumerate() {
        let s = score_fn(r);
        let build_s = r.build_time_s;
        let build_str = if build_s > 60.0 {
            format!("{:.1} min", build_s / 60.0)
        } else {
            format!("{:.1}s", build_s)
        };
        report.push_str(&format!(
            "| {} | {} | {} | {:.0} us | {:.0} | {:.1} | {:.0}% | {:.1} |\n",
            i + 1,
            r.label,
            build_str,
            r.latency_mean_us,
            r.qps,
            r.peak_ram_mb,
            r.recall_k10 * 100.0,
            s
        ));
    }

    // ─── Analisis de n_probe en IVF-PQ ────────────────────────────────────
    report.push_str("\n---\n## Impacto de `n_probe` en IVF-PQ\n\n");
    report.push_str("| n_probe | Latencia (us) | QPS | Recall@10 | RAM (MB) | Score |\n");
    report.push_str("|---------|---------------|-----|-----------|----------|-------|\n");

    // Ordenar IVF por n_probe (extraer n_probe de la etiqueta "IVF-PQ nlist=X M=Y probe=Z")
    ivf_results.sort_by(|a, b| {
        let a_np = a
            .label
            .rsplit("probe=")
            .next()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        let b_np = b
            .label
            .rsplit("probe=")
            .next()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        a_np.cmp(&b_np)
    });

    for r in &ivf_results {
        let np = r
            .label
            .rsplit("probe=")
            .next()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        let s = score_fn(r);
        let recall_str = if r.recall_k10 >= RECALL_THRESHOLD {
            format!("**{:.0}%** ✅", r.recall_k10 * 100.0)
        } else {
            format!("{:.0}%", r.recall_k10 * 100.0)
        };
        report.push_str(&format!(
            "| {} | {:.0} | {:.0} | {} | {:.1} | {:.1} |\n",
            np, r.latency_mean_us, r.qps, recall_str, r.peak_ram_mb, s
        ));
    }

    // Analisis textual
    report.push_str("\n### Analisis\n\n");

    // Encontrar primer n_probe que cruza 90%
    let first_90 = ivf_results
        .iter()
        .find(|r| r.recall_k10 >= RECALL_THRESHOLD);
    if let Some(entry) = first_90 {
        let np = entry
            .label
            .rsplit("probe=")
            .next()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        report.push_str(&format!(
            "- El umbral del 90% de Recall@10 se alcanza con **n_probe={}**.\n",
            np
        ));
    } else {
        report.push_str("- Ninguna configuracion de n_probe logro alcanzar el 90% de Recall@10 ");
        report
            .push_str("con vectores aleatorios. Esto es esperable: los ANN indexes no explotan\n");
        report
            .push_str("estructura inexistente en datos ruidosos. Con embeddings reales (texto),\n");
        report.push_str("el recall seria significativamente mayor.\n");
    }

    // Comparacion n_probe bajo vs alto
    if let (Some(low), Some(high)) = (ivf_results.first(), ivf_results.last()) {
        let low_np = low
            .label
            .rsplit("probe=")
            .next()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        let high_np = high
            .label
            .rsplit("probe=")
            .next()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        report.push_str(&format!(
            "- n_probe={}: {:.0} us, {:.0}% recall vs n_probe={}: {:.0} us, {:.0}% recall.\n",
            low_np,
            low.latency_mean_us,
            low.recall_k10 * 100.0,
            high_np,
            high.latency_mean_us,
            high.recall_k10 * 100.0,
        ));
        let speedup = high.qps / low.qps;
        let recall_gain = (high.recall_k10 - low.recall_k10) * 100.0;
        report.push_str(&format!(
            "- Aumentar n_probe de {} a {} mejora recall en {:.0} puntos pero reduce QPS {:.1}x.\n",
            low_np, high_np, recall_gain, speedup
        ));
    }

    report
        .push_str("\n---\n*Reporte generado automaticamente por el sistema de Tunning Autonomo*\n");
    report
}
