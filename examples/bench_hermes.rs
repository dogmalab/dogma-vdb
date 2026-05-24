//! Lightweight benchmark on hermes-agent with file sampling.
//! Usage: HERMES_AGENT_PATH=... cargo run --release --example bench_hermes
use dogma_vdb::distance::Metric;
use dogma_vdb::index::{
    BruteForceIndex, HnswConfig, HnswIndex, Index, IvfPqConfig, IvfPqIndex,
};
use dogma_vdb::smart_chunker::SmartChunker;
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

const DIM: usize = 64;
const QUERIES: usize = 100;
const TOP_K: usize = 10;
const SAMPLE_FILES: usize = 100;

fn embed_text(text: &str) -> Vec<f32> {
    use std::hash::{Hash, Hasher};
    let mut vec = Vec::with_capacity(DIM);
    for i in 0..DIM {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        text.hash(&mut h);
        (i as u64).hash(&mut h);
        vec.push((h.finish() as f64 / u64::MAX as f64) as f32);
    }
    vec
}

fn sample_files(root: &Path, max: usize) -> Vec<std::path::PathBuf> {
    let mut files = Vec::new();
    let mut dirs = vec![root.to_path_buf()];
    while let Some(dir) = dirs.pop() {
        if files.len() >= max { break; }
        let Ok(entries) = std::fs::read_dir(&dir) else { continue; };
        for entry in entries.flatten() {
            if files.len() >= max { break; }
            let path = entry.path();
            if path.file_name().and_then(|s| s.to_str()).is_some_and(|s| s.starts_with('.')) {
                continue;
            }
            if path.is_dir() {
                dirs.push(path);
            } else if path.is_file() && files.len() < max {
                files.push(path);
            }
        }
    }
    files
}

fn main() {
    let hermes_path = std::env::var("HERMES_AGENT_PATH").unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/home/arggil".into());
        format!("{home}/Documents/DEV-WORKSPACE/hermes-agent")
    });

    eprintln!("=== dogma-vdb Benchmark (sampled) ===");
    eprintln!("Source: {hermes_path}");
    eprintln!("Sample: up to {SAMPLE_FILES} files\n");

    // ── Ingest ──
    let chunker = SmartChunker::default();
    let binary_exts: &[&str] = &[
        "png", "jpg", "jpeg", "gif", "ico", "woff2", "woff", "ttf", "otf", "eot",
        "pdf", "zip", "gz", "pyc", "mp3", "mp4", "webm",
    ];

    let mut all_docs = Vec::new();
    let mut file_count = 0u64;
    let mut skipped = 0u64;

    let t0 = Instant::now();
    for path in &sample_files(Path::new(&hermes_path), SAMPLE_FILES) {
        if file_count >= SAMPLE_FILES as u64 { break; }
        let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
        if binary_exts.contains(&ext) { skipped += 1; continue; }
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => { skipped += 1; continue; }
        };
        if content.trim().is_empty() { skipped += 1; continue; }
        file_count += 1;
        let rel = path.strip_prefix(&hermes_path).unwrap_or(path);
        let base_id = rel.to_string_lossy().replace(['/', '\\', '.'], "-");
        let docs = chunker.chunk_to_docs(path, &content, &base_id, HashMap::new());
        all_docs.extend(docs);
    }
    let ingest_t = t0.elapsed();

    eprintln!("Ingested: {file_count} files, {} chunks, {:.2}s, {skipped} skipped",
        all_docs.len(), ingest_t.as_secs_f64());

    // ── Embed ──
    let t0 = Instant::now();
    for doc in &mut all_docs {
        doc.embedding = embed_text(&doc.text);
    }
    let embed_t = t0.elapsed();
    eprintln!("Embed: {} docs, {:.2}s", all_docs.len(), embed_t.as_secs_f64());
    let doc_count = all_docs.len();

    // ── Index build ──
    eprintln!("\n── Index Build ──");
    let (bf, t_bf) = {
        let t = Instant::now();
        let mut idx = BruteForceIndex::new_with(Metric::Cosine, false, false);
        idx.insert(&all_docs);
        (idx, t.elapsed())
    };
    eprintln!("  BruteForce: {:>8.2} ms", t_bf.as_secs_f64() * 1000.0);

    let (hnsw, t_hnsw) = {
        let t = Instant::now();
        let mut idx = HnswIndex::new(HnswConfig {
            m: 16, ef_construction: 200, ef_search: 50,
            metric: Metric::Cosine, flat_embeddings: false,
            sq: false, sq_rescore: false,
        });
        idx.insert(&all_docs);
        (idx, t.elapsed())
    };
    eprintln!("  HNSW:       {:>8.2} ms", t_hnsw.as_secs_f64() * 1000.0);

    let (ivfpq, t_ivfpq) = {
        let t = Instant::now();
        let config = IvfPqConfig {
            n_list: 256.min(doc_count), m_subspaces: 8, n_probe: 8,
            metric: Metric::Cosine, rerank_enabled: false,
        };
        let _ = config.validate();
        let mut idx = IvfPqIndex::new(config);
        idx.insert(&all_docs);
        (idx, t.elapsed())
    };
    drop(all_docs);
    eprintln!("  IVF-PQ:     {:>8.2} ms", t_ivfpq.as_secs_f64() * 1000.0);

    // ── Query bench ──
    let queries: Vec<Vec<f32>> = (0..QUERIES)
        .map(|_| (0..DIM).map(|i| {
            let v = ((i as u64 * 6364136223846793005).wrapping_mul(QUERIES as u64 + 1) >> 33) as f64 / 1e9;
            v as f32
        }).collect())
        .collect();

    eprintln!("\n── Query Benchmark ({QUERIES} queries, top_k={TOP_K}) ──");

    fn bench(label: &str, index: &impl Index, bf: &BruteForceIndex,
             queries: &[Vec<f32>], top_k: usize, build_ms: f64) {
        let mut latencies = Vec::with_capacity(queries.len());
        let mut recall = 0.0;
        for q in queries {
            let t = Instant::now();
            let results = index.search(q, top_k);
            latencies.push(t.elapsed().as_secs_f64() * 1_000_000.0);
            let bf_results = bf.search(q, top_k);
            let bf_ids: std::collections::HashSet<&str> =
                bf_results.iter().map(|r| r.document.id.as_str()).collect();
            let hits = results.iter().filter(|r| bf_ids.contains(r.document.id.as_str())).count();
            recall += hits as f64 / top_k as f64;
        }
        latencies.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = latencies[latencies.len() / 2];
        let p95 = latencies[(latencies.len() as f64 * 0.95) as usize];
        let mean = latencies.iter().sum::<f64>() / latencies.len() as f64;
        recall /= queries.len() as f64;
        eprintln!("  {:<12} build={:>7.2}ms  p50={:>8.1}μs  p95={:>8.1}μs  mean={:>8.1}μs  recall={:>5.1}%",
            label, build_ms, p50, p95, mean, recall * 100.0);
    }

    bench("BruteForce", &bf, &bf, &queries, TOP_K, t_bf.as_secs_f64() * 1000.0);
    bench("HNSW",       &hnsw, &bf, &queries, TOP_K, t_hnsw.as_secs_f64() * 1000.0);
    bench("IVF-PQ",     &ivfpq, &bf, &queries, TOP_K, t_ivfpq.as_secs_f64() * 1000.0);

    // Speedup vs BF
    eprintln!("\n── Speedup vs BruteForce ──");
    let bf_mean = latencies_for(&bf, &queries, TOP_K);
    for (label, idx) in [("HNSW", &hnsw as &dyn Index), ("IVF-PQ", &ivfpq as &dyn Index)] {
        let m = latencies_for(idx, &queries, TOP_K);
        let speedup = bf_mean / m;
        eprintln!("  {label:<12} {:.1}× {}", speedup,
            if speedup > 1.0 { "faster than BF" } else { "slower than BF" });
    }
    eprintln!("\n=== Done ===");
}

fn latencies_for(index: &dyn Index, queries: &[Vec<f32>], top_k: usize) -> f64 {
    let mut sum = 0.0;
    for q in queries {
        let t = Instant::now();
        index.search(q, top_k);
        sum += t.elapsed().as_secs_f64() * 1_000_000.0;
    }
    sum / queries.len() as f64
}
