// Test secuencial: ingerir hermes-agent, probar cada índice UNO POR UNO
// con liberación de memoria entre cada uno y telemetría de RSS.
//!
//! Uso: cargo run --release --example test_sequential
//!      RUST_LOG=info cargo run --release --example test_sequential
//!   (con --features chunker-syntax para chunking con tree-sitter)
//!
//! Nota: jemalloc desactivado para diagnóstico — ver issue del chunker

use dogma_vdb::distance::Metric;
use dogma_vdb::doc::Document;
use dogma_vdb::index::{
    BruteForceIndex, HnswConfig, HnswIndex, Index, IvfPqConfig, IvfPqIndex,
};
use dogma_vdb::memory;
use dogma_vdb::smart_chunker::SmartChunker;
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;
use std::panic;
use std::time::Instant;

// ── Config ──────────────────────────────────────────────────────────────
const DIM: usize = 64;
const TOP_K: usize = 10;
const QUERIES: usize = 20;
const SAMPLE_FILES: usize = 200;

// ── Telemetría: RSS ─────────────────────────────────────────────────────
fn rss_kb() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines().find_map(|l| {
                if l.starts_with("VmRSS:") {
                    l.split_whitespace()
                        .nth(1)
                        .and_then(|v| v.parse().ok())
                } else {
                    None
                }
            })
        })
        .unwrap_or(0)
}

fn print_mem(label: &str) {
    let rss = rss_kb();
    eprintln!("  📊 RSS {label}: {:.1} MB", rss as f64 / 1024.0);
}

fn rss_delta(before: u64, label: &str) {
    let after = rss_kb();
    let delta = after as i64 - before as i64;
    eprintln!(
        "  📊 RSS tras {label}: {:.1} MB (Δ{}{:.1} MB)",
        after as f64 / 1024.0,
        if delta >= 0 { "+" } else { "" },
        delta as f64 / 1024.0
    );
}

// ── Embedder hash-based (sin dependencias externas) ─────────────────────
fn embed_text(text: &str) -> Vec<f32> {
    use std::hash::{Hash, Hasher};
    (0..DIM)
        .map(|i| {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            text.hash(&mut h);
            i.hash(&mut h);
            (h.finish() as f64 / u64::MAX as f64) as f32
        })
        .collect()
}

// ── Colector de archivos ───────────────────────────────────────────────
fn collect_files(root: &Path, max: usize) -> Vec<std::path::PathBuf> {
    let mut files = Vec::new();
    let mut dirs = vec![root.to_path_buf()];
    while let Some(dir) = dirs.pop() {
        if files.len() >= max {
            break;
        }
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            if files.len() >= max {
                break;
            }
            let path = entry.path();
            if path
                .file_name()
                .and_then(|s| s.to_str())
                .map_or(false, |s| s.starts_with('.'))
            {
                continue;
            }
            if path.is_dir() {
                dirs.push(path);
            } else if path.is_file() {
                files.push(path);
            }
        }
    }
    files
}

// ── Benchmark de un índice ─────────────────────────────────────────────
fn bench_index(
    label: &str,
    docs: &[Document],
    queries: &[Vec<f32>],
    build_index: impl Fn() -> Box<dyn Index>,
) {
    eprintln!("\n── {label} ──");
    let rss_before = rss_kb();

    // Build
    let t0 = Instant::now();
    let mut index = build_index();
    index.insert(docs);
    let build_time = t0.elapsed();
    rss_delta(rss_before, &format!("build {label}"));

    // Warmup
    if !queries.is_empty() {
        for _ in 0..3 {
            index.search(&queries[0], TOP_K);
        }
    }

    // Query latency
    let mut latencies = Vec::with_capacity(QUERIES);
    let t0 = Instant::now();
    for i in 0..QUERIES {
        let q = &queries[i % queries.len()];
        let start = Instant::now();
        index.search(q, TOP_K);
        latencies.push(start.elapsed().as_secs_f64() * 1_000_000.0);
    }
    let total_s = t0.elapsed().as_secs_f64();
    let mean_us = if !latencies.is_empty() {
        latencies.iter().sum::<f64>() / latencies.len() as f64
    } else {
        0.0
    };

    eprintln!(
        "  Build: {:.3}s  |  Query: {:.0} us mean  |  QPS: {:.0}",
        build_time.as_secs_f64(),
        mean_us,
        QUERIES as f64 / total_s
    );

    // Liberar
    drop(index);
    rss_delta(rss_before, &format!("drop {label}"));
    eprintln!("  → Memoria liberada correctamente");
}

// ── Main ───────────────────────────────────────────────────────────────
fn main() {
    let hermes_path = std::env::var("HERMES_AGENT_PATH").unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/home/arggil".into());
        format!("{home}/Documents/DEV-WORKSPACE/hermes-agent")
    });

    eprintln!("═══════════════════════════════════════════════════");
    eprintln!("  dogma-vdb Test Secuencial — hermes-agent");
    eprintln!("═══════════════════════════════════════════════════");
    print_mem("inicio");

    // ── 1. Colectar archivos ──────────────────────────────────────────
    eprintln!("\n── [1/4] Colectando archivos ──");
    let files = collect_files(Path::new(&hermes_path), SAMPLE_FILES);
    eprintln!("  {} archivos sampleados de {}", files.len(), hermes_path);
    print_mem("tras colectar");

    // ── 2. Chunking ────────────────────────────────────────────────────
    eprintln!("\n── [2/4] Chunking ──");
    let chunker = SmartChunker::default();
    let binary_exts: &[&str] = &[
        "png", "jpg", "jpeg", "gif", "ico", "woff2", "woff", "ttf", "otf",
        "eot", "pdf", "zip", "gz", "pyc", "mp3", "mp4", "webm",
    ];

    let t0 = Instant::now();
    let mut all_docs = Vec::new();
    let mut file_count = 0u64;
    let mut skipped = 0u64;
    for path in &files {
        // MemoryGuard periódico cada 10 archivos
        file_count += 1;

        // LOG POR ARCHIVO (flush forzado para ver el último antes del crash)
        eprintln!("  📄 [{file_count}/{}] {:?}", files.len(),
            path.strip_prefix(&hermes_path).unwrap_or(path));
        std::io::stderr().flush().ok();

        if file_count % 10 == 0 {
            eprintln!("  📊 {}/{} archivos, {} chunks, RSS: {:.1} MB",
                file_count, files.len(), all_docs.len(),
                rss_kb() as f64 / 1024.0);
            std::io::stderr().flush().ok();
            if let Err(e) = memory::ensure_memory() {
                eprintln!("  ❌ MemoryGuard detuvo chunking en archivo {}/{}: {e}",
                    file_count, files.len());
                std::io::stderr().flush().ok();
                break;
            }
        }
        let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
        if binary_exts.contains(&ext) {
            skipped += 1;
            continue;
        }
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => {
                skipped += 1;
                continue;
            }
        };
        if content.trim().is_empty() {
            skipped += 1;
            continue;
        }
        let rel = path.strip_prefix(&hermes_path).unwrap_or(path);
        let base_id = rel.to_string_lossy().replace(['/', '\\', '.'], "-");
        
        // Wrapped en catch_unwind para capturar panics
        let docs_result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            chunker.chunk_to_docs(path, &content, &base_id, HashMap::new())
        }));
        let docs = match docs_result {
            Ok(d) => d,
            Err(e) => {
                eprintln!("  ❌ PANIC en chunk_to_docs para {:?}: {:?}", rel, e);
                std::io::stderr().flush().ok();
                Vec::new()
            }
        };
        all_docs.extend(docs);
    }
    let ingest_t = t0.elapsed();
    eprintln!(
        "  {} archivos procesados, {} chunks, {} saltados, {:.2}s",
        files.len(),
        all_docs.len(),
        skipped,
        ingest_t.as_secs_f64()
    );
    print_mem("tras chunking");

    // ── 3. Embedding ───────────────────────────────────────────────────
    eprintln!("\n── [3/4] Embedding ({DIM}-dim hash) ──");
    if let Err(e) = memory::ensure_memory() {
        eprintln!("  ❌ Memory guard detuvo embedding: {e}");
        return;
    }
    let t0 = Instant::now();
    for doc in &mut all_docs {
        doc.embedding = embed_text(&doc.text);
    }
    let embed_t = t0.elapsed();
    eprintln!(
        "  {} docs embebidos en {:.2}s",
        all_docs.len(),
        embed_t.as_secs_f64()
    );
    print_mem("tras embedding");

    // ── 4. Queries de prueba ───────────────────────────────────────────
    let queries: Vec<Vec<f32>> = (0..QUERIES)
        .map(|_| {
            (0..DIM)
                .map(|i| {
                    let v = ((i as u64 * 6364136223846793005)
                        .wrapping_mul(QUERIES as u64 + 1)
                        >> 33) as f64
                        / 1e9;
                    v as f32
                })
                .collect()
        })
        .collect();

    // ── 5. Índices UNO POR UNO ─────────────────────────────────────────
    eprintln!("\n── [4/4] Benchmarks secuenciales ──");

    // 5a. BruteForce
    bench_index("BruteForce", &all_docs, &queries, || {
        Box::new(BruteForceIndex::new(Metric::Cosine))
    });
    print_mem("post-BF");

    // 5b. BruteForce + SQ
    bench_index("BruteForce+SQ", &all_docs, &queries, || {
        Box::new(BruteForceIndex::new_with(Metric::Cosine, true, false))
    });
    print_mem("post-BF+SQ");

    // 5c. HNSW (ef=50)
    bench_index("HNSW M=16 ef=50", &all_docs, &queries, || {
        Box::new(HnswIndex::new(HnswConfig {
            m: 16,
            ef_construction: 100,
            ef_search: 50,
            metric: Metric::Cosine,
            flat_embeddings: false,
            sq: false,
            sq_rescore: false,
        }))
    });
    print_mem("post-HNSW");

    // 5d. HNSW (ef=200, alta precisión)
    bench_index("HNSW M=16 ef=200", &all_docs, &queries, || {
        Box::new(HnswIndex::new(HnswConfig {
            m: 16,
            ef_construction: 200,
            ef_search: 200,
            metric: Metric::Cosine,
            flat_embeddings: false,
            sq: false,
            sq_rescore: false,
        }))
    });
    print_mem("post-HNSW-hr");

    // 5e. HNSW + SQ + Rescore
    bench_index("HNSW+SQ+Rescore", &all_docs, &queries, || {
        Box::new(HnswIndex::new(HnswConfig {
            m: 16,
            ef_construction: 100,
            ef_search: 50,
            metric: Metric::Cosine,
            flat_embeddings: false,
            sq: true,
            sq_rescore: true,
        }))
    });
    print_mem("post-HNSW-SQ");

    // 5f. IVF-PQ
    let n_list = (all_docs.len() / 4).max(4).min(256);
    bench_index("IVF-PQ", &all_docs, &queries, || {
        Box::new(IvfPqIndex::new(IvfPqConfig {
            n_list,
            m_subspaces: 8,
            n_probe: 8,
            metric: Metric::Cosine,
            ..Default::default()
        }))
    });
    print_mem("post-IVF-PQ");

    // Liberar todo
    drop(all_docs);
    drop(queries);
    print_mem("final (todo liberado)");

    eprintln!("\n═══════════════════════════════════════════════════");
    eprintln!("  ✅ Test completado sin crashes");
    eprintln!("═══════════════════════════════════════════════════");
}
