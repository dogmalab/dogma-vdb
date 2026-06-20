//! Query pipeline: embed query text, search collection, print results.

use anyhow::{Context, Result};
use dogma_vdb::config::{PerformanceProfile, QueryPipelineConfig};
use dogma_vdb::embedding::Embedder as CoreEmbedder;
use dogma_vdb::index::bm25::Bm25Index;
use dogma_vdb::prelude::*;
use std::path::Path;

use super::ingest::{FastEmbedAdapter, HashEmbedder};

/// Create embedder for query (same options as ingest).
fn create_query_embedder(use_hash: bool, dim: usize) -> Result<Box<dyn CoreEmbedder>> {
    if use_hash {
        Ok(Box::new(HashEmbedder::new(dim)))
    } else {
        Ok(Box::new(FastEmbedAdapter::new()?))
    }
}

/// Run a vector-only query.
fn vector_query(
    col: &Collection,
    embedder: &dyn CoreEmbedder,
    query_text: &str,
    k: usize,
) -> Result<Vec<ScoredDocument>> {
    let query_vec = embedder
        .embed(query_text)
        .map_err(|e| anyhow::anyhow!("Embedding query failed: {e}"))?;
    Ok(col.search(&query_vec, k))
}

/// Run a hybrid query (vector + BM25 + RRF) with on-disk BM25 caching.
fn hybrid_query(
    col: &Collection,
    embedder: &dyn CoreEmbedder,
    query_text: &str,
    k: usize,
) -> Result<Vec<ScoredDocument>> {
    let query_vec = embedder
        .embed(query_text)
        .map_err(|e| anyhow::anyhow!("Embedding query failed: {e}"))?;

    let bm25_path = col.path().with_extension("bm25");

    let bm25 = if bm25_path.exists() {
        match Bm25Index::load(&bm25_path) {
            Ok(cached) if cached.len() == col.len() => {
                log::info!("BM25 loaded from cache ({})", bm25_path.display());
                cached
            }
            Ok(_) => {
                log::info!("BM25 cache stale, rebuilding...");
                build_bm25(col, &bm25_path)
            }
            Err(e) => {
                log::warn!("BM25 cache corrupt ({e}), rebuilding...");
                build_bm25(col, &bm25_path)
            }
        }
    } else {
        log::info!("Building BM25 index for hybrid search...");
        build_bm25(col, &bm25_path)
    };

    let results = col.hybrid_search(
        &query_vec,
        query_text,
        Some(&bm25),
        None,
        &QueryPipelineConfig {
            profile: PerformanceProfile::HybridProduction,
            top_k: k,
        },
    );
    Ok(results)
}

fn build_bm25(col: &Collection, bm25_path: &Path) -> Bm25Index {
    let mut bm25 = Bm25Index::new();
    for (i, doc) in col.documents().enumerate() {
        bm25.insert(i, &doc.text);
    }
    log::info!("BM25 ready ({} docs), saving cache...", bm25.len());
    if let Err(e) = bm25.save(bm25_path) {
        log::warn!("Failed to save BM25 cache: {e}");
    }
    bm25
}

/// Print search results to stdout.
fn print_results(results: &[ScoredDocument], label: &str) {
    if results.is_empty() {
        println!("  (no results)");
        return;
    }
    println!("\n── {label} ──");
    for (i, r) in results.iter().enumerate() {
        let text_preview: String = r.document.text.chars().take(100).collect();
        let text_preview = if r.document.text.len() > 100 {
            format!("{text_preview}...")
        } else {
            text_preview
        };
        let source = r
            .document
            .metadata_val("structure")
            .map(|s| format!(" [{s}]"))
            .unwrap_or_default();
        println!(
            "  [{:2}] score={:.4}{}  id={}",
            i + 1,
            r.score,
            source,
            r.document.id,
        );
        println!("       {}", text_preview);
    }
}

/// Run the query command.
#[allow(clippy::too_many_arguments)]
pub fn run_query(
    collection: &str,
    query_text: &str,
    k: usize,
    index_type: &str,
    metric: &str,
    use_hash: bool,
    dim: usize,
    hybrid: bool,
) -> Result<()> {
    let col_path = Path::new(collection);
    if !col_path.exists() {
        anyhow::bail!("Collection does not exist: {collection}");
    }

    let col = Collection::open_with(col_path, index_type, metric)
        .with_context(|| format!("Failed to open collection {collection}"))?;

    if col.is_empty() {
        anyhow::bail!("Collection is empty");
    }

    log::info!(
        "Collection: {} ({} docs, {} dim)",
        col.name(),
        col.len(),
        col.documents().next().map(|d| d.dimension()).unwrap_or(0),
    );

    let embedder = create_query_embedder(use_hash, dim)?;

    let t0 = std::time::Instant::now();
    let results = if hybrid {
        log::info!("Hybrid search (vector + BM25 + RRF)");
        hybrid_query(&col, embedder.as_ref(), query_text, k)?
    } else {
        log::info!("Pure vector search");
        vector_query(&col, embedder.as_ref(), query_text, k)?
    };
    let elapsed = t0.elapsed();

    println!("\n=======================================");
    println!("  Query: \"{query_text}\"");
    println!("  Results: {}/{}", results.len(), k);
    println!("  Time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
    println!("=======================================");

    if hybrid {
        print_results(&results, "Hybrid (vector + BM25 + RRF)");
    } else {
        print_results(&results, "Vector");
    }
    println!();

    Ok(())
}

/// Show collection information.
pub fn run_info(collection: &str, index_type: &str, metric: &str) -> Result<()> {
    let col_path = Path::new(collection);
    if !col_path.exists() {
        anyhow::bail!("Collection does not exist: {collection}");
    }

    let col = Collection::open_with(col_path, index_type, metric)?;

    let first_doc = col.documents().next();
    let dim = first_doc.map(|d| d.dimension()).unwrap_or(0);
    let metadata_keys = first_doc.map(|d| d.metadata.len()).unwrap_or(0);

    let mut lang_counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for doc in col.documents() {
        let lang = doc.metadata_val("language").unwrap_or("unknown");
        *lang_counts.entry(lang).or_insert(0) += 1;
    }

    println!("\n=======================================");
    println!("  dogma-vdb rag info");
    println!("=======================================");
    println!("  Collection: {}", col.name());
    println!("  Path:       {}", col_path.display());
    println!("  Documents:  {}", col.len());
    println!("  Dimension:  {}", dim);
    println!("  Index:      {index_type}");
    println!("  Metric:     {metric}");
    println!("  Metadata:   {metadata_keys} keys");
    println!();
    println!("── Distribution by language ──");
    let mut langs: Vec<(&str, usize)> = lang_counts.into_iter().collect();
    langs.sort_by_key(|l| std::cmp::Reverse(l.1));
    for (lang, count) in &langs {
        let pct = (*count as f64 / col.len() as f64) * 100.0;
        println!("  {:<15} {:>6} ({:>5.1}%)", lang, count, pct);
    }
    println!("=======================================\n");

    Ok(())
}
