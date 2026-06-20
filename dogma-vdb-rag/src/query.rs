//! Query pipeline: embed query text → search collection → print results.

use anyhow::{Context, Result};
use dogma_vdb::config::{PerformanceProfile, QueryPipelineConfig};
use dogma_vdb::embedding::Embedder as CoreEmbedder;
use dogma_vdb::index::bm25::Bm25Index;
use dogma_vdb::prelude::*;
use std::path::Path;

/// Create embedder for query (same options as ingest).
fn create_query_embedder(use_hash: bool, dim: usize) -> Result<Box<dyn CoreEmbedder>> {
    if use_hash {
        Ok(Box::new(super::ingest::HashEmbedder::new(dim)))
    } else {
        Ok(Box::new(super::ingest::FastEmbedAdapter::new()?))
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
        .map_err(|e| anyhow::anyhow!("Error al embedizar consulta: {e}"))?;
    Ok(col.search(&query_vec, k))
}

/// Run a hybrid query (vector + BM25 + RRF) with on-disk BM25 caching.
///
/// The BM25 index is cached as `{collection}.bm25` next to the `.vdb` file.
/// It is automatically rebuilt when the document count changes.
fn hybrid_query(
    col: &Collection,
    embedder: &dyn CoreEmbedder,
    query_text: &str,
    k: usize,
) -> Result<Vec<ScoredDocument>> {
    let query_vec = embedder
        .embed(query_text)
        .map_err(|e| anyhow::anyhow!("Error al embedizar consulta: {e}"))?;

    // Determine BM25 cache path next to the .vdb file
    let bm25_path = col.path().with_extension("bm25");

    // Try loading cached BM25 index
    let bm25 = if bm25_path.exists() {
        match Bm25Index::load(&bm25_path) {
            Ok(cached) if cached.len() == col.len() => {
                log::info!("BM25 cargado desde caché ({})", bm25_path.display());
                cached
            }
            Ok(cached) => {
                log::info!(
                    "BM25 caché obsoleto ({} docs en caché vs {} en colección), reconstruyendo…",
                    cached.len(),
                    col.len()
                );
                let mut bm25 = Bm25Index::new();
                for (i, doc) in col.documents().enumerate() {
                    bm25.insert(i, &doc.text);
                }
                if let Err(e) = bm25.save(&bm25_path) {
                    log::warn!("No se pudo guardar BM25 caché: {e}");
                }
                bm25
            }
            Err(e) => {
                log::warn!("BM25 caché corrupto ({e}), reconstruyendo…");
                let mut bm25 = Bm25Index::new();
                for (i, doc) in col.documents().enumerate() {
                    bm25.insert(i, &doc.text);
                }
                if let Err(e) = bm25.save(&bm25_path) {
                    log::warn!("No se pudo guardar BM25 caché: {e}");
                }
                bm25
            }
        }
    } else {
        log::info!("Construyendo índice BM25 para búsqueda híbrida…");
        let mut bm25 = Bm25Index::new();
        for (i, doc) in col.documents().enumerate() {
            bm25.insert(i, &doc.text);
        }
        log::info!("BM25 listo ({} docs), guardando en caché…", bm25.len());
        if let Err(e) = bm25.save(&bm25_path) {
            log::warn!("No se pudo guardar BM25 caché: {e}");
        }
        bm25
    };

    let results = col.hybrid_search(
        &query_vec,
        query_text,
        Some(&bm25),
        None, // no reranker for CLI
        &QueryPipelineConfig {
            profile: PerformanceProfile::HybridProduction,
            top_k: k,
        },
    );
    Ok(results)
}

/// Print search results to stdout.
fn print_results(results: &[ScoredDocument], label: &str) {
    if results.is_empty() {
        println!("  (sin resultados)");
        return;
    }
    println!("\n── {label} ──");
    for (i, r) in results.iter().enumerate() {
        let text_preview: String = r.document.text.chars().take(100).collect();
        let text_preview = if r.document.text.len() > 100 {
            format!("{}…", text_preview)
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
        anyhow::bail!("La colección no existe: {collection}");
    }

    // Open collection
    let col = Collection::open_with(col_path, index_type, metric)
        .with_context(|| format!("Error al abrir colección {collection}"))?;

    if col.is_empty() {
        anyhow::bail!("La colección está vacía");
    }

    log::info!(
        "Colección: {} ({} docs, {} dim)",
        col.name(),
        col.len(),
        col.documents().next().map(|d| d.dimension()).unwrap_or(0),
    );

    // Create embedder
    let embedder = create_query_embedder(use_hash, dim)?;

    // Search
    let t0 = std::time::Instant::now();
    let results = if hybrid {
        log::info!("Búsqueda híbrida (vector + BM25 + RRF)");
        hybrid_query(&col, embedder.as_ref(), query_text, k)?
    } else {
        log::info!("Búsqueda vectorial pura");
        vector_query(&col, embedder.as_ref(), query_text, k)?
    };
    let elapsed = t0.elapsed();

    // Print results
    println!("\n═══════════════════════════════════════════");
    println!("  Consulta: \"{query_text}\"");
    println!("  Resultados: {}/{}", results.len(), k);
    println!("  Tiempo: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
    println!("═══════════════════════════════════════════");

    if hybrid {
        print_results(&results, "Híbrido (vector + BM25 + RRF)");
    } else {
        print_results(&results, "Vectorial");
    }
    println!();

    Ok(())
}

/// Show collection information.
pub fn run_info(collection: &str, index_type: &str, metric: &str) -> Result<()> {
    let col_path = Path::new(collection);
    if !col_path.exists() {
        anyhow::bail!("La colección no existe: {collection}");
    }

    let col = Collection::open_with(col_path, index_type, metric)?;

    let first_doc = col.documents().next();
    let dim = first_doc.map(|d| d.dimension()).unwrap_or(0);
    let metadata_keys = first_doc.map(|d| d.metadata.len()).unwrap_or(0);

    // Count by language/strategy
    let mut lang_counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for doc in col.documents() {
        let lang = doc.metadata_val("language").unwrap_or("unknown");
        *lang_counts.entry(lang).or_insert(0) += 1;
    }

    println!("\n═══════════════════════════════════════════");
    println!("  dogma-vdb-rag info");
    println!("═══════════════════════════════════════════");
    println!("  Colección: {}", col.name());
    println!("  Ruta:      {}", col_path.display());
    println!("  Documentos: {}", col.len());
    println!("  Dimensión: {}", dim);
    println!("  Índice:    {index_type}");
    println!("  Métrica:   {metric}");
    println!("  Metadatos: {metadata_keys} keys");
    println!();
    println!("── Distribución por estrategia ──");
    let mut langs: Vec<(&str, usize)> = lang_counts.into_iter().collect();
    langs.sort_by_key(|l| std::cmp::Reverse(l.1));
    for (lang, count) in &langs {
        let pct = (*count as f64 / col.len() as f64) * 100.0;
        println!("  {:<15} {:>6} ({:>5.1}%)", lang, count, pct);
    }
    println!("═══════════════════════════════════════════\n");

    Ok(())
}
