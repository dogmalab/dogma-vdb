//! Watch service: monitor source directory for file changes, auto re-chunk,
//! re-embed, and re-index documents into the collection.

use crate::ingest::{create_embedder, embed_docs};
use anyhow::{Context, Result};
use dogma_vdb::collection::Collection;
use dogma_vdb::doc::Document;
use dogma_vdb::embedding::Embedder as CoreEmbedder;
use dogma_vdb::smart_chunker::SmartChunker;
use notify::{Config, Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Debounce state: track pending re-index and timer.
struct WatchState {
    needs_reindex: bool,
    last_index: Instant,
}

/// Re-index all documents: re-scan source, chunk, embed, then atomically
/// replace the collection.
fn do_reindex(
    col: &mut Option<Collection>,
    embedder: &dyn CoreEmbedder,
    source: &Path,
    indices: &[String],
    chunker: &SmartChunker,
    index_type: &str,
    metric: &str,
) -> Result<usize> {
    let t0 = Instant::now();

    // ── Step 1: Walk + Chunk ───────────────────────────────────────────
    let mut all_docs: Vec<Document> = Vec::new();
    let mut dirs = vec![source.to_path_buf()];
    while let Some(dir) = dirs.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
            if name.starts_with('.') && name != ".gitignore" {
                continue;
            }
            if path.is_dir() {
                if matches!(
                    name,
                    ".git" | "node_modules" | "target" | ".venv" | "__pycache__"
                ) {
                    continue;
                }
                dirs.push(path);
            } else if path.is_file() {
                let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
                if indices.is_empty() || indices.iter().any(|e| e == ext) {
                    let content = match std::fs::read_to_string(&path) {
                        Ok(c) => c,
                        Err(_) => continue,
                    };
                    if content.trim().is_empty() {
                        continue;
                    }
                    let base_id = path
                        .strip_prefix(source)
                        .unwrap_or(&path)
                        .to_string_lossy()
                        .replace(['/', '\\', '.'], "-");
                    let docs = chunker.chunk_to_docs(&path, &content, &base_id, Default::default());
                    all_docs.extend(docs);
                }
            }
        }
    }

    if all_docs.is_empty() {
        log::info!("No hay documentos para indexar");
        return Ok(0);
    }
    log::info!("{} chunks generados", all_docs.len());

    // ── Step 2: Embed ──────────────────────────────────────────────────
    log::info!("Embedding {} documentos…", all_docs.len());
    embed_docs(&mut all_docs, embedder)?;
    log::info!("Embedding completado en {:.2}s", t0.elapsed().as_secs_f64());

    // ── Step 3: Atomic replace collection ──────────────────────────────
    let col_path = col.as_ref().map(|c| c.path().to_path_buf());
    // Drop old collection (releases file lock)
    *col = None;

    if let Some(ref path) = col_path {
        // Delete old file so open_with starts fresh
        if path.exists() {
            std::fs::remove_file(path).with_context(|| format!("Error al eliminar {path:?}"))?;
        }
        // Create fresh collection and insert
        let mut new_col = Collection::open_with(path, index_type, metric)
            .with_context(|| "Error al recrear colección")?;
        new_col
            .insert_batch(&all_docs)
            .with_context(|| "Error al insertar documentos en re-index")?;
        *col = Some(new_col);
    }

    log::info!(
        "✅ Re-index completado: {} documentos en {:.2}s",
        all_docs.len(),
        t0.elapsed().as_secs_f64()
    );
    Ok(all_docs.len())
}

/// Start the watch loop. Returns when terminated (Ctrl+C).
#[allow(clippy::too_many_arguments)]
pub fn run_watch(
    source: &str,
    collection: &str,
    extensions: &str,
    index_type: &str,
    metric: &str,
    use_hash: bool,
    dim: usize,
    debounce_ms: u64,
    initial_ingest: bool,
) -> Result<()> {
    let col_path = PathBuf::from(collection);
    let source_path = PathBuf::from(source);

    if !source_path.is_dir() {
        anyhow::bail!("El directorio fuente no existe: {source}");
    }

    // Create parent dirs for collection
    if let Some(parent) = col_path.parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("No se pudo crear {parent:?}"))?;
        }
    }

    // Parse extensions
    let ext_filters: Vec<String> = extensions
        .split(',')
        .map(|s| s.trim().to_lowercase())
        .filter(|s| !s.is_empty())
        .collect();

    // Open or create collection
    let mut col: Option<Collection> = Some(
        Collection::open_with(&col_path, index_type, metric)
            .with_context(|| format!("Error al abrir/crear colección {collection}"))?,
    );

    // Create embedder
    let embedder = create_embedder(use_hash, dim)?;

    // Create chunker
    let chunker = SmartChunker::default();

    // ── Initial ingest ──────────────────────────────────────────────────
    if col.as_ref().map_or(true, |c| c.is_empty()) {
        if initial_ingest {
            log::info!("Ingestión inicial…");
            do_reindex(
                &mut col,
                embedder.as_ref(),
                &source_path,
                &ext_filters,
                &chunker,
                index_type,
                metric,
            )?;
            log::info!(
                "Ingestión inicial completada ({} docs)",
                col.as_ref().map(|c| c.len()).unwrap_or(0)
            );
        } else {
            log::info!("Colección vacía, esperando cambios…");
        }
    } else {
        log::info!(
            "Colección existente con {} documentos",
            col.as_ref().map(|c| c.len()).unwrap_or(0)
        );
    }

    // ── File watcher setup ──────────────────────────────────────────────
    log::info!(
        "👀 Vigilando: {} (cada cambio → re-chunk + re-embed + re-index)",
        source
    );
    log::info!("  Debounce: {}ms", debounce_ms);
    log::info!("  Extensiones: {:?}", ext_filters);

    let (tx, rx) = mpsc::channel::<Result<Event, notify::Error>>();
    let mut watcher = RecommendedWatcher::new(tx, Config::default())
        .with_context(|| "No se pudo crear el file watcher")?;

    watcher
        .watch(&source_path, RecursiveMode::Recursive)
        .with_context(|| format!("Error al vigilar {source}"))?;

    // Ctrl+C handler
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        log::info!("👋 Ctrl+C recibido — terminando watcher…");
        r.store(false, Ordering::SeqCst);
    })
    .map_err(|e| anyhow::anyhow!("Error al instalar Ctrl+C handler: {e}"))?;

    // Shared debounce state
    let state = Arc::new(Mutex::new(WatchState {
        needs_reindex: false,
        last_index: Instant::now(),
    }));
    let debounce = Duration::from_millis(debounce_ms);

    // ── Event loop ──────────────────────────────────────────────────────
    while running.load(Ordering::SeqCst) {
        // Check for debounced re-index
        {
            let mut st = state.lock().unwrap();
            if st.needs_reindex && st.last_index.elapsed() >= debounce {
                st.needs_reindex = false;
                st.last_index = Instant::now();
                drop(st);

                log::info!("🔄 Cambio detectado — re-indexando…");
                match do_reindex(
                    &mut col,
                    embedder.as_ref(),
                    &source_path,
                    &ext_filters,
                    &chunker,
                    index_type,
                    metric,
                ) {
                    Ok(n) => log::info!("✅ Re-index completado: {n} documentos"),
                    Err(e) => log::error!("❌ Error en re-index: {e}"),
                }
                continue;
            }
        }

        // Wait for file events (with timeout for debounce check)
        match rx.recv_timeout(Duration::from_millis(500)) {
            Ok(Ok(event)) => {
                handle_fs_event(&event, &ext_filters, &state);
            }
            Ok(Err(e)) => {
                log::warn!("Error de watcher: {e}");
            }
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                log::info!("Watcher desconectado — terminando");
                break;
            }
        }
    }

    log::info!("👋 Watch finalizado.");
    Ok(())
}

/// Handle a single filesystem event.
fn handle_fs_event(event: &Event, extensions: &[String], state: &Arc<Mutex<WatchState>>) {
    let relevant = matches!(
        event.kind,
        EventKind::Modify(_) | EventKind::Create(_) | EventKind::Remove(_)
    );
    if !relevant {
        return;
    }

    let has_relevant = event.paths.iter().any(|p| {
        let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
        extensions.is_empty() || extensions.iter().any(|e| e == ext)
    });
    if !has_relevant {
        return;
    }

    let action = match event.kind {
        EventKind::Modify(_) => "modificado",
        EventKind::Create(_) => "creado",
        EventKind::Remove(_) => "eliminado",
        _ => "?",
    };
    for p in &event.paths {
        log::info!("📄 {}: {}", action, p.display());
    }

    let mut st = state.lock().unwrap();
    st.needs_reindex = true;
}
