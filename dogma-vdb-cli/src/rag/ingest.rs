//! RAG pipeline: walk directory, chunk, embed, index into Collection.

use anyhow::{Context, Result};
use dogma_vdb::doc::Document;
use dogma_vdb::embedding::Embedder as CoreEmbedder;
use dogma_vdb::error::Error as VdbError;
use dogma_vdb::prelude::*;
use dogma_vdb::smart_chunker::{SmartChunker, SmartChunkerConfig};
use dogma_vdb_embed::Embedder as FastEmbedTrait;
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

/// Binary file extensions to skip during ingestion.
const BINARY_EXTS: &[&str] = &[
    "png", "jpg", "jpeg", "gif", "ico", "woff2", "woff", "ttf", "otf", "eot", "pdf", "zip", "gz",
    "bz2", "xz", "zst", "tar", "pyc", "pyo", "so", "dll", "dylib", "exe", "bin", "mp3", "mp4",
    "webm", "avi", "mov", "ogg", "wav", "wasm",
];

/// Adapter: wraps FastEmbedder into CoreEmbedder.
pub struct FastEmbedAdapter {
    inner: dogma_vdb_embed_fastembed::FastEmbedder,
}

impl FastEmbedAdapter {
    pub fn new() -> Result<Self> {
        let inner = dogma_vdb_embed_fastembed::FastEmbedder::new()
            .map_err(|e| anyhow::anyhow!("FastEmbed init failed: {e}"))?;
        Ok(Self { inner })
    }
}

impl CoreEmbedder for FastEmbedAdapter {
    fn embed(&self, text: &str) -> std::result::Result<Vec<f32>, VdbError> {
        self.inner
            .embed(text)
            .map_err(|e| VdbError::Internal(e.to_string()))
    }

    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    fn embed_batch(&self, texts: &[&str]) -> std::result::Result<Vec<Vec<f32>>, VdbError> {
        self.inner
            .embed_batch(texts)
            .map_err(|e| VdbError::Internal(e.to_string()))
    }
}

/// A simple hash-based embedder for testing / no-ONNX scenarios.
pub struct HashEmbedder {
    dim: usize,
}

impl HashEmbedder {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl CoreEmbedder for HashEmbedder {
    fn embed(&self, text: &str) -> std::result::Result<Vec<f32>, VdbError> {
        use std::hash::{Hash, Hasher};
        Ok((0..self.dim)
            .map(|i| {
                let mut h = std::collections::hash_map::DefaultHasher::new();
                text.hash(&mut h);
                i.hash(&mut h);
                (h.finish() as f64 / u64::MAX as f64) as f32
            })
            .collect())
    }

    fn dimension(&self) -> usize {
        self.dim
    }
}

/// Walk a directory recursively, collecting non-binary file paths.
pub fn collect_files(
    root: &Path,
    extensions: &[String],
    max: Option<usize>,
) -> Vec<std::path::PathBuf> {
    let mut files = Vec::new();
    if !root.is_dir() {
        log::warn!("{:?} is not a valid directory", root);
        return files;
    }
    let mut dirs = vec![root.to_path_buf()];
    while let Some(dir) = dirs.pop() {
        if max.is_some_and(|m| files.len() >= m) {
            break;
        }
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            if max.is_some_and(|m| files.len() >= m) {
                break;
            }
            let path = entry.path();
            if path
                .file_name()
                .and_then(|s| s.to_str())
                .is_some_and(|s| s.starts_with('.') && s != ".gitignore")
            {
                continue;
            }
            if path.is_dir() {
                let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
                if matches!(
                    name,
                    ".git" | "node_modules" | "target" | ".venv" | "__pycache__"
                ) {
                    continue;
                }
                dirs.push(path);
            } else if path.is_file() {
                if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                    if extensions.is_empty() || extensions.iter().any(|e| e == ext) {
                        files.push(path);
                    }
                }
            }
        }
    }
    files
}

/// Create an embedder: FastEmbed (ONNX) or Hash fallback.
pub fn create_embedder(use_hash: bool, dim: usize) -> Result<Box<dyn CoreEmbedder>> {
    if use_hash {
        log::info!("Using hash embedder (dim={})", dim);
        Ok(Box::new(HashEmbedder::new(dim)))
    } else {
        log::info!("Initializing FastEmbed (all-MiniLM-L6-v2, 384-dim)...");
        log::info!("  (first run downloads ~90 MB ONNX model)");
        let adapter = FastEmbedAdapter::new()?;
        log::info!("FastEmbed ready (dim={})", adapter.dimension());
        Ok(Box::new(adapter))
    }
}

/// Run the full ingest pipeline.
pub fn run_ingest(
    source: &str,
    output: &str,
    extensions_str: &str,
    index_type: &str,
    metric: &str,
    use_hash: bool,
    dim: usize,
) -> Result<()> {
    let source_path = Path::new(source);
    if !source_path.is_dir() {
        anyhow::bail!("Source directory does not exist: {source}");
    }

    let output_path = Path::new(output);
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let extensions: Vec<String> = extensions_str
        .split(',')
        .map(|s| s.trim().to_lowercase())
        .filter(|s| !s.is_empty())
        .collect();

    let embedder = create_embedder(use_hash, dim)?;
    let chunker = SmartChunker::new(SmartChunkerConfig::default());

    log::info!("Collecting files from {source} (ext: {:?})", extensions);
    let t0 = Instant::now();
    let files = collect_files(source_path, &extensions, None);
    log::info!(
        "{} files found in {:.2}s",
        files.len(),
        t0.elapsed().as_secs_f64()
    );

    if files.is_empty() {
        log::warn!("No files found with the specified extensions");
        return Ok(());
    }

    log::info!("Chunking {} files...", files.len());
    let t0 = Instant::now();
    let mut all_docs: Vec<Document> = Vec::new();
    let mut skipped = 0u64;
    let mut file_count = 0u64;

    for path in &files {
        file_count += 1;
        let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
        if BINARY_EXTS.contains(&ext) {
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

        let base_id = path
            .strip_prefix(source_path)
            .unwrap_or(path)
            .to_string_lossy()
            .replace(['/', '\\', '.'], "-");

        let docs = chunker.chunk_to_docs(path, &content, &base_id, HashMap::new());
        all_docs.extend(docs);

        if file_count % 50 == 0 {
            log::info!(
                "  Processed {file_count}/{} files, {} chunks",
                files.len(),
                all_docs.len()
            );
        }
    }

    log::info!(
        "Chunking completed: {} chunks from {} files ({} skipped) in {:.2}s",
        all_docs.len(),
        files.len(),
        skipped,
        t0.elapsed().as_secs_f64(),
    );

    if all_docs.is_empty() {
        log::warn!("No chunks generated — empty collection");
        return Ok(());
    }

    log::info!("Embedding {} chunks...", all_docs.len());
    let t0 = Instant::now();
    for doc in all_docs.iter_mut() {
        let emb = embedder
            .embed(&doc.text)
            .with_context(|| format!("embedding failed for {}", doc.id))?;
        doc.embedding = emb;
    }
    log::info!("Embedding completed in {:.2}s", t0.elapsed().as_secs_f64());

    log::info!("Indexing into {output} (index={index_type}, metric={metric})...");
    let t0 = Instant::now();
    let mut col = Collection::open_with(output_path, index_type, metric)
        .with_context(|| format!("Failed to open/create collection at {output}"))?;

    let existing = col.len();
    if existing > 0 {
        log::info!(
            "Existing collection with {existing} docs — adding {} new",
            all_docs.len()
        );
    }

    col.insert_batch(&all_docs)
        .with_context(|| "Failed to insert documents")?;

    log::info!(
        "Indexing completed: {} documents in {:.2}s",
        col.len(),
        t0.elapsed().as_secs_f64(),
    );

    println!("\n=======================================");
    println!("  dogma-vdb rag ingest — COMPLETED");
    println!("=======================================");
    println!("  Source:     {source}");
    println!("  Collection: {output}");
    println!("  Files:      {}", files.len());
    println!("  Chunks:     {}", all_docs.len());
    println!("  Dimension:  {}", embedder.dimension());
    println!("  Index:      {index_type}/{metric}");
    println!("  Skipped:    {skipped}");
    println!();
    println!("  To query:");
    println!("    dogma-vdb rag query \"{output}\" <your-query>");
    println!("=======================================\n");

    Ok(())
}
