//! RAG pipeline subcommand (feature = "rag").

pub mod ingest;
pub mod query;
pub mod watch;

use clap::Subcommand;

#[derive(Subcommand)]
pub enum Commands {
    /// Ingest source files into a .vdb collection (chunk + embed + index)
    Ingest {
        /// Source directory to scan
        source: String,
        /// Output .vdb file path
        #[arg(short, long, default_value = "data/rag.vdb")]
        output: String,
        /// File extensions to process (comma-separated, default: rs,md,toml,txt)
        #[arg(short, long, default_value = "rs,md,toml,txt")]
        extensions: String,
        /// Index type: bruteforce, hnsw, ivf_pq
        #[arg(long, default_value = "bruteforce")]
        index: String,
        /// Distance metric: cosine, dot, euclidean
        #[arg(long, default_value = "cosine")]
        metric: String,
        /// Use hash-based embedder instead of FastEmbed (no ONNX)
        #[arg(long)]
        hash: bool,
        /// Embedding dimension (only used with --hash, default: 64)
        #[arg(long, default_value_t = 64)]
        dim: usize,
    },
    /// Query a .vdb collection with semantic search
    Query {
        /// Path to the .vdb collection
        collection: String,
        /// Query text
        query: String,
        /// Number of results (default: 10)
        #[arg(short, long, default_value_t = 10)]
        k: usize,
        /// Index type
        #[arg(long, default_value = "bruteforce")]
        index: String,
        /// Distance metric
        #[arg(long, default_value = "cosine")]
        metric: String,
        /// Use hash embedder (must match what was used during ingest)
        #[arg(long)]
        hash: bool,
        /// Embedding dimension (only used with --hash)
        #[arg(long, default_value_t = 64)]
        dim: usize,
        /// Hybrid search (vector + BM25 + RRF)
        #[arg(long)]
        hybrid: bool,
    },
    /// Watch a source directory for changes and auto-reindex
    Watch {
        /// Source directory to watch
        source: String,
        /// Output .vdb file path
        #[arg(short, long, default_value = "data/rag.vdb")]
        output: String,
        /// File extensions to watch (comma-separated)
        #[arg(short, long, default_value = "rs,md,toml,txt")]
        extensions: String,
        /// Index type
        #[arg(long, default_value = "bruteforce")]
        index: String,
        /// Distance metric
        #[arg(long, default_value = "cosine")]
        metric: String,
        /// Use hash embedder
        #[arg(long)]
        hash: bool,
        /// Embedding dimension for hash embedder
        #[arg(long, default_value_t = 64)]
        dim: usize,
        /// Debounce interval in milliseconds
        #[arg(long, default_value_t = 500)]
        debounce_ms: u64,
        /// Skip initial scan (just watch for changes)
        #[arg(long)]
        no_initial: bool,
    },
    /// Show collection information
    Info {
        /// Path to the .vdb collection
        collection: String,
        /// Index type
        #[arg(long, default_value = "bruteforce")]
        index: String,
        /// Distance metric
        #[arg(long, default_value = "cosine")]
        metric: String,
    },
}

/// Embed a set of documents in-place using the given embedder.
pub fn embed_docs(
    docs: &mut [dogma_vdb::doc::Document],
    embedder: &dyn dogma_vdb::embedding::Embedder,
) -> anyhow::Result<()> {
    let dim = embedder.dimension();
    for doc in docs.iter_mut() {
        if doc.embedding.len() != dim {
            let emb = embedder
                .embed(&doc.text)
                .map_err(|e| anyhow::anyhow!("embedding failed for doc {}: {e}", doc.id))?;
            doc.embedding = emb;
        }
    }
    Ok(())
}
