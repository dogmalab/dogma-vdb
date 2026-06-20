//! dogma-vdb-rag — RAG pipeline for dogma-vdb.
//!
//! Ingests source directories (chunk + embed + index),
//! queries with semantic search, and watches for file changes.
//!
//! # Usage
//!
//! ```bash
//! # Ingest a source directory
//! dogma-vdb-rag ingest ./src --output docs.vdb
//!
//! # Query the collection
//! dogma-vdb-rag query docs.vdb "how does HNSW work?"
//!
//! # Watch for changes and auto-reindex
//! dogma-vdb-rag watch ./src --output docs.vdb
//!
//! # Show collection info
//! dogma-vdb-rag info docs.vdb
//! ```

mod ingest;
mod query;
mod watch;

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "dogma-vdb-rag", about = "RAG pipeline for dogma-vdb")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
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

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();

    let cli = Cli::parse();
    match cli.command {
        Commands::Ingest {
            source,
            output,
            extensions,
            index,
            metric,
            hash,
            dim,
        } => ingest::run_ingest(&source, &output, &extensions, &index, &metric, hash, dim),
        Commands::Query {
            collection,
            query,
            k,
            index,
            metric,
            hash,
            dim,
            hybrid,
        } => query::run_query(&collection, &query, k, &index, &metric, hash, dim, hybrid),
        Commands::Watch {
            source,
            output,
            extensions,
            index,
            metric,
            hash,
            dim,
            debounce_ms,
            no_initial,
        } => watch::run_watch(
            &source,
            &output,
            &extensions,
            &index,
            &metric,
            hash,
            dim,
            debounce_ms,
            !no_initial,
        ),
        Commands::Info {
            collection,
            index,
            metric,
        } => query::run_info(&collection, &index, &metric),
    }
}
