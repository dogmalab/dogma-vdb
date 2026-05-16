//! dogma-vdb-cli — command-line interface for .vdb collections.
//!
//! # Examples
//!
//! ```bash
//! # Show collection info
//! dogma-vdb-cli info my_data.vdb
//!
//! # List documents
//! dogma-vdb-cli list my_data.vdb
//!
//! # Search (embedding as comma-separated floats)
//! dogma-vdb-cli query my_data.vdb "0.1,0.2,0.3" --k 10
//!
//! # Ingest a document
//! dogma-vdb-cli ingest my_data.vdb --id doc-1 --text "Hello, world!"
//!
//! # Delete documents
//! dogma-vdb-cli delete my_data.vdb doc-1 doc-2
//! ```

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use dogma_vdb::prelude::*;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "dogma-vdb-cli", about = "Portable vector database CLI")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Show collection statistics
    Info {
        /// Path to the .vdb file
        path: PathBuf,
        /// Index type (bruteforce, hnsw)
        #[arg(long, default_value = "bruteforce")]
        index_type: String,
        /// Distance metric (cosine, dot, euclidean)
        #[arg(long, default_value = "cosine")]
        metric: String,
    },
    /// List all documents in a collection
    List {
        /// Path to the .vdb file
        path: PathBuf,
        /// Index type
        #[arg(long, default_value = "bruteforce")]
        index_type: String,
        /// Distance metric
        #[arg(long, default_value = "cosine")]
        metric: String,
    },
    /// Search a collection
    Query {
        /// Path to the .vdb file
        path: PathBuf,
        /// Query embedding as comma-separated floats (e.g. "0.1,0.2,0.3")
        embedding: String,
        /// Number of results to return
        #[arg(long, default_value_t = 10)]
        k: usize,
        /// Index type
        #[arg(long, default_value = "bruteforce")]
        index_type: String,
        /// Distance metric
        #[arg(long, default_value = "cosine")]
        metric: String,
    },
    /// Ingest a document into a collection
    Ingest {
        /// Path to the .vdb file
        path: PathBuf,
        /// Document ID
        #[arg(long)]
        id: Option<String>,
        /// Document text content
        #[arg(long)]
        text: Option<String>,
        /// Path to a file to read as document content
        #[arg(long)]
        file: Option<String>,
        /// Embedding as comma-separated floats (optional)
        #[arg(long)]
        embedding: Option<String>,
        /// Index type
        #[arg(long, default_value = "bruteforce")]
        index_type: String,
        /// Distance metric
        #[arg(long, default_value = "cosine")]
        metric: String,
    },
    /// Delete documents by ID
    Delete {
        /// Path to the .vdb file
        path: PathBuf,
        /// Document ID(s) to delete
        ids: Vec<String>,
        /// Index type
        #[arg(long, default_value = "bruteforce")]
        index_type: String,
        /// Distance metric
        #[arg(long, default_value = "cosine")]
        metric: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Info {
            path,
            index_type,
            metric,
        } => cmd_info(&path, &index_type, &metric),
        Commands::List {
            path,
            index_type,
            metric,
        } => cmd_list(&path, &index_type, &metric),
        Commands::Query {
            path,
            embedding,
            k,
            index_type,
            metric,
        } => cmd_query(&path, &embedding, k, &index_type, &metric),
        Commands::Ingest {
            path,
            id,
            text,
            file,
            embedding,
            index_type,
            metric,
        } => cmd_ingest(&path, id, text, file, embedding, &index_type, &metric),
        Commands::Delete {
            path,
            ids,
            index_type,
            metric,
        } => cmd_delete(&path, &ids, &index_type, &metric),
    }
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

fn open_collection(path: &PathBuf, index_type: &str, metric: &str) -> Result<Collection> {
    Collection::open_with(path, index_type, metric)
        .with_context(|| format!("failed to open collection at {}", path.display()))
}

fn cmd_info(path: &PathBuf, index_type: &str, metric: &str) -> Result<()> {
    let col = open_collection(path, index_type, metric)?;
    let path_str = path.to_string_lossy();
    println!("Collection: {path_str}");
    println!("  Name:      {}", col.name());
    println!("  Documents: {}", col.len());
    println!("  Index:     {index_type}");
    println!("  Metric:    {metric}");
    println!("  Empty:     {}", col.is_empty());
    Ok(())
}

fn cmd_list(path: &PathBuf, index_type: &str, metric: &str) -> Result<()> {
    let col = open_collection(path, index_type, metric)?;
    if col.is_empty() {
        println!("Collection is empty.");
        return Ok(());
    }
    println!("Documents in {}:", col.name());
    for (i, doc) in col.documents().enumerate() {
        let text_preview: &str = if doc.text.len() > 60 {
            &doc.text[..57]
        } else {
            &doc.text
        };
        let dim = doc.dimension();
        let has_emb = if dim > 0 {
            format!("embedding [{dim}]")
        } else {
            "no embedding".into()
        };
        let meta = doc.metadata.len();
        println!(
            "  [{i}] id={}  {}  text=\"{text_preview}\"  metadata={meta} keys",
            doc.id, has_emb
        );
    }
    Ok(())
}

fn cmd_query(
    path: &PathBuf,
    embedding_str: &str,
    k: usize,
    index_type: &str,
    metric: &str,
) -> Result<()> {
    let col = open_collection(path, index_type, metric)?;
    if col.is_empty() {
        println!("Collection is empty — no results.");
        return Ok(());
    }
    let query = parse_embedding(embedding_str)?;
    let results = col.search(&query, k);
    if results.is_empty() {
        println!("No results found.");
        return Ok(());
    }
    println!("Top {k} results:");
    for (i, r) in results.iter().enumerate() {
        let text_preview: &str = if r.document.text.len() > 80 {
            &r.document.text[..77]
        } else {
            &r.document.text
        };
        println!(
            "  [{i}] score={:.6}  id={}  text=\"{text_preview}\"",
            r.score, r.document.id
        );
    }
    Ok(())
}

fn cmd_ingest(
    path: &PathBuf,
    id: Option<String>,
    text: Option<String>,
    file: Option<String>,
    embedding_str: Option<String>,
    index_type: &str,
    metric: &str,
) -> Result<()> {
    let mut col = open_collection(path, index_type, metric)?;

    // Resolve document ID
    let doc_id = id.unwrap_or_else(|| {
        let n = col.len() + 1;
        format!("doc-{n}")
    });

    // Resolve document text
    let doc_text: String = if let Some(t) = text {
        t
    } else if let Some(f) = file {
        std::fs::read_to_string(&f).with_context(|| format!("failed to read file {}", f))?
    } else {
        anyhow::bail!("provide --text or --file for document content");
    };

    // Optional embedding
    let mut builder = Document::builder(&doc_id, doc_text);
    if let Some(emb_str) = embedding_str {
        let emb = parse_embedding(&emb_str)?;
        builder = builder.embedding(emb);
    }

    let doc = builder.build();
    col.insert(doc)?;
    println!(
        "Inserted document '{doc_id}' into {} ({col_len} docs)",
        col.name(),
        col_len = col.len()
    );
    Ok(())
}

fn cmd_delete(path: &PathBuf, ids: &[String], index_type: &str, metric: &str) -> Result<()> {
    let mut col = open_collection(path, index_type, metric)?;
    let str_ids: Vec<&str> = ids.iter().map(|s| s.as_str()).collect();
    let deleted = col.delete(&str_ids)?;
    println!("Deleted {deleted} document(s) from {}.", col.name());
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Parse a comma-separated float string into a Vec<f32>.
fn parse_embedding(s: &str) -> Result<Vec<f32>> {
    s.split(',')
        .map(|token| {
            token
                .trim()
                .parse::<f32>()
                .with_context(|| format!("invalid float in embedding: '{token}'"))
        })
        .collect()
}
