//! Basic usage example for dogma-vdb.
//!
//! Run: cargo run --example basic --features watch

use dogma_vdb::prelude::*;
use std::path::Path;

fn main() -> Result<()> {
    // Create or open a collection
    let path = Path::new("/tmp/example.vdb");
    let mut col = Collection::open_with(path, "bruteforce", "cosine")?;

    // Insert documents with embeddings
    col.insert(
        Document::builder("doc1", "Rust is safe and fast")
            .embedding(vec![1.0, 0.0, 0.0])
            .metadata("lang", "en")
            .build(),
    )?;
    col.insert(
        Document::builder("doc2", "Python is easy to write")
            .embedding(vec![0.0, 1.0, 0.0])
            .metadata("lang", "en")
            .build(),
    )?;
    println!("Inserted 2 documents (total: {})", col.len());

    // Search
    let results = col.search(&[1.0, 0.0, 0.0], 5);
    println!("\nSearch results for [1.0, 0.0, 0.0]:");
    for (i, r) in results.iter().enumerate() {
        println!(
            "  [{i}] score={:.4}  id={}  text={}",
            r.score, r.document.id, r.document.text
        );
    }

    // Filtered search
    let results = col.search_filtered(&[1.0, 0.0, 0.0], 5, &|d: &Document| {
        d.metadata_val("lang") == Some("en")
    });
    println!("\nFiltered (lang=en): {} results", results.len());

    // Update
    col.update(
        Document::builder("doc1", "Rust is safe, fast, and productive")
            .embedding(vec![1.0, 0.0, 0.0])
            .metadata("lang", "en")
            .build(),
    )?;
    println!("\nUpdated doc1");

    // Delete
    let deleted = col.delete(&["doc2"])?;
    println!("Deleted {deleted} document(s)");

    // Show final state
    println!("\nFinal collection: {} document(s)", col.len());
    for d in col.documents() {
        println!("  id={} text=\"{}\"", d.id, d.text);
    }

    // Cleanup
    let _ = std::fs::remove_file(path);
    Ok(())
}
