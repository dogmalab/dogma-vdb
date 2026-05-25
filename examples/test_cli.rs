// Test aislado: chunkear cli.py
use dogma_vdb::smart_chunker::{ChunkStrategy, SmartChunker};
use std::fs;

fn main() {
    eprintln!("Leyendo cli.py...");
    let content = fs::read_to_string(
        "/home/arggil/Documents/DEV-WORKSPACE/hermes-agent/cli.py"
    ).unwrap();
    eprintln!("Leído: {} bytes, {} líneas", content.len(), content.lines().count());

    let chunker = SmartChunker::default();
    eprintln!("Chunking como Python...");
    let chunks = chunker.chunk_text(&content, ChunkStrategy::Code);
    eprintln!("Chunks: {}", chunks.len());
    for (i, c) in chunks.iter().enumerate() {
        eprintln!("  chunk {i}: {} bytes, struct={:?}", c.text.len(), c.structure);
    }
    eprintln!("✅ OK!");
}
