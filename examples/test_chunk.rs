//! Minimal test: chunk ONE file and measure RSS
use dogma_vdb::smart_chunker::SmartChunker;
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

fn rss_kb() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| s.lines().find_map(|l| {
            if l.starts_with("VmRSS:") {
                l.split_whitespace().nth(1).and_then(|v| v.parse().ok())
            } else { None }
        }))
        .unwrap_or(0)
}

fn main() {
    eprintln!("[test] RSS before SmartChunker: {} MB", rss_kb() / 1024);
    std::io::stderr().flush().ok();

    let chunker = SmartChunker::default();
    eprintln!("[test] RSS after SmartChunker: {} MB", rss_kb() / 1024);
    std::io::stderr().flush().ok();

    let path = std::env::args().nth(1).expect("need file path");
    let content = std::fs::read_to_string(&path).expect("read file");
    eprintln!("[test] File: {} ({} bytes)", path, content.len());
    eprintln!("[test] RSS before chunk: {} MB", rss_kb() / 1024);
    std::io::stderr().flush().ok();

    let docs = chunker.chunk_to_docs(Path::new(&path), &content, "test", HashMap::new());
    eprintln!("[test] Chunks: {}", docs.len());
    eprintln!("[test] RSS after chunk: {} MB", rss_kb() / 1024);
    std::io::stderr().flush().ok();

    for (i, doc) in docs.iter().enumerate() {
        eprintln!("  chunk[{}]: {} bytes, id={}", i, doc.text.len(), doc.id);
        if i >= 10 { eprintln!("  ... (showing first 10 only)"); break; }
    }
}
