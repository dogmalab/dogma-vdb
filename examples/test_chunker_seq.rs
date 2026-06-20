// Test: chunk .md + small .py to check hang after N files
use dogma_vdb::doc::Document;
use dogma_vdb::smart_chunker::SmartChunker;
use std::collections::HashMap;
use std::path::Path;

fn main() {
    eprintln!("═══ Sequential Chunker Test ═══");

    let root = Path::new("/home/arggil/Documents/DEV-WORKSPACE/hermes-agent");
    let chunker = SmartChunker::default();
    let mut all_docs: Vec<Document> = Vec::new();

    // 15 .md files (RELEASE_*.md)
    for i in 0..15 {
        let path = root.join(format!("RELEASE_v0.{}.md", i));
        if !path.exists() {
            continue;
        }
        let content = std::fs::read_to_string(&path).unwrap();
        let base_id = format!("release-{}", i);
        let docs = chunker.chunk_to_docs(&path, &content, &base_id, HashMap::new());
        let n = docs.len();
        all_docs.extend(docs);
        eprintln!(
            "  [{}/15] {:?} -> {} chunks (total: {})",
            i + 1,
            path.file_name().unwrap(),
            n,
            all_docs.len()
        );
    }

    // batch_runner.py (54 KB)
    let path = root.join("batch_runner.py");
    if path.exists() {
        let content = std::fs::read_to_string(&path).unwrap();
        let docs = chunker.chunk_to_docs(&path, &content, "batch-runner", HashMap::new());
        let n = docs.len();
        all_docs.extend(docs);
        eprintln!(
            "  [py] batch_runner.py -> {} chunks (total: {})",
            n,
            all_docs.len()
        );
    }

    // cli.py (532 KB)
    eprintln!("  Processing cli.py...");
    let path = root.join("cli.py");
    let content = std::fs::read_to_string(&path).unwrap();
    eprintln!(
        "  Read cli.py: {} bytes, {} lines",
        content.len(),
        content.lines().count()
    );
    let docs = chunker.chunk_to_docs(&path, &content, "cli-py", HashMap::new());
    let n = docs.len();
    all_docs.extend(docs);
    eprintln!("  cli.py OK -> {} chunks (total: {})", n, all_docs.len());

    eprintln!("═══ Test OK: {} chunks total ═══", all_docs.len());
}
