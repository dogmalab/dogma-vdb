//! Test the full ingest pipeline from bench_hermes: walk, read, chunk all files
use dogma_vdb::smart_chunker::SmartChunker;
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

fn rss_kb() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines().find_map(|l| {
                if l.starts_with("VmRSS:") {
                    l.split_whitespace().nth(1).and_then(|v| v.parse().ok())
                } else {
                    None
                }
            })
        })
        .unwrap_or(0)
}

fn collect_files(root: &Path) -> Vec<std::path::PathBuf> {
    let mut files = Vec::new();
    let mut dirs = vec![root.to_path_buf()];
    while let Some(dir) = dirs.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path
                .file_name()
                .and_then(|s| s.to_str())
                .map_or(false, |s| s.starts_with('.'))
            {
                continue;
            }
            if path.is_dir() {
                dirs.push(path);
            } else if path.is_file() {
                files.push(path);
            }
        }
    }
    files
}

fn main() {
    eprintln!("[TEST] RSS start: {} MB", rss_kb() / 1024);
    std::io::stderr().flush().ok();

    let root = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/arggil/Documents/DEV-WORKSPACE/hermes-agent".into());
    let files = collect_files(Path::new(&root));
    eprintln!("[TEST] Files: {}, RSS: {} MB", files.len(), rss_kb() / 1024);
    std::io::stderr().flush().ok();

    let chunker = SmartChunker::default();
    let binary_exts: &[&str] = &[
        "png", "jpg", "jpeg", "gif", "ico", "woff2", "woff", "ttf", "otf", "eot", "pdf", "zip",
        "gz", "pyc", "mp3", "mp4", "webm",
    ];

    let mut all_docs = Vec::new();
    let mut skipped = 0u64;

    for (i, path) in files.iter().enumerate() {
        let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
        if binary_exts.contains(&ext) {
            skipped += 1;
            continue;
        }

        if i >= 100 {
            eprintln!("[TEST] Stopping after 100 files to check RSS");
            break;
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

        let rel = path.strip_prefix(&root).unwrap_or(path);
        let base_id = rel.to_string_lossy().replace(['/', '\\', '.'], "-");
        let docs = chunker.chunk_to_docs(path, &content, &base_id, HashMap::new());
        all_docs.extend(docs);
    }

    eprintln!(
        "[TEST] After {} files: {} docs, RSS: {} MB",
        100,
        all_docs.len(),
        rss_kb() / 1024
    );
    std::io::stderr().flush().ok();
}
