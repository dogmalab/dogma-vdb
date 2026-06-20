use dogma_vdb::smart_chunker::SmartChunker;
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

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
    let root = std::env::args().nth(1).expect("need path");
    let chunker = SmartChunker::default();
    let binary_exts: &[&str] = &[
        "png", "jpg", "jpeg", "gif", "ico", "woff2", "woff", "ttf", "otf", "eot", "pdf", "zip",
        "gz", "pyc", "mp3", "mp4", "webm",
    ];

    let files = collect_files(Path::new(&root));
    eprintln!("FILES: {}", files.len());

    let mut all_docs = Vec::new();
    let start = Instant::now();

    for (idx, path) in files.iter().enumerate() {
        let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
        if binary_exts.contains(&ext) {
            continue;
        }

        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => continue,
        };
        if content.trim().is_empty() {
            continue;
        }

        let rel = path.strip_prefix(&root).unwrap_or(path);
        let base_id = rel.to_string_lossy().replace(['/', '\\', '.'], "-");
        let docs = chunker.chunk_to_docs(path, &content, &base_id, HashMap::new());
        all_docs.extend(docs);

        if idx % 500 == 0 && idx > 0 {
            let elapsed = start.elapsed().as_secs_f64();
            let ratio = all_docs.len() as f64 / idx as f64;
            eprintln!(
                "[{idx}/{}] {} docs, {:.1} docs/file, {:.1}s",
                files.len(),
                all_docs.len(),
                ratio,
                elapsed
            );
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    eprintln!(
        "DONE: {} files, {} chunks, {:.2}s",
        files.len(),
        all_docs.len(),
        elapsed
    );
    eprintln!("MEM: docs={:.2}MB", all_docs.len() * 2000 / 1024 / 1024);
}
