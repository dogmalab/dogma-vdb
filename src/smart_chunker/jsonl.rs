//! JSONL chunker — each non-empty line is a chunk.

use crate::smart_chunker::SmartChunk;

/// Each non-empty line becomes a separate chunk.
#[derive(Debug)]
pub struct JsonLinesChunker;

impl JsonLinesChunker {
    pub fn chunk(&self, text: &str, _max_size: usize) -> Vec<SmartChunk> {
        if text.is_empty() {
            return vec![SmartChunk {
                text: String::new(),
                structure: None,
                level: 0,
                start_line: 0,
                end_line: 0,
            }];
        }
        text.lines()
            .enumerate()
            .filter(|(_, line)| !line.trim().is_empty())
            .map(|(i, line)| SmartChunk {
                text: line.trim().to_string(),
                structure: None,
                level: 0,
                start_line: i,
                end_line: i + 1,
            })
            .collect()
    }
}
