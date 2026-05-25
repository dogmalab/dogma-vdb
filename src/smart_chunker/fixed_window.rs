//! Fixed‑window chunker — splits any text into fixed‑sized byte windows.
//!
//! Replaces the previous Markdown, JSONL, and plain-text strategies.
//! No structural awareness (no heading detection, no line‑by‑line).
//! Just a safe, simple sliding window over bytes with UTF‑8 boundary
//! protection and configurable overlap.
//!
//! ### UTF‑8 safety
//!
//! Every boundary computation uses `.is_char_boundary()` to guarantee
//! zero panics on multi‑byte character sequences.

use crate::smart_chunker::{subdivide_by_lines, SmartChunk};

/// Splits text into fixed‑size byte windows with configurable overlap.
///
/// # Panics
///
/// Never — all slicing is guarded by `is_char_boundary()`.
#[derive(Debug)]
pub struct FixedWindowChunker {
    overlap: usize,
}

impl FixedWindowChunker {
    pub fn new(overlap: usize) -> Self {
        Self { overlap }
    }

    /// Chunk `text` into windows of at most `max_size` bytes.
    pub fn chunk(&self, text: &str, max_size: usize) -> Vec<SmartChunk> {
        if text.is_empty() {
            return vec![SmartChunk {
                text: String::new(),
                structure: None,
                level: 0,
                start_line: 0,
                end_line: 0,
            }];
        }

        if text.len() <= max_size {
            return vec![SmartChunk {
                text: text.to_string(),
                structure: None,
                level: 0,
                start_line: 0,
                end_line: text.lines().count(),
            }];
        }

        let lines: Vec<&str> = text.lines().collect();
        let total = lines.len();
        let len = text.len();

        let mut chunks = Vec::new();
        let mut start = 0;
        let mut cur_line = 0;

        while start < len {
            if start + max_size >= len {
                chunks.push(SmartChunk {
                    text: text[start..].to_string(),
                    structure: None,
                    level: 0,
                    start_line: cur_line,
                    end_line: total,
                });
                break;
            }

            // Ideal byte offset, snapped to the nearest UTF‑8 char boundary
            let raw_ideal = start + max_size;
            let ideal = if text.is_char_boundary(raw_ideal) {
                raw_ideal
            } else {
                (start..raw_ideal)
                    .rfind(|&i| text.is_char_boundary(i))
                    .unwrap_or(raw_ideal)
            };

            let nlines = text[start..ideal].lines().count();
            chunks.push(SmartChunk {
                text: text[start..ideal].to_string(),
                structure: None,
                level: 0,
                start_line: cur_line,
                end_line: cur_line + nlines,
            });

            cur_line += nlines.max(1);

            // Advance with overlap, always on a char boundary
            let next = ideal.saturating_sub(self.overlap);
            start = if next > start {
                if text.is_char_boundary(next) {
                    next
                } else {
                    (next + 1..=ideal)
                        .find(|&i| text.is_char_boundary(i))
                        .unwrap_or(ideal)
                }
            } else {
                // Force progress to avoid infinite loop
                ideal
            };
        }

        // Subdivide any chunk that still exceeds max_size (unlikely but safe)
        let mut out = Vec::with_capacity(chunks.len());
        for c in chunks {
            if c.text.len() > max_size {
                out.extend(subdivide_by_lines(&c, max_size));
            } else {
                out.push(c);
            }
        }
        out
    }
}
