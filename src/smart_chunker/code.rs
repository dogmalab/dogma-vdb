//! Code chunker — splits source files by top-level definitions.
//!
//! Regex patterns are compiled **once** in the constructor.

use crate::smart_chunker::{subdivide_by_lines, SmartChunk};
use regex_lite::Regex;

/// Splits code by function/class/struct boundaries using language-specific
/// regex patterns.
///
/// If a definition exceeds `max_size`, it is subdivided by lines.
/// If no structural boundaries are found, it falls back to paragraph chunking.
#[derive(Debug)]
pub struct CodeChunker {
    /// Pre-compiled patterns, ordered by priority (lowest level first).
    patterns: Vec<Regex>,
}

impl CodeChunker {
    /// Compile all patterns once.
    pub fn new(raw: &[&str]) -> Self {
        let patterns: Vec<Regex> = raw.iter().filter_map(|p| Regex::new(p).ok()).collect();
        Self { patterns }
    }

    /// Chunk `text` into structural chunks.
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

        let lines: Vec<&str> = text.lines().collect();
        let total = lines.len();

        // Find structural boundaries: (line_index, name, level)
        let mut bounds: Vec<(usize, String, usize)> = Vec::new();

        for (i, line) in lines.iter().enumerate() {
            for (level, re) in self.patterns.iter().enumerate() {
                if let Some(caps) = re.captures(line) {
                    let name = caps
                        .get(1)
                        .map(|m| m.as_str().to_string())
                        .unwrap_or_else(|| line.trim().to_string());
                    bounds.push((i, name, level));
                    break;
                }
            }
        }

        // No structural boundaries → fall back to paragraphs
        if bounds.is_empty() {
            return paragraph_fallback(text, max_size);
        }

        let mut chunks: Vec<SmartChunk> = Vec::with_capacity(bounds.len());
        for w in bounds.windows(2) {
            let (start, ref name, level) = w[0];
            let (end, _, _) = w[1];
            chunks.push(SmartChunk {
                text: lines[start..end].join("\n"),
                structure: Some(name.clone()),
                level,
                start_line: start,
                end_line: end,
            });
        }
        // Last chunk
        if let Some(&(start, ref name, level)) = bounds.last() {
            chunks.push(SmartChunk {
                text: lines[start..].join("\n"),
                structure: Some(name.clone()),
                level,
                start_line: start,
                end_line: total,
            });
        }

        // Subdivide oversized chunks
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

/// Fallback paragraph chunking used when no code structure is detected.
fn paragraph_fallback(text: &str, max_size: usize) -> Vec<SmartChunk> {
    if text.is_empty() || text.len() <= max_size {
        return vec![SmartChunk {
            text: text.to_string(),
            structure: None,
            level: 0,
            start_line: 0,
            end_line: text.lines().count(),
        }];
    }
    let mut chunks = Vec::new();
    let mut start = 0;
    let len = text.len();
    let mut cur_line = 0;
    let lines: Vec<&str> = text.lines().collect();
    let total = lines.len();

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
        let ideal = start + max_size;
        let end = text[start..ideal]
            .rfind("\n\n")
            .map(|p| start + p + 2)
            .filter(|&e| e > start)
            .unwrap_or(ideal);

        let nlines = text[start..end].lines().count();
        chunks.push(SmartChunk {
            text: text[start..end].to_string(),
            structure: None,
            level: 0,
            start_line: cur_line,
            end_line: cur_line + nlines,
        });
        cur_line += nlines.max(1);
        start = end;
    }
    chunks
}
