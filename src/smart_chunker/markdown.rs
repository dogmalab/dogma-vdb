//! Markdown chunker — splits by heading boundaries.

use crate::smart_chunker::{subdivide_by_lines, SmartChunk};
use regex_lite::Regex;

/// Chunks markdown by heading boundaries (`# `, `## `, etc.).
///
/// Falls back to paragraph chunking if no headings are found.
#[derive(Debug)]
pub struct MarkdownChunker;

impl MarkdownChunker {
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

        // Find headings: ^(#+)\s+(.+)
        let re = Regex::new(r"^(#+)\s+(.+)$").unwrap();
        let mut headings: Vec<(usize, String, usize)> = Vec::new();
        for (i, line) in lines.iter().enumerate() {
            if let Some(caps) = re.captures(line) {
                let level = caps.get(1).map(|m| m.as_str().len()).unwrap_or(1);
                let title = caps.get(2).map(|m| m.as_str()).unwrap_or("");
                headings.push((i, title.to_string(), level));
            }
        }

        if headings.is_empty() {
            return paragraph_fallback(text, max_size);
        }

        let mut chunks: Vec<SmartChunk> = Vec::with_capacity(headings.len());
        for w in headings.windows(2) {
            let (start, ref title, level) = w[0];
            let (end, _, _) = w[1];
            chunks.push(SmartChunk {
                text: lines[start..end].join("\n"),
                structure: Some(title.clone()),
                level,
                start_line: start,
                end_line: end,
            });
        }
        if let Some(&(start, ref title, level)) = headings.last() {
            chunks.push(SmartChunk {
                text: lines[start..].join("\n"),
                structure: Some(title.clone()),
                level,
                start_line: start,
                end_line: total,
            });
        }

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

/// Fallback paragraph chunking.
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
