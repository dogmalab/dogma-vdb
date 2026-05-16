//! Paragraph chunker — splits generic text by `\n\n` boundaries.

use crate::smart_chunker::SmartChunk;

/// Splits text by paragraph breaks with configurable overlap.
#[derive(Debug)]
pub struct ParagraphChunker {
    overlap: usize,
}

impl ParagraphChunker {
    pub fn new(overlap: usize) -> Self {
        Self { overlap }
    }

    pub fn chunk(&self, text: &str, max_size: usize) -> Vec<SmartChunk> {
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

            let consumed = text[start..end.saturating_sub(self.overlap)]
                .lines()
                .count()
                .max(1);
            cur_line += consumed;
            start = end.saturating_sub(self.overlap);
            if start >= end {
                start = end;
            }
        }
        chunks
    }
}
