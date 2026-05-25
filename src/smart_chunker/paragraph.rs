//! Paragraph chunker — splits generic text by `\n\n` boundaries.

use crate::embedding::Embedder;
use crate::smart_chunker::SmartChunk;

/// Splits text by paragraph breaks with configurable overlap.
/// Also provides [`chunk_semantic`] for embedding‑similarity–based splitting.
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

            // Ideal byte offset, adjusted to a valid UTF-8 char boundary
            let raw_ideal = start + max_size;
            let ideal = if text.is_char_boundary(raw_ideal) {
                raw_ideal
            } else {
                // Walk back to the nearest char boundary before raw_ideal
                (start..raw_ideal)
                    .rfind(|&i| text.is_char_boundary(i))
                    .unwrap_or(raw_ideal)
            };

            // Find the last paragraph break (\n\n) within [start..ideal)
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

            // Consumed line count (w/ overlap), always on char boundaries
            let consumed_end = end.saturating_sub(self.overlap);
            let consumed_end = if text.is_char_boundary(consumed_end) {
                consumed_end
            } else {
                (start..consumed_end)
                    .rfind(|&i| text.is_char_boundary(i))
                    .unwrap_or(start)
            };
            let consumed = text[start..consumed_end]
                .lines()
                .count()
                .max(1);
            cur_line += consumed;

            // Advance `start` — NEVER let it regress or stall (infinite-loop fix).
            // Also ensure it lands on a char boundary (UTF-8 panic fix).
            let next = end.saturating_sub(self.overlap);
            start = if next > start {
                if text.is_char_boundary(next) {
                    next
                } else {
                    // Walk forward to the next char boundary (at most `end`)
                    (next + 1..=end)
                        .find(|&i| text.is_char_boundary(i))
                        .unwrap_or(end)
                }
            } else {
                // Force minimum progress to break out of any stall
                end
            };
        }
        chunks
    }

    /// Chunk text using embedding similarity: sentence → embedding → cosine
    /// distance → cut where contextual similarity drops below a threshold.
    ///
    /// Falls back to [`chunk`](Self::chunk) when the embedder returns fewer
    /// embeddings than sentences (e.g. empty batch result).
    pub fn chunk_semantic(
        &self,
        text: &str,
        max_size: usize,
        embedder: &dyn Embedder,
    ) -> Vec<SmartChunk> {
        // Split into sentences via regex
        let re = regex_lite::Regex::new(r"[.!?]\s+|[\r\n]+").unwrap();
        let sentences: Vec<&str> = re
            .split(text)
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();

        if sentences.len() < 2 {
            return self.chunk(text, max_size);
        }

        // Embed sentences in batches
        let refs: Vec<&str> = sentences.iter().copied().collect();
        let embeddings = match embedder.embed_batch(&refs) {
            Ok(e) => e,
            Err(_) => return self.chunk(text, max_size),
        };
        if embeddings.len() < 2 {
            return self.chunk(text, max_size);
        }

        // Compute cosine distances between adjacent sentences
        let mut cut_points = Vec::new();
        let mut acc_len = 0usize;
        let threshold = 0.35f32;

        for i in 0..embeddings.len() - 1 {
            let dot: f32 = embeddings[i]
                .iter()
                .zip(embeddings[i + 1].iter())
                .map(|(a, b)| a * b)
                .sum();
            let mag_a: f32 = embeddings[i].iter().map(|x| x * x).sum::<f32>().sqrt();
            let mag_b: f32 = embeddings[i + 1].iter().map(|x| x * x).sum::<f32>().sqrt();
            let cos =
                if mag_a > 0.0 && mag_b > 0.0 { dot / (mag_a * mag_b) } else { 1.0 };

            // Approximate byte length of the current sentence
            acc_len += sentences[i].len() + 1; // +1 for separator

            if cos < threshold && acc_len >= max_size / 2 {
                cut_points.push(i + 1);
                acc_len = 0;
            }
        }

        if cut_points.is_empty() {
            return self.chunk(text, max_size);
        }

        // Build chunks from cut points
        let mut chunks = Vec::with_capacity(cut_points.len() + 1);
        let mut cursor = 0;
        let mut line_cursor = 0;
        let lines: Vec<&str> = text.lines().collect();
        let total = lines.len();

        for &cp in &cut_points {
            let end = cp.min(sentences.len());
            let segment: String = sentences[cursor..end].join(" ");
            let _nlines = segment.lines().count();
            chunks.push(SmartChunk {
                text: segment,
                structure: None,
                level: 0,
                start_line: line_cursor,
                end_line: line_cursor + _nlines,
            });
            line_cursor += _nlines.max(1);
            cursor = end;
        }
        // Last segment
        if cursor < sentences.len() {
            let segment: String = sentences[cursor..].join(" ");
            let _nlines = segment.lines().count();
            chunks.push(SmartChunk {
                text: segment,
                structure: None,
                level: 0,
                start_line: line_cursor,
                end_line: total,
            });
        }

        chunks
    }
}
