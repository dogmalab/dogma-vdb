//! Semantic chunker — splits dense prose by embedding similarity.
//!
//! Algorithm: sentence → embedding → cosine distance → cut where
//! contextual similarity drops below a threshold.
//!
//! Designed for books, essays, and long‑form text where structural
//! markers (headings, paragraphs) are absent or unreliable.

use crate::distance::cosine;
use crate::embedding::Embedder;
use crate::smart_chunker::SmartChunk;
use regex_lite::Regex;

/// Splits text at semantic boundaries detected via embedding similarity.
///
/// # Algorithm
///
/// 1. Split the text into sentences.
/// 2. Embed every sentence with the provided `Embedder`.
/// 3. Compute cosine distance between adjacent sentence pairs.
/// 4. Cut wherever the distance exceeds `threshold`, **and** the
///    accumulated characters ≥ `min_chunk`.
pub struct SemanticChunker {
    embedder: Box<dyn Embedder>,
    /// Cosine‑distance threshold (default ≈ 0.35).
    /// Higher = fewer cuts (larger chunks).
    threshold: f32,
    /// Minimum characters before a cut is allowed.
    min_chunk: usize,
}

impl std::fmt::Debug for SemanticChunker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SemanticChunker")
            .field("threshold", &self.threshold)
            .field("min_chunk", &self.min_chunk)
            .field("dimension", &self.embedder.dimension())
            .finish()
    }
}

impl SemanticChunker {
    /// Build a new semantic chunker.
    ///
    /// `threshold` is a cosine distance: 0.0 = identical, 1.0+ = orthogonal.
    /// Start with 0.35 and tune from there.
    pub fn new(embedder: Box<dyn Embedder>, threshold: f32) -> Self {
        Self {
            embedder,
            threshold: threshold.max(0.05),
            min_chunk: 128,
        }
    }

    /// Set a minimum chunk size (in characters).
    pub fn with_min_chunk(mut self, min: usize) -> Self {
        self.min_chunk = min;
        self
    }

    /// Chunk `text` into semantic segments.
    ///
    /// Uses embedding similarity to find topic boundaries regardless
    /// of document size. Returns a single chunk if the text has fewer
    /// than 2 sentences, or if embedding fails.
    pub fn chunk(&self, text: &str, _max_size: usize) -> Vec<SmartChunk> {
        if text.is_empty() {
            return vec![single_chunk(text)];
        }

        let sentences = split_sentences(text);
        if sentences.len() < 2 {
            return vec![single_chunk(text)];
        }

        // Embed all sentences
        let refs: Vec<&str> = sentences.iter().map(|s| s.as_str()).collect();
        let embeddings = match self.embedder.embed_batch(&refs) {
            Ok(v) => v,
            Err(_) => return vec![single_chunk(text)],
        };
        if embeddings.len() != sentences.len() {
            return vec![single_chunk(text)];
        }

        // Compute adjacent distances and find cut points
        let mut boundaries: Vec<usize> = Vec::new();
        let mut running_chars: usize = 0;

        for i in 0..sentences.len() - 1 {
            let sim = cosine(&embeddings[i], &embeddings[i + 1]);
            let dist = 1.0 - sim; // cosine distance

            running_chars += sentences[i].len();

            if dist > self.threshold && running_chars >= self.min_chunk {
                boundaries.push(i + 1); // cut right after sentence i
                running_chars = 0;
            }
        }

        // Build chunks from boundaries
        let mut chunks: Vec<SmartChunk> = Vec::with_capacity(boundaries.len() + 1);
        let mut start = 0;
        for &end in &boundaries {
            let chunk_text: String = sentences[start..end].join(" ");
            let nlines = chunk_text.lines().count();
            chunks.push(SmartChunk {
                text: chunk_text,
                structure: None,
                level: 0,
                start_line: 0,
                end_line: nlines,
            });
            start = end;
        }
        // Last segment
        let chunk_text: String = sentences[start..].join(" ");
        let nlines = chunk_text.lines().count();
        chunks.push(SmartChunk {
            text: chunk_text,
            structure: None,
            level: 0,
            start_line: 0,
            end_line: nlines,
        });
        chunks
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn single_chunk(text: &str) -> SmartChunk {
    SmartChunk {
        text: text.to_string(),
        structure: None,
        level: 0,
        start_line: 0,
        end_line: text.lines().count(),
    }
}

/// Split text into sentences using punctuation and paragraph breaks.
fn split_sentences(text: &str) -> Vec<String> {
    let re = Regex::new(r"([.!?])\s+|(\n\s*\n)+").unwrap();
    let mut sentences: Vec<String> = Vec::new();
    let mut last = 0;

    for m in re.find_iter(text) {
        let sentence = text[last..m.end()].trim();
        if !sentence.is_empty() {
            sentences.push(sentence.to_string());
        }
        last = m.end();
    }
    // Last remainder
    let remainder = text[last..].trim();
    if !remainder.is_empty() {
        sentences.push(remainder.to_string());
    }

    // Filter out very short fragments (single words, whitespace)
    sentences.retain(|s| s.chars().count() >= 4);
    sentences
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyEmbedder;

    impl Embedder for DummyEmbedder {
        fn embed(&self, _text: &str) -> crate::error::Result<Vec<f32>> {
            Ok(vec![0.1; 8])
        }
        fn dimension(&self) -> usize {
            8
        }
        fn embed_batch(&self, texts: &[&str]) -> crate::error::Result<Vec<Vec<f32>>> {
            // Return identical vectors for the first 3 sentences,
            // then orthogonal for the rest -> cut after sentence 3
            Ok(texts
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    if i < 3 {
                        vec![1.0; 8] // identical
                    } else {
                        vec![0.0; 8] // orthogonal
                    }
                })
                .collect())
        }
    }

    #[test]
    fn test_semantic_chunker_splits_at_breaks() {
        let embedder = Box::new(DummyEmbedder);
        let chunker = SemanticChunker::new(embedder, 0.5).with_min_chunk(0);

        // 5 sentences: first 3 are "similar", last 2 are "different"
        let text = "First sentence here. Second follows. Third continues. Now a big shift. Totally different topic.";
        let chunks = chunker.chunk(text, 9999);

        assert!(chunks.len() >= 2, "expected 2+ chunks, got {}", chunks.len());
        // First 3 sentences should be in chunk 0
        assert!(chunks[0].text.contains("First"));
        assert!(chunks[0].text.contains("Third"));
    }

    #[test]
    fn test_semantic_chunker_single_chunk_short() {
        let embedder = Box::new(DummyEmbedder);
        let chunker = SemanticChunker::new(embedder, 0.5);
        let chunks = chunker.chunk("Short text.", 9999);
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn test_semantic_chunker_empty_text() {
        let embedder = Box::new(DummyEmbedder);
        let chunker = SemanticChunker::new(embedder, 0.5);
        let chunks = chunker.chunk("", 9999);
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn test_split_sentences() {
        let s = "Hello world. This is a test! And another? Yes.";
        let parts = split_sentences(s);
        assert!(parts.len() >= 3, "got {} parts", parts.len());
    }
}
