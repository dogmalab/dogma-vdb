//! Lightweight BM25 inverted index for hybrid text + vector search.
//!
//! Tokenises by splitting on non-alphanumeric characters and lowercasing.
//! Uses standard BM25Okapi with k₁ = 1.2, b = 0.75.
//!
//! # Example
//! ```
//! use dogma_vdb::index::bm25::Bm25Index;
//!
//! let mut idx = Bm25Index::new();
//! idx.insert(0, "Rust is fast and safe");
//! idx.insert(1, "Python is slow but ergonomic");
//! let results = idx.search("fast", 2);
//! assert_eq!(results[0].0, 0);
//! ```

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

/// BM25 constants (Okapi BM25 default).
const K1: f32 = 1.2;
const B: f32 = 0.75;

/// A lightweight BM25 inverted index.
///
/// Stores term frequencies per document, document lengths, and the
/// average document length for the IDF / length-normalisation terms.
#[derive(Serialize, Deserialize)]
pub struct Bm25Index {
    /// Inverted list: term → Vec<(doc_id, frequency_in_doc)>.
    inverted: HashMap<String, Vec<(usize, u32)>>,
    /// Total number of indexed documents.
    num_docs: usize,
    /// Length (in tokens) of each document.
    doc_lengths: Vec<usize>,
    /// Sum of all document lengths.
    total_length: usize,
}

impl Default for Bm25Index {
    fn default() -> Self {
        Self::new()
    }
}

impl Bm25Index {
    /// Create an empty BM25 index.
    pub fn new() -> Self {
        Self {
            inverted: HashMap::new(),
            num_docs: 0,
            doc_lengths: Vec::new(),
            total_length: 0,
        }
    }

    /// Index a document by its integer ID and text.
    ///
    /// The document ID should match the positional index in the
    /// corresponding `Vec<Document>` used by the vector index.
    pub fn insert(&mut self, doc_id: usize, text: &str) {
        let tokens = tokenise(text);
        let len = tokens.len();
        self.doc_lengths.push(len);
        self.total_length += len;

        let mut freq: HashMap<&str, u32> = HashMap::new();
        for t in &tokens {
            *freq.entry(t).or_insert(0) += 1;
        }
        for (term, count) in freq {
            self.inverted
                .entry(term.to_string())
                .or_default()
                .push((doc_id, count));
        }
        self.num_docs += 1;
    }

    /// Search with BM25, returning `Vec<(doc_id, score)>` sorted descending.
    ///
    /// Returns the top `top_k` results.  Empty query returns `vec![]`.
    pub fn search(&self, query: &str, top_k: usize) -> Vec<(usize, f32)> {
        if query.trim().is_empty() || self.num_docs == 0 {
            return Vec::new();
        }

        let tokens = tokenise(query);
        let avgdl = self.total_length as f32 / self.num_docs as f32;

        // Accumulate BM25 scores per doc
        let mut scores: HashMap<usize, f32> = HashMap::new();

        for term in &tokens {
            let idf = self.idf(term);
            if idf <= 0.0 {
                continue;
            }
            if let Some(postings) = self.inverted.get(term) {
                for &(doc_id, tf) in postings {
                    let doc_len = self.doc_lengths[doc_id] as f32;
                    let numerator = tf as f32 * (K1 + 1.0);
                    let denominator = tf as f32 + K1 * (1.0 - B + B * doc_len / avgdl);
                    *scores.entry(doc_id).or_insert(0.0) += idf * numerator / denominator;
                }
            }
        }

        let mut results: Vec<(usize, f32)> = scores.into_iter().collect();
        results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }

    /// Number of indexed documents.
    pub fn len(&self) -> usize {
        self.num_docs
    }

    pub fn is_empty(&self) -> bool {
        self.num_docs == 0
    }

    /// Persist the BM25 index to a JSON file at `path`.
    ///
    /// The saved file can be loaded later with [`Bm25Index::load`].
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::Error::Io`] if the file cannot be written,
    /// or [`crate::error::Error::Internal`] if serialisation fails.
    pub fn save(&self, path: &Path) -> crate::error::Result<()> {
        let json = serde_json::to_string(self).map_err(|e| {
            crate::error::Error::Internal(format!("Failed to serialise BM25 index: {e}"))
        })?;
        std::fs::write(path, &json).map_err(|e| crate::error::Error::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        Ok(())
    }

    /// Load a BM25 index from a JSON file previously saved with [`Bm25Index::save`].
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::Error::Io`] if the file cannot be read,
    /// or [`crate::error::Error::Internal`] if the JSON is malformed.
    pub fn load(path: &Path) -> crate::error::Result<Self> {
        let json = std::fs::read_to_string(path).map_err(|e| crate::error::Error::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        let idx: Self = serde_json::from_str(&json).map_err(|e| {
            crate::error::Error::Internal(format!("Failed to deserialise BM25 index: {e}"))
        })?;
        Ok(idx)
    }

    // ── helpers ──────────────────────────────────────────────────────────

    /// Inverse document frequency: ln(1 + (N - df + 0.5) / (df + 0.5))
    fn idf(&self, term: &str) -> f32 {
        let df = self.inverted.get(term).map(|v| v.len()).unwrap_or(0) as f32;
        if df == 0.0 {
            return 0.0;
        }
        let n = self.num_docs as f32;
        ((n - df + 0.5) / (df + 0.5)).ln() + 1.0
    }
}

/// Tokenise: split on non-alphanumeric, lowercase, non-empty.
fn tokenise(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bm25_basic() {
        let mut idx = Bm25Index::new();
        idx.insert(0, "the cat sat on the mat");
        idx.insert(1, "the dog chased the cat");
        idx.insert(2, "the bird flew away");

        let r = idx.search("cat", 10);
        assert_eq!(r.len(), 2);
        // Both docs have tf=1 for "cat". Doc 1 is shorter (5 vs 6 tokens)
        // so BM25 gives it a higher score (shorter doc = higher weight per match).
        assert_eq!(r[0].0, 1);
        assert_eq!(r[1].0, 0);
    }

    #[test]
    fn test_bm25_empty_query() {
        let mut idx = Bm25Index::new();
        idx.insert(0, "hello world");
        assert!(idx.search("", 5).is_empty());
        assert!(idx.search("   ", 5).is_empty());
    }

    #[test]
    fn test_bm25_empty_index() {
        let idx = Bm25Index::new();
        assert!(idx.search("hello", 5).is_empty());
    }

    #[test]
    fn test_bm25_rank_by_relevance() {
        let mut idx = Bm25Index::new();
        idx.insert(0, "Rust for systems programming");
        idx.insert(1, "Rust is fast and safe");
        idx.insert(2, "Python is interpreted");
        // "Rust" appears in docs 0 and 1
        let r = idx.search("Rust", 5);
        assert_eq!(r.len(), 2);
        // doc 0 has 4 tokens, doc 1 has 5. Shorter doc → higher BM25.
        assert_eq!(r[0].0, 0);
    }

    #[test]
    fn test_tokenise() {
        let t = tokenise("Hello, World! Rust-is-safe.");
        assert_eq!(t, vec!["hello", "world", "rust", "is", "safe"]);
    }

    #[test]
    fn test_bm25_save_load() {
        let mut idx = Bm25Index::new();
        idx.insert(0, "the cat sat on the mat");
        idx.insert(1, "the dog chased the cat");

        let dir = std::env::temp_dir();
        let path = dir.join("__test_bm25_save_load.bm25");

        // Clean up any leftover from previous runs
        let _ = std::fs::remove_file(&path);

        idx.save(&path).expect("save should succeed");

        let loaded = Bm25Index::load(&path).expect("load should succeed");
        assert_eq!(idx.len(), loaded.len());

        // Search results must be identical
        let r1 = idx.search("cat", 10);
        let r2 = loaded.search("cat", 10);
        assert_eq!(r1, r2);

        // Clean up
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_bm25_load_nonexistent() {
        let path = std::env::temp_dir().join("__test_bm25_nonexistent.bm25");
        let _ = std::fs::remove_file(&path);
        assert!(Bm25Index::load(&path).is_err());
    }
}
