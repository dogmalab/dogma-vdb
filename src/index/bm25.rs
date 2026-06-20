//! Lightweight BM25 inverted index for hybrid text + vector search.
//!
//! Tokenises by splitting on non-alphanumeric characters and lowercasing.
//! Uses standard BM25Okapi with k₁ = 1.2, b = 0.75.
//!
//! Supports **phrase queries** via quoted syntax: `search("\"part time\"")`
//! matches only documents where "part" and "time" appear consecutively.
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

/// A single posting entry: which document contains this term, at which positions.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Posting {
    doc_id: usize,
    positions: Vec<u32>,
}

/// A lightweight BM25 inverted index.
///
/// Stores term positions per document for phrase query support,
/// document lengths, and the average document length for IDF /
/// length-normalisation terms.
#[derive(Serialize, Deserialize)]
pub struct Bm25Index {
    /// Inverted list: term → Vec<Posting> (positional postings).
    inverted: HashMap<String, Vec<Posting>>,
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
        let tokens_with_pos = tokenise_with_positions(text);
        let len = tokens_with_pos.len();
        self.doc_lengths.push(len);
        self.total_length += len;

        // Build term → positions map
        let mut term_positions: HashMap<String, Vec<u32>> = HashMap::new();
        for (token, pos) in &tokens_with_pos {
            term_positions.entry(token.clone()).or_default().push(*pos);
        }

        for (term, positions) in term_positions {
            self.inverted
                .entry(term)
                .or_default()
                .push(Posting { doc_id, positions });
        }
        self.num_docs += 1;
    }

    /// Search with BM25, returning `Vec<(doc_id, score)>` sorted descending.
    ///
    /// - Regular queries: bag-of-words, scores each term independently.
    /// - Phrase queries: wrap in double quotes, e.g. `"part time"`.
    ///   Only matches documents where the terms appear consecutively.
    ///
    /// Returns the top `top_k` results. Empty query returns `vec![]`.
    pub fn search(&self, query: &str, top_k: usize) -> Vec<(usize, f32)> {
        if query.trim().is_empty() || self.num_docs == 0 {
            return Vec::new();
        }

        // Detect phrase query: "quoted text"
        let trimmed = query.trim();
        if trimmed.starts_with('"') && trimmed.ends_with('"') && trimmed.len() > 2 {
            return self.search_phrase(&trimmed[1..trimmed.len() - 1], top_k);
        }

        // Standard bag-of-words search
        let tokens = tokenise(trimmed);
        let avgdl = self.total_length as f32 / self.num_docs as f32;

        let mut scores: HashMap<usize, f32> = HashMap::new();

        for term in &tokens {
            let idf = self.idf(term);
            if idf <= 0.0 {
                continue;
            }
            if let Some(postings) = self.inverted.get(term) {
                for posting in postings {
                    let tf = posting.positions.len() as u32;
                    let doc_len = self.doc_lengths[posting.doc_id] as f32;
                    let numerator = tf as f32 * (K1 + 1.0);
                    let denominator = tf as f32 + K1 * (1.0 - B + B * doc_len / avgdl);
                    *scores.entry(posting.doc_id).or_insert(0.0) += idf * numerator / denominator;
                }
            }
        }

        let mut results: Vec<(usize, f32)> = scores.into_iter().collect();
        results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }

    /// Search for an exact phrase: terms must appear consecutively.
    ///
    /// Returns `Vec<(doc_id, score)>` sorted descending by BM25 score.
    fn search_phrase(&self, phrase: &str, top_k: usize) -> Vec<(usize, f32)> {
        let phrase_tokens = tokenise(phrase);
        if phrase_tokens.is_empty() {
            return Vec::new();
        }

        if phrase_tokens.len() == 1 {
            // Single term — fall back to standard search
            return self.search(&phrase_tokens[0], top_k);
        }

        let avgdl = self.total_length as f32 / self.num_docs as f32;

        // Find documents that contain ALL phrase terms
        let first_term = &phrase_tokens[0];
        let first_postings = match self.inverted.get(first_term) {
            Some(p) => p,
            None => return Vec::new(),
        };

        let mut candidate_docs: HashMap<usize, Vec<&Posting>> = HashMap::new();
        for posting in first_postings {
            candidate_docs
                .entry(posting.doc_id)
                .or_default()
                .push(posting);
        }

        // Intersect with remaining terms
        for term in &phrase_tokens[1..] {
            let postings = match self.inverted.get(term) {
                Some(p) => p,
                None => return Vec::new(),
            };
            let term_doc_ids: std::collections::HashSet<usize> =
                postings.iter().map(|p| p.doc_id).collect();
            candidate_docs.retain(|doc_id, _| term_doc_ids.contains(doc_id));
        }

        // For each candidate, verify consecutive positions
        let mut scores: HashMap<usize, f32> = HashMap::new();

        for (&doc_id, first_postings) in &candidate_docs {
            // Get all positions for each term in this document
            let mut term_positions: Vec<Vec<u32>> = Vec::new();
            term_positions.push(
                first_postings
                    .iter()
                    .flat_map(|p| p.positions.iter().copied())
                    .collect(),
            );
            for term in &phrase_tokens[1..] {
                if let Some(postings) = self.inverted.get(term) {
                    let positions: Vec<u32> = postings
                        .iter()
                        .filter(|p| p.doc_id == doc_id)
                        .flat_map(|p| p.positions.iter().copied())
                        .collect();
                    term_positions.push(positions);
                }
            }

            // Check if any starting position forms a consecutive sequence
            if phrase_matches(&term_positions) {
                // Score with BM25 (sum of individual term scores)
                let mut score = 0.0f32;
                for term in &phrase_tokens {
                    let idf = self.idf(term);
                    if idf <= 0.0 {
                        continue;
                    }
                    if let Some(postings) = self.inverted.get(term) {
                        for posting in postings {
                            if posting.doc_id == doc_id {
                                let tf = posting.positions.len() as f32;
                                let doc_len = self.doc_lengths[doc_id] as f32;
                                let numerator = tf * (K1 + 1.0);
                                let denominator = tf + K1 * (1.0 - B + B * doc_len / avgdl);
                                score += idf * numerator / denominator;
                            }
                        }
                    }
                }
                *scores.entry(doc_id).or_insert(0.0) += score;
            }
        }

        let mut results: Vec<(usize, f32)> = scores.into_iter().collect();
        results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }

    /// Number of indexed documents.
    #[must_use]
    pub fn len(&self) -> usize {
        self.num_docs
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.num_docs == 0
    }

    /// Persist the BM25 index to a JSON file at `path`.
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

/// Check if term positions form a consecutive sequence starting from any
/// position of the first term.
fn phrase_matches(term_positions: &[Vec<u32>]) -> bool {
    if term_positions.is_empty() || term_positions[0].is_empty() {
        return false;
    }

    for &start in &term_positions[0] {
        let valid = (start + 1..)
            .zip(term_positions[1..].iter())
            .all(|(expected, positions)| positions.contains(&expected));
        if valid {
            return true;
        }
    }
    false
}

/// Tokenise with positions: split on non-alphanumeric, lowercase, non-empty.
fn tokenise_with_positions(text: &str) -> Vec<(String, u32)> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .enumerate()
        .map(|(i, s)| (s.to_lowercase(), i as u32))
        .collect()
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
        let r = idx.search("Rust", 5);
        assert_eq!(r.len(), 2);
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
        let _ = std::fs::remove_file(&path);

        idx.save(&path).expect("save should succeed");

        let loaded = Bm25Index::load(&path).expect("load should succeed");
        assert_eq!(idx.len(), loaded.len());

        let r1 = idx.search("cat", 10);
        let r2 = loaded.search("cat", 10);
        assert_eq!(r1, r2);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_bm25_load_nonexistent() {
        let path = std::env::temp_dir().join("__test_bm25_nonexistent.bm25");
        let _ = std::fs::remove_file(&path);
        assert!(Bm25Index::load(&path).is_err());
    }

    // ── Phrase query tests ──────────────────────────────────────────────

    #[test]
    fn test_bm25_phrase_basic() {
        let mut idx = Bm25Index::new();
        idx.insert(0, "part time job");
        idx.insert(1, "time to find a part");
        idx.insert(2, "full time position");

        // "part time" matches only doc 0 (consecutive)
        let r = idx.search("\"part time\"", 10);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].0, 0);
    }

    #[test]
    fn test_bm25_phrase_no_match_reversed() {
        let mut idx = Bm25Index::new();
        idx.insert(0, "part time job");
        idx.insert(1, "time to find a part");

        // "time part" does NOT match "part time"
        let r = idx.search("\"time part\"", 10);
        assert!(r.is_empty());
    }

    #[test]
    fn test_bm25_phrase_partial_match() {
        let mut idx = Bm25Index::new();
        idx.insert(0, "part time job");
        idx.insert(1, "part of something");

        // "part time" — only doc 0 has both consecutively
        let r = idx.search("\"part time\"", 10);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].0, 0);
    }

    #[test]
    fn test_bm25_phrase_single_term() {
        let mut idx = Bm25Index::new();
        idx.insert(0, "part time job");
        idx.insert(1, "something else");

        // Single term in quotes falls back to bag-of-words
        let r = idx.search("\"part\"", 10);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].0, 0);
    }

    #[test]
    fn test_bm25_phrase_save_load() {
        let mut idx = Bm25Index::new();
        idx.insert(0, "part time job");
        idx.insert(1, "time to find a part");

        let dir = std::env::temp_dir();
        let path = dir.join("__test_bm25_phrase_save_load.bm25");
        let _ = std::fs::remove_file(&path);

        idx.save(&path).expect("save should succeed");
        let loaded = Bm25Index::load(&path).expect("load should succeed");

        let r1 = idx.search("\"part time\"", 10);
        let r2 = loaded.search("\"part time\"", 10);
        assert_eq!(r1, r2);
        assert_eq!(r1.len(), 1);
        assert_eq!(r1[0].0, 0);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_bm25_phrase_longer() {
        let mut idx = Bm25Index::new();
        idx.insert(0, "the quick brown fox jumps over the lazy dog");
        idx.insert(1, "quick and brown but not fox");
        idx.insert(2, "the lazy brown dog sleeps");

        // "quick brown fox" matches only doc 0 (consecutive)
        let r = idx.search("\"quick brown fox\"", 10);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].0, 0);
    }

    #[test]
    fn test_bm25_phrase_with_non_consecutive() {
        let mut idx = Bm25Index::new();
        idx.insert(0, "quick and brown fox");
        idx.insert(1, "quick brown fox");

        // "quick brown fox" — doc 1 has consecutive, doc 0 does not
        let r = idx.search("\"quick brown fox\"", 10);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].0, 1);
    }
}
