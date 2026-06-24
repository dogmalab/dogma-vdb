//! Static keyword dictionaries for SIMIL category inference.
//!
//! Keywords are organized by semantic category. The `KeywordIndex`
//! pre-computes embeddings for fast cosine-similarity lookup.

use std::sync::Arc;

use crate::embedding::Embedder;

/// Semantic category for a SIMIL node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmlCategory {
    Flow,
    Query,
    Invariant,
    Type,
}

/// Pre-computed keyword index for vector-based classification.
pub struct KeywordIndex {
    pub keywords: Vec<String>,
    pub categories: Vec<SmlCategory>,
    pub embeddings: Vec<Vec<f32>>,
}

// --- Static dictionaries ---

pub const FLOW_KEYWORDS: &[&str] = &[
    "process",
    "transform",
    "pipeline",
    "run",
    "execute",
    "handle",
    "convert",
    "build",
    "generate",
    "emit",
    "produce",
    "load",
    "write",
    "send",
    "receive",
    "sync",
    "flush",
    "compact",
    "deploy",
    "migrate",
    "stream",
    "ingest",
    "export",
    "import",
    "compile",
    "build",
    "package",
];

pub const QUERY_KEYWORDS: &[&str] = &[
    "check", "validate", "verify", "query", "find", "search", "test", "inspect", "audit",
    "measure", "count", "list", "get", "has", "is", "should", "expect", "match", "filter",
    "select",
];

pub const INVARIANT_KEYWORDS: &[&str] = &[
    "must",
    "always",
    "never",
    "require",
    "assert",
    "guard",
    "ensure",
    "reject",
    "fail",
    "abort",
    "panic",
    "lock",
    "freeze",
    "immutable",
    "pinned",
    "mandatory",
    "forbidden",
    "obligatory",
];

pub const TYPE_KEYWORDS: &[&str] = &[
    "entity",
    "model",
    "schema",
    "struct",
    "record",
    "value",
    "object",
    "type",
    "class",
    "interface",
    "enum",
    "module",
    "service",
    "resource",
    "config",
    "definition",
];

/// Build a keyword index by embedding all keywords with the given embedder.
///
/// Returns a `KeywordIndex` suitable for cosine-similarity classification.
pub fn build_keyword_index(embedder: &Arc<dyn Embedder>) -> KeywordIndex {
    let mut keywords = Vec::new();
    let mut categories = Vec::new();

    for &kw in FLOW_KEYWORDS {
        keywords.push(kw.to_string());
        categories.push(SmlCategory::Flow);
    }
    for &kw in QUERY_KEYWORDS {
        keywords.push(kw.to_string());
        categories.push(SmlCategory::Query);
    }
    for &kw in INVARIANT_KEYWORDS {
        keywords.push(kw.to_string());
        categories.push(SmlCategory::Invariant);
    }
    for &kw in TYPE_KEYWORDS {
        keywords.push(kw.to_string());
        categories.push(SmlCategory::Type);
    }

    let refs: Vec<&str> = keywords.iter().map(|s| s.as_str()).collect();
    let embeddings = embedder.embed_batch(&refs).unwrap_or_default();

    KeywordIndex {
        keywords,
        categories,
        embeddings,
    }
}

impl KeywordIndex {
    /// Find the closest keyword category for a query vector.
    ///
    /// Returns `(category, score)` or `None` if the index is empty.
    pub fn classify(&self, query: &[f32]) -> Option<(SmlCategory, f32)> {
        if self.embeddings.is_empty() || query.is_empty() {
            return None;
        }

        let mut best_category = SmlCategory::Flow;
        let mut best_score = f32::NEG_INFINITY;

        for (kw_embedding, category) in self.embeddings.iter().zip(&self.categories) {
            let score = cosine_similarity(query, kw_embedding);
            if score > best_score {
                best_score = score;
                best_category = *category;
            }
        }

        Some((best_category, best_score))
    }
}

/// Cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut mag_a = 0.0f32;
    let mut mag_b = 0.0f32;
    for i in 0..len {
        dot += a[i] * b[i];
        mag_a += a[i] * a[i];
        mag_b += b[i] * b[i];
    }
    let denom = mag_a.sqrt() * mag_b.sqrt();
    if denom < 1e-8 {
        0.0
    } else {
        dot / denom
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_category_counts() {
        assert!(!FLOW_KEYWORDS.is_empty());
        assert!(!QUERY_KEYWORDS.is_empty());
        assert!(!INVARIANT_KEYWORDS.is_empty());
        assert!(!TYPE_KEYWORDS.is_empty());
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0, 2.0, 3.0];
        let score = cosine_similarity(&v, &v);
        assert!((score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let score = cosine_similarity(&a, &b);
        assert!(score.abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_empty() {
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
        assert_eq!(cosine_similarity(&[1.0], &[]), 0.0);
    }

    #[test]
    fn test_keyword_index_classify_empty() {
        let idx = KeywordIndex {
            keywords: vec![],
            categories: vec![],
            embeddings: vec![],
        };
        assert!(idx.classify(&[1.0, 2.0]).is_none());
    }
}
