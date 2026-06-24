//! Inference engines for SML classification.
//!
//! Two-pass approach:
//! 1. `HeuristicInferencer` — keyword/regex patterns (always active, zero-cost)
//! 2. `SemanticInferencer` — embedding cosine similarity (optional, refines ambiguous cases)

use std::sync::Arc;

use crate::embedding::Embedder;
use crate::sml::keywords::{KeywordIndex, SmlCategory};

// --- Type mapping ---

/// Map verbose type strings to SIMIL shorthand.
pub fn map_type(ty: &str) -> &str {
    let t = ty.trim();
    match t {
        "String" | "str" | "&str" | "&'a str" | "string" | "CString" | "OsString" => "str",
        "i8" | "i16" | "i32" | "i64" | "i128" | "u8" | "u16" | "u32" | "u64" | "u128" | "int"
        | "isize" | "usize" => "int",
        "f32" | "f64" | "float" | "decimal" => "float",
        "PathBuf" | "Path" | "pathlib.Path" | "path" => "path",
        "bool" | "Boolean" | "boolean" => "bool",
        t if t.starts_with("Vec<") || t.starts_with("list[") || t.starts_with("Array<") => "List",
        t if t.starts_with("Option<") || t.ends_with('?') => "T?",
        t if t.starts_with("Enum[") => t,
        t if t.starts_with("HashMap<") || t.starts_with("dict[") => "Map",
        _ => t,
    }
}

// --- HeuristicInferencer ---

/// Keyword-based classification (always active, nanosecond cost).
pub struct HeuristicInferencer;

impl Default for HeuristicInferencer {
    fn default() -> Self {
        Self::new()
    }
}

impl HeuristicInferencer {
    pub fn new() -> Self {
        Self
    }

    /// Classify a chunk's structure name into a SIMIL category.
    pub fn classify_structure(&self, structure: &str) -> SmlCategory {
        let lower = structure.to_lowercase();
        let s = lower.as_str();

        // Type keywords
        if s.contains("struct")
            || s.contains("class")
            || s.contains("enum")
            || s.contains("trait")
            || s.contains("interface")
            || s.contains("type ")
            || s.contains("model")
            || s.contains("entity")
        {
            return SmlCategory::Type;
        }

        // Flow keywords
        if s.contains("fn ")
            || s.contains("def ")
            || s.contains("func ")
            || s.contains("function")
            || s.contains("proc ")
            || s.contains("method")
            || s.contains("impl")
            || s.contains("handle")
            || s.contains("process")
        {
            return SmlCategory::Flow;
        }

        // Default: if it starts with PascalCase, it's a type
        if structure.chars().next().is_some_and(|c| c.is_uppercase()) {
            return SmlCategory::Type;
        }

        // snake_case default: flow
        SmlCategory::Flow
    }

    /// Check if a line contains invariant/axiom keywords.
    pub fn is_invariant_line(&self, line: &str) -> bool {
        let lower = line.to_lowercase();
        let keywords = [
            "must ",
            "always ",
            "never ",
            "require",
            "assert",
            "guard",
            "ensure",
            "reject",
            "forbidden",
            "mandatory",
            "obligatory",
        ];
        keywords.iter().any(|kw| lower.contains(kw))
    }

    /// Extract attribute from a `name: Type` pattern.
    /// Returns `(name, type)` or `None`.
    pub fn extract_attr<'a>(&self, line: &'a str) -> Option<(&'a str, &'a str)> {
        let line = line.trim();
        // Skip comments, doc strings, keywords
        if line.starts_with("//")
            || line.starts_with('>')
            || line.starts_with('!')
            || line.starts_with('*')
            || line.starts_with('@')
            || line.starts_with("pub ")
            || line.starts_with("fn ")
            || line.starts_with("def ")
            || line.starts_with("func ")
        {
            return None;
        }

        // Look for `name: Type` pattern
        let colon = line.find(':')?;
        if colon == 0 {
            return None;
        }
        let name = line[..colon].trim();
        let rest = line[colon + 1..].trim();

        // Extract type (up to comma, brace, comment, or end)
        let ty_end = rest
            .find(',')
            .or_else(|| rest.find('}'))
            .or_else(|| rest.find("//"))
            .unwrap_or(rest.len());
        let ty = rest[..ty_end].trim();

        if name.is_empty() || ty.is_empty() {
            return None;
        }

        // Validate name looks like an identifier
        if name
            .chars()
            .next()
            .is_some_and(|c| c.is_alphabetic() || c == '_')
            && name.chars().all(|c| c.is_alphanumeric() || c == '_')
        {
            Some((name, ty))
        } else {
            None
        }
    }

    /// Extract a doc string from a `> "..."` or `/// ...` line.
    pub fn extract_doc<'a>(&self, line: &'a str) -> Option<&'a str> {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("> \"") {
            rest.strip_suffix('"')
        } else if let Some(rest) = line.strip_prefix("/// ") {
            Some(rest)
        } else if line == "///" {
            Some("")
        } else {
            None
        }
    }

    /// Extract a link target from a `*target` line.
    pub fn extract_link<'a>(&self, line: &'a str) -> Option<&'a str> {
        let line = line.trim();
        if let Some(target) = line.strip_prefix('*') {
            let target = target.trim();
            if !target.is_empty() {
                return Some(target);
            }
        }
        None
    }
}

// --- SemanticInferencer ---

/// Embedding-based classification for ambiguous text chunks.
pub struct SemanticInferencer {
    embedder: Arc<dyn Embedder>,
    threshold: f32,
}

impl SemanticInferencer {
    pub fn new(embedder: Arc<dyn Embedder>) -> Self {
        Self {
            embedder,
            threshold: 0.7,
        }
    }

    pub fn with_threshold(embedder: Arc<dyn Embedder>, threshold: f32) -> Self {
        Self {
            embedder,
            threshold,
        }
    }

    /// Classify a text chunk by embedding it and comparing against a keyword index.
    ///
    /// Returns `(category, score)` if above threshold, `None` otherwise.
    pub fn classify(&self, text: &str, index: &KeywordIndex) -> Option<(SmlCategory, f32)> {
        let embedding = self.embedder.embed(text).ok()?;
        let (category, score) = index.classify(&embedding)?;
        if score >= self.threshold {
            Some((category, score))
        } else {
            None
        }
    }

    /// Classify a batch of texts against the keyword index.
    pub fn classify_batch(
        &self,
        texts: &[&str],
        index: &KeywordIndex,
    ) -> Vec<Option<(SmlCategory, f32)>> {
        let embeddings = self.embedder.embed_batch(texts).unwrap_or_default();

        embeddings
            .iter()
            .map(|emb| {
                let (category, score) = index.classify(emb)?;
                if score >= self.threshold {
                    Some((category, score))
                } else {
                    None
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_map_type_string() {
        assert_eq!(map_type("String"), "str");
        assert_eq!(map_type("&str"), "str");
        assert_eq!(map_type("string"), "str");
    }

    #[test]
    fn test_map_type_int() {
        assert_eq!(map_type("i32"), "int");
        assert_eq!(map_type("u64"), "int");
        assert_eq!(map_type("usize"), "int");
    }

    #[test]
    fn test_map_type_float() {
        assert_eq!(map_type("f32"), "float");
        assert_eq!(map_type("f64"), "float");
    }

    #[test]
    fn test_map_type_complex() {
        assert_eq!(map_type("Vec<String>"), "List");
        assert_eq!(map_type("Option<i32>"), "T?");
        assert_eq!(map_type("Enum[ADMIN, USER]"), "Enum[ADMIN, USER]");
        assert_eq!(map_type("HashMap<String, i32>"), "Map");
    }

    #[test]
    fn test_classify_structure() {
        let h = HeuristicInferencer::new();
        assert_eq!(h.classify_structure("struct User"), SmlCategory::Type);
        assert_eq!(h.classify_structure("class Foo"), SmlCategory::Type);
        assert_eq!(h.classify_structure("fn main"), SmlCategory::Flow);
        assert_eq!(h.classify_structure("def process"), SmlCategory::Flow);
        assert_eq!(h.classify_structure("func handle"), SmlCategory::Flow);
        assert_eq!(h.classify_structure("UserProfile"), SmlCategory::Type);
        assert_eq!(h.classify_structure("process_data"), SmlCategory::Flow);
    }

    #[test]
    fn test_is_invariant_line() {
        let h = HeuristicInferencer::new();
        assert!(h.is_invariant_line("Must always validate"));
        assert!(h.is_invariant_line("This is required"));
        assert!(h.is_invariant_line("Never allow empty values"));
        assert!(!h.is_invariant_line("This is a normal comment"));
    }

    #[test]
    fn test_extract_attr() {
        let h = HeuristicInferencer::new();
        assert_eq!(h.extract_attr("name: String"), Some(("name", "String")));
        assert_eq!(h.extract_attr("pub name: String"), None);
        assert_eq!(h.extract_attr("// comment: foo"), None);
        assert_eq!(
            h.extract_attr("email: String, // user email"),
            Some(("email", "String"))
        );
    }

    #[test]
    fn test_extract_doc() {
        let h = HeuristicInferencer::new();
        assert_eq!(
            h.extract_doc("> \"User account entity\""),
            Some("User account entity")
        );
        assert_eq!(h.extract_doc("/// This is a doc"), Some("This is a doc"));
        assert_eq!(h.extract_doc("///"), Some(""));
        assert_eq!(h.extract_doc("normal text"), None);
    }

    #[test]
    fn test_extract_link() {
        let h = HeuristicInferencer::new();
        assert_eq!(h.extract_link("*UserProfile"), Some("UserProfile"));
        assert_eq!(h.extract_link("* Profile"), Some("Profile"));
        assert_eq!(h.extract_link("normal text"), None);
        assert_eq!(h.extract_link("*"), None);
    }
}
