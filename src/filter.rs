//! Document filter helpers for metadata-based search filtering.
//!
//! # Example
//! ```
//! use dogma_vdb::filter;
//! use dogma_vdb::doc::Document;
//!
//! let doc = Document::builder("a", "hello")
//!     .metadata("lang", "en")
//!     .metadata("source", "book.pdf")
//!     .build();
//!
//! let is_english = filter::metadata_eq("lang", "en");
//! assert!(is_english(&doc));
//!
//! let is_pdf = filter::metadata_contains("source", ".pdf");
//! assert!(is_pdf(&doc));
//! ```

use crate::doc::Document;

/// A document filter function type.
pub type Filter = Box<dyn Fn(&Document) -> bool>;

/// Returns `true` if the document has a metadata key whose value equals `value`.
pub fn metadata_eq(key: &str, value: &str) -> impl Fn(&Document) -> bool {
    let k = key.to_string();
    let v = value.to_string();
    move |doc: &Document| doc.metadata_val(&k) == Some(&v)
}

/// Returns `true` if the document has a metadata key whose value contains `substr`.
pub fn metadata_contains(key: &str, substr: &str) -> impl Fn(&Document) -> bool {
    let k = key.to_string();
    let s = substr.to_string();
    move |doc: &Document| doc.metadata_val(&k).is_some_and(|v| v.contains(&s))
}

/// Returns `true` if the document has a metadata key (regardless of value).
pub fn metadata_exists(key: &str) -> impl Fn(&Document) -> bool {
    let k = key.to_string();
    move |doc: &Document| doc.metadata.contains_key(&k)
}

/// Combines multiple filters with AND logic.
pub fn all_of(filters: Vec<Filter>) -> impl Fn(&Document) -> bool {
    move |doc: &Document| filters.iter().all(|f| f(doc))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_doc() -> Document {
        Document::builder("t1", "text")
            .metadata("lang", "en")
            .metadata("source", "book.pdf")
            .build()
    }

    #[test]
    fn test_metadata_eq_match() {
        let f = metadata_eq("lang", "en");
        assert!(f(&test_doc()));
    }

    #[test]
    fn test_metadata_eq_no_match() {
        let f = metadata_eq("lang", "es");
        assert!(!f(&test_doc()));
    }

    #[test]
    fn test_metadata_eq_missing_key() {
        let f = metadata_eq("author", "arggil");
        assert!(!f(&test_doc()));
    }

    #[test]
    fn test_metadata_contains_match() {
        let f = metadata_contains("source", ".pdf");
        assert!(f(&test_doc()));
    }

    #[test]
    fn test_metadata_contains_no_match() {
        let f = metadata_contains("source", ".txt");
        assert!(!f(&test_doc()));
    }

    #[test]
    fn test_metadata_exists_true() {
        let f = metadata_exists("lang");
        assert!(f(&test_doc()));
    }

    #[test]
    fn test_metadata_exists_false() {
        let f = metadata_exists("author");
        assert!(!f(&test_doc()));
    }

    #[test]
    fn test_all_of_all_match() {
        let filters = vec![
            Box::new(metadata_eq("lang", "en")) as Filter,
            Box::new(metadata_exists("source")) as Filter,
        ];
        let f = all_of(filters);
        assert!(f(&test_doc()));
    }

    #[test]
    fn test_all_of_one_fails() {
        let filters = vec![
            Box::new(metadata_eq("lang", "en")) as Filter,
            Box::new(metadata_eq("lang", "es")) as Filter,
        ];
        let f = all_of(filters);
        assert!(!f(&test_doc()));
    }
}
