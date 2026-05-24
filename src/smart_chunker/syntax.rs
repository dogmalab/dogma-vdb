//! Tree‑sitter–based syntax chunker.
//!
//! Walks the AST to find top‑level function, struct, class, trait,
//! impl definitions with surgical precision.
//!
//! Feature‑gated behind `chunker-syntax`. Falls back to regex when
//! parsing fails or the language has no grammar loaded.

use crate::smart_chunker::{subdivide_by_lines, FileType, SmartChunk};
use std::sync::Mutex;
use tree_sitter::{Node, Parser, TreeCursor};

// ---------------------------------------------------------------------------
// Per‑language definition specs
// ---------------------------------------------------------------------------

type DefSpec = (&'static str, usize);

const RUST_DEFS: &[DefSpec] = &[
    ("function_item", 0),
    ("struct_item", 0),
    ("enum_item", 0),
    ("trait_item", 0),
    ("union_item", 0),
    ("type_item", 0),
    ("impl_item", 0),
    ("const_item", 0),
    ("static_item", 0),
    ("mod_item", 0),
    ("macro_definition", 0),
];

const PYTHON_DEFS: &[DefSpec] = &[
    ("function_definition", 0),
    ("class_definition", 0),
    ("decorated_definition", 0),
];

const JS_DEFS: &[DefSpec] = &[
    ("function_declaration", 0),
    ("generator_function_declaration", 0),
    ("class_declaration", 0),
    ("lexical_declaration", 0),
    ("interface_declaration", 0),
    ("type_alias_declaration", 0),
    ("enum_declaration", 0),
];

const GO_DEFS: &[DefSpec] = &[
    ("function_declaration", 0),
    ("method_declaration", 0),
    ("type_declaration", 0),
];

// ---------------------------------------------------------------------------
// SyntaxChunker
// ---------------------------------------------------------------------------

/// AST‑driven chunker for source code.
///
/// Uses tree‑sitter parsers to locate top‑level definitions with exact
/// line ranges and precise names.
pub struct SyntaxChunker {
    rust: Mutex<Parser>,
    python: Mutex<Parser>,
    javascript: Mutex<Parser>,
    go: Mutex<Parser>,
    rust_defs: std::collections::HashSet<&'static str>,
    py_defs: std::collections::HashSet<&'static str>,
    js_defs: std::collections::HashSet<&'static str>,
    go_defs: std::collections::HashSet<&'static str>,
}

impl std::fmt::Debug for SyntaxChunker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SyntaxChunker")
            .field("rust_defs", &self.rust_defs)
            .field("py_defs", &self.py_defs)
            .field("js_defs", &self.js_defs)
            .field("go_defs", &self.go_defs)
            .finish()
    }
}

impl SyntaxChunker {
    /// Create parsers for all four languages.
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let mut rust = Parser::new();
        rust.set_language(&tree_sitter_rust::language())?;
        let mut python = Parser::new();
        python.set_language(&tree_sitter_python::language())?;
        let mut javascript = Parser::new();
        javascript.set_language(&tree_sitter_javascript::language())?;
        let mut go = Parser::new();
        go.set_language(&tree_sitter_go::language())?;

        let to_set = |specs: &[DefSpec]| specs.iter().map(|(k, _)| *k).collect();
        Ok(Self {
            rust: Mutex::new(rust),
            python: Mutex::new(python),
            javascript: Mutex::new(javascript),
            go: Mutex::new(go),
            rust_defs: to_set(RUST_DEFS),
            py_defs: to_set(PYTHON_DEFS),
            js_defs: to_set(JS_DEFS),
            go_defs: to_set(GO_DEFS),
        })
    }

    /// Chunk `text` by top‑level definitions for `file_type`.
    ///
    /// Returns an empty vec when parsing fails or no definitions are found,
    /// letting the caller fall back to the regex‑based `CodeChunker`.
    pub fn chunk(&self, text: &str, file_type: FileType, max_size: usize) -> Vec<SmartChunk> {
        let (parser, defs) = match file_type {
            FileType::Rust => (&self.rust, &self.rust_defs),
            FileType::Python => (&self.python, &self.py_defs),
            FileType::JavaScript => (&self.javascript, &self.js_defs),
            FileType::Go => (&self.go, &self.go_defs),
            _ => return vec![],
        };

        let tree = match parser.lock().unwrap().parse(text, None) {
            Some(t) => t,
            None => return vec![],
        };

        let lines: Vec<&str> = text.lines().collect();
        if lines.is_empty() {
            return vec![];
        }
        let total = lines.len();

        // Walk top‑level children to collect definition boundaries
        let mut bounds: Vec<(usize, String, usize)> = Vec::new();
        let mut cursor = tree.root_node().walk();
        if cursor.goto_first_child() {
            collect_top_defs(&mut cursor, text, defs, &mut bounds);
        }

        if bounds.is_empty() {
            return vec![];
        }

        // Build chunks between consecutive boundaries (same algorithm as CodeChunker)
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

// ---------------------------------------------------------------------------
// AST traversal helpers
// ---------------------------------------------------------------------------

/// Walk top‑level sibling nodes, collecting those that match `defs`.
fn collect_top_defs(
    cursor: &mut TreeCursor,
    source: &str,
    defs: &std::collections::HashSet<&'static str>,
    out: &mut Vec<(usize, String, usize)>,
) {
    loop {
        let kind = cursor.node().kind();
        if defs.contains(kind) {
            let name = extract_def_name(cursor.node(), source);
            let start = cursor.node().start_position().row;
            out.push((start, name, 0));
        }
        if !cursor.goto_next_sibling() {
            break;
        }
    }
}

/// Extract the canonical name of a definition node.
fn extract_def_name(node: Node, source: &str) -> String {
    // decorated_definition → look inside for the actual definition
    if node.kind() == "decorated_definition" {
        let mut cursor = node.walk();
        if cursor.goto_first_child() {
            // Find the definition child (function_definition or class_definition)
            loop {
                let child = cursor.node();
                if matches!(child.kind(), "function_definition" | "class_definition") {
                    if let Some(name) = child
                        .child_by_field_name("name")
                        .and_then(|n| n.utf8_text(source.as_bytes()).ok())
                    {
                        return name.to_string();
                    }
                }
                if !cursor.goto_next_sibling() {
                    break;
                }
            }
        }
    }

    // impl_item → format as "impl Type" or "impl Trait for Type"
    if node.kind() == "impl_item" {
        if let Some(trait_node) = node.child_by_field_name("trait") {
            let t = trait_node.utf8_text(source.as_bytes()).unwrap_or("");
            if let Some(type_node) = node.child_by_field_name("type") {
                let ty = type_node.utf8_text(source.as_bytes()).unwrap_or("");
                return format!("impl {} for {}", t, ty);
            }
            return format!("impl {}", t);
        }
        if let Some(type_node) = node.child_by_field_name("type") {
            let ty = type_node.utf8_text(source.as_bytes()).unwrap_or("");
            return format!("impl {}", ty);
        }
    }

    // Standard: field "name"
    if let Some(name_node) = node.child_by_field_name("name") {
        if let Ok(name) = name_node.utf8_text(source.as_bytes()) {
            return name.to_string();
        }
    }

    // Fallback: first non‑whitespace line of the node
    if let Ok(text) = node.utf8_text(source.as_bytes()) {
        let first = text.lines().next().unwrap_or(text).trim();
        return first.to_string();
    }

    node.kind().to_string()
}
