//! SML Compiler: orchestrates heuristics + semantic inference to produce SML nodes.

use std::sync::Arc;

use crate::embedding::Embedder;
use crate::smart_chunker::SmartChunk;
use crate::sml::ast::*;
use crate::sml::infer::{HeuristicInferencer, SemanticInferencer};
use crate::sml::keywords::{KeywordIndex, SmlCategory};

use std::sync::OnceLock;

/// SIMIL compiler. Combines deterministic heuristics with optional
/// semantic inference to transform code/text chunks into SML AST nodes.
pub struct SmlCompiler {
    heuristic: HeuristicInferencer,
    semantic: Option<SemanticInferencer>,
    keyword_index: OnceLock<KeywordIndex>,
}

impl Default for SmlCompiler {
    fn default() -> Self {
        Self::new()
    }
}

impl SmlCompiler {
    /// Create a compiler with heuristic-only mode (no embedder).
    pub fn new() -> Self {
        Self {
            heuristic: HeuristicInferencer::new(),
            semantic: None,
            keyword_index: OnceLock::new(),
        }
    }

    /// Create a compiler with semantic inference enabled.
    pub fn with_embedder(embedder: Arc<dyn Embedder>) -> Self {
        Self {
            heuristic: HeuristicInferencer::new(),
            semantic: Some(SemanticInferencer::new(embedder)),
            keyword_index: OnceLock::new(),
        }
    }

    #[allow(dead_code)]
    fn keyword_index(&self, embedder: &Arc<dyn Embedder>) -> &KeywordIndex {
        self.keyword_index
            .get_or_init(|| crate::sml::keywords::build_keyword_index(embedder))
    }

    /// Compile a single SmartChunk into a SML node.
    pub fn compile(&self, chunk: &SmartChunk, source: &str) -> SmlNode {
        match &chunk.structure {
            Some(structure) => self.compile_code_chunk(chunk, structure, source),
            None => self.compile_text_chunk(chunk, source),
        }
    }

    /// Compile a batch of SmartChunks into SML nodes.
    pub fn compile_batch(&self, chunks: &[SmartChunk], source: &str) -> Vec<SmlNode> {
        chunks.iter().map(|c| self.compile(c, source)).collect()
    }

    // --- Code chunk compilation ---

    fn compile_code_chunk(&self, chunk: &SmartChunk, structure: &str, source: &str) -> SmlNode {
        let category = self.heuristic.classify_structure(structure);

        match category {
            SmlCategory::Type => self.build_type_node(chunk, structure, source),
            SmlCategory::Flow => self.build_flow_node(chunk, structure, source),
            _ => self.build_type_node(chunk, structure, source),
        }
    }

    fn build_type_node(&self, chunk: &SmartChunk, structure: &str, source: &str) -> SmlNode {
        let name = extract_struct_name(structure).to_string();
        let start = chunk.start_line.min(source.len());
        let end = chunk.end_line.min(source.len());
        let chunk_source = &source[start..end];

        let doc = find_doc_above(chunk, source).map(|s| s.to_string());
        let attrs = extract_attrs_from_region(chunk_source);
        let links = extract_links_from_region(chunk_source);
        let invariants = extract_invariants_from_region(chunk_source);

        SmlNode::Type {
            name,
            doc,
            attrs,
            links,
            invariants,
        }
    }

    fn build_flow_node(&self, chunk: &SmartChunk, structure: &str, source: &str) -> SmlNode {
        let name = extract_fn_name(structure).to_string();
        let start = chunk.start_line.min(source.len());
        let end = chunk.end_line.min(source.len());
        let chunk_source = &source[start..end];

        let params = extract_params_from_region(chunk_source);
        let body = extract_steps_from_region(chunk_source);

        SmlNode::Flow { name, params, body }
    }

    // --- Text chunk compilation ---

    fn compile_text_chunk(&self, chunk: &SmartChunk, source: &str) -> SmlNode {
        let start = chunk.start_line.min(source.len());
        let end = chunk.end_line.min(source.len());
        let chunk_source = &source[start..end];

        // Try semantic refinement if available
        if let Some(ref semantic) = self.semantic {
            if let Some(idx) = self.keyword_index.get() {
                if let Some((category, _score)) = semantic.classify(chunk_source, idx) {
                    match category {
                        SmlCategory::Invariant => {
                            return self.build_invariant_node_from_text(chunk_source);
                        }
                        SmlCategory::Flow => {
                            return self.build_flow_node_from_text(chunk_source);
                        }
                        SmlCategory::Type => {
                            return self.build_type_node_from_text(chunk_source);
                        }
                        SmlCategory::Query => {
                            return self.build_flow_node_from_text(chunk_source);
                        }
                    }
                }
            }
        }

        // Fallback: heuristic keyword detection on sentences
        self.heuristic_classify_text(chunk_source)
    }

    fn heuristic_classify_text(&self, text: &str) -> SmlNode {
        // Check if any sentence has invariant keywords
        for line in text.lines() {
            let trimmed = line.trim();
            if !trimmed.is_empty() && self.heuristic.is_invariant_line(trimmed) {
                return self.build_invariant_node_from_text(text);
            }
        }

        // Check if it looks like a type definition (has PascalCase words)
        let first_word = text
            .lines()
            .next()
            .unwrap_or("")
            .split_whitespace()
            .next()
            .unwrap_or("");
        if first_word.chars().next().is_some_and(|c| c.is_uppercase()) && first_word.len() > 2 {
            return self.build_type_node_from_text(text);
        }

        // Default: flow node
        self.build_flow_node_from_text(text)
    }

    fn build_type_node_from_text(&self, text: &str) -> SmlNode {
        let name = extract_first_identifier(text).to_string();
        let doc = first_sentence(text).map(|s| s.to_string());

        SmlNode::Type {
            name,
            doc,
            attrs: vec![],
            links: vec![],
            invariants: vec![],
        }
    }

    fn build_flow_node_from_text(&self, text: &str) -> SmlNode {
        let name = extract_first_identifier(text).to_string();
        let steps = extract_steps_from_text(text);

        SmlNode::Flow {
            name,
            params: vec![],
            body: steps,
        }
    }

    fn build_invariant_node_from_text(&self, text: &str) -> SmlNode {
        let name = extract_first_identifier(text).to_string();
        let doc = first_sentence(text).map(|s| s.to_string());

        let invariants: Vec<SmlInvariant> = text
            .lines()
            .filter(|line| self.heuristic.is_invariant_line(line.trim()))
            .filter_map(|line| {
                let cond = extract_condition_from_sentence(line.trim());
                let action = extract_action_from_sentence(line.trim());
                if !cond.is_empty() && !action.is_empty() {
                    Some(SmlInvariant {
                        condition: cond,
                        action,
                    })
                } else {
                    None
                }
            })
            .collect();

        SmlNode::Type {
            name,
            doc,
            attrs: vec![],
            links: vec![],
            invariants,
        }
    }
}

// --- Helpers: extract from structure names ---

fn extract_struct_name(structure: &str) -> &str {
    let without_prefix = structure.strip_prefix("pub ").unwrap_or(structure);
    let without_prefix = without_prefix
        .strip_prefix("pub(crate) ")
        .unwrap_or(without_prefix);

    for prefix in &[
        "struct ",
        "class ",
        "enum ",
        "trait ",
        "interface ",
        "type ",
        "impl ",
    ] {
        if let Some(rest) = without_prefix.strip_prefix(prefix) {
            return rest
                .split(&['{', '(', '<', ' ', '\t'][..])
                .next()
                .unwrap_or(rest);
        }
    }

    structure
        .split(&['{', '(', '<', ' ', '\t'][..])
        .next()
        .unwrap_or(structure)
}

fn extract_fn_name(structure: &str) -> &str {
    let without_prefix = structure.strip_prefix("pub ").unwrap_or(structure);
    let without_prefix = without_prefix
        .strip_prefix("pub(crate) ")
        .unwrap_or(without_prefix);
    let without_prefix = without_prefix
        .strip_prefix("async ")
        .unwrap_or(without_prefix);

    for prefix in &["fn ", "def ", "func ", "function ", "proc "] {
        if let Some(rest) = without_prefix.strip_prefix(prefix) {
            return rest
                .split(&['(', '<', ' ', '\t'][..])
                .next()
                .unwrap_or(rest);
        }
    }

    structure
        .split(&['(', '<', ' ', '\t'][..])
        .next()
        .unwrap_or(structure)
}

// --- Helpers: extract from source regions ---

fn find_doc_above<'a>(chunk: &SmartChunk, source: &'a str) -> Option<&'a str> {
    if chunk.start_line == 0 {
        return None;
    }

    let lines: Vec<&str> = source.lines().collect();
    let mut doc_lines = Vec::new();

    let mut i = chunk.start_line.saturating_sub(1);
    while i > 0 {
        let line = lines[i].trim();
        if line.starts_with("///") || line.starts_with("//!") {
            let doc_text = line
                .strip_prefix("///")
                .or_else(|| line.strip_prefix("//!"))
                .unwrap_or(line)
                .trim();
            doc_lines.insert(0, doc_text);
            i -= 1;
        } else if line.starts_with('#') && line.contains("doc") {
            let doc_text = line.trim_start_matches('#').trim();
            doc_lines.insert(0, doc_text);
            i -= 1;
        } else {
            break;
        }
    }

    if doc_lines.is_empty() {
        None
    } else {
        let joined = doc_lines.join(" ");
        source
            .find(&joined)
            .map(|start| &source[start..start + joined.len()])
    }
}

fn extract_attrs_from_region(text: &str) -> Vec<SmlAttr> {
    let h = HeuristicInferencer::new();
    text.lines()
        .filter_map(move |line| {
            let (name, ty) = h.extract_attr(line)?;
            let required = line.trim().ends_with('!');
            let optional = line.trim().ends_with('?');
            let ty = map_type(ty).to_string();
            Some(SmlAttr {
                name: name.to_string(),
                ty,
                optional,
                required,
            })
        })
        .collect()
}

fn extract_links_from_region(text: &str) -> Vec<SmlLink> {
    let h = HeuristicInferencer::new();
    text.lines()
        .filter_map(|line| {
            h.extract_link(line).map(|target| SmlLink {
                target: target.to_string(),
            })
        })
        .collect()
}

fn extract_invariants_from_region(text: &str) -> Vec<SmlInvariant> {
    let h = HeuristicInferencer::new();
    text.lines()
        .filter(|line| h.is_invariant_line(line.trim()))
        .filter_map(|line| {
            let trimmed = line.trim();
            let cond = extract_condition_from_sentence(trimmed);
            let action = extract_action_from_sentence(trimmed);
            if !cond.is_empty() && !action.is_empty() {
                Some(SmlInvariant {
                    condition: cond,
                    action,
                })
            } else {
                None
            }
        })
        .collect()
}

fn extract_params_from_region(text: &str) -> Vec<SmlAttr> {
    text.lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            if trimmed.starts_with("//")
                || trimmed.starts_with('>')
                || trimmed.starts_with('!')
                || trimmed.starts_with("let ")
                || trimmed.starts_with("return")
                || trimmed.starts_with("if ")
            {
                return None;
            }

            let colon = trimmed.find(':')?;
            if colon == 0 {
                return None;
            }
            let name = trimmed[..colon].trim();
            let rest = trimmed[colon + 1..].trim();
            let ty_end = rest
                .find(',')
                .or_else(|| rest.find(')'))
                .or_else(|| rest.find('{'))
                .unwrap_or(rest.len());
            let ty = rest[..ty_end].trim();

            if name.is_empty() || ty.is_empty() {
                return None;
            }

            let optional = ty.starts_with("Option<") || ty.ends_with('?');
            let ty = map_type(ty).to_string();

            Some(SmlAttr {
                name: name.to_string(),
                ty,
                optional,
                required: false,
            })
        })
        .collect()
}

fn extract_steps_from_region(text: &str) -> Vec<SmlStep> {
    text.lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with("//") || trimmed.starts_with('#') {
                return None;
            }

            if trimmed.starts_with("if ")
                || trimmed.starts_with('?')
                || trimmed.starts_with("match ")
            {
                let condition = trimmed
                    .strip_prefix("if ")
                    .or_else(|| trimmed.strip_prefix('?'))
                    .or_else(|| trimmed.strip_prefix("match "))
                    .unwrap_or(trimmed);
                let condition = condition.trim_end_matches('{').trim().to_string();

                let actions: Vec<String> = trimmed
                    .split("->")
                    .skip(1)
                    .map(|s| s.trim().trim_end_matches('{').trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();

                if !actions.is_empty() {
                    Some(SmlStep {
                        condition: Some(condition),
                        pipe: actions,
                    })
                } else {
                    Some(SmlStep {
                        condition: Some(condition),
                        pipe: vec![],
                    })
                }
            } else if trimmed.starts_with("->") || trimmed.starts_with("return") {
                let action = trimmed
                    .strip_prefix("->")
                    .or_else(|| trimmed.strip_prefix("return"))
                    .unwrap_or(trimmed)
                    .trim()
                    .trim_end_matches(';')
                    .to_string();
                Some(SmlStep {
                    condition: None,
                    pipe: vec![action],
                })
            } else {
                None
            }
        })
        .collect()
}

fn extract_steps_from_text(text: &str) -> Vec<SmlStep> {
    text.lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                return None;
            }

            let h = HeuristicInferencer::new();
            if h.is_invariant_line(trimmed) {
                let cond = extract_condition_from_sentence(trimmed);
                let action = extract_action_from_sentence(trimmed);
                if !cond.is_empty() && !action.is_empty() {
                    return Some(SmlStep {
                        condition: Some(cond),
                        pipe: vec![action],
                    });
                }
            }

            if trimmed.contains("->") {
                let actions: Vec<String> = trimmed
                    .split("->")
                    .skip(1)
                    .map(|s| s.trim().trim_end_matches(';').to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
                if !actions.is_empty() {
                    return Some(SmlStep {
                        condition: None,
                        pipe: actions,
                    });
                }
            }

            None
        })
        .collect()
}

// --- Helpers: text extraction ---

fn extract_first_identifier(text: &str) -> &str {
    text.split_whitespace()
        .next()
        .unwrap_or("Unknown")
        .trim_matches(&['"', '\'', '.', ',', ';', ':', '!', '?'][..])
}

fn first_sentence(text: &str) -> Option<&str> {
    let line = text.lines().next()?.trim();
    if line.is_empty() {
        return None;
    }
    line.find('.').map(|i| &line[..i + 1]).or(Some(line))
}

fn extract_condition_from_sentence(text: &str) -> String {
    let lower = text.to_lowercase();
    let keywords = [
        "must ", "always ", "never ", "require", "assert", "guard", "ensure",
    ];
    for kw in &keywords {
        if let Some(pos) = lower.find(kw) {
            let rest = &text[pos..];
            let end = rest
                .find(" then ")
                .or_else(|| rest.find(", "))
                .or_else(|| rest.find(". "))
                .unwrap_or(rest.len());
            return rest[..end].trim().to_string();
        }
    }
    text.trim().to_string()
}

fn extract_action_from_sentence(text: &str) -> String {
    let lower = text.to_lowercase();

    for marker in &[" then ", " -> ", " will ", " should "] {
        if let Some(pos) = lower.find(marker) {
            let rest = text[pos + marker.len()..].trim();
            let end = rest
                .find(". ")
                .or_else(|| rest.find(", "))
                .or_else(|| rest.find(';'))
                .unwrap_or(rest.len());
            return rest[..end].trim().to_string();
        }
    }

    text.lines().last().unwrap_or(text).trim().to_string()
}

fn map_type(ty: &str) -> &str {
    crate::sml::infer::map_type(ty)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_struct_name() {
        assert_eq!(extract_struct_name("struct UserAccount"), "UserAccount");
        assert_eq!(extract_struct_name("pub struct Foo"), "Foo");
        assert_eq!(extract_struct_name("class Bar"), "Bar");
        assert_eq!(extract_struct_name("UserProfile"), "UserProfile");
    }

    #[test]
    fn test_extract_fn_name() {
        assert_eq!(extract_fn_name("fn main"), "main");
        assert_eq!(extract_fn_name("pub fn process_data"), "process_data");
        assert_eq!(extract_fn_name("async def handle"), "handle");
        assert_eq!(extract_fn_name("func run"), "run");
    }

    #[test]
    fn test_compiler_new() {
        let compiler = SmlCompiler::new();
        assert!(compiler.semantic.is_none());
    }

    #[test]
    fn test_extract_first_identifier() {
        assert_eq!(
            extract_first_identifier("DeploymentPolicy rules"),
            "DeploymentPolicy"
        );
        assert_eq!(extract_first_identifier("hello world"), "hello");
    }

    #[test]
    fn test_first_sentence() {
        assert_eq!(
            first_sentence("This is a policy. It governs deployments."),
            Some("This is a policy.")
        );
        assert_eq!(
            first_sentence("Single line without period"),
            Some("Single line without period")
        );
        assert_eq!(first_sentence(""), None);
    }

    #[test]
    fn test_compile_struct() {
        let compiler = SmlCompiler::new();
        let chunk = SmartChunk {
            text: "pub struct UserAccount {\n    pub username: String,\n}".into(),
            structure: Some("pub struct UserAccount".into()),
            level: 0,
            start_line: 0,
            end_line: 3,
        };
        let source = "pub struct UserAccount {\n    pub username: String,\n}";
        let node = compiler.compile(&chunk, source);
        assert_eq!(node.kind(), "type");
        assert_eq!(node.name(), "UserAccount");
    }

    #[test]
    fn test_compile_fn() {
        let compiler = SmlCompiler::new();
        let chunk = SmartChunk {
            text: "fn process_data(input: String) {\n    // process\n}".into(),
            structure: Some("fn process_data".into()),
            level: 0,
            start_line: 0,
            end_line: 3,
        };
        let source = "fn process_data(input: String) {\n    // process\n}";
        let node = compiler.compile(&chunk, source);
        assert_eq!(node.kind(), "flow");
        assert_eq!(node.name(), "process_data");
    }
}
