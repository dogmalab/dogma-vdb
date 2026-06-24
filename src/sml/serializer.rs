//! Serializer: SIMIL AST → string.

use crate::sml::ast::{SmlAttr, SmlInvariant, SmlNode, SmlStep};

/// Serialize a single SIMIL node to its string representation.
pub fn serialize(node: &SmlNode) -> String {
    let mut out = String::with_capacity(128);
    serialize_node(node, &mut out);
    out
}

/// Serialize a batch of SIMIL nodes, separated by newlines.
pub fn serialize_batch(nodes: &[SmlNode]) -> String {
    let mut out = String::with_capacity(nodes.len() * 128);
    for (i, node) in nodes.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        serialize_node(node, &mut out);
    }
    out
}

fn serialize_node(node: &SmlNode, out: &mut String) {
    match node {
        SmlNode::Type {
            name,
            doc,
            attrs,
            links,
            invariants,
        } => {
            out.push_str("type ");
            out.push_str(name);
            out.push('\n');

            if let Some(d) = doc {
                out.push_str("> \"");
                out.push_str(d);
                out.push_str("\"\n");
            }

            for attr in attrs {
                serialize_attr(attr, out);
            }

            for link in links {
                out.push('*');
                out.push_str(&link.target);
                out.push('\n');
            }

            for inv in invariants {
                serialize_invariant(inv, out);
            }
        }
        SmlNode::Flow { name, params, body } => {
            out.push_str("-> ");
            out.push_str(name);
            out.push('(');

            for (i, p) in params.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(&p.name);
                if p.required {
                    out.push('!');
                } else if p.optional {
                    out.push('?');
                }
            }

            out.push_str(") :\n");

            for step in body {
                serialize_step(step, out);
            }
        }
    }
}

fn serialize_attr(attr: &SmlAttr, out: &mut String) {
    out.push('@');
    out.push_str(&attr.name);
    out.push(':');
    out.push_str(&attr.ty);
    if attr.required {
        out.push('!');
    } else if attr.optional {
        out.push('?');
    }
    out.push('\n');
}

fn serialize_invariant(inv: &SmlInvariant, out: &mut String) {
    out.push_str("! ");
    out.push_str(&inv.condition);
    out.push_str(" >> -> ");
    out.push_str(&inv.action);
    out.push('\n');
}

fn serialize_step(step: &SmlStep, out: &mut String) {
    out.push_str("   ");
    if let Some(cond) = &step.condition {
        out.push('?');
        out.push(' ');
        out.push_str(cond);
        out.push_str(" >> ");
    }
    for (i, action) in step.pipe.iter().enumerate() {
        if i > 0 {
            out.push_str(" >> ");
        }
        out.push_str("-> ");
        out.push_str(action);
    }
    out.push('\n');
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sml::ast::*;

    #[test]
    fn test_serialize_type_minimal() {
        let node = SmlNode::Type {
            name: "Config".into(),
            doc: None,
            attrs: vec![],
            links: vec![],
            invariants: vec![],
        };
        assert_eq!(serialize(&node), "type Config\n");
    }

    #[test]
    fn test_serialize_type_full() {
        let node = SmlNode::Type {
            name: "UserAccount".into(),
            doc: Some("User identity entity".into()),
            attrs: vec![
                SmlAttr {
                    name: "username".into(),
                    ty: "str".into(),
                    optional: false,
                    required: true,
                },
                SmlAttr {
                    name: "email".into(),
                    ty: "str".into(),
                    optional: false,
                    required: false,
                },
                SmlAttr {
                    name: "role".into(),
                    ty: "Enum[ADMIN, USER]".into(),
                    optional: false,
                    required: false,
                },
            ],
            links: vec![SmlLink {
                target: "Profile".into(),
            }],
            invariants: vec![SmlInvariant {
                condition: "@role:ADMIN".into(),
                action: "grant_full_access".into(),
            }],
        };
        let s = serialize(&node);
        assert!(s.contains("type UserAccount"));
        assert!(s.contains("> \"User identity entity\""));
        assert!(s.contains("@username:str!"));
        assert!(s.contains("@email:str"));
        assert!(s.contains("@role:Enum[ADMIN, USER]"));
        assert!(s.contains("*Profile"));
        assert!(s.contains("! @role:ADMIN >> -> grant_full_access"));
    }

    #[test]
    fn test_serialize_flow() {
        let node = SmlNode::Flow {
            name: "process_data".into(),
            params: vec![
                SmlAttr {
                    name: "input".into(),
                    ty: "str".into(),
                    optional: false,
                    required: true,
                },
                SmlAttr {
                    name: "config".into(),
                    ty: "path".into(),
                    optional: true,
                    required: false,
                },
            ],
            body: vec![
                SmlStep {
                    condition: Some("input.valid".into()),
                    pipe: vec!["validate".into(), "transform".into()],
                },
                SmlStep {
                    condition: None,
                    pipe: vec!["store".into()],
                },
            ],
        };
        let s = serialize(&node);
        assert!(s.contains("-> process_data(input!, config?) :"));
        assert!(s.contains("? input.valid >> -> validate >> -> transform"));
        assert!(s.contains("-> store"));
    }

    #[test]
    fn test_serialize_batch() {
        let nodes = vec![
            SmlNode::Type {
                name: "A".into(),
                doc: None,
                attrs: vec![],
                links: vec![],
                invariants: vec![],
            },
            SmlNode::Type {
                name: "B".into(),
                doc: None,
                attrs: vec![],
                links: vec![],
                invariants: vec![],
            },
        ];
        let s = serialize_batch(&nodes);
        assert!(s.contains("type A"));
        assert!(s.contains("type B"));
        let lines: Vec<&str> = s.lines().collect();
        assert!(lines.len() >= 2);
    }
}
