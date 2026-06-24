//! AST nodes for SIMIL representation.

/// Top-level SIMIL node.
///
/// `name` is owned (`String`) because it is derived from structure
/// names that may not share the source text lifetime.
/// Doc, attrs, links, and invariants borrow from the source text.
pub enum SmlNode {
    /// `type Name` — entity, struct, class, enum, module.
    Type {
        name: String,
        doc: Option<String>,
        attrs: Vec<SmlAttr>,
        links: Vec<SmlLink>,
        invariants: Vec<SmlInvariant>,
    },
    /// `-> name(params) :` — function, process, flow.
    Flow {
        name: String,
        params: Vec<SmlAttr>,
        body: Vec<SmlStep>,
    },
}

/// Attribute: `@name:type`.
pub struct SmlAttr {
    pub name: String,
    pub ty: String,
    pub optional: bool,
    pub required: bool,
}

/// Link: `*target`.
pub struct SmlLink {
    pub target: String,
}

/// Invariant: `! condition >> -> action`.
pub struct SmlInvariant {
    pub condition: String,
    pub action: String,
}

/// Step in a flow body: `? condition >> -> pipe[0] >> -> pipe[1]`.
pub struct SmlStep {
    pub condition: Option<String>,
    pub pipe: Vec<String>,
}

impl SmlNode {
    /// Returns the node type name for debugging/display.
    pub fn kind(&self) -> &'static str {
        match self {
            SmlNode::Type { .. } => "type",
            SmlNode::Flow { .. } => "flow",
        }
    }

    /// Returns the node name.
    pub fn name(&self) -> &str {
        match self {
            SmlNode::Type { name, .. } => name,
            SmlNode::Flow { name, .. } => name,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_kind_and_name() {
        let node = SmlNode::Type {
            name: "UserAccount".into(),
            doc: Some("User identity entity".into()),
            attrs: vec![SmlAttr {
                name: "username".into(),
                ty: "str".into(),
                optional: false,
                required: true,
            }],
            links: vec![],
            invariants: vec![],
        };
        assert_eq!(node.kind(), "type");
        assert_eq!(node.name(), "UserAccount");
    }

    #[test]
    fn test_flow_node() {
        let node = SmlNode::Flow {
            name: "process_data".into(),
            params: vec![SmlAttr {
                name: "input".into(),
                ty: "str".into(),
                optional: false,
                required: false,
            }],
            body: vec![SmlStep {
                condition: Some("input.valid".into()),
                pipe: vec!["validate".into(), "transform".into()],
            }],
        };
        assert_eq!(node.kind(), "flow");
        assert_eq!(node.name(), "process_data");
    }
}
