//! Optional MCP server (feature = "mcp").
//!
//! Exposes tools (`vecdb_query`, `vecdb_ingest`, `vecdb_list`,
//! `vecdb_delete`) via the Model Context Protocol so that any
//! MCP‑compatible agent (Claude Desktop, Cursor, opencode, …) can
//! query the vector database.

use crate::error::Result;
use std::path::PathBuf;

/// Configuration for [`serve_stdio`] and [`serve_http`].
#[derive(Debug, Clone)]
pub struct McpConfig {
    pub db_dir: PathBuf,
    pub transport: McpTransport,
}

#[derive(Debug, Clone, PartialEq)]
pub enum McpTransport {
    Stdio,
    Http { port: u16 },
    WebSocket { port: u16 },
}

/// Run the MCP server over stdio (default).
///
/// The server reads JSON‑RPC 2.0 requests from stdin and writes
/// responses to stdout.
pub async fn serve_stdio(_config: McpConfig) -> Result<()> {
    todo!()
}

/// Run the MCP server over HTTP / WebSocket.
pub async fn serve_http(_config: McpConfig) -> Result<()> {
    todo!()
}
