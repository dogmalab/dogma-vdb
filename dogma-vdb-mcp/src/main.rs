//! dogma-vdb-mcp — MCP server for dogma-vdb.
//!
//! Exposes tools for querying, ingesting, listing, and deleting
//! documents in .vdb collections.  Compatible with any MCP client
//! (Claude Desktop, Cursor, opencode, etc.).

use rmcp::{
    handler::server::wrapper::{Json, Parameters},
    schemars, serve_server, tool, tool_router,
    transport::IntoTransport,
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tokio::io::{stdin, stdout};
use tracing_subscriber::EnvFilter;

// ---------------------------------------------------------------------------
// Input / output types
// ---------------------------------------------------------------------------

fn default_k() -> usize {
    10
}

// -- Collection params (reused) -------------------------------------------

#[derive(Deserialize, schemars::JsonSchema, Default)]
struct CollectionParams {
    #[schemars(description = "Path to the .vdb collection file")]
    path: String,
    #[schemars(description = "Index type: bruteforce or hnsw")]
    #[serde(default = "default_index_type")]
    index_type: String,
    #[schemars(description = "Distance metric: cosine, dot, euclidean")]
    #[serde(default = "default_metric")]
    metric: String,
}

fn default_index_type() -> String {
    "bruteforce".into()
}
fn default_metric() -> String {
    "cosine".into()
}

// -- Query -----------------------------------------------------------------

#[derive(Deserialize, schemars::JsonSchema, Default)]
struct QueryParams {
    path: String,
    #[schemars(description = "Query embedding as JSON array of floats, e.g. [0.1, 0.2, 0.3]")]
    embedding: Vec<f64>,
    #[schemars(description = "Number of results (default: 10)")]
    #[serde(default = "default_k")]
    k: usize,
    #[serde(default = "default_index_type")]
    index_type: String,
    #[serde(default = "default_metric")]
    metric: String,
}

#[derive(Serialize, schemars::JsonSchema)]
struct QueryResultItem {
    score: f64,
    id: String,
    text: String,
}

#[derive(Serialize, schemars::JsonSchema)]
struct QueryOutput {
    results: Vec<QueryResultItem>,
    count: usize,
}

// -- Ingest ----------------------------------------------------------------

#[derive(Deserialize, schemars::JsonSchema, Default)]
struct IngestParams {
    path: String,
    id: String,
    text: String,
    #[serde(default)]
    embedding: Option<Vec<f64>>,
    #[serde(default = "default_index_type")]
    index_type: String,
    #[serde(default = "default_metric")]
    metric: String,
}

#[derive(Serialize, schemars::JsonSchema)]
struct IngestOutput {
    id: String,
    document_count: usize,
}

// -- Delete ----------------------------------------------------------------

#[derive(Deserialize, schemars::JsonSchema, Default)]
struct DeleteParams {
    path: String,
    ids: Vec<String>,
    #[serde(default = "default_index_type")]
    index_type: String,
    #[serde(default = "default_metric")]
    metric: String,
}

#[derive(Serialize, schemars::JsonSchema)]
struct DeleteOutput {
    deleted: usize,
}

// -- List ------------------------------------------------------------------

#[derive(Serialize, schemars::JsonSchema)]
struct ListDocumentItem {
    id: String,
    text: String,
    dimension: usize,
    metadata_keys: usize,
}

#[derive(Serialize, schemars::JsonSchema)]
struct ListOutput {
    documents: Vec<ListDocumentItem>,
    count: usize,
}

// -- Info ------------------------------------------------------------------

#[derive(Serialize, schemars::JsonSchema)]
struct InfoOutput {
    name: String,
    document_count: usize,
    index_type: String,
    metric: String,
}

// ---------------------------------------------------------------------------
// MCP Server
// ---------------------------------------------------------------------------

#[derive(Default, Clone)]
struct VdbServer;

fn open_collection(
    path: &str,
    index_type: &str,
    metric: &str,
) -> Result<dogma_vdb::collection::Collection, String> {
    dogma_vdb::collection::Collection::open_with(PathBuf::from(path), index_type, metric)
        .map_err(|e| format!("open failed: {e}"))
}

#[tool_router(server_handler)]
impl VdbServer {
    #[tool(
        name = "vecdb_query",
        description = "Search a .vdb collection by embedding vector. Returns the k most similar documents."
    )]
    fn query(
        &self,
        Parameters(params): Parameters<QueryParams>,
    ) -> Result<Json<QueryOutput>, String> {
        let col = open_collection(&params.path, &params.index_type, &params.metric)?;
        let query_f32: Vec<f32> = params.embedding.iter().map(|&x| x as f32).collect();
        let results = col.search(&query_f32, params.k);

        let items: Vec<QueryResultItem> = results
            .into_iter()
            .map(|r| {
                let text = if r.document.text.len() > 200 {
                    format!("{}...", &r.document.text[..197])
                } else {
                    r.document.text
                };
                QueryResultItem {
                    score: r.score as f64,
                    id: r.document.id,
                    text,
                }
            })
            .collect();

        let count = items.len();
        Ok(Json(QueryOutput {
            results: items,
            count,
        }))
    }

    #[tool(
        name = "vecdb_ingest",
        description = "Insert a document into a .vdb collection."
    )]
    fn ingest(
        &self,
        Parameters(params): Parameters<IngestParams>,
    ) -> Result<Json<IngestOutput>, String> {
        let mut col = open_collection(&params.path, &params.index_type, &params.metric)?;
        let mut builder = dogma_vdb::doc::Document::builder(&params.id, params.text);
        if let Some(emb) = params.embedding {
            builder = builder.embedding(emb.into_iter().map(|x| x as f32).collect());
        }
        col.insert(builder.build())
            .map_err(|e| format!("insert failed: {e}"))?;
        Ok(Json(IngestOutput {
            id: params.id,
            document_count: col.len(),
        }))
    }

    #[tool(
        name = "vecdb_delete",
        description = "Delete documents from a .vdb collection by their IDs."
    )]
    fn delete(
        &self,
        Parameters(params): Parameters<DeleteParams>,
    ) -> Result<Json<DeleteOutput>, String> {
        let mut col = open_collection(&params.path, &params.index_type, &params.metric)?;
        let ids_ref: Vec<&str> = params.ids.iter().map(|s| s.as_str()).collect();
        let deleted = col
            .delete(&ids_ref)
            .map_err(|e| format!("delete failed: {e}"))?;
        Ok(Json(DeleteOutput { deleted }))
    }

    #[tool(
        name = "vecdb_list",
        description = "List all documents in a .vdb collection."
    )]
    fn list(
        &self,
        Parameters(params): Parameters<CollectionParams>,
    ) -> Result<Json<ListOutput>, String> {
        let col = open_collection(&params.path, &params.index_type, &params.metric)?;
        let documents: Vec<ListDocumentItem> = col
            .documents()
            .map(|d| {
                let text = if d.text.len() > 100 {
                    format!("{}...", &d.text[..97])
                } else {
                    d.text.clone()
                };
                ListDocumentItem {
                    id: d.id.clone(),
                    text,
                    dimension: d.dimension(),
                    metadata_keys: d.metadata.len(),
                }
            })
            .collect();
        let count = documents.len();
        Ok(Json(ListOutput { documents, count }))
    }

    #[tool(
        name = "vecdb_info",
        description = "Show metadata and statistics about a .vdb collection."
    )]
    fn info(
        &self,
        Parameters(params): Parameters<CollectionParams>,
    ) -> Result<Json<InfoOutput>, String> {
        let col = open_collection(&params.path, &params.index_type, &params.metric)?;
        Ok(Json(InfoOutput {
            name: col.name().to_string(),
            document_count: col.len(),
            index_type: params.index_type,
            metric: params.metric,
        }))
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let server = VdbServer;

    tracing::info!("Starting dogma-vdb MCP server (stdio transport)");
    tracing::info!("Tools: vecdb_query, vecdb_ingest, vecdb_delete, vecdb_list, vecdb_info");

    serve_server(server, (stdin(), stdout()).into_transport())
        .await
        .map_err(|e| anyhow::anyhow!("MCP server exited: {e}"))?;

    Ok(())
}
