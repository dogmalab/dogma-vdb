//! dogma-vdb-mcp — MCP server for dogma-vdb.
//!
//! Exposes tools for querying, ingesting, listing, and deleting
//! documents in .vdb collections.  Compatible with any MCP client
//! (Claude Desktop, Cursor, opencode, etc.).
//!
//! ## Transports
//!
//! * `stdio` (default) — JSON-RPC over stdin/stdout.
//! * `sse`    — Streamable HTTP / SSE transport (requires `sse` feature).

use rmcp::{
    handler::server::wrapper::{Json, Parameters},
    schemars, serve_server, tool, tool_router,
    transport::IntoTransport,
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::OnceLock;
use tokio::io::{stdin, stdout};
use tracing_subscriber::EnvFilter;

mod rerank_adapter;

use dogma_vdb::rerank::Reranker;

// ---------------------------------------------------------------------------
// CLI argument parsing (minimal, no clap dependency)
// ---------------------------------------------------------------------------

struct CliArgs {
    transport: String,
    #[cfg(feature = "sse")]
    port: u16,
}

fn parse_args() -> CliArgs {
    let args: Vec<String> = std::env::args().collect();
    let mut transport = String::from("stdio");
    #[cfg(feature = "sse")]
    let mut port = 8080u16;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--transport" => {
                i += 1;
                if i < args.len() {
                    transport = args[i].clone();
                }
            }
            #[cfg(feature = "sse")]
            "--port" => {
                i += 1;
                if i < args.len() {
                    if let Ok(p) = args[i].parse::<u16>() {
                        port = p;
                    }
                }
            }
            _ => {} // ignore unknown flags
        }
        i += 1;
    }
    CliArgs {
        transport,
        #[cfg(feature = "sse")]
        port,
    }
}

// ---------------------------------------------------------------------------
// Global reranker (lazily initialised once)
// ---------------------------------------------------------------------------

static RERANKER: OnceLock<rerank_adapter::DogmaRerankerAdapter> = OnceLock::new();

fn get_reranker() -> Option<&'static rerank_adapter::DogmaRerankerAdapter> {
    RERANKER.get()
}

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
    #[schemars(description = "Enable two-stage reranking (default: false)")]
    #[serde(default)]
    rerank: bool,
    #[schemars(description = "Original query text (required when rerank=true)")]
    #[serde(default)]
    query_text: Option<String>,
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

        let search_k = if params.rerank {
            params.k * 5
        } else {
            params.k
        };
        let query_f32: Vec<f32> = params.embedding.iter().map(|&x| x as f32).collect();
        let results = col.search(&query_f32, search_k);

        // Stage 2: reranking (Cross-Encoder) if enabled
        if params.rerank {
            let query_text = params.query_text.as_deref().unwrap_or("").to_string();
            if query_text.is_empty() {
                return Err("rerank requires 'query_text' parameter".into());
            }

            // Extract raw documents from search results
            let mut docs: Vec<dogma_vdb::doc::Document> =
                results.into_iter().map(|r| r.document).collect();

            // Apply reranker
            let reranker = get_reranker().ok_or_else(|| {
                "rerank requested but no reranker initialised. \
                 Start the server with DOGMA_RERANK=1 or omit rerank"
                    .to_string()
            })?;
            reranker
                .rerank(&query_text, &mut docs)
                .map_err(|e| format!("rerank failed: {e}"))?;

            // Truncate to the requested k after reranking
            docs.truncate(params.k);

            // Rebuild ScoredDocuments (scores are now from the Cross-Encoder)
            // Use the reranker's scores as `score` for transparency.
            let items: Vec<QueryResultItem> = docs
                .into_iter()
                .map(|d| {
                    let text = if d.text.len() > 200 {
                        format!("{}...", &d.text[..197])
                    } else {
                        d.text
                    };
                    QueryResultItem {
                        score: 0.0, // Cross-Encoder scores not preserved in this flow
                        id: d.id,
                        text,
                    }
                })
                .collect();
            let count = items.len();
            Ok(Json(QueryOutput {
                results: items,
                count,
            }))
        } else {
            // No reranking — return results as-is
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
// SSE / Streamable HTTP transport (behind `sse` feature)
// ---------------------------------------------------------------------------

#[cfg(feature = "sse")]
async fn run_sse_server(server: VdbServer, port: u16) -> anyhow::Result<()> {
    use axum::{
        body::Body,
        extract::{Request, State},
        response::Response,
        routing::any,
        BoxError, Router,
    };
    use http_body_util::BodyExt;
    use rmcp::transport::streamable_http_server::{
        session::local::LocalSessionManager, StreamableHttpServerConfig, StreamableHttpService,
    };
    use std::convert::Infallible;

    let config = StreamableHttpServerConfig::default().disable_allowed_hosts();
    let session_manager = std::sync::Arc::new(LocalSessionManager::default());

    let svc = StreamableHttpService::new(move || Ok(server.clone()), session_manager, config);

    async fn sse_handler(
        State(svc): State<StreamableHttpService<VdbServer, LocalSessionManager>>,
        req: Request,
    ) -> Response {
        let resp = svc.handle(req).await;
        let (parts, body) = resp.into_parts();
        // Convert BoxBody<Bytes, Infallible> -> Body (axum)
        // Since Infallible can never happen, we coerce it to BoxError.
        let body = Body::new(body.map_err(|e: Infallible| -> BoxError { match e {} }));
        Response::from_parts(parts, body)
    }

    let app = Router::new()
        .route("/{*path}", any(sse_handler))
        .with_state(svc);

    let addr = format!("0.0.0.0:{}", port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;

    tracing::info!("Starting dogma-vdb MCP server (SSE transport) on http://{addr}");
    tracing::info!("Tools: vecdb_query, vecdb_ingest, vecdb_delete, vecdb_list, vecdb_info");
    tracing::info!("MCP Streamable HTTP endpoint — POST /  (application/json)");
    tracing::info!("SSE endpoint — GET / (Accept: text/event-stream)");

    axum::serve(listener, app).await?;
    Ok(())
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

    // Initialise the Cross-Encoder reranker when `DOGMA_RERANK=1` is set.
    // Attempts to load a real ONNX model from DOGMA_RERANK_MODEL_PATH and
    // DOGMA_RERANK_TOKENIZER_PATH env vars.  Falls back to StubReranker
    // (deterministic mock) if the model can't be found.
    if std::env::var("DOGMA_RERANK").as_deref() == Ok("1") {
        let model_path = std::env::var("DOGMA_RERANK_MODEL_PATH")
            .unwrap_or_else(|_| "models/bge-reranker-base/model.onnx".into());
        let tokenizer_path = std::env::var("DOGMA_RERANK_TOKENIZER_PATH")
            .unwrap_or_else(|_| "models/bge-reranker-base/tokenizer.json".into());

        match dogma_vdb_rerank::OnnxReranker::new(&model_path, &tokenizer_path, 512, 2) {
            Ok(onnx) => {
                let reranker = rerank_adapter::DogmaRerankerAdapter::new(Box::new(onnx));
                let _ = RERANKER.set(reranker);
                tracing::info!("Reranker initialised (OnnxReranker, model={model_path})");
            }
            Err(e) => {
                let reranker = rerank_adapter::DogmaRerankerAdapter::new(Box::new(
                    dogma_vdb_rerank::StubReranker,
                ));
                let _ = RERANKER.set(reranker);
                tracing::warn!(
                    "OnnxReranker failed to load ({e}); using StubReranker (mock) instead. \
                     Set DOGMA_RERANK_MODEL_PATH and DOGMA_RERANK_TOKENIZER_PATH"
                );
            }
        }
    } else {
        tracing::info!("Reranker disabled (set DOGMA_RERANK=1 to enable)");
    }

    let args = parse_args();
    let server = VdbServer;

    match args.transport.as_str() {
        "sse" => {
            #[cfg(feature = "sse")]
            {
                run_sse_server(server, args.port).await?;
            }
            #[cfg(not(feature = "sse"))]
            {
                anyhow::bail!(
                    "SSE transport requires the 'sse' feature. \
                     Rebuild with `cargo build --features sse`"
                );
            }
        }
        _ => {
            tracing::info!("Starting dogma-vdb MCP server (stdio transport)");
            tracing::info!(
                "Tools: vecdb_query, vecdb_ingest, vecdb_delete, vecdb_list, vecdb_info"
            );

            serve_server(server, (stdin(), stdout()).into_transport())
                .await
                .map_err(|e| anyhow::anyhow!("MCP server exited: {e}"))?;
        }
    }

    Ok(())
}
