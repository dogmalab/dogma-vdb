//! Runtime configuration loaded from `config.toml` or env vars.
//!
//! Config sources (first match wins):
//! 1. `$XDG_CONFIG_HOME/dogma-vdb/config.toml` (or `~/.config/dogma-vdb/config.toml`)
//! 2. `./config.toml` in the working directory
//! 3. Environment variables with `DOGMA_VDB_` prefix
//! 4. Built-in defaults

use once_cell::sync::Lazy;
use serde::Deserialize;
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Top-level config
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    #[serde(default)]
    pub general: General,
    #[serde(default)]
    pub chunker: ChunkerConfig,
    #[serde(default)]
    pub collection: CollectionConfig,
    #[serde(default)]
    pub watch: WatchConfig,
    #[serde(default)]
    pub mcp: McpConfig,
    #[serde(default)]
    pub embedder: EmbedderConfig,
    #[serde(default)]
    pub logging: LoggingConfig,
}

// ---------------------------------------------------------------------------
// Sub-configs
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Deserialize, Clone)]
pub struct General {
    #[serde(default = "General::default_debug")]
    pub debug: bool,
}
impl General {
    const fn default_debug() -> bool {
        false
    }
}

#[derive(Debug, Default, Deserialize, Clone)]
pub struct ChunkerConfig {
    #[serde(default = "ChunkerConfig::default_chunk_size")]
    pub chunk_size: usize,
    #[serde(default = "ChunkerConfig::default_overlap")]
    pub overlap: usize,
    #[serde(default = "ChunkerConfig::default_separator")]
    pub separator: String,
}
impl ChunkerConfig {
    const fn default_chunk_size() -> usize {
        4096
    }
    const fn default_overlap() -> usize {
        128
    }
    fn default_separator() -> String {
        "\n\n".to_string()
    }
}

#[derive(Debug, Default, Deserialize, Clone)]
pub struct CollectionConfig {
    #[serde(default = "CollectionConfig::default_path")]
    pub path: PathBuf,
    #[serde(default = "CollectionConfig::default_index_type")]
    pub index_type: String,
    #[serde(default = "CollectionConfig::default_metric")]
    pub index_metric: String,
    // HNSW-specific (only used when index_type == "hnsw")
    #[serde(default = "CollectionConfig::default_hnsw_m")]
    pub hnsw_m: usize,
    #[serde(default = "CollectionConfig::default_hnsw_ef_construction")]
    pub hnsw_ef_construction: usize,
    #[serde(default = "CollectionConfig::default_hnsw_ef_search")]
    pub hnsw_ef_search: usize,
    #[serde(default)]
    pub hnsw_flat_embeddings: bool,
    // Annoy-specific (only used when index_type == "annoy")
    #[serde(default = "CollectionConfig::default_annoy_n_trees")]
    pub annoy_n_trees: usize,
    #[serde(default = "CollectionConfig::default_annoy_search_k")]
    pub annoy_search_k: i32,
    // Scalar Quantization (orthogonal — applies to any backend)
    #[serde(default)]
    pub sq: bool,
    /// When `sq=true` and this is `true`, run exact f32 rescoring on
    /// the top `k*2` quantized results to recover recall (default: `false`).
    #[serde(default)]
    pub sq_rescore: bool,
}
impl CollectionConfig {
    fn default_path() -> PathBuf {
        PathBuf::from("data/default.vdb")
    }
    fn default_index_type() -> String {
        "bruteforce".into()
    }
    fn default_metric() -> String {
        "cosine".into()
    }
    fn default_hnsw_m() -> usize {
        16
    }
    fn default_hnsw_ef_construction() -> usize {
        200
    }
    fn default_hnsw_ef_search() -> usize {
        50
    }
    fn default_annoy_n_trees() -> usize {
        10
    }
    fn default_annoy_search_k() -> i32 {
        -1
    }
}

#[derive(Debug, Default, Deserialize, Clone)]
pub struct WatchConfig {
    #[serde(default = "WatchConfig::default_enabled")]
    pub enabled: bool,
    #[serde(default)]
    pub source_dirs: Vec<PathBuf>,
    #[serde(default)]
    pub extensions: Vec<String>,
    #[serde(default = "WatchConfig::default_debounce")]
    pub debounce_ms: u64,
}
impl WatchConfig {
    const fn default_enabled() -> bool {
        false
    }
    fn default_debounce() -> u64 {
        500
    }
}

#[derive(Debug, Default, Deserialize, Clone)]
pub struct McpConfig {
    #[serde(default = "McpConfig::default_enabled")]
    pub enabled: bool,
    #[serde(default)]
    pub transport: McpTransport,
    #[serde(default = "McpConfig::default_port")]
    pub port: u16,
}
impl McpConfig {
    const fn default_enabled() -> bool {
        false
    }
    const fn default_port() -> u16 {
        5000
    }
}

#[derive(Debug, Default, Deserialize, Clone, PartialEq)]
pub enum McpTransport {
    #[default]
    Stdio,
    Http,
    WebSocket,
}

#[derive(Debug, Default, Deserialize, Clone)]
pub struct EmbedderConfig {
    #[serde(default = "EmbedderConfig::default_model")]
    pub model: String,
    #[serde(default = "EmbedderConfig::default_device")]
    pub device: String,
    #[serde(default = "EmbedderConfig::default_batch")]
    pub batch_size: usize,
}
impl EmbedderConfig {
    fn default_model() -> String {
        "default".into()
    }
    fn default_device() -> String {
        "cpu".into()
    }
    fn default_batch() -> usize {
        32
    }
}

#[derive(Debug, Default, Deserialize, Clone)]
pub struct LoggingConfig {
    #[serde(default = "LoggingConfig::default_level")]
    pub level: String,
    #[serde(default)]
    pub output: Option<PathBuf>,
}
impl LoggingConfig {
    fn default_level() -> String {
        "info".into()
    }
}

// ---------------------------------------------------------------------------
// Lazy global config
// ---------------------------------------------------------------------------

pub static CONFIG: Lazy<Config> = Lazy::new(|| {
    // 1. XDG / home config
    let home_dir = std::env::var("HOME").ok().map(PathBuf::from);
    let xdg_config = std::env::var("XDG_CONFIG_HOME")
        .ok()
        .map(PathBuf::from)
        .or(home_dir.clone())
        .map(|p| p.join(".config").join("dogma-vdb").join("config.toml"));

    // 2. Local config.toml
    let local_config = PathBuf::from("config.toml");

    let cfg_str: Option<String> = xdg_config
        .and_then(|p| std::fs::read_to_string(p).ok())
        .or_else(|| std::fs::read_to_string(&local_config).ok());

    let mut cfg: Config = if let Some(ref s) = cfg_str {
        toml::from_str(s).unwrap_or_default()
    } else {
        Config::default()
    };

    // 3. Override with DOGMA_VDB_* env vars
    for (key, val) in std::env::vars() {
        if !key.starts_with("DOGMA_VDB_") {
            continue;
        }
        let k = &key[10..]; // strip prefix
        match k {
            "CHUNKER_CHUNK_SIZE" => {
                cfg.chunker.chunk_size = val.parse().unwrap_or(cfg.chunker.chunk_size)
            }
            "CHUNKER_OVERLAP" => cfg.chunker.overlap = val.parse().unwrap_or(cfg.chunker.overlap),
            "CHUNKER_SEPARATOR" => cfg.chunker.separator = val,
            "COLLECTION_PATH" => cfg.collection.path = PathBuf::from(val),
            "WATCH_ENABLED" => cfg.watch.enabled = val == "true",
            "MCP_ENABLED" => cfg.mcp.enabled = val == "true",
            "MCP_PORT" => cfg.mcp.port = val.parse().unwrap_or(cfg.mcp.port),
            "LOG_LEVEL" => cfg.logging.level = val,
            "COLLECTION_INDEX_TYPE" => cfg.collection.index_type = val,
            "COLLECTION_METRIC" => cfg.collection.index_metric = val,
            "COLLECTION_HNSW_M" => {
                cfg.collection.hnsw_m = val.parse().unwrap_or(cfg.collection.hnsw_m)
            }
            "COLLECTION_HNSW_EF_CONSTRUCTION" => {
                cfg.collection.hnsw_ef_construction =
                    val.parse().unwrap_or(cfg.collection.hnsw_ef_construction)
            }
            "COLLECTION_HNSW_EF_SEARCH" => {
                cfg.collection.hnsw_ef_search = val.parse().unwrap_or(cfg.collection.hnsw_ef_search)
            }
            "COLLECTION_HNSW_FLAT" => cfg.collection.hnsw_flat_embeddings = val == "true",
            "COLLECTION_ANNOY_N_TREES" => {
                cfg.collection.annoy_n_trees = val.parse().unwrap_or(cfg.collection.annoy_n_trees)
            }
            "COLLECTION_ANNOY_SEARCH_K" => {
                cfg.collection.annoy_search_k = val.parse().unwrap_or(cfg.collection.annoy_search_k)
            }
            "COLLECTION_SQ" => cfg.collection.sq = val == "true",
            "COLLECTION_SQ_RESCORE" => cfg.collection.sq_rescore = val == "true",
            _ => {}
        }
    }

    cfg
});

impl Default for Config {
    fn default() -> Self {
        Self {
            general: General { debug: false },
            chunker: ChunkerConfig {
                chunk_size: 4096,
                overlap: 128,
                separator: "\n\n".into(),
            },
            collection: CollectionConfig {
                path: PathBuf::from("data/default.vdb"),
                index_type: "bruteforce".into(),
                index_metric: "cosine".into(),
                hnsw_m: 16,
                hnsw_ef_construction: 200,
                hnsw_ef_search: 50,
                hnsw_flat_embeddings: false,
                annoy_n_trees: 10,
                annoy_search_k: -1,
                sq: false,
                sq_rescore: false,
            },
            watch: WatchConfig {
                enabled: false,
                source_dirs: Vec::new(),
                extensions: Vec::new(),
                debounce_ms: 500,
            },
            mcp: McpConfig {
                enabled: false,
                transport: McpTransport::Stdio,
                port: 5000,
            },
            embedder: EmbedderConfig {
                model: "default".into(),
                device: "cpu".into(),
                batch_size: 32,
            },
            logging: LoggingConfig {
                level: "info".into(),
                output: None,
            },
        }
    }
}
