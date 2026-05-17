# dogma-vdb

Portable vector database in JSONL format. Rustic, zero-cost, MCP-ready.

**Status**: Alpha — core compiles, **152 tests pass**. 4 index backends
(BruteForce, HNSW, Annoy, SQ), CLI, MCP server, file watcher.

## Features

- **4 index backends**: BruteForce (exact), HNSW (approximate graph),
  Annoy (random projection forest), SQ (scalar quantization — orthogonal)
- **Flat embeddings**: contiguous `Vec<f32>` for ~2.5× speedup at scale
- **SQ (i8 quantization)**: ~4× less memory, ~2× faster distance, optional rescore
- **CRUD**: insert, batch insert, delete, update
- **Metadata filtering**: `metadata_eq`, `metadata_contains`, `metadata_exists`, `all_of`
- **JSONL format**: debuggable with `cat`, `grep`, `sed`, versionable with `git`
- **No server**: file-based, zero config, no daemon
- **MCP-ready**: optional MCP server (stdio) for Claude Desktop / Cursor / opencode
- **Watch mode**: optional file watcher for auto-reindexing
- **Pure Rust**: HNSW, Annoy, SQ algorithms are custom implementations

## Quick Start

```rust
use dogma_vdb::prelude::*;

// Collection opens (or creates) a .vdb file, index type from config
let mut col = Collection::open("my_data.vdb")?;
col.insert(Document::new("doc-1", "Rust is fast"))?;
let results = col.search(&[0.1, 0.2, 0.3], 5);
```

## Benchmarks (5K docs, 128-dim)

| Backend | us/query | Recall |
|---------|:--------:|:------:|
| BruteForce | 1,542 | 100% |
| HNSW (ef=50) | 38 | 100% |
| HNSW+Flat | 43 | 100% |
| HNSW+SQ | 65 | ~95% (w/ rescore) |
| Annoy (10 trees) | 3,942 | 100% |
| BF+SQ | 1,569 | ~95% (w/ rescore) |

## Config

```toml
[collection]
index_type = "hnsw"              # bruteforce | hnsw | annoy
index_metric = "cosine"          # cosine | dot | euclidean

# HNSW
hnsw_m = 16
hnsw_ef_construction = 200
hnsw_ef_search = 50
hnsw_flat_embeddings = false

# Annoy
annoy_n_trees = 10
annoy_search_k = -1

# SQ (orthogonal — applies to any backend)
sq = false
sq_rescore = false
```

## Index Backends

| Backend | Algorithm | Type | Incremental | Best for |
|---------|-----------|------|:-----------:|----------|
| BruteForce | Linear scan O(n·d) | Exact | Yes | < 10K docs |
| HNSW | Hierarchical NSW graph | Approx | Yes | < 100K docs |
| Annoy | Random projection forest | Approx | Batch | Static datasets |
| SQ | Scalar quantization i8 | Orthogonal | Yes | Any backend, -4× RAM |

## Crates

| Crate | Description |
|-------|-------------|
| `dogma-vdb` | Core library (storage, index, collection, chunking) |
| `dogma-vdb-cli` | CLI tool (info, list, query, ingest, delete) |
| `dogma-vdb-embed` | Embedder trait definition |
| `dogma-vdb-embed-fastembed` | Fastembed (ONNX) integration (skeleton) |
| `dogma-vdb-mcp` | MCP server over stdio |

## Build & Test

```bash
cargo check
cargo test          # 152 tests
cargo clippy -- -D warnings
cargo fmt --check
cargo run --release --example bench
```

## License

MIT OR Apache-2.0
