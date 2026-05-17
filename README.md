# dogma-vdb

Portable vector database in JSONL format. Rustic, zero-cost, MCP-ready.

**Status**: Beta — core compiles, **155 tests pass**, SIMD-accelerated,
binary native format, 4 index backends, CLI, MCP server, file watcher,
FastEmbed ONNX integration, LangChain MCP adapter.

## Features

- **SIMD-accelerated** — dot product, cosine, and euclidean via `wide` crate
  (SSE/AVX2 on x86, NEON on ARM). HNSW search ~3.6x faster, build ~4.3x.
- **Binary native format** — header + metadata + raw f32 embeddings contiguous.
  ~2.3x smaller, ~7x faster save/load vs JSONL. Auto-migration from old format.
- **4 index backends**: BruteForce (exact), HNSW (approximate graph),
  Annoy (random projection forest), SQ (scalar quantization — orthogonal)
- **Flat embeddings**: contiguous `Vec<f32>` for ~2.5× speedup at scale
- **SQ (i8 quantization)**: ~4× less memory, optional f32 rescore
- **CRUD**: insert, batch insert, delete, update
- **Metadata filtering**: `metadata_eq`, `metadata_contains`, `metadata_exists`, `all_of`
- **JSONL export**: `collection.export_jsonl()` for debug with `cat`, `grep`, `sed`
- **No server**: file-based, zero config, no daemon
- **MCP-ready**: optional MCP server (stdio) for Claude Desktop / Cursor / opencode
- **LangChain MCP adapter**: `examples/langchain_mcp.py` — zero-code integration
- **Watch mode**: optional file watcher for auto-reindexing
- **FastEmbed ONNX**: `FastEmbedder` with all-MiniLM-L6-v2 (384-dim, ~90MB model)
- **Pure Rust**: HNSW, Annoy, SQ algorithms are custom implementations

## Quick Start

```rust
use dogma_vdb::prelude::*;

let mut col = Collection::open("my_data.vdb")?;
col.insert(Document::new("doc-1", "Rust is fast"))?;
let results = col.search(&[0.1, 0.2, 0.3], 5);

// Export for debugging
col.export_jsonl("my_data.jsonl")?;
```

## Benchmarks (5K docs, 128-dim, SIMD on)

| Backend | us/query | Recall | Notes |
|---------|:--------:|:------:|-------|
| HNSW (ef=50) | **77** | 100% | 3.6x vs no-SIMD |
| HNSW+SQ | 71 | 0-60% | 4x menos RAM |
| HNSW+Flat | 79 | 100% | Cache win >100K |
| BruteForce | 1,460 | 100% | Exacto |
| BF+SQ | 1,584 | 40% | 4x menos RAM |
| Annoy (10 trees) | 3,216 | 100% | Build 3ms |

**Build time**: HNSW 1.5s (4.3x vs no-SIMD), Annoy 3ms

**Storage benchmark** (5K docs 384-dim):
| Format | Size | Save | Load |
|--------|:---:|:----:|:----:|
| Binary | 8.2 MB | 7.1 ms | 9.1 ms |
| JSONL | 18.6 MB | 55 ms | 57 ms |

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
| `dogma-vdb-cli` | CLI tool (info, list, query, ingest, delete, export) |
| `dogma-vdb-embed` | Embedder trait definition |
| `dogma-vdb-embed-fastembed` | Fastembed (ONNX) integration (384-dim MiniLM-L6-v2) |
| `dogma-vdb-mcp` | MCP server over stdio |

## Build & Test

```bash
cargo check --workspace
cargo test          # 155 tests
cargo clippy -- -D warnings
cargo fmt --check
cargo run --release --example bench
cargo audit
```

## Security

- Zero `unsafe` blocks in production code
- No shell/command execution
- No hardcoded secrets
- No network dependencies in core
- All file operations use typed errors (no panics in production paths)
- `cargo audit` clean (2 allowed warnings, both transitive via fastembed)

## License

MIT OR Apache-2.0
