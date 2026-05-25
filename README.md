# dogma-vdb

Portable vector database in JSONL format. Rustic, zero-cost, MCP-ready.

**Status**: Beta — core compiles, **202 tests pass**, SIMD-accelerated,
binary native format v2 (mmap-ready), 3 index backends + SQ orthogonal,
CLI, MCP server, file watcher, FastEmbed ONNX integration, LangChain MCP adapter,
Cross-Encoder reranking pipeline, **SmartChunker with 5 strategies + Tree-sitter AST + Semantic embeddings**.

---

## 🌟 What's New (Recent Milestones)

The indexing strategy and storage ecosystem have been redesigned:

- **IVF-PQ backend** replaces the retired Annoy index — Product Quantization
  for extreme memory savings (~300 KB for 5K docs, 8× less than HNSW/BF)
- **`VectorStorage` trait** decouples vector storage from index lifecycle
- **`MmapBackedStorage`** — memory-mapped zero-copy loading (~0ms cold start)
- **Binary format v2** — 32-byte aligned padding for AVX2-safe SIMD
- **HNSW+SQ recall fixed** — from 0-60% to 90% (midpoint bias + rescore)
- **IVF-PQ tuning** — SIMD-aligned `m_subspaces` (multiple of 8), auto-tuning
  with rerank flag halves probe count for speed
- **Two-stage reranking** — new `dogma-vdb-rerank` crate with `CrossEncoderReranker`
  trait, `StubReranker`, and MCP integration via `DOGMA_RERANK=1`

---

## 🛠️ Storage Architecture

Collection
```
  ├── storage: BinStorage                  ← JSONL + binary persistence
  ├── emb_storage: Arc<dyn VectorStorage>  ← Contiguous embeddings (shared)
  └── index: Box<dyn Index>
        └── storage: Arc<dyn VectorStorage>  ← Auto-injected by collection
              │
              ▼
     [VectorStorage Trait]
        ├── MemoryBackedStorage   (tests, volatile pipelines)
        └── MmapBackedStorage     (production, ~0ms load via virtual memory)
```

### Cold-load instant (~0ms) via mmap

Thanks to `MmapBackedStorage` (backed by `memmap2`), the CLI and MCP server
no longer pay the I/O penalty of reading, parsing, and cloning embeddings
into RAM on every start. The native binary file is mapped directly into the
OS virtual address space — startup drops from **9ms → ~0ms**.

### 32-byte alignment (AVX2-ready)

The binary v2 writer injects dynamic padding bytes after the JSON/TOML
metadata header, guaranteeing the flat vector section starts at a 32-byte
aligned offset. This allows CPU intrinsics (AVX2, AVX-512, NEON) to
operate at maximum speed without alignment faults.

---

## Features

- **SIMD-accelerated** — dot product, cosine, euclidean via `wide` crate
  (SSE/AVX2 on x86, NEON on ARM). HNSW search ~3.6× faster, build ~4.3×.
- **Binary native format v2** — header + metadata + raw f32 embeddings contiguous.
  32-byte aligned. ~2.3× smaller, ~7× faster save/load vs JSONL. Auto-migration.
- **3 index backends**: BruteForce (exact), HNSW (approximate graph),
  IVF-PQ (inverted file + product quantization)
- **SQ (i8 scalar quantization)**: orthogonal — applies to any backend.
  ~4× less memory, optional f32 rescore
- **Flat embeddings**: contiguous `Vec<f32>` for ~2.5× speedup at scale
- **CRUD**: insert, batch insert, delete, update
- **Metadata filtering**: `metadata_eq`, `metadata_contains`, `metadata_exists`, `all_of`
- **JSONL export**: `collection.export_jsonl()` for debug with `cat`, `grep`, `sed`
- **No server**: file-based, zero config, no daemon
- **MCP-ready**: optional MCP server (stdio) for Claude Desktop / Cursor / opencode
- **Two-stage reranking** — MCP query supports `rerank=true` for Cross-Encoder
  rescoring after vector retrieval. Enabled via `DOGMA_RERANK=1`
- **Cross-Encoder reranking crate** — `dogma-vdb-rerank` with agnostic
  `CrossEncoderReranker` trait (no dogma-vbd dependency)
- **IVF-PQ SIMD-aligned** — `m_subspaces` must be multiple of 8 for AVX2-safe
  lookup tables. Validated at construction time.
- **Rerank-aware IVF-PQ** — when `rerank_enabled=true`, IVF-PQ halves its
  probe count to favour speed; recall is recovered by the Cross-Encoder pass.
- **LangChain MCP adapter**: `examples/langchain_mcp.py` — zero-code integration
- **Watch mode**: optional file watcher for auto-reindexing
- **FastEmbed ONNX**: `FastEmbedder` with all-MiniLM-L6-v2 (384-dim, ~90MB model)
- **Pure Rust**: HNSW, IVF-PQ, SQ, reranker are custom implementations
- **Zero unsafe** in production logic — unsafe blocks strictly isolated to
  byte-conversion in the storage trait
- **SIGBUS defensive docs**: explicit documentation on `MmapBackedStorage`

---

## 🧠 Smart Chunker — 7 Strategies

| Strategy | Engine | Best for |
|----------|--------|----------|
| **Code** (Rust/Python/JS/Go) | Tree-sitter AST *or* regex fallback | Source files with functions, classes, structs |
| **Markdown** | Heading hierarchy (`#`, `##`, `###`) | Docs, blogs, wikis |
| **JSONL** | Line-by-line | Structured logs, datasets |
| **Paragraph** | Double-newline boundaries | Plain text, prose |
| **Semantic** | Embedding cosine distance | Dense prose, books, essays without headings |

```bash
# AST chunking (feature-gated):
cargo build --features chunker-syntax

# Semantic chunking (requires embedder):
use dogma_vdb::smart_chunker::SmartChunker;
let chunker = SmartChunker::default()
    .with_semantic(Box::new(embedder));
let chunks = chunker.chunk_text(long_essay, FileType::Semantic);
```

> **Sub-chunkers stay sequential.** Concurrency lives only at the batch level (`rayon::par_iter()` over `InputFile` slices).

---

## 📊 Comparison With Other Vector Databases

| Dimension | dogma-vdb 🦀 | ChromaDB 🐍 | LanceDB 🗄️ | Qdrant 🦀 | USearch ⚡ |
|-----------|:-----------:|:-----------:|:----------:|:---------:|:----------:|
| Runtime | **Binary** (native) | Python 300 MB | Pip + Arrow | Docker | C library |
| Deps (core) | **3** | ~200 | ~150 | ~100 | **0** |
| Formato | **JSONL + Bin v2** | SQLite+Parquet | Lance columnar | Binario | Binario |
| Async | **No** (sync) | Sync API | Tokio | Tokio | **No** |
| MCP nativo | ✅ **Sí** | ❌ | ❌ | ❌ | ❌ |
| Chunking | ✅ 7 estrategias | ❌ split_text | ❌ | ❌ | ❌ |
| mmap ~0ms | ✅ MmapBacked | ❌ | ✅ Lance | ❌ | ❌ |
| SQ ortogonal | ✅ Cualquier backend | ❌ | ❌ | ❌ | ❌ |
| Reranking | ✅ Cross-Encoder | ❌ | ❌ | ❌ | ❌ |

| Bench (5K/128d) | dogma-vdb HNSW | ChromaDB | Qdrant |
|:----------------|:--------------:|:--------:|:------:|
| Query (μs) | **77** | ~4,000 | ~200 |
| RAM 5K docs | ~3 MB | ~50 MB | ~8 MB |
| Carga fría | **~0 ms** (mmap) | ~250 ms | ~100 ms |
| Recall@10 | **100%** | ~95% | ~99% |

> dogma-vdb es a las vector DBs lo que **SQLite** es a las relacionales: embebido, zero-config, debuggeable con herramientas UNIX.

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
| **HNSW (ef=50)** | **77** | **100%** | 3.6× vs no-SIMD |
| HNSW+SQ+Rescore | 73 | 90% | ~4× less RAM |
| HNSW+Flat | 79 | 100% | Cache win >100K |
| **IVF-PQ** (n_list=16) | **128** | **95%** | ~300 KB RAM (8× less) |
| BruteForce | 1,460 | 100% | Exact |
| BF+SQ | 1,584 | 40% | 4× less RAM |

**Build time**: HNSW 1.5s (4.3× vs no-SIMD), IVF-PQ 14ms (K-Means + PQ)

**Storage benchmark** (5K docs 384-dim):

| Format | Size | Save | Load |
|--------|:---:|:----:|:----:|
| Binary v2 (mmap) | 8.2 MB | 7.1 ms | **~0 ms** |
| JSONL | 18.6 MB | 55 ms | 57 ms |

---

## ⚡ Index Backend Matrix

| Backend | Algorithm | Type | Incremental | RAM (5K docs) | Speed | Ideal for |
|---------|-----------|------|:-----------:|:-------------:|:-----:|-----------|
| BruteForce | Linear scan O(n·d) | Exact | ✅ | Full | 1,460 us/q | < 10K docs |
| HNSW | Hierarchical NSW graph | Approx | ✅ | High | 77 us/q | High recall (< 100K docs) |
| IVF-PQ | Inverted file + Product Quantization | Approx | ❌ (batch) | Minimal (~300 KB) | 128 us/q | Max resource savings |
| SQ (i8) | Scalar quantization | Orthogonal | ✅ | 4× reduction | Varies | Savings layer on any index |

> **HNSW + SQ + Rescore** achieves **90% recall** with identical 73 us/query
> latency, thanks to corrected midpoint bias and per-document scale/bias.

### IVF-PQ tuning

| Parameter | Default | Description |
|-----------|:-------:|-------------|
| `n_list` | 100 | Number of IVF centroids (K-Means) |
| `n_probe` | 5 | Clusters to probe per search |
| `m_subspaces` | 32 | PQ sub-spaces (must be multiple of 8 for SIMD) |

When `DOGMA_RERANK=1` is set, IVF-PQ auto-reduces its probe count by half
(minimum 2) to prioritise speed, relying on the Cross-Encoder reranker to
recover recall in the second stage.

---

## Config

```toml
[collection]
index_type = "hnsw"              # bruteforce | hnsw | ivf_pq
index_metric = "cosine"          # cosine | dot | euclidean

# HNSW
hnsw_m = 16
hnsw_ef_construction = 200
hnsw_ef_search = 50
hnsw_flat_embeddings = false

# IVF-PQ
ivf_pq_n_clusters = 100          # n_list — K-Means centroids
ivf_pq_n_subvectors = 32         # m_subspaces — PQ sub-vectors (multiple of 8)
ivf_pq_n_probe = 5               # clusters to probe per search

# SQ (orthogonal — applies to any backend)
sq = false
sq_rescore = false
```

---

## Crates

| Crate | Description |
|-------|-------------|
| `dogma-vdb` | Core library (storage, index, collection, chunking) |
| `dogma-vdb-cli` | CLI tool (info, list, query, ingest, delete, export) |
| `dogma-vdb-embed` | Embedder trait definition |
| `dogma-vdb-embed-fastembed` | Fastembed (ONNX) integration (384-dim MiniLM-L6-v2) |
| `dogma-vdb-mcp` | MCP server over stdio |
| `dogma-vdb-rerank` | Agnostic Cross-Encoder reranking (`CrossEncoderReranker` trait) |

### Feature Flags

| Flag | Enables | Deps added |
|------|---------|-----------|
| *(none)* | Core only — 3 deps | serde, serde_json, thiserror |
| `watch` | File watcher | notify, crossbeam-channel |
| `chunker-syntax` | Tree-sitter AST code chunking | tree-sitter + 4 grammars |

---

## Build & Test

```bash
cargo check --workspace
cargo test          # 202 tests
cargo test --features chunker-syntax  # 202 + tree-sitter tests
cargo clippy --all-targets
cargo fmt --check
cargo run --release --bin dogma-vdb-benchmarks 2>/dev/null || echo "(benchmarks need data)"
```

---

## Security

- Zero `unsafe` blocks in production logic — strictly isolated to
  byte-alignment conversion in `VectorStorage` trait
- `MmapBackedStorage` includes explicit SIGBUS guard documentation
- No shell/command execution
- No hardcoded secrets
- No network dependencies in core
- All file operations use typed errors (no panics in production paths)
- `cargo audit` clean (2 allowed warnings, both transitive via fastembed)

## License

MIT OR Apache-2.0
