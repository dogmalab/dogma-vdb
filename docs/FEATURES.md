# dogma-vdb — Feature Reference & Configuration Guide

> Complete reference of all features, their configuration, and usage.
> Last updated: 2026-06-23

---

## 1. Quick Reference

| Feature | Status | Feature Flag | Default |
|---------|--------|:------------:|:-------:|
| BruteForce Index | ✅ Stable | — | on |
| HNSW Index | ✅ Stable | — | on |
| IVF-PQ Index | ✅ Stable | — | on |
| Scalar Quantization (SQ) | ✅ Stable | — | on |
| BM25 Text Search | ✅ Stable | — | on |
| Reciprocal Rank Fusion (RRF) | ✅ Stable | — | on |
| Hybrid Search Pipeline | ✅ Stable | — | on |
| Metadata Filtering | ✅ Stable | — | on |
| SmartChunker (3 strategies) | ✅ Stable | — | on |
| Binary Storage v2 (mmap) | ✅ Stable | — | on |
| MCP Server (stdio) | ✅ Stable | — | on |
| CLI | ✅ Stable | — | on |
| Cross-Encoder Reranking | ✅ Stable | — | on |
| SIMIL Ingestion Parser | ✅ New | `sml` | off |
| StorageStrategy | ✅ New | `sml` | off |
| File Watcher | ✅ Stable | `watch` | off |
| Syntax Chunker (tree-sitter) | ✅ Stable | `chunker-syntax` | off |
| Python Bindings (PyO3) | ✅ New | — | on |

---

## 2. Index Backends

### 2.1 BruteForceIndex

Exact O(n·d) linear scan. Best for < 10K documents.

```toml
[collection]
index_type = "bruteforce"
index_metric = "cosine"      # cosine | dot | euclidean
```

| Feature | Support |
|---------|---------|
| Incremental insert | ✅ |
| SQ compression | ✅ |
| SQ rescore | ✅ |
| Pre-filtering | ✅ (filter before distance) |
| mmap storage | ✅ |

### 2.2 HnswIndex

Approximate O(log n) via hierarchical navigable small world graph.

```toml
[collection]
index_type = "hnsw"
index_metric = "cosine"

# HNSW-specific parameters
hnsw_m = 16                      # connections per node
hnsw_ef_construction = 200       # build quality
hnsw_ef_search = 50              # query quality (higher = more recall)
hnsw_flat_embeddings = false     # contiguous Vec<f32> for cache efficiency
```

| Feature | Support |
|---------|---------|
| Incremental insert | ✅ |
| SQ compression | ✅ |
| SQ rescore | ✅ |
| Pre-filtering | ❌ (post-filter with k×3 multiplier) |
| mmap storage | ✅ |

### 2.3 IvfPqIndex

Inverted file + Product Quantization. Extreme memory savings.

```toml
[collection]
index_type = "ivf_pq"
index_metric = "cosine"

# IVF-PQ specific
ivf_pq_n_clusters = 256         # K-Means centroids (n_list)
ivf_pq_n_subvectors = 32        # PQ sub-vectors (must be multiple of 8)
ivf_pq_n_probe = 8              # clusters to probe per search
```

| Feature | Support |
|---------|---------|
| Incremental insert | ❌ (full rebuild) |
| SQ compression | ✅ (orthogonal layer) |
| Tombstone delete | ✅ |
| Auto-tuning (rerank flag) | ✅ |
| Persistence (atomic) | ✅ |
| Compaction | ✅ |

### 2.4 Scalar Quantization (SQ)

Orthogonal optimization layer. Compresses f32 → i8. Works with any backend.

```toml
[collection]
sq = true
sq_rescore = true    # recovers recall from 40% to 90%
```

- 4× less RAM
- ~2× faster search
- Rescore: top-k×2 candidates recalculated with f32

---

## 3. Storage

### 3.1 Binary Format v2

```
Offset  Size  Field
0       4     magic: "DVDB"
4       4     version: u32 LE (2)
8       4     dim: u32 LE
12      4     count: u32 LE
16      8     emb_offset: u64 LE
24      —     metadata blocks (id, text, k-v metadata)
emb_offset  —  embeddings f32 LE contiguous (32-byte aligned)
```

- 32-byte alignment for AVX2 SIMD
- ~2.3× smaller, ~7× faster save/load vs JSONL
- Zero-copy via `MmapBackedStorage`

### 3.2 VectorStorage Trait

```rust
pub trait VectorStorage: Send + Sync {
    fn as_bytes(&self) -> &[u8];
    fn as_embeddings(&self) -> &[f32];
    fn flush(&self) -> Result<()>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
}
```

| Implementation | Use case |
|----------------|----------|
| `MemoryBackedStorage` | Tests, volatile pipelines |
| `MmapBackedStorage` | Production, ~0ms cold load |

---

## 4. SIMIL Ingestion Parser (`feature = "sml"`)

Compiles source code and plain text into SIMIL (System Intent Markup Language)
manifests stored in `Document.metadata["sml"]`.

### 4.1 How It Works

```
Input text/code
    ↓
SmartChunker.chunk_text()     → SmartChunks (structure, line ranges)
    ↓
SmlCompiler.compile_batch()   → SmlNodes (Type, Flow, Invariants)
    ↓
serializer::serialize_batch() → String (SIMIL manifest)
    ↓
Document.metadata["sml"]      ← injected automatically
```

### 4.2 Inference Pipeline (Double Pass)

| Pass | Engine | When | Speed |
|------|--------|------|-------|
| 1. Heuristic | Keywords + regex patterns | Always | nanoseconds |
| 2. Semantic | Embedding cosine similarity | If embedder injected | microseconds |

The heuristics detect:
- **Types**: `struct`, `class`, `enum`, `trait`, PascalCase names
- **Flows**: `fn`, `def`, `func`, snake_case names
- **Invariants**: `must`, `always`, `require`, `assert`, `guard`
- **Attributes**: `name: Type` patterns
- **Links**: `*target` references

### 4.3 Usage

```rust
use dogma_vdb::sml::{SmlCompiler, ingest};

// Standalone compilation
let compiler = SmlCompiler::new();
let metadata = ingest(source_code, "main.rs", &compiler);
// metadata["sml"] = "type UserAccount\n> \"...\"\n@username:str!\n..."

// With semantic inference (requires embedder)
let compiler = SmlCompiler::with_embedder(Arc::new(fastembedder));
let metadata = ingest(plain_text, "policy.txt", &compiler);
```

### 4.4 Automatic Integration

When `feature = "sml"` is enabled, `Collection::insert()` and
`Collection::insert_batch()` automatically:

1. Compile each document's text to SIMIL
2. Inject the manifest into `metadata["sml"]`
3. Apply `StorageStrategy` (keep or clear text)

```rust
let mut col = Collection::open_with("code.vdb", "hnsw", "cosine")?;
col.insert(Document::new("doc-1", "pub struct Foo { ... }"))?;
// Document is stored with metadata["sml"] automatically
```

### 4.5 Configuration

```toml
# StorageStrategy is independent of feature flag
# (always available in config, only active when sml feature is on)

[collection]
storage_strategy = "hybrid"       # default: keep original text
# storage_strategy = "symbolic_pure"  # clear text after SML extraction
```

Environment variable:
```bash
export DOGMA_VDB_COLLECTION_STORAGE_STRATEGY=symbolic_pure
```

---

## 5. StorageStrategy

Controls what is stored per document after SML extraction.

### 5.1 Hybrid Mode (Default)

```
Document {
    id: "doc-1",
    text: "pub struct UserAccount { ... }",     ← KEPT
    embedding: [0.1, 0.2, ...],
    metadata: { "sml": "type UserAccount\n..." } ← INJECTED
}
```

**Use case**: RAG pipelines where the LLM needs the original text/code
for context, but also benefits from SIMIL metadata for filtering.

### 5.2 SymbolicPure Mode

```
Document {
    id: "doc-1",
    text: "",                                    ← CLEARED
    embedding: [0.1, 0.2, ...],
    metadata: { "sml": "type UserAccount\n..." } ← INJECTED
}
```

**Use case**: Security-sensitive environments where the original text
contains secrets/IP, or ultra-lightweight edge deployments where every
byte counts.

**Impact on other features**:
- BM25: invisible (zero tokens) — SML takes over text search role
- Reranker: receives empty text — disabled in this mode
- CLI/MCP list: shows empty text — display SML instead

---

## 6. SmartChunker — 3 Strategies

### 6.1 Code Strategy

Auto-detected for: `.rs`, `.py`, `.js`, `.ts`, `.go`

Splits by top-level definitions using pre-compiled regex patterns:
- Rust: `fn`, `struct`, `enum`, `trait`, `impl`
- Python: `def`, `class`, decorators
- JavaScript/TypeScript: `function`, `class`, `interface`, `type`
- Go: `func`, `type struct`, `type interface`

### 6.2 Paragraph Strategy

Splits by `\n\n` boundaries with configurable overlap.
Optional semantic chunking: splits by cosine similarity between
adjacent sentence embeddings (threshold 0.35).

```rust
let chunker = SmartChunker::default()
    .with_semantic(Box::new(embedder));
let chunks = chunker.chunk_text(text, ChunkStrategy::Paragraph);
```

### 6.3 FixedWindow Strategy

Default for: `.md`, `.txt`, `.jsonl`, `.json`, `.yaml`, `.toml`, `.sh`

Fixed-size byte windows with UTF-8 safe boundaries.

---

## 7. Hybrid Search Pipeline

```
Query
  ├── Vector search (HNSW/IVF-PQ/BF) → top_k × multiplier candidates
  ├── BM25 text search (optional)     → top_k × multiplier candidates
  ├── RRF fusion (if both active)     → top_k × 2 candidates
  └── Reranker (optional)             → final top_k
```

### Performance Profiles

| Profile | Vector | BM25 | Reranker | Multiplier | Use case |
|---------|:------:|:----:|:--------:|:----------:|----------|
| `PrecisionLocal` | ✅ | ✅ | ✅ | 5× | Development, MCP |
| `HybridProduction` | ✅ | ✅ | ✅ | 3× | Production at scale |
| `MaxSpeed` | ✅ | ❌ | ❌ | 2× | IoT, edge |

---

## 8. Watch Mode (`feature = "watch"`)

Monitor directories and auto-index files as they change.

```toml
[watch]
enabled = true
source_dirs = ["docs/"]
extensions = ["md", "rs", "py", "js"]
debounce_ms = 500
```

```bash
cargo build --features watch
```

---

## 9. MCP Server

Exposes vector search to MCP-compatible agents (Claude Desktop, Cursor, opencode).

### Tools

| Tool | Description |
|------|-------------|
| `vecdb_query` | Vector search with optional reranking |
| `vecdb_ingest` | Insert a document |
| `vecdb_delete` | Delete by ID list |
| `vecdb_list` | List documents |
| `vecdb_info` | Collection stats |

### Reranking

```bash
DOGMA_RERANK=1 ./target/release/dogma-vdb-mcp
```

---

## 10. Configuration Reference

### 10.1 Full `config.toml`

```toml
[general]
debug = false

[collection]
index_type = "bruteforce"       # bruteforce | hnsw | ivf_pq
index_metric = "cosine"        # cosine | dot | euclidean
storage_strategy = "hybrid"    # hybrid | symbolic_pure

# HNSW
hnsw_m = 16
hnsw_ef_construction = 200
hnsw_ef_search = 50
hnsw_flat_embeddings = false

# IVF-PQ
ivf_pq_n_clusters = 256
ivf_pq_n_subvectors = 32
ivf_pq_n_probe = 8

# SQ
sq = false
sq_rescore = false

[chunker]
chunk_size = 4096
overlap = 128
separator = "\n\n"

[watch]
enabled = false
source_dirs = []
extensions = []
debounce_ms = 500

[mcp]
enabled = false
transport = "stdio"
port = 5000

[embedder]
model = "default"
device = "cpu"
batch_size = 32

[logging]
level = "info"
```

### 10.2 Environment Variables

All config fields can be overridden with `DOGMA_VDB_` prefix:

| Variable | Type | Example |
|----------|------|---------|
| `DOGMA_VDB_COLLECTION_INDEX_TYPE` | string | `hnsw` |
| `DOGMA_VDB_COLLECTION_METRIC` | string | `cosine` |
| `DOGMA_VDB_COLLECTION_STORAGE_STRATEGY` | string | `symbolic_pure` |
| `DOGMA_VDB_COLLECTION_HNSW_M` | usize | `32` |
| `DOGMA_VDB_COLLECTION_HNSW_EF_CONSTRUCTION` | usize | `300` |
| `DOGMA_VDB_COLLECTION_HNSW_EF_SEARCH` | usize | `100` |
| `DOGMA_VDB_COLLECTION_HNSW_FLAT` | bool | `true` |
| `DOGMA_VDB_COLLECTION_IVF_PQ_N_CLUSTERS` | usize | `100` |
| `DOGMA_VDB_COLLECTION_IVF_PQ_N_SUBVECTORS` | usize | `16` |
| `DOGMA_VDB_COLLECTION_IVF_PQ_N_PROBE` | usize | `4` |
| `DOGMA_VDB_COLLECTION_SQ` | bool | `true` |
| `DOGMA_VDB_COLLECTION_SQ_RESCORE` | bool | `true` |
| `DOGMA_VDB_WATCH_ENABLED` | bool | `true` |
| `DOGMA_VDB_MCP_ENABLED` | bool | `true` |
| `DOGMA_VDB_MCP_PORT` | u16 | `8080` |
| `DOGMA_VDB_LOG_LEVEL` | string | `debug` |
| `DOGMA_VDB_CHUNKER_CHUNK_SIZE` | usize | `2048` |
| `DOGMA_VDB_CHUNKER_OVERLAP` | usize | `64` |

### 10.3 Feature Flags

| Flag | Command | Deps added | Description |
|------|---------|-----------|-------------|
| `sml` | `cargo build --features sml` | none | SIMIL ingestion parser + StorageStrategy |
| `watch` | `cargo build --features watch` | notify, crossbeam-channel | File system watcher |
| `chunker-syntax` | `cargo build --features chunker-syntax` | tree-sitter + grammars | AST-based syntax chunking |

```bash
# Build with all features
cargo build --features "sml,watch,chunker-syntax"

# Run tests with SML
cargo test --features sml

# Run tests with all features
cargo test --all-features
```

---

## 11. Crates

| Crate | Description | Dependencies |
|-------|-------------|-------------|
| `dogma-vdb` | Core library | serde, serde_json, thiserror, rayon, wide, bytemuck, memmap2, once_cell, toml, log, regex-lite |
| `dogma-vdb-cli` | CLI tool | dogma-vdb, clap, anyhow |
| `dogma-vdb-embed` | Embedder trait | zero deps |
| `dogma-vdb-embed-fastembed` | FastEmbed ONNX | dogma-vdb-embed, fastembed |
| `dogma-vdb-mcp` | MCP server | dogma-vdb, rmcp, tokio |
| `dogma-vdb-rerank` | Cross-Encoder | ort, tokenizers, ndarray |
| `dogma-vdb-rag` | RAG pipeline | dogma-vdb, fastembed |
| `dogma-vdb-python` | Python bindings | pyo3 |
