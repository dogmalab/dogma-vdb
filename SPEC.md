# dogma-vdb — Functional Specification

> Based on source code audit (2026-05-25). Covers the main crate
> `dogma-vdb` + 6 workspace crates: CLI, MCP, embed, fastembed, rerank, benchmarks.

---

## 1. Overview

Portable vector database in Rust. Binary `.vdb` format with
legacy JSONL format auto-detection. Zero async in core, no server,
config-driven.

**Problem**: ChromaDB is heavy (300 MB pip), LanceDB is complex
(50K LOC, 200+ deps). Something tiny, portable, debuggable
with `cat`/`grep`/`sed` is needed, that runs anywhere with a single binary.

**Target user**: Developers who need local ANN for RAG
or datasets < 100K vectors, without wanting to spin up servers or install
Python.

---

## 2. Index Backends

Each backend implements the `Index` trait:

```rust
pub trait Index: Send + Sync {
    fn insert(&mut self, docs: &[Document]);
    fn delete(&mut self, ids: &[&str]) -> usize;
    fn search(&self, query: &[f32], k: usize) -> Vec<ScoredDocument>;
    fn search_filtered(&self, query: &[f32], k: usize, filter: &(dyn Fn(&Document) -> bool + Sync)) -> Vec<ScoredDocument>;
    fn documents(&self) -> &[Document];
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool { self.len() == 0 }
    fn set_storage(&mut self, _storage: Arc<dyn VectorStorage>) {}
}
```

### RF-01: BruteForceIndex
- **Description**: Exact search O(n·d) by linear scan.
- **Optional SQ**: compresses f32→i8 for ~4× less RAM, ~2× faster.
- **Optional Rescore**: recalculates top-k*2 with f32 to recover recall.
- **Pre-filtering**: applies filter BEFORE computing distance (efficient).
- **Status**: IMPLEMENTED (~517 LOC).

### RF-02: HnswIndex
- **Description**: Approximate search O(log n) via hierarchical graph (Malkov & Yashunin 2016).
- **Config**: `M` (connections per node), `ef_construction`, `ef_search`.
- **Deterministic random level**: SplitMix64 (no `rand` dependency).
- **flat_embeddings**: Contiguous `Vec<f32>` instead of `Vec<Vec<f32>>` (reduces cache misses).
- **Optional SQ**: graph built with exact f32, search with i8.
- **Optional Rescore**: top-k*2 candidates → f32 rescore (recall 90%+).
- **Diversity heuristic**: avoids cycles at layer 0.
- **Status**: IMPLEMENTED (~1,061 LOC).

### RF-03: IvfPqIndex
- **Description**: IVF + Product Quantization. Partitions space with K-Means, compresses sub-vectors to u8.
- **Config**: `n_list`, `n_probe`, `m_subspaces` (multiple of 8), `metric`, `rerank_enabled`.
- **Build** (batch):
  1. K-Means on embeddings (k-means++ init, max 20 iterations).
  2. Assignment to centroid + partition into sub-vectors.
  3. PQ codebook: 256 centroids per subspace.
  4. u8 codes per document.
- **Search**: query→centroid distance, asymmetric LUTs, u8 code scan.
- **Auto-tuning**: if `rerank_enabled=true`, effective n_probe is halved.
- **Insert/Delete**: full index rebuild.
- **Persistence**: atomic (write-tmp + rename), soft-delete, compaction.
- **Validation**: `m_subspaces % 8 == 0` (SIMD alignment).
- **Status**: IMPLEMENTED (~921 LOC index + 724 LOC persistence).

### RF-04: SQ — Scalar Quantization
- **Description**: Orthogonal optimization layer. Compresses f32→i8. 4× less RAM, 2× faster.
- **Algorithm**: `scale = (max - min) / 255.0`, `bias = (max + min) / 2.0`, `i8 = clamp((f32 - bias) / scale, -128, 127)`.
- **Functions**: `quantize()`, `quantize_query()`, `dot_i8()`, `score_i8()`, `rescore()`.
- **Orthogonal**: works with BruteForce, HNSW and IVF-PQ.
- **Document always stores f32**: SQ is only for in-memory search.
- **Status**: IMPLEMENTED (~294 LOC).

### RF-05: BM25 Text Index
- **Description**: Lightweight inverted text index for hybrid search.
- **Formula**: Standard BM25Okapi (k₁=1.2, b=0.75).
- **Tokenization**: split on non-alphanumeric + lowercase.
- **API**: `search(text, k) -> Vec<(doc_index, score)>`.
- **Status**: IMPLEMENTED (~194 LOC).

### RF-06: RRF — Reciprocal Rank Fusion
- **Description**: Combines two ranked lists (vector + BM25) using standard RRF.
- **Formula**: `score(d) = Σ(1 / (k + rank_i(d)))` where k=60.
- **Status**: IMPLEMENTED (~122 LOC).

---

## 3. Storage

### RF-07: BinStorage v2
- **Binary format** (`.vdb`):
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
- **Padding**: 32-byte alignment for AVX2.
- **Auto-detection**: Collection.open() reads magic bytes `DVDB` — if it doesn't match, returns error.
- **embedding_region()**: reads only header (24 bytes) to get offset/dim/count without loading metadata into RAM.
- **Status**: IMPLEMENTED (~563 LOC).

### RF-08: Export JSONL (via Collection)
- **Format**: JSONL (one line per document) — self-describing, debuggable with `cat`/`grep`.
- **Usage**: `collection.export_jsonl(path)` exports documents to JSONL for debug.
- **Note**: the core no longer loads JSONL. Export is inline via `serde_json::to_string`.
- **Status**: IMPLEMENTED (in collection.rs).

### RF-09: VectorStorage Trait
- **Purpose**: abstract contiguous embedding storage (RAM or mmap).
- **API**:
  ```rust
  pub trait VectorStorage: Send + Sync {
      fn as_bytes(&self) -> &[u8];
      fn as_embeddings(&self) -> &[f32];  // unsafe: u8→f32 reinterpret (isolated, audited)
      fn flush(&self) -> Result<()>;
      fn len(&self) -> usize;
      fn is_empty(&self) -> bool;
  }
  ```
- **MemoryBackedStorage**: `Vec<u8>` backend. For tests, volatile pipelines.
- **MmapBackedStorage**: memmap2 (~0ms load, OS pages on demand).
  - `open(path)` — maps the complete embedding region.
  - Includes `advise(memmap2::Advice::Random)` to reduce page faults on sequential reads.
- **Status**: IMPLEMENTED (~297 LOC trait + 563 LOC storage).

---

## 4. Collection — High-Level API

### RF-10: Collection
```rust
Collection::open(path) -> Result<Self>                      // config-driven
Collection::open_with(path, index_type, metric) -> Result<Self>  // override
Collection::insert(doc) -> Result<()>                         // single insert + persist
Collection::insert_batch(docs) -> Result<()>                  // batch insert + persist
Collection::delete(ids) -> Result<usize>                       // delete + persist
Collection::update(doc) -> Result<()>                         // delete + insert
Collection::search(query, k) -> Vec<ScoredDocument>           // vector search
Collection::search_query(embedder, text, k) -> Vec<ScoredDocument>  // text→embed→search
Collection::search_filtered(query, k, filter) -> Vec<ScoredDocument>  // with filter
Collection::hybrid_search(query_vec, query_text, bm25, reranker, pipeline) -> Vec<ScoredDocument>
Collection::export_jsonl(path) -> Result<()>                  // export to JSONL
Collection::documents() -> Iterator<Item = &Document>
Collection::embedding_storage() -> Option<&Arc<dyn VectorStorage>>
Collection::len() / is_empty() / name() / path()
```
- **Auto-detection**: binary vs legacy JSONL format in `open()`.
- **3 backends**: configurable via `index_type` (bruteforce|hnsw|ivf_pq).
- **Status**: IMPLEMENTED (~808 LOC).

### RF-11: Hybrid Search Pipeline
1. **Extract**: `candidate_multiplier × top_k` from each active engine (vector + BM25).
2. **Fuse**: RRF if both engines active, keeps `2 × top_k`.
3. **Rerank**: if `PerformanceProfile` enables it and a reranker is available, reorder with Cross-Encoder.
4. **Performance Profiles**: `PrecisionLocal`, `ProduccionHibrido`, `VelocidadExtrema`.
   - `use_bm25()`, `use_reranker()`, `candidate_multiplier()`.
- **Status**: IMPLEMENTED (integration in Collection + profile config).

---

## 5. Distances (SIMD)

### RF-12: Distance Metrics
- **Cosine** [−1, 1]: normalized cosine similarity.
- **Dot**: direct dot product.
- **Euclidean** [0, ∞): euclidean distance (negated internally).
- **SIMD**: `wide` crate (f32x8 = SSE/AVX2 on x86, NEON on ARM).
- **Fallback**: graceful for elements < 8. No `unsafe`.
- **score_i8**: distance in integer arithmetic for SQ.
- **Status**: IMPLEMENTED (~277 LOC).

---

## 6. Document Model

### RF-13: Document
- `id: String` — unique identifier.
- `text: String` — textual content.
- `embedding: Vec<f32>` — embedding vector (can be empty).
- `metadata: HashMap<String, String>` — arbitrary key-value pairs.
- **Fluent Builder**: `Document::builder(id, text).embedding(vec).metadata(k, v).build()`.
- **Serde**: Serialize + Deserialize (JSONL-ready).
- **Status**: IMPLEMENTED (~205 LOC).

---

## 7. Metadata Filtering

### RF-14: Filter API
- `Filter` = `Box<dyn Fn(&Document) -> bool>`.
- `metadata_eq(key, value)` — exact equality.
- `metadata_contains(key, substr)` — substring match.
- `metadata_exists(key)` — key present.
- `all_of(filters)` — logical AND.
- Inline closures: `|doc| doc.metadata_val("lang") == Some("en")`.
- **Behavior by backend**:
  - BruteForce: pre-filter (before distance).
  - HNSW/IVF-PQ: post-filter with multiplier k×3–5.
- **Status**: IMPLEMENTED (~122 LOC).

---

## 8. Smart Chunker

### RF-15: SmartChunker
- Auto-detects `ChunkStrategy` by extension:
  - `.rs`, `.py`, `.js`, `.ts`, `.go` → `Code`.
  - `.txt`, `.md`, `.jsonl`, `.json`, `.yaml`, `.toml`, `.sh` → `FixedWindow`.
  - Any other → `FixedWindow`.
  - `Paragraph` can be explicitly assigned for semantic chunking.
- Each chunk: `SmartChunk { text, structure: Option<String>, level: usize, start_line, end_line }`.
- **Status**: IMPLEMENTED (~682 LOC main module).

### RF-16: CodeChunker (regex)
- **Description**: Splits source code by top-level definitions.
- **Pre-compiled patterns**: Rust (`fn`, `impl`, `struct`, `enum`, `trait`, `mod`),
  Python (`def`, `class`), JS/TS (`function`, `class`, `const`, `interface`), Go (`func`, `type`).
- **Auto-dispatch**: detects language by content (keywords) if the extension is ambiguous.
- **Fallback**: If no pattern matches, subdivision by lines.
- **Status**: IMPLEMENTED (~153 LOC).

### RF-17: ParagraphChunker (Integrated Semantic)
- **Description**: Splits generic text by `\n\n` with configurable overlap.
- **Safety fix**: start never goes backward + char boundaries guaranteed (prevents infinite loops and UTF-8 panics).
- **Semantic chunking**: `chunk_semantic()` method that uses `Embedder` to split by cosine similarity between adjacent sentences (threshold 0.35).
  - Fallback: without embedder or on failure → paragraph chunking.
- **Status**: IMPLEMENTED (~208 LOC).

### RF-18: FixedWindowChunker
- **Description**: Splits any text into fixed-size windows with overlap.
- **Replaces**: the previous Markdown, JSONL, and plain text strategies.
- **UTF-8 safety**: all splits use `.is_char_boundary()` — zero panics.
- **Subdivision**: if a chunk exceeds `max_size`, it's subdivided by lines.
- **Status**: IMPLEMENTED (~120 LOC).

---

## 9. Embedder Trait

### RF-22: Embedder
```rust
pub trait Embedder: Send + Sync {
    fn embed(&self, text: &str) -> Result<Vec<f32>>;
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
    fn dimension(&self) -> usize;
}
```
- **Implementations**: `dogma-vdb-embed-fastembed` (ONNX via fastembed.rs).
- **Status**: IMPLEMENTED (trait: ~28 LOC, fastembed: ~80 LOC).

---

## 10. Reranker Trait

### RF-23: Reranker
```rust
pub trait Reranker: Send + Sync {
    fn rerank(&self, query: &str, documents: &mut Vec<Document>) -> Result<()>;
}
```
- **NoRerank**: default no-op implementation (leaves order intact).
- **Integration**: used by `hybrid_search()` when the profile enables it.
- **ONNX Implementation**: in `dogma-vdb-rerank` (Cross-Encoder via ort + tokenizers).
- **Status**: IMPLEMENTED (trait: ~69 LOC, ONNX runtime: ~177 LOC).

---

## 11. Watch Mode

### RF-24: File Watcher (feature = "watch")
- **Dependencies**: `notify` v8 + `crossbeam-channel`.
- **Events**: `WatchEvent::Modified(path)`, `WatchEvent::FileError(path, error)`.
- **start_watching()**: spawns thread, returns `Receiver<WatchEvent>`.
- **Config**: source_dirs, extensions, debounce_ms (default 500).
- **walkdir()**: recursive scan with extension filter.
- **Status**: IMPLEMENTED (~316 LOC).

---

## 12. Configuration

### RF-25: Config System
- **Sources** (first match wins):
  1. `$XDG_CONFIG_HOME/dogma-vdb/config.toml` (~/.config/dogma-vdb/config.toml)
  2. `./config.toml` in working directory
  3. Environment variables `DOGMA_VDB_` (e.g. `DOGMA_VDB_DEBUG=true`)
  4. Hardcoded default values
- **Global**: `lazy_static CONFIG` (OnceCell/Lazy).
- **Sections**:
  ```toml
  [general]
  debug = false

  [collection]
  index_type = "bruteforce"    # bruteforce | hnsw | ivf_pq
  index_metric = "cosine"      # cosine | dot | euclidean
  sq = false
  sq_rescore = false
  hnsw_m = 16
  hnsw_ef_construction = 200
  hnsw_ef_search = 50
  hnsw_flat_embeddings = false
  ivf_pq_n_clusters = 256
  ivf_pq_n_subvectors = 8
  ivf_pq_n_probe = 8

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
  transport = "stdio"          # stdio | http | websocket
  port = 5000

  [embedder]
  model = "default"
  device = "cpu"
  batch_size = 32

  [logging]
  level = "info"
  ```
- **PerformanceProfile**: `PrecisionLocal` (5x, rerank yes), `ProduccionHibrido` (3x, rerank yes), `VelocidadExtrema` (2x, rerank no).
- **QueryPipelineConfig**: `profile` + `top_k`.
- **Status**: IMPLEMENTED (~397 LOC).

---

## 13. Memory

### RF-26: Memory Guard
- **Purpose**: prevent OOM on large operations (insert, build_index, chunking).
- **Source**: `/proc/meminfo` on Linux.
- **Levels**: `Normal`, `Low` (free < 15%), `Critical` (free < 5%).
- **ensure_memory()**: aborts operation if `Critical`, warns if `Low`.
- **Status**: IMPLEMENTED (~170 LOC).

---

## 14. Crate `dogma-vdb-cli` — Command Line Interface

### RF-27: CLI
- **Dependencies**: `clap`, `dogma-vdb`, `serde_json`, `anyhow`.
- **Commands**:
  - `query <path> <k> <query_vec...>` — vector search.
  - `ingest <path> <jsonl>` — insert documents from JSONL.
  - `delete <path> <id>` — delete by ID.
  - `list <path>` — list documents.
  - `info <path>` — collection statistics.
- **Status**: IMPLEMENTED (~335 LOC).

---

## 15. Crate `dogma-vdb-mcp` — MCP Server

### RF-28: MCP Server
- **Dependencies**: `rmcp`, `tokio`, `serde`, `tracing`, `dogma-vdb`, `dogma-vdb-rerank`.
- **Transport**: stdio (for now).
- **Tools**:
  - `vecdb_query` — search vectors.
  - `vecdb_ingest` — insert documents.
  - `vecdb_delete` — delete by ID.
  - `vecdb_list` — list documents.
  - `vecdb_info` — statistics.
- **Reranker**: integrates `OnnxReranker` when `DOGMA_RERANK=1`.
- **Status**: IMPLEMENTED (~406 LOC server + ~109 LOC rerank adapter).

---

## 16. Crate `dogma-vdb-rerank` — Cross-Encoder Reranker

### RF-29: ONNX Reranker
- **Dependencies**: `ort` (ONNX Runtime), `tokenizers`, `rayon`, `ndarray`.
- **Model**: Cross-Encoder (MiniLM-L6-v2) downloaded from HuggingFace.
- **API**: `compute_scores(query, texts) -> Vec<(usize, f32)>`.
- **Batch**: parallelism with rayon.
- **Status**: IMPLEMENTED (~177 LOC).

---

## 17. Crate `dogma-vdb-benchmarks` — Grid Benchmark

### RF-30: Benchmark Grid
- **Purpose**: automatically find configuration sweet spots.
- **Grid**: size variations (10K–100K), dimension (128–768), HNSW (M, ef), IVF-PQ (nlist, M_sub).
- **Metrics**: Recall@1/10/100, QPS, latency (mean/p50/p95/p99), RAM, build time.
- **Score**: `QPS / RAM_MB` for configs with recall ≥ 90%.
- **Reports**: `BENCHMARK.md` + `TUNING_REPORT.md` with top-3 sweet spots.
- **Status**: IMPLEMENTED (~1,171 LOC).

---

## 18. IVF-PQ Persistence

### RF-31: Atomic Persistence
- **Protocol**: write to `.tmp`, `sync_all()`, `rename()` → the final file only appears when writing is complete.
- **Format**: JSON metadata + JSONL embeddings + mmap.
- **Soft-delete**: mark documents as deleted instead of rewriting everything.
- **Compaction**: rewrites the file without deleted documents, frees space.
- **Status**: IMPLEMENTED (~724 LOC).

---

## 19. Security Model

dogma-vdb is a **local CLI tool / embeddable library**:

| Component | Exposure | Risk |
|-----------|----------|:----:|
| Core library | None (only user code) | 0 |
| CLI | Local, user invokes explicitly | 0 |
| MCP stdio | Local processes authorized by user | Low |
| Watcher | Directories configured by user | Low |
| fastembed | Downloads models from HuggingFace | Low |

**Principles**:
- No `unsafe` in production code (isolated to `as_embeddings()` in VectorStorage).
- MmapBackedStorage includes defensive documentation against SIGBUS.
- No system command execution.
- No hardcoded secrets.
- No networking in core (MCP server is a separate binary and stdio by default).
- No external dependencies for core algorithms (HNSW, IVF-PQ, SQ — pure Rust).

---

## 20. Feature Flags

| Flag | Dependencies | Purpose |
|------|-------------|:--------|
| `watch` | notify, crossbeam-channel | File watcher |
| *(default)* | serde, serde_json, thiserror, rayon, wide, bytemuck, memmap2, once_cell, toml, log, regex-lite | Minimum core |

---

## 21. Tests and Coverage

- **Unit tests**: in each module (`#[cfg(test)]`).
- **Integration tests**: `tests/integration.rs`.
- **Documentation tests**: doc-tests across all public API.
- **Total**: 192 tests (192 pass, clippy clean).
- **Benchmarks**: exhaustive grid benchmark + benchmark with real ONNX embeddings.

---

## 22. Roadmap — Future Features

See `ARCH-SPEC.md` section 12 for detailed post-beta items. Priorities:

| Priority | Feature | Impact |
|----------|---------|:------:|
| High | Parallel IVF-PQ build (rayon) | High |
| High | Python bindings (PyO3) | High |
| High | Native LangChain VectorStore | High |
| Medium | SIMD for PQ lookup (wide) | Medium |
| Medium | Fuzz testing | Medium |
| Medium | Efficient CRUD update (in-place) | Medium |
| Medium | Multi-index search | Medium |
| Medium | Additional embedding models | Medium |
| Medium | File locking (fs2) anti-SIGBUS | Medium |
| Low | Parquet export format | Low |
| Low | CLI REPL mode | Low |
| Low | Benchmarks in CI | Low |
