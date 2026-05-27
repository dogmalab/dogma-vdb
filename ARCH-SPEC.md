# Architecture — dogma-vdb

## 1. Architectural Principles

1. **1 Rust file = 1 component**. Each component has a single responsibility.
2. **Index trait as boundary**. Backends are interchangeable via Box<dyn Index>.
3. **SQ is orthogonal**. It doesn't change the API, only the storage/distance.
4. **VectorStorage decouples vectors from indices**. Backends receive
   injected contiguous embeddings, without knowing their origin (RAM or mmap).
5. **No external dependencies for core algorithms**. HNSW, IVF-PQ, SQ are pure Rust.
6. **Config-driven**. Everything parameterized via config.toml, no hardcoding.

---

## 2. Architecture Diagram

```
                         Collection
                             |
                     Box<dyn Index>
                      /     |     \
               /            |            \
        BruteForce      HnswIndex      IvfPqIndex
             |               |              |
             +-- SQ? --------+-- SQ? -------+-- SQ?
             |               |              |
        Arc<dyn VectorStorage> (shared)
          /               \
   MemoryBacked      MmapBacked
   (Vec<f32>)        (memmap2)

  BinStorage (persistence JSONL + binary v2)
```

**SQ**: when `sq=true`, each backend uses `score_i8()` with scale/bias
per document. The graph/topology is built with original f32, the
search can use i8 with optional rescore.

---

## 3. File Structure

```text
src/
  lib.rs                  # Mod declarations + prelude
  doc.rs                  # Document struct + builder
  distance.rs             # Metric, score(), dot(), cosine(), euclidean(), score_i8()
  error.rs                # Error types
  storage/
    mod.rs                # BinStorage (binary v2 read/write) + JsonlStorage
    traits.rs             # VectorStorage trait + MemoryBackedStorage + MmapBackedStorage
  collection.rs           # Collection API (open, insert, search, hybrid_search, etc.)
  config.rs               # Config load from TOML + env vars (global CONFIG)
  filter.rs               # Metadata filter helpers
  embedding.rs            # Embedder trait (for text→vec)
  memory.rs               # Memory guard (pressure detection from /proc/meminfo)
  rerank.rs               # Reranker trait + NoRerank default
  chunker.rs              # Simple text chunker (legacy)
  watch.rs                # File watcher (notify v8, feature = "watch")
  index/
    mod.rs                # Index trait + factory + re-exports
    brute_force.rs        # BruteForceIndex
    hnsw.rs               # HnswIndex + HnswConfig
    ivf_pq.rs             # IvfPqIndex + IvfPqConfig
    ivf_pq_persistence.rs # Atomic persistence, soft-delete, compaction
    sq.rs                 # SQ helpers: quantize(), score_i8(), rescore()
    bm25.rs               # BM25 inverted text index
    rrf.rs                # Reciprocal Rank Fusion
  smart_chunker/
    mod.rs                # SmartChunker: auto-detect strategy, dispatch
    code.rs               # CodeChunker (regex-based)
    paragraph.rs           # ParagraphChunker + chunk_semantic (merged)
    fixed_window.rs        # FixedWindowChunker (replaces markdown, jsonl, text)
```

---

## 4. SQ — Scalar Quantization

### 4.1. Quantization Algorithm (corrected)

For each embedding `v` of dimension `d`:

1. Compute `min_d` and `max_d` **per document** (not global).
2. `midpoint = (max_d + min_d) / 2.0` (bias centered in the range).
3. `scale = (max_d - min_d) / 255.0`.
4. `v_i8[i] = clamp(round((v[i] - bias) / scale), -128, 127)`.

The midpoint as bias replaces the previous `min`, guaranteeing that
values close to 0 are correctly mapped to the symmetric i8 range.

### 4.2. Distance in i8

```
dot_i8(a_i8, b_i8) = sum_i(a_i8[i] * b_i8[i])  // linear scale
```

For ANN search where only the ranking matters, the constant
scale factors do not affect the order.

### 4.3. Rescoring (optional)

To recover precision, after obtaining top-k with i8, rescore
the top-k*2 with original f32. This adds ~20% overhead but improves recall
from 40% → 90% in HNSW+SQ.

### 4.4. Integration by Backend

**BruteForce + SQ**: iterate embedding_i8, compute dot_i8, sort.
If rescore=true, take top-k*2, rescore with f32.

**HNSW + SQ**: The graph is built with original f32 distance
(guarantees correct topology). `search_layer()` uses `score_i8()`.
With rescore: top-k*2 candidates → f32 rescore. Recall: 90%.

**IVF-PQ + SQ**: K-Means centroids are computed in f32. The
cluster assignment and asymmetric distance are done in f32.
SQ is an additional orthogonal layer on top of PQ codes.

### 4.5. Where SQ Lives

In `src/index/sq.rs`:

```rust
/// Quantize an f32 embedding to i8 with per-document scale.
pub fn quantize(embedding: &[f32], scale: f32, bias: f32) -> Vec<i8>;

/// Quantize the query for i8 search.
pub fn quantize_query(query: &[f32], scale: f32, bias: f32) -> Vec<i8>;

/// Dot product in i8 (SIMD-friendly).
pub fn dot_i8(a: &[i8], b: &[i8]) -> i32;

/// i8 score converted to f32.
pub fn score_i8(query_i8: &[i8], doc_i8: &[i8], scale: f32, bias: f32) -> f32;

/// Recalculate exact score with f32 for rescoring.
pub fn rescore(query: &[f32], docs: &[&Document], metric: Metric) -> Vec<ScoredDocument>;
```

It is not an index or a wrapper — it is a utility module. Each backend
uses it when `sq=true`.

---

## 5. IVF-PQ — Inverted File + Product Quantization

### 5.1. Data Structure

```rust
pub struct IvfPqIndex {
    documents: Vec<Document>,       // metadata, text, embedding (f32)
    centroids: Vec<Vec<f32>>,       // n_list K-Means centroids (f32)
    pq_codebook: Vec<Vec<Vec<f32>>>, // m_subspaces sub-codebooks (256 x (d/m) each)
    codes: Vec<Vec<u8>>,            // PQ codes per document (m_subspaces bytes each)
    assignments: Vec<usize>,        // cluster assignment per document
    config: IvfPqConfig,
    storage: Arc<dyn VectorStorage>, // shared contiguous embeddings
}

pub struct IvfPqConfig {
    pub n_list: usize,               // number of centroids (default: 100)
    pub m_subspaces: usize,          // number of sub-vectors (default: 32, multiple of 8)
    pub n_probe: usize,              // clusters to explore (default: 5)
    pub metric: Metric,
    pub rerank_enabled: bool,        // auto-tuning: reduces n_probe by half
}
```

### 5.2. Build Algorithm (batch)

```
fn build(docs: &[Document]) -> IvfPqIndex:
    1. K-Means over all embeddings (max 20 iterations):
       a. Initialize nlist centroids with k-means++
       b. Assign each embedding to the nearest centroid
       c. Recompute centroids as the average of their points
       d. Repeat until convergence or max iterations

    2. Product Quantization:
       a. For each embedding dimension, divide into m sub-vectors
          of size d/m
       b. For each subspace, run K-Means with 256 centroids
          (sub-vector codebook)
       c. For each document, encode each sub-vector as the index
          u8 of the nearest centroid in that subspace

    3. Store: centroids (f32), pq_codebook (f32), codes (u8),
       assignments (usize)
```

### 5.3. Search

```
fn search(query, k) -> Vec<ScoredDocument>:
    1. Compute query distance to all nlist centroids.
       Select the nprobe nearest ones.

    2. For each of the nprobe clusters:
       a. Precompute distance table (LUT) between query and the
          256 centroids of each PQ subspace.
       b. Scan the u8 codes of the documents in that cluster:
          approx_distance = sum_m(LUT[m][code[m]])
       c. Keep global top-k with min-heap.

    3. Sort candidates by distance, return top-k.
```

### 5.4. Complexity

| Operation | Complexity |
|-----------|------------|
| Build (K-Means) | O(n_list · n · d · iter) |
| Build (PQ) | O(256 · m_subspaces · (d/m_subspaces) · n) = O(256 · d · n) |
| Search | O(n_list · d + effective_probe · (n/n_list) · m_subspaces) |

where `effective_probe = n_probe` if `rerank_enabled=false`, or
`(n_probe / 2).max(2)` if `rerank_enabled=true`.

### 5.5. Memory

For 5K docs 128-dim, n_list=100, m_subspaces=32:
- Centroids: 100 × 128 × 4B = 51 KB
- PQ codebook: 32 × 256 × (128/32) × 4B = 128 KB
- Codes: 5K × 32B = 160 KB
- Assignments: 5K × 8B = 40 KB
- **Total: ~380 KB** (~8× less than HNSW/BF)

---

## 6. VectorStorage Trait

### 6.1. Definition

```rust
pub trait VectorStorage: Send + Sync {
    fn as_bytes(&self) -> &[u8];
    fn as_embeddings(&self) -> &[f32];
    fn flush(&self) -> Result<()>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
}
```

### 6.2. Implementations

**MemoryBackedStorage**:
```rust
pub struct MemoryBackedStorage {
    data: Vec<u8>,
}
```
- Contiguous `Vec<u8>` in RAM (f32 embeddings reinterpreted via `as_embeddings()`).
- `as_embeddings()` returns `&[f32]` via `unsafe { from_raw_parts() }` (isolated, audited).
- `from_f32_slice()`: constructs from `&[f32]` (copy).
- For tests, volatile pipelines, warm cache.

**MmapBackedStorage**:
```rust
pub struct MmapBackedStorage {
    _file: std::fs::File,    // keeps fd alive
    mmap: memmap2::Mmap,     // mapping of the entire file
}
```
- Loads ~0ms: the OS pages on demand.
- Binary v2 format with 32-byte padding.
- `as_embeddings()` reinterprets the mapped bytes as `&[f32]`.
- `advise(Advice::Random)` to eliminate readahead page faults.
- ⚠️ SIGBUS: if an external process truncates the file, the kernel kills
  the process. Documented in code.

### 6.3. Integration in Collection

```rust
pub struct Collection {
    name: String,
    documents: Vec<Document>,
    index: Box<dyn Index>,
    storage: BinStorage,
    emb_storage: Arc<dyn VectorStorage>,
}

fn open(path) -> Result<Self> {
    let storage = BinStorage::load(path)?;       // read metadata + docs
    let emb_storage = match config.use_mmap {
        true => MmapBackedStorage::new(path)?,
        false => MemoryBackedStorage::from_docs(&storage.documents),
    };
    let emb_storage = Arc::new(emb_storage);

    let index = build_index(cfg, emb_storage.clone())?;
    index.insert(&storage.documents)?;

    Ok(Collection { name, documents, index, storage, emb_storage })
}
```

---

## 7. Flat Embeddings in HNSW

### 7.1. Storage

```rust
pub struct HnswIndex {
    documents: Vec<Document>,       // metadata, text
    embeddings_flat: Vec<f32>,      // only if flat_embeddings=true
    dim: usize,                     // only if flat_embeddings=true
    storage: Arc<dyn VectorStorage>, // contiguous embeddings (always present)
    // ... rest same
}
```

### 7.2. Helper

```rust
fn embedding(&self, node_id: usize) -> &[f32] {
    if self.config.flat_embeddings {
        let start = node_id * self.dim;
        &self.embeddings_flat[start..start + self.dim]
    } else {
        self.storage.get(node_id)
    }
}
```

### 7.3. Insertion

When `flat_embeddings=true`, insert_one() does:
1. Extends `embeddings_flat` with the new embedding.
2. The embedding also lives in `storage` (shared VectorStorage).

Design decision: flat_embeddings is only for in-memory search.
The binary format always stores the full f32 embedding (portability).

### 7.4. Delete with Flat

When a document is deleted with flat, `embeddings_flat` must be rebuilt
from the remaining documents (O(n·d) cost once, equivalent to what the
graph rebuild in delete already does).

---

## 8. Factory Strategy

In `index/mod.rs`:

```rust
fn build_index(cfg: &CollectionConfig, storage: Arc<dyn VectorStorage>) -> Box<dyn Index> {
    let mut index: Box<dyn Index> = match cfg.index_type {
        "hnsw" => Box::new(HnswIndex::new(HnswConfig { ... }, storage)),
        "ivf_pq" => Box::new(IvfPqIndex::new(IvfPqConfig { ... }, storage)),
        _ => Box::new(BruteForceIndex::new(metric, storage)),
    };

    // SQ is not a wrapper — each backend receives the sq flag
    // and acts accordingly in its search/insert methods.
}
```

---

## 9. Dependencies

### Current
- serde, serde_json, thiserror — core
- rayon — parallel BruteForce
- toml, once_cell, log — config
- memmap2 — MmapBackedStorage
- bytemuck — safe f32↔[u8] reinterpret
- wide — SIMD-accelerated distance functions
- regex-lite — smart chunker patterns
- notify, crossbeam-channel — watcher (feature)

### No external dependencies for core algorithms
- HNSW: SplitMix64 (already implemented in core)
- IVF-PQ: K-Means and PQ are pure Rust (stdlib)
- SQ: pure Rust (stdlib)

### Optional
- `rand` (dev-dependency)

---

## 10. Target Metrics

| Backend | 5K docs 128-dim | 50K docs 768-dim | 100K docs 384-dim |
|---------|:---------------:|:----------------:|:-----------------:|
| BruteForce | 1,460 us | ~200 ms | ~400 ms |
| HNSW | 77 us | ~500 us | ~1 ms |
| IVF-PQ | 128 us | ~2 ms | ~4 ms |
| HNSW+SQ+Rescore | 73 us | ~350 us | ~700 us |

Estimated RAM for 100K docs 384-dim:
- f32 embeddings: 100K × 384 × 4 = ~153 MB
- HNSW graphs: ~200 MB additional (connections)
- IVF-PQ: ~1.5 MB (centroids + codebook + codes)
- SQ i8: ~38 MB (i8 only, no graphs)

---

## 11. Implementation Priority Completed

1. ~~**HNSW + flat_embeddings**~~ (completed)
2. ~~**SQ module**~~ (completed)
3. ~~**SQ integration** in BruteForce and HNSW~~ (completed, recall 90%)
4. ~~**Annoy**~~ (replaced by IVF-PQ)
5. ~~**IVF-PQ backend**~~ (completed, ~8× RAM savings)
6. ~~**VectorStorage trait**~~ (completed, mmap ~0ms load)
7. ~~**Benchmarks**~~ (updated)

---

## 12. Future Enrichment (Post-Beta)

### 12.1. Security

| Item | Priority | Description |
|------|:--------:|-------------|
| MCP HTTP auth | Medium | Security for future HTTP implementation of the MCP server in a separate crate |
| File locking (fs2) | Medium | OS-level locking to prevent SIGBUS from concurrent writes |
| Watcher path sandbox | Low | Validate that `source_dirs` is within a configured base directory |
| Model checksum verification | Low | Verify SHA256 checksum of downloaded ONNX models |
| Audit CI hardening | Low | Configure `cargo audit` to fail only on real vulnerabilities |

### 12.2. Performance and Scalability

| Item | Impact | Description |
|------|:------:|-------------|
| **Parallel IVF-PQ build** | High | Parallelize K-Means and PQ build with rayon. ~1 session |
| **SIMD for PQ lookup** | Medium | Accelerate asymmetric distance with SIMD (wide crate) |
| **HNSW parallel insert** | Medium | Batch insert with lock-free graph. ~2 sessions |
| **Multi-index search** | Low | Search in HNSW + IVF-PQ and merge results |

### 12.3. Formats and Portability

| Item | Impact | Description |
|------|:------:|-------------|
| **Parquet format** | Medium | Export to Apache Parquet for data science interoperability |
| **Import from ChromaDB/LanceDB** | Medium | Migration script from other vector formats |
| **zstd compression in binary** | Low | Optional zstd compression for binary v3 format |

### 12.4. Integrations

| Item | Impact | Description |
|------|:------:|-------------|
| **Python bindings (PyO3)** | High | `pip install dogma-vdb` with full Python API. ~3-4 sessions |
| **Native LangChain VectorStore** | High | Python provider implementing LangChain VectorStore using MCP subprocess |
| **Additional embedding models** | Medium | Support ONNX models other than MiniLM-L6-v2 (BGE, GTE, etc.) |
| **Llamarada / mistral.rs** | Medium | Embedding via llama.cpp for local models |

### 12.5. Operations

| Item | Impact | Description |
|------|:------:|-------------|
| **Efficient CRUD update** | Medium | Currently delete+insert rewrites everything. Make update in-place |
| **Snapshot / versioning** | Low | Keep N previous versions of the .vdb for rollback |
| **CLI REPL** | Low | Interactive mode to explore collections from terminal |
| **Collection statistics** | Low | Report vector distribution, outliers, clustering |

### 12.6. Testing and CI

| Item | Impact | Description |
|------|:------:|-------------|
| **Fuzz testing** | Medium | Fuzzing of data input (malformed embeddings, corrupt metadata) |
| **Benchmarks in CI** | Low | Run bench.rs in CI and compare with previous commit |
| **MCP integration test** | Low | E2E test that starts MCP server, connects, runs queries |
| **Proptest for indices** | Low | Property test: search(k) always returns <= k results |
