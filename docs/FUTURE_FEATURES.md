# 🗺️ Architecture Roadmap: dogma-vdb (Next Steps)

> Stable baseline: 257 tests, IVF-PQ Recall@10 = 74% with K-Means++
> Date: 2026-06-23 | Commit: bcc75c8

---

## 1. 🧠 Autonomous Smart-Tuning (`src/tuning/agent.rs`)

**Goal:** Eliminate the need for manual vector parameter configuration,
dynamically indexing based on available hardware and data volume.

### Decision Mechanism (SLM / Heuristic in Rust)

The tuner detects the environment at runtime and selects the optimal
configuration without user intervention.

#### System Inputs

```rust
pub enum TargetPriority {
    HighRecall,      // Maximum precision (default for MCP / interactive queries)
    UltraLowLatency, // Minimum latency (batch / CI pipeline)
    Balanced,        // Default trade-off
}

pub struct SystemProfile {
    pub free_ram_mb: u64,         // From MemoryGuard::check_memory()
    pub total_ram_mb: u64,        // From /proc/meminfo
    pub cpu_cores: usize,         // From available_parallelism()
    pub dataset_size: usize,      // Number of documents
    pub dimension: usize,         // Embedding dimensionality
}
```

#### Dynamic Control Matrix

| Condition | Backend | Config | Rationale |
|-----------|---------|--------|-----------|
| RAM < 8GB || dataset > 50K | `IVF-PQ` | `n_list=256, M=32, n_probe=8, sq=true` | Protect OS from OOM. PQ + SQ = ~32× compression vs f32 |
| RAM > 16GB || dataset < 10K | `HNSW` | `M=64, ef_construction=200, ef_search=100, f32` | Prioritizes recall. Memory is not a limitation |
| 8GB < RAM < 16GB | `HNSW+SQ` | `M=32, ef=100, sq=true, sq_rescore=true` | Balance. SQ rescore recovers recall |
| dataset < 1K (any RAM) | `BruteForce` | — | Exact, O(n·d) trivial. No build overhead |

#### Hot Tuning

Once the index is built, the system monitors p99 latency
of the last N queries and adjusts search parameters dynamically:

```rust
struct HotTuningState {
    recent_latencies_p99: Vec<f64>,   // rolling window (last 100 queries)
    current_ef_search: usize,         // for HNSW
    current_n_probe: usize,           // for IVF-PQ
    target_latency_us: f64,           // e.g., 500 μs
}

impl HotTuningState {
    /// If p99 > target_latency, reduce ef_search / n_probe.
    /// If p99 << target_latency and recall is sufficient, increase parameters.
    fn tick(&mut self, measured_p99_us: f64) -> TuningAdjustment;
}
```

#### Integration

```rust
// In collection.rs
pub fn open_with_tuning(path, priority: TargetPriority) -> Result<Self>;

// Or via config.toml
// [tuning]
// priority = "high_recall"
// auto_tune = true
```

### Files to modify/create

| File | Action | Estimated LOC |
|------|--------|:------------:|
| `src/tuning/mod.rs` | New — SystemProfile, TargetPriority, decision matrix | ~120 |
| `src/tuning/agent.rs` | New — HotTuningState, rolling window, dynamic adjustment | ~80 |
| `src/config.rs` | Modify — add `[tuning]` section | ~20 |
| `src/collection.rs` | Modify — integrate `open_with_tuning()` and `auto_tune` | ~40 |
| **Total** | | **~260 LOC** |

### New dependencies

- None. Everything is pure Rust + stdlib (percentile calculations, rolling window).

### Success criteria

```
Without manual tuning, on a machine with 8 GB RAM and 100K docs 384-dim:
  - IVF-PQ auto-selected
  - Build < 30s
  - Query p50 < 500 μs
  - No OOM
```

---

## 2. 🔀 Async Concurrent Watcher v2 (`src/watch/actor.rs`)

**Goal:** Elevate the real-time monitoring system to the same low-level
standard as the rest of the engine, eliminating the file open/close
bottleneck on each event.

### Current watcher problem (v1)

```
Notify event → open collection → insert docs → store → close
                   ↑ each event re-opens and persists the entire .vdb
```

For 100 simultaneously modified files, this produces **100 opens,
100 inserts, 100 full disk writes**. With `debounce_ms=500`,
the coalescence window is fixed, not adaptive.

### Actor-Based Architecture (v2)

#### 1. Live Co-Shared Instance

The watcher no longer receives a `PathBuf` to re-open the `.vdb` file.
It receives an `Arc<RwLock<Collection>>` to interact with the database
open in RAM (mmap).

```rust
pub struct LiveCollection {
    collection: Arc<RwLock<Collection>>,
    /// Channel to notify the actor when to flush
    flush_tx: Sender<()>,
}

pub fn start_watching_v2(
    config: WatchConfig,
    collection: LiveCollection,
) -> Receiver<WatchEvent>;
```

#### 2. Coalescence Table (Advanced Debounce)

```rust
struct CoalescenceTable {
    /// Map of path → last event time
    pending: HashMap<PathBuf, Instant>,
    /// Window duration (configurable per event type)
    window: Duration,
}

impl CoalescenceTable {
    /// Registers an event. If one already exists for the same path in the
    /// current window, it is overwritten (coalesced).
    fn push(&mut self, path: PathBuf) -> bool;  // true = new, false = coalesced

    /// Returns paths that have exceeded the coalescence window.
    fn drain_ready(&mut self) -> Vec<PathBuf>;
}
```

#### 3. Decoupled Flush Pipeline

```
[notify events] → CoalescenceTable → hot inject (Collection in RAM)
                                           ↓
                              silence > 2s? → collection.flush() to disk
```

- Modified chunks are **hot-injected** into the live collection
  in memory (without re-opening the file).
- Disk flush (`flush()`) is executed **lazily**
  only when the notify event channel has been silent for
  more than 2 seconds, avoiding I/O storms.
- `LiveCollection::flush()` uses `BinStorage::store()` atomic (tmp + rename).

### Flow Diagram

```
                    ┌──────────────────────┐
                    │   notify::Watcher     │
                    │  (crossbeam-channel)  │
                    └──────────┬───────────┘
                               │ raw events
                               ▼
                    ┌──────────────────────┐
                    │  CoalescenceTable    │
                    │  (500ms window)      │
                    │  HashSet<PathBuf>    │
                    └──────────┬───────────┘
                               │ coalesced paths
                               ▼
                    ┌──────────────────────┐
                    │  hot inject          │
                    │  Arc<RwLock<Coll>>   │
                    │  chunker + insert    │
                    └──────────┬───────────┘
                               │
                    ┌──────────┴───────────┐
                    │  Silence timer        │
                    │  > 2s no events?      │
                    └──────────┬───────────┘
                               │ yes
                               ▼
                    ┌──────────────────────┐
                    │  flush() to disk     │
                    │  (tmp + rename)      │
                    └──────────────────────┘
```

### Files to modify/create

| File | Action | Estimated LOC |
|------|--------|:------------:|
| `src/watch/actor.rs` | New — actor loop, CoalescenceTable, decoupled flush | ~150 |
| `src/watch/mod.rs` | Modify — re-export `start_watching_v2`, keep v1 | ~30 |
| `src/watch.rs` | Delete or convert to delegation to `watch/actor.rs` | — |
| **Total** | | **~180 LOC** |

### New dependencies

- None. `crossbeam-channel` and `notify` are already in the `watch` feature.
- `Arc<RwLock>` comes from stdlib.

### Success criteria

```
With 500 .rs files:
  - Initial scan: < 2s
  - Modification of 50 simultaneous files: 1 disk flush, not 50
  - Hot injection latency: < 10 ms per file
  - Collection always responds to queries during ingestion
```

---

## 3. 🌐 Industrial Network Layer & Transport (`src/network/`)

**Goal:** Expose the vector engine to microservices and enterprise remote
integrations via high-performance protocols, under the
feature gate `#[cfg(feature = "net")]`.

### 3.1. gRPC Server (`src/network/grpc.rs`)

Native implementation using the [`tonic`](https://crates.io/crates/tonic) crate.
Definition of a `.proto` schema optimized for bidirectional streaming
of dense vectors (`f32` / `i8`) and metadata payloads.

```protobuf
service DogmaVdb {
  // Simple vector search
  rpc Search(SearchRequest) returns (SearchResponse);

  // Hybrid search (vector + BM25 + reranker)
  rpc HybridSearch(HybridSearchRequest) returns (SearchResponse);

  // Streaming document ingestion
  rpc Ingest(stream Document) returns (IngestResponse);

  // Deletion by ID
  rpc Delete(DeleteRequest) returns (DeleteResponse);
}

message SearchRequest {
  string collection_path = 1;
  repeated float embedding = 2;
  uint32 k = 3;
  string index_type = 4;   // bruteforce | hnsw | ivf_pq
  string metric = 5;       // cosine | dot | euclidean
  bool rerank = 6;
  string query_text = 7;   // required if rerank = true
}
```

#### Architecture

```
                    ┌─────────────────────────────┐
                    │      tonic gRPC Server       │
                    │   (http2, multiplexed)       │
                    └──┬──────────┬───────────┬────┘
                       │          │           │
              ┌────────┴──┐ ┌─────┴──────┐ ┌──┴─────────┐
              │ Search    │ │ Hybrid     │ │ Ingest     │
              │ Handler   │ │ Handler    │ │ Handler    │
              └────┬──────┘ └─────┬──────┘ └─────┬──────┘
                   │              │               │
                   └──────┬──────┴───────────────┘
                          │
                    ┌─────┴──────┐
                    │ Collection │
                    │ (mmap)     │
                    └────────────┘
```

#### Python Client

```python
import grpc
from dogmavdb_pb2 import SearchRequest
from dogmavdb_pb2_grpc import DogmaVdbStub

channel = grpc.insecure_channel("localhost:50051")
client = DogmaVdbStub(channel)
response = client.Search(SearchRequest(
    collection_path="data/docs.vdb",
    embedding=[0.1, 0.2, 0.3],
    k=10,
    index_type="hnsw",
    metric="cosine",
))
```

### 3.2. MCP via HTTP / WebSockets (`src/network/mcp_http.rs`)

Evolve the current MCP server transport (stdio) to SSE
(Server-Sent Events) or WebSockets over HTTP. Allows multiple
remote agents (Claude Desktop, Cursor, opencode, scripts) to connect
simultaneously without being tied to the lifecycle of a single terminal
process.

#### Current state of the MCP server

```
today:   serve_server(server, (stdin(), stdout()).into_transport())
tomorrow: serve_server(server, http_into_transport("0.0.0.0:5000"))
```

The `rmcp` crate we already use supports HTTP/SSE transport. What's missing
is adding `axum` as the HTTP server and choosing the transport via CLI flag.

```bash
# Today (stdio only):
dogma-vdb-mcp

# Tomorrow (HTTP / WebSocket):
dogma-vdb-mcp --transport http --port 5000
dogma-vdb-mcp --transport ws --port 5000
```

#### Concurrent connections diagram

```
        ┌──────────────┐
        │ Claude       │────┐
        │ Desktop      │    │
        └──────────────┘    │   ┌────────────────────┐
                            ├──►│  dogma-vdb MCP      │
        ┌──────────────┐    │   │  Server (HTTP/WS)   │
        │ Cursor       │────┘   │  :5000              │
        └──────────────┘        │                     │
                                │  Collection (mmap)  │
        ┌──────────────┐        └────────────────────┘
        │ opencode     │────┐
        └──────────────┘    │
                            │
        ┌──────────────┐    │
        │ Python SDK   │────┘
        └──────────────┘
```

### 3.3. Feature Gate

```toml
[features]
net = ["dep:tonic", "dep:axum", "dep:rmcp"]
```

### Files to modify/create

| File | Action | Estimated LOC |
|------|--------|:------------:|
| `proto/dogmavdb.proto` | New — protobuf definition | ~80 |
| `src/network/mod.rs` | New — feature gate + re-exports | ~10 |
| `src/network/grpc.rs` | New — tonic server handlers | ~200 |
| `src/network/mcp_http.rs` | New — HTTP/WS transport for MCP | ~100 |
| `dogma-vdb-mcp/src/main.rs` | Modify — `--transport` + `--port` flags | ~50 |
| `Cargo.toml` | Modify — `net` feature + optional deps | ~10 |
| `build.rs` | New — .proto compilation | ~30 |
| **Total** | | **~480 LOC** |

### New dependencies

| Crate | Version | Purpose |
|-------|:-------:|---------|
| `tonic` | 0.12+ | gRPC server/framework |
| `prost` | 0.13+ | `.proto` to Rust compilation |
| `axum` | 0.8+ | HTTP server for MCP transport |
| `rmcp` | 1.x | *(already in workspace, extend transport)* |

### Success criteria

```
- gRPC: 10,000 queries/s on single core, 10K docs 384-dim collection
- MCP HTTP: 5 simultaneous agents, no interference between connections
- MCP WebSocket: latency < 100 μs overhead over actual query time
- net feature off by default: 0 impact on binaries without networking
```

---

## Summary of estimated effort

| Feature | LOC | New dependencies | Priority |
|---------|:---:|:----------------:|:--------:|
| Autonomous Smart-Tuning | ~260 | 0 | High |
| Concurrent Watcher v2 | ~180 | 0 | Medium |
| Network Layer (gRPC + MCP HTTP) | ~480 | 4 (tonic, prost, axum, rmcp) | Medium |
| **Total** | **~920** | | |

---

## Current baseline (pre-requisite for both features)

| Metric | Value |
|--------|:-----:|
| Total LOC | 8,594 |
| Tests | 257 pass, 0 fail |
| Compilation | 0 errors, 0 warnings |
| IVF-PQ Recall@10 (real embeddings) | 74.0% |
| IVF-PQ Latency p50 | 344 μs |
| K-Means++ | Implemented (D² weighting) |
| Feature flags | `sml`, `watch` (off by default) |
| Storage format | Binary v2 (DVDB), no JSONL |
| Chunker strategies | 3: Code, Paragraph, FixedWindow |
