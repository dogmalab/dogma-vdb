# AGENTS.md — Rules for Implementing dogma-vdb

## Project Philosophy

```
dogma-vdb = Rust + JSONL + serde_json
            (min deps, portable, no server)
```

Every line of code must justify its existence. We prefer **50 clear lines** over 200 "architecturally flexible" lines.

---

## ✅ CURRENT STATUS (2026-05-17)

### Core crate (`dogma-vdb`) — COMPILES and 152 TESTS PASS

| Module | File | Lines | Tests | Status |
|--------|---------|-------|-------|--------|
| Document | `src/doc.rs` | 205 | 8 | Complete |
| Error | `src/error.rs` | 45 | - | Complete |
| Distance | `src/distance.rs` | 209 | 16 | Complete |
| Filter | `src/filter.rs` | 122 | 9 | Complete |
| Storage (JSONL) | `src/storage.rs` | 307 | 15 | Complete |
| Collection | `src/collection.rs` | ~530 | 15 | Complete |
| Runtime Config | `src/config.rs` | ~320 | - | Complete |
| Chunker | `src/chunker.rs` | 247 | 8 | Complete |
| Embedder trait | `src/embedding.rs` | 28 | - | Complete |
| SmartChunker | `src/smart_chunker/` | ~560 | 20+ | Complete |
| Index trait | `src/index/mod.rs` | 67 | - | Complete |
| Index (BruteForce) | `src/index/brute_force.rs` | 440 | 18 | Complete |
| Index (HNSW) | `src/index/hnsw.rs` | ~840 | 21 | Complete |
| Index (IVF-PQ) | `src/index/ivf_pq.rs` | ~400 | 8 | New |
| Index (Annoy) | ~~`src/index/annoy.rs`~~ | — | — | **Removed** |
| SQ module | `src/index/sq.rs` | ~230 | 8 | Complete |
| Watcher | `src/watch.rs` | 56 | - | **SKELETON** (`todo!()`) |
| MCP Server | `src/mcp.rs` | 36 | - | **SKELETON** (`todo!()`) |

### Sub-crates

| Crate | File | Status |
|-------|---------|--------|
| `dogma-vdb-cli` | `cli/src/main.rs` | Complete (info, list, query, ingest, delete) |
| `dogma-vdb-mcp` | `mcp/src/main.rs` | Complete (vecdb_query, ingest, delete, list, info) |
| `dogma-vdb-embed` | `embed/src/lib.rs` | Complete (trait definition) |
| `dogma-vdb-embed-fastembed` | `embed-fastembed/src/lib.rs` | Complete (FastEmbedder with ONNX MiniLM-L6-v2) |

### Tests
- Unit: 139 pass
- Integration: 9 pass
- Doc-tests: 8 pass, 2 ignored
- **Total: 156 tests, 0 failures**

---

## ✅ What We DO

### 1. Idiomatic Rust — no fluff

- **Ownership first**. Borrow (`&`) by default, owned (`T`) only when the callee needs ownership.
- **`Into<T>` in constructors** for zero-cost flexibility.
- **`impl Trait` in parameters** (monomorphization) instead of `Box<dyn Trait>` unless you need real dynamic dispatch.
- **`sort_unstable`** over `sort`. We don't need stability.
- **`#[inline]`** only in 1-3 line functions that are in hotspots (distances, dot product).
- **`debug_assert_eq!`** for preconditions that should only be checked in debug mode.

### 2. Small code — each file < 300 lines

Maximum 300 lines per file (with exceptions for test-heavy files:
`storage.rs` 307, `smart_chunker/mod.rs` 536 which includes ~200 lines of tests).
If a module grows larger, it gets split.

### 3. Minimal dependencies — ask before adding

**Required core deps (currently):**
- `serde` + `serde_json` + `thiserror` — essential
- `regex-lite` — smart chunker (regex lightweight)
- `once_cell` + `toml` + `log` — runtime config

**Optional deps (features):**
- `watch` → `notify` + `crossbeam-channel`
- `mcp` → `rmcp` + `tokio` + `tracing` + `clap`

### 4. Testing from the start

- Every module has `#[cfg(test)] mod tests` at the end.
- Integration tests in `tests/` use real temporary files.
- Tests must pass **without network** or external services.
- All new tests must compile and pass in CI.

### 5. JSONL Format — the center of the design

```
.vdb file
├── Line 1: {"id":"doc-1","text":"...","embedding":[0.1,...],"metadata":{...}}
├── Line 2: {"id":"doc-2","text":"...","embedding":[...],"metadata":{...}}
└── ...
```

- **Each line is independent** — can use `grep`, `sed`, `head`.
- **Append-only** by design — appending is O(1). Updating requires rewriting.
- **`serde_json::from_str`** line by line (streaming with BufReader).

### 6. Small and focused traits

```rust
pub trait Embedder: Send + Sync {
    fn embed(&self, text: &str) -> Result<Vec<f32>>;
    fn dimension(&self) -> usize;
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> { /* default */ }
}
```

### 7. Useful documentation

- `///` with usage examples across the entire public API.
- `# Examples` in docstrings (run with `cargo test --doc`).
- `#[must_use]` on functions whose result should not be ignored.

### 8. Comments in English, minimal, purposeful

- **All comments must be in English.** No Spanish, no mixed-language comments.
- **Reduce comments to the minimum necessary.** The code should speak about the application, not the comments. Every comment must justify its existence — if the code is self-explanatory, delete the comment.
- `//` comments document *why*, not *what*. Let types, variable names, and function signatures express *what*.
- `///` doc comments document the public API for consumers. They are not the place for implementation notes.

---

## ❌ What We DON'T Do

### 1. DON'T add dependencies to core without discussing it

If someone wants to just read `.vdb` files without async or HTTP, they should be able to do so with minimal deps.

### 2. No premature abstraction

```rust
// WRONG — abstracting for the sake of abstraction
trait DistanceCalculator { fn compute(&self, a: &[f32], b: &[f32]) -> f32; }

// RIGHT — a concrete, reusable function
pub fn cosine(a: &[f32], b: &[f32]) -> f32;
```

We start with `BruteForceIndex` and if HNSW is needed later, it gets added as another implementor of the `Index` trait.

### 3. DON'T clone unnecessarily

```rust
// WRONG
fn search(&self, query: Vec<f32>) -> Vec<Document> { let query = query.clone(); ... }

// RIGHT
fn search(&self, query: &[f32]) -> Vec<ScoredDocument>;
```

### 4. DON'T unwrap() in production

```rust
// WRONG
let doc = docs.iter().find(|d| d.id == "x").unwrap();

// RIGHT
let doc = docs.iter().find(|d| d.id == "x")
    .ok_or_else(|| Error::DocumentNotFound("x".into()))?;
```

`unwrap()` only in tests and examples.

### 5. No over-engineered structures

- No `async` in the core. If async is needed, it goes in the `mcp` or `cli` crate.
- No procedural macros.
- No `unsafe` unless strictly necessary and measured.
- No unnecessary generics.

### 6. DON'T ignore clippy warnings

CI fails with `-D warnings`. Silencing warnings with `#[allow(...)]` only if there is a justified and documented reason.

### 7. DON'T write comments that repeat the code

```rust
// WRONG — comment repeats what the code already says
// Increment the counter by one
counter += 1;

// RIGHT — comment explains why, not what
// Skip padding bytes added by the serializer
offset += ALIGNMENT_PAD;
```

### 8. ANN Index (HNSW) — rules

The approximate index complements `BruteForceIndex` without replacing it:

- **Pure Rust implementation** — no new external dependencies
- **Same API** — implements the existing `Index` trait
- **Configurable parameters** in `HnswConfig`: `M` (connections), `ef_construction` (build quality), `ef_search` (query quality)
- **Predictable memory**: each node stores its vector + neighbors per layer
- **`ef_search` controls the trade-off**: higher value = more recall, less speed
- **Collection can use either**: injected via `HnswConfig` instead of `Metric`

```rust
let mut index = HnswIndex::new(HnswConfig {
    M: 16,
    ef_construction: 200,
    ef_search: 50,
    metric: Metric::Cosine,
});
index.insert(&docs);
let results = index.search(&query, 10);
```

Expected performance vs BruteForce:

| Dataset | BruteForce | HNSW (ef=50) | HNSW (ef=200) |
|---------|-----------|--------------|---------------|
| 1K      | 0.5ms     | 0.2ms        | 0.5ms         |
| 10K     | 5ms       | 0.5ms        | 2ms           |
| 100K    | 50ms      | 1ms          | 5ms           |
| 1M      | 500ms     | 3ms          | 15ms          |
| Recall  | 100%      | ~90-95%      | ~98-99%       |

---

## Tools We Have

### From core (always available)

| Tool | Purpose |
|---|---|
| `std::fs` | Read/write .vdb files |
| `std::io::{BufReader, BufWriter}` | Streaming line by line |
| `std::collections::HashMap` | Document metadata |
| `serde_json` | Serialize/deserialize JSONL |
| `thiserror` | Typed errors |
| `regex_lite` | Smart chunking by file type |

### From Rust stdlib (no extra dependencies)

```rust
f32::sqrt()          // → vector magnitude
f32::powi()          // → euclidean distance
f32::abs()           // → tolerances
.iter().zip()        // → dot product
.map().sum()         // → sum of products
.sort_unstable_by()  // → sort by score
File::open()         // → read .vdb
File::create()       // → write .vdb
OpenOptions::append()// → append to .vdb
Path::exists()       // → does the file exist?
Path::extension()    // → filter by extension
Path::file_stem()    // → collection name
```

### With optional features

| Feature | Extra tools |
|---|---|
| `watch` | `notify` (inotify/kqueue), `crossbeam-channel` |
| `mcp` | `rmcp`, `tokio`, `tracing`, `clap` |

---

## Typical Module Structure

```rust
//! 1. One-line docstring with the purpose.

// 2. Grouped imports: stdlib, external, crate
use std::path::PathBuf;
use crate::error::Result;

// 3. Public types (struct, enum, trait)
pub struct Foo { ... }
pub trait Bar { ... }

// 4. Implementations
impl Foo { ... }
impl Bar for Foo { ... }

// 5. Public helper functions (if applicable)
pub fn helper() { ... }

// 6. Tests (at end of file)
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_foo() { ... }
}
```

---

## How We Evaluate New Code

1. **Compiles with `cargo check --all-features`** ✅
2. **No clippy errors** (`cargo clippy --all-features -- -D warnings`) ✅
3. **Tests pass** (`cargo test --all-features`) ✅
4. **No new dependencies** in core (or justified) ✅
5. **Correct formatting** (`cargo fmt --all -- --check`) ✅

If everything passes, the code can be merged.

---

## Pending (Roadmap)

- [x] Implement HNSW index (`src/index/hnsw.rs`)
- [x] Collection can use HNSW via config
- [x] Full CRUD (insert, delete, update)
- [x] CLI (info, list, query, ingest, delete)
- [x] MCP server (vecdb_query, ingest, delete, list, info)
- [x] Comparative benchmarks (all backends)
- [x] HNSW flat_embeddings
- [x] SQ module + integration in BF and HNSW
- [x] SQ rescore (recover recall with f32)
- [x] IVF-PQ index (inverted file + product quantization)
- [x] Config env vars for all fields
- [ ] Implement `watch.rs` (file system watcher, feature = "watch")
- [ ] Implement `mcp.rs` (MCP server, feature = "mcp")
- [x] Implement real FastEmbed (`dogma-vdb-embed-fastembed`)
- [x] Multi-crate workspace (root Cargo.toml)
- [ ] Complete examples in `examples/`

---

*Last updated: 2026-05-16*
