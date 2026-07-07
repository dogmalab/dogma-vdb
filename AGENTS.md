# AGENTS.md — Rules for Implementing dogma-vdb

> The state harness of the [Dogma](https://github.com/dogmalab/.github) platform.

This document contains the **harness-specific** rules for working on
`dogma-vdb`. The general rules of the platform — the Four Questions,
the contribution process, the security reporting flow — live in the
org-level [CONTRIBUTING.md](https://github.com/dogmalab/.github/blob/main/CONTRIBUTING.md).
The why behind these rules is in the
[MANIFESTO.md](https://github.com/dogmalab/.github/blob/main/MANIFESTO.md).

If a rule from this file conflicts with a rule from the org-level
docs, the org-level docs win. If a rule here is missing from the
org-level docs, that is a bug — open an issue to fix it.

---

## Project philosophy

```
dogma-vdb = Rust + JSONL + serde_json
            (min deps, portable, no server)
```

Every line of code must justify its existence. We prefer **50 clear
lines** over 200 "architecturally flexible" lines. The state harness
is a library that other harnesses depend on. Complexity in the state
harness is a tax on every pattern that composes it.

---

## Current status (2026-06-23)

### Core crate (`dogma-vdb`)

| Module | File | Lines | Tests |
|---|---|---|---|
| Document | `src/doc.rs` | ~210 | 8 |
| Error | `src/error.rs` | ~50 | - |
| Distance | `src/distance.rs` | 209 | 16 |
| Filter | `src/filter.rs` | 122 | 9 |
| Storage (binary) | `src/storage/mod.rs` | ~590 | 20 |
| Storage traits | `src/storage/traits.rs` | ~270 | 7 |
| Collection | `src/collection.rs` | ~810 | 20 |
| Runtime Config | `src/config.rs` | ~420 | - |
| Chunker | `src/chunker.rs` | 247 | 8 |
| Embedder trait | `src/embedding.rs` | 28 | - |
| SmartChunker | `src/smart_chunker/` | ~685 | 20+ |
| Memory guard | `src/memory.rs` | ~170 | 4 |
| Reranker trait | `src/rerank.rs` | 69 | 2 |
| Index trait | `src/index/mod.rs` | ~115 | - |
| Index (BruteForce) | `src/index/brute_force.rs` | ~516 | 18 |
| Index (HNSW) | `src/index/hnsw.rs` | ~1055 | 21 |
| Index (IVF-PQ) | `src/index/ivf_pq.rs` | ~1100 | 21 |
| Index (BM25) | `src/index/bm25.rs` | ~267 | 7 |
| Index (RRF) | `src/index/rrf.rs` | ~125 | 5 |
| K-Means | `src/index/kmeans.rs` | ~207 | 7 |
| SQ module | `src/index/sq.rs` | ~230 | 8 |
| SIMIL | `src/sml/` | ~1040 | 31 |
| Watcher | `src/watch.rs` | ~316 | - |

### Sub-crates

| Crate | Status |
|---|---|
| `dogma-vdb-cli` | Complete (+ `rag` feature flag) |
| `dogma-vdb-mcp` | Complete (vecdb_query, ingest, delete, list, info) |
| `dogma-vdb-embed` | Complete (trait definition) |
| `dogma-vdb-embed-fastembed` | Complete (FastEmbedder with ONNX MiniLM-L6-v2) |
| `dogma-vdb-rerank` | Complete (OnnxReranker) |
| `dogma-vdb-rag` | Complete (kept for backward compat) |

### Tests

- Unit: 181 pass
- Integration: 8 pass
- Doc-tests: 9 pass, 3 ignored
- **Total: 198 tests, 0 failures**

The README's headline "257 tests" is a marketing figure from an
earlier snapshot. The authoritative number is the one above, and
it is updated as part of every release.

---

## Harness-specific rules

These are the rules that apply to `dogma-vdb` and not necessarily
to the other harnesses. The general rules — formatting, English
comments, no premature abstraction, code style — live in
[CONTRIBUTING.md](https://github.com/dogmalab/.github/blob/main/CONTRIBUTING.md#code-style-per-harness).

### 1. No async in the core

The state harness is a library. The user calls it from their code.
If their code is sync, the state harness blocks. If their code is
async, the state harness waits. Adding async to the core would
force every caller to think about async.

The state harness is a database. SQLite is sync. RocksDB is sync.
The state harness is sync. Async belongs in the agent harness and
the network harness, where it talks to LLMs and HTTP clients.

### 2. No `unsafe` in production logic, with one exception

`#![deny(unsafe_code)]` is set at the crate root. The only `unsafe`
blocks allowed are in `src/storage/traits.rs` for byte reinterpret
between `Vec<f32>` and `&[u8]`. These are documented, isolated, and
covered by tests. The byte reinterpret is the foundation of the
binary v2 mmap format; without it, the `MmapBackedStorage` cold
start would be impossible.

The current count is **5** `unsafe` blocks across the codebase.
The improvement plan is to consolidate them into a single
`mmap_file()` helper. See
[`docs/plans/2026-06-05-dogma-improvement-plan.md`](./docs/plans/2026-06-05-dogma-improvement-plan.md).

### 3. JSONL is the source of truth

The binary v2 mmap format is a cache. Every `Collection` can be
re-exported to JSONL and re-read. The two are synchronized on every
write. This is the **JSONL/Binary duality** that lets `cat`,
`grep`, and `jq` work on state harness files without any tool from
the state harness itself.

### 4. File size limit: 300 lines

No file in the state harness exceeds 300 lines (with documented
exceptions for test-heavy files: `storage.rs` 307, `smart_chunker/mod.rs`
536 which includes ~200 lines of tests). When a module grows past
this, it is split.

### 5. Minimal core dependencies

The state harness core has 11 dependencies:

- `serde` + `serde_json` + `thiserror` — essential
- `regex-lite` — smart chunker
- `rayon` — parallel iteration
- `wide` — SIMD-accelerated dot product / distance
- `bytemuck` — safe f32<->[u8] reinterpret
- `memmap2` — zero-copy memory-mapped I/O
- `once_cell` + `toml` + `log` — runtime config

Adding any of these to core requires a strong case in an issue.
Optional features may add more:

- `sml` → SIMIL ingestion parser (no new deps, uses existing tools)
- `watch` → `notify` + `crossbeam-channel`
- `chunker-syntax` → `tree-sitter` + language grammars
- `mcp` → `rmcp`, `tokio`, `tracing`, `clap`
- `cli` → `clap`, `tracing`, `tokio`

### 6. No new dependencies in core without an RFC

Per the org-level [CONTRIBUTING.md](https://github.com/dogmalab/.github/blob/main/CONTRIBUTING.md),
every new dependency in core requires a `[rfc]` issue with a
written justification. Adding a dependency is a long-term commitment
to maintenance, security updates, and supply-chain risk.

### 7. ANN rules (HNSW, IVF-PQ)

The approximate indexes complement `BruteForceIndex` without
replacing it:

- **Pure Rust implementation** — no external ANN libraries.
- **Same API** — implements the existing `Index` trait.
- **Configurable parameters** — `HnswConfig`, `IvfPqConfig` are
  documented in [`ARCH-SPEC.md`](./ARCH-SPEC.md).
- **Predictable memory** — documented per-index in `ARCH-SPEC.md`.

We do not compete with ScaNN on recall at the 1M-vector scale. We
compete on portability, inspectability, and "the file is the
service". If your use case needs ScaNN-class recall, you are not
the target audience; please use ScaNN directly.

### 8. SIGBUS awareness in `MmapBackedStorage`

When a file is mmap'd and the underlying file is truncated, the
process receives a SIGBUS on the next access. The state harness
documents this in the `MmapBackedStorage` API; users are expected
to treat the mmap as read-only while the process is running.
See the [README section](./README.md#cold-load-instant-0ms-via-mmap)
for the full warning.

### 9. English-only comments

All comments in the state harness are in English. The historical
exception is a small number of older modules where Spanish comments
were left in place from earlier authorship. New code must be in
English. See the org-level CONTRIBUTING for the rationale.

---

## How we evaluate new code

The five gates, in order:

1. **Compiles** with `cargo check --all-features`.
2. **Clippy clean** with `cargo clippy --all-features -- -D warnings`.
3. **Tests pass** with `cargo test --all-features`.
4. **No new dependencies** in core (or justified in an `[rfc]`).
5. **Correct formatting** with `cargo fmt --all -- --check`.

If everything passes, the code can be merged. If something fails,
fix it locally before opening a PR. CI runs the same checks.

---

## What we DON'T do

These are the state-harness-specific anti-rules. The platform-level
anti-rules are in the
[MANIFESTO.md](https://github.com/dogmalab/.github/blob/main/MANIFESTO.md#what-we-are-not).

- **No server mode.** A long-running HTTP server fronting the state
  harness. Breaks the "one file, no daemon" promise.
- **No new mandatory dependencies.** Every new dep is opt-in via a
  feature flag.
- **No async in core.** See rule 1.
- **No `unsafe` outside `storage/traits.rs`.** See rule 2.
- **No client/server API.** The state harness is a library, not a
  service. The network harness is the service.
- **No automatic schema migration.** The binary v2 format version
  is checked on load; incompatible versions return
  `Error::IncompatibleVersion`. Manual migration is the user's
  choice.

---

## Roadmap (state-harness-specific)

The platform-level roadmap is in the org-level
[ROADMAP.md](https://github.com/dogmalab/.github/blob/main/ROADMAP.md).
This section lists the items specific to the state harness that
are in flight or recently shipped.

### Recently shipped

- [x] SIMIL ingestion parser (`feature = "sml"`)
- [x] StorageStrategy (`Hybrid` and `SymbolicPure`)
- [x] IVF-PQ with SIMD-aligned `m_subspaces`
- [x] HNSW+SQ recall fix (0-60% → 90% with rescore)
- [x] `VectorStorage` trait decoupling
- [x] `MmapBackedStorage` (memory-mapped zero-copy, ~0ms cold start)
- [x] Binary v2 format (32-byte AVX2 alignment)

### In flight

- [ ] Multi-file collections (one `.vdb` directory per collection)
- [ ] Streaming JSONL export for large collections
- [ ] Parquet export as an opt-in (for migration from other systems)

### Queued (post-MVP)

- [ ] `dogma-vdb-mcp` over HTTP/SSE (feature flag, not core)
- [ ] Fuzz testing of the JSONL parser
- [ ] In-place update of documents (currently requires delete + insert)
- [ ] Multi-index search (run query against BF + HNSW + IVF-PQ, RRF)

### Rejected

- ❌ Server mode. See "What we DON'T do".
- ❌ Compete with ScaNN. See rule 7.
- ❌ Replace JSONL with Parquet. The duality is the point.
- ❌ Auto-tune `HnswConfig` at runtime. The user configures.

---

## See also

- [README.md](./README.md) — the public-facing documentation.
- [SPEC.md](./SPEC.md) — the formal specification.
- [ARCH-SPEC.md](./ARCH-SPEC.md) — architecture decisions.
- [RCA_GUIDE.md](./RCA_GUIDE.md) — how we audit our own benchmarks.
- [docs/FEATURES.md](./docs/FEATURES.md) — feature reference.
- Org-level docs:
  [MANIFESTO](https://github.com/dogmalab/.github/blob/main/MANIFESTO.md),
  [STRATEGY](https://github.com/dogmalab/.github/blob/main/STRATEGY.md),
  [CONTRIBUTING](https://github.com/dogmalab/.github/blob/main/CONTRIBUTING.md),
  [FAQ](https://github.com/dogmalab/.github/blob/main/FAQ.md),
  [ROADMAP](https://github.com/dogmalab/.github/blob/main/ROADMAP.md),
  [GLOSSARY](https://github.com/dogmalab/.github/blob/main/GLOSSARY.md).

---

*Last updated: 2026-07-07*
