# dogma-vdb

Portable vector database in JSONL format.

**Status**: Alpha — core compiles, 107 tests pass. Stubs for CLI, MCP server, and watcher.

## Design

- **JSONL format** — every `.vdb` file is plain JSONL, inspectable with `cat`, `grep`, `sed`
- **No server** — file-based, zero config, no daemon
- **Append-only** — insert is O(1)
- **MCP-ready** — optional MCP server for Claude Desktop / Cursor / opencode
- **Watch mode** — optional file watcher for auto-reindexing

## Quick Start

```rust
use dogma_vdb::prelude::*;

let mut col = Collection::open("my_data.vdb")?;
col.insert(Document::new("doc-1", "Rust is fast"))?;
let results = col.search(&[0.1, 0.2, 0.3], 5, Metric::Cosine);
```

## Crates

| Crate | Description |
|-------|-------------|
| `dogma-vdb` | Core library (storage, index, collection, chunking) |
| `dogma-vdb-cli` | CLI tool (skeleton) |
| `dogma-vdb-embed` | Embedder trait definition |
| `dogma-vdb-embed-fastembed` | Fastembed (ONNX) integration (skeleton) |
| `dogma-vdb-mcp` | MCP server (skeleton) |

## Build & Test

```bash
cargo check
cargo test          # 107 tests
cargo clippy -- -D warnings
cargo fmt -- --check
```

## License

MIT OR Apache-2.0
