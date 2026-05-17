# AGENTS.md — Reglas para Implementar dogma-vdb

## Filosofia del Proyecto

```
dogma-vdb = Rust + JSONL + serde_json
            (minimo deps, portable, sin servidor)
```

Cada linea de codigo debe justificar su existencia. Preferimos **50 lineas claras** a 200 lineas "arquitectonicamente flexibles".

---

## ✅ ESTADO ACTUAL (2026-05-17)

### Core crate (`dogma-vdb`) — COMPILA y 152 TESTS PASAN

| Modulo | Archivo | Lines | Tests | Estado |
|--------|---------|-------|-------|--------|
| Document | `src/doc.rs` | 205 | 8 | Completo |
| Error | `src/error.rs` | 45 | - | Completo |
| Distance | `src/distance.rs` | 209 | 16 | Completo |
| Filter | `src/filter.rs` | 122 | 9 | Completo |
| Storage (JSONL) | `src/storage.rs` | 307 | 15 | Completo |
| Collection | `src/collection.rs` | ~530 | 15 | Completo |
| Runtime Config | `src/config.rs` | ~320 | - | Completo |
| Chunker | `src/chunker.rs` | 247 | 8 | Completo |
| Embedder trait | `src/embedding.rs` | 28 | - | Completo |
| SmartChunker | `src/smart_chunker/` | ~560 | 20+ | Completo |
| Index trait | `src/index/mod.rs` | 67 | - | Completo |
| Index (BruteForce) | `src/index/brute_force.rs` | 440 | 18 | Completo |
| Index (HNSW) | `src/index/hnsw.rs` | ~840 | 21 | Completo |
| Index (IVF-PQ) | `src/index/ivf_pq.rs` | ~400 | 8 | Nuevo |
| Index (Annoy) | ~~`src/index/annoy.rs`~~ | — | — | **Eliminado** |
| SQ module | `src/index/sq.rs` | ~230 | 8 | Completo |
| Watcher | `src/watch.rs` | 56 | - | **SKELETON** (`todo!()`) |
| MCP Server | `src/mcp.rs` | 36 | - | **SKELETON** (`todo!()`) |

### Sub-crates

| Crate | Archivo | Estado |
|-------|---------|--------|
| `dogma-vdb-cli` | `cli/src/main.rs` | Completo (info, list, query, ingest, delete) |
| `dogma-vdb-mcp` | `mcp/src/main.rs` | Completo (vecdb_query, ingest, delete, list, info) |
| `dogma-vdb-embed` | `embed/src/lib.rs` | Completo (trait definition) |
| `dogma-vdb-embed-fastembed` | `embed-fastembed/src/lib.rs` | Completo (FastEmbedder con ONNX MiniLM-L6-v2) |

### Tests
- Unitarios: 139 pasan
- Integracion: 9 pasan
- Doc-tests: 8 pasan, 2 ignorados
- **Total: 156 tests, 0 fallos**

---

## ✅ Lo Que SI Hacemos

### 1. Rust idiomatico — sin rodeos

- **Ownership ante todo**. Tomar prestado (`&`) por defecto, owned (`T`) solo cuando el callee necesita ser dueno.
- **`Into<T>` en constructores** para flexibilidad sin costo.
- **`impl Trait` en parametros** (monomorfizacion) en lugar de `Box<dyn Trait>` a menos que necesites dynamic dispatch real.
- **`sort_unstable`** sobre `sort`. No necesitamos estabilidad.
- **`#[inline]`** solo en funciones de 1-3 lineas que estan en hotspots (distances, dot product).
- **`debug_assert_eq!`** para precondiciones que solo deben chequearse en debug.

### 2. Codigo pequeno — cada archivo < 300 lineas

Maximo 300 lineas por archivo (con excepciones para test-heavy:
`storage.rs` 307, `smart_chunker/mod.rs` 536 que incluye ~200 de tests).
Si un modulo crece mas, se divide.

### 3. Dependencias minimas — preguntar antes de anadir

**Deps obligatorias del core actual:**
- `serde` + `serde_json` + `thiserror` — esenciales
- `regex-lite` — smart chunker (regex lightweight)
- `once_cell` + `toml` + `log` — config runtime

**Deps opcionales (features):**
- `watch` → `notify` + `crossbeam-channel`
- `mcp` → `rmcp` + `tokio` + `tracing` + `clap`

### 4. Pruebas desde el principio

- Cada modulo tiene `#[cfg(test)] mod tests` al final.
- Tests de integracion en `tests/` usan archivos temporales reales.
- Los tests deben pasar **sin red** ni servicios externos.
- Todos los tests nuevos deben compilar y pasar en CI.

### 5. Formato JSONL — el centro del diseno

```
.vdb file
├── Line 1: {"id":"doc-1","text":"...","embedding":[0.1,...],"metadata":{...}}
├── Line 2: {"id":"doc-2","text":"...","embedding":[...],"metadata":{...}}
└── ...
```

- **Cada linea es independiente** — se puede hacer `grep`, `sed`, `head`.
- **Append-only** por diseno — anadir es O(1). Actualizar requiere reescribir.
- **`serde_json::from_str`** linea por linea (streaming con BufReader).

### 6. Traits pequenos y enfocados

```rust
pub trait Embedder: Send + Sync {
    fn embed(&self, text: &str) -> Result<Vec<f32>>;
    fn dimension(&self) -> usize;
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> { /* default */ }
}
```

### 7. Documentacion util

- `///` con ejemplos de uso en toda la API publica.
- `# Examples` en docstrings (se ejecutan con `cargo test --doc`).
- `#[must_use]` en funciones cuyo resultado no deberia ignorarse.

---

## ❌ Lo Que NO Hacemos

### 1. NO anadir dependencias al core sin discutirlo

Si alguien quiere solo leer archivos `.vdb` sin async ni HTTP, que pueda hacerlo con deps minimas.

### 2. NO premature abstraction

```rust
// MAL — abstraer por abstraer
trait DistanceCalculator { fn compute(&self, a: &[f32], b: &[f32]) -> f32; }

// BIEN — una funcion, concreta, reutilizable
pub fn cosine(a: &[f32], b: &[f32]) -> f32;
```

Empezamos con `BruteForceIndex` y si hace falta HNSW luego, se anade como otro implementador del trait `Index`.

### 3. NO clonar sin necesidad

```rust
// MAL
fn search(&self, query: Vec<f32>) -> Vec<Document> { let query = query.clone(); ... }

// BIEN
fn search(&self, query: &[f32]) -> Vec<ScoredDocument>;
```

### 4. NO unwrap() en produccion

```rust
// MAL
let doc = docs.iter().find(|d| d.id == "x").unwrap();

// BIEN
let doc = docs.iter().find(|d| d.id == "x")
    .ok_or_else(|| Error::DocumentNotFound("x".into()))?;
```

`unwrap()` solo en tests y ejemplos.

### 5. NO estructuras sobreingenieria

- Sin `async` en el core. Si se necesita async, va en el crate `mcp` o `cli`.
- Sin macros procedurales.
- Sin `unsafe` a menos que sea estrictamente necesario y medido.
- Sin genericos innecesarios.

### 6. NO ignorar los warnings de clippy

El CI falla con `-D warnings`. Silenciar warnings con `#[allow(...)]` solo si hay una razon justificada y documentada.

### 7. ANN Index (HNSW) — reglas

El index aproximado complementa a `BruteForceIndex` sin reemplazarlo:

- **Implementacion pura en Rust** — sin nuevas dependencias externas
- **Misma API** — implementa el trait `Index` existente
- **Parametros configurables** en `HnswConfig`: `M` (conexiones), `ef_construction` (calidad build), `ef_search` (calidad query)
- **Memoria predecible**: cada nodo guarda su vector + vecinos por capa
- **`ef_search` controla el balance**: mayor valor = mas recall, menos velocidad
- **Collection puede usar cualquiera**: se inyecta via `HnswConfig` en lugar de `Metric`

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

Rendimiento esperado vs BruteForce:

| Dataset | BruteForce | HNSW (ef=50) | HNSW (ef=200) |
|---------|-----------|--------------|---------------|
| 1K      | 0.5ms     | 0.2ms        | 0.5ms         |
| 10K     | 5ms       | 0.5ms        | 2ms           |
| 100K    | 50ms      | 1ms          | 5ms           |
| 1M      | 500ms     | 3ms          | 15ms          |
| Recall  | 100%      | ~90-95%      | ~98-99%       |

---

## Herramientas Que Tenemos

### Del core (siempre disponibles)

| Herramienta | Para que |
|---|---|
| `std::fs` | Leer/escribir archivos .vdb |
| `std::io::{BufReader, BufWriter}` | Streaming linea por linea |
| `std::collections::HashMap` | Metadata de documentos |
| `serde_json` | Serializar/deserializar JSONL |
| `thiserror` | Errores tipados |
| `regex_lite` | Smart chunking por tipo de archivo |

### De la stdlib de Rust (sin dependencias extra)

```rust
f32::sqrt()          // → magnitud de vectores
f32::powi()          // → distancia euclideana
f32::abs()           // → tolerancias
.iter().zip()        // → dot product
.map().sum()         // → suma de productos
.sort_unstable_by()  // → ordenar por score
File::open()         // → leer .vdb
File::create()       // → escribir .vdb
OpenOptions::append()// → append al .vdb
Path::exists()       // → ¿existe el archivo?
Path::extension()    // → filtrar por extension
Path::file_stem()    // → nombre de coleccion
```

### Con features opcionales

| Feature | Herramientas extra |
|---|---|
| `watch` | `notify` (inotify/kqueue), `crossbeam-channel` |
| `mcp` | `rmcp`, `tokio`, `tracing`, `clap` |

---

## Estructura Tipica de un Modulo

```rust
//! 1. Docstring de una linea con el proposito.

// 2. Imports agrupados: stdlib, externos, crate
use std::path::PathBuf;
use crate::error::Result;

// 3. Tipos publicos (struct, enum, trait)
pub struct Foo { ... }
pub trait Bar { ... }

// 4. Implementaciones
impl Foo { ... }
impl Bar for Foo { ... }

// 5. Funciones publicas helpers (si aplica)
pub fn helper() { ... }

// 6. Tests (al final del archivo)
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_foo() { ... }
}
```

---

## Como Evaluamos Codigo Nuevo

1. **Compila con `cargo check --all-features`** ✅
2. **Sin errores de clippy** (`cargo clippy --all-features -- -D warnings`) ✅
3. **Tests pasan** (`cargo test --all-features`) ✅
4. **Sin dependencias nuevas** en el core (o justificadas) ✅
5. **Formato correcto** (`cargo fmt --all -- --check`) ✅

Si cumple todo, el codigo puede mergearse.

---

## Pendiente (Roadmap)

- [x] Implementar HNSW index (`src/index/hnsw.rs`)
- [x] Collection puede usar HNSW via config
- [x] CRUD completo (insert, delete, update)
- [x] CLI (info, list, query, ingest, delete)
- [x] MCP server (vecdb_query, ingest, delete, list, info)
- [x] Benchmarks comparativos (todos los backends)
- [x] HNSW flat_embeddings
- [x] SQ module + integracion en BF y HNSW
- [x] SQ rescore (recuperar recall con f32)
- [x] IVF-PQ index (inverted file + product quantization)
- [x] Config env vars para todos los campos
- [ ] Implementar `watch.rs` (file system watcher, feature = "watch")
- [ ] Implementar `mcp.rs` (MCP server, feature = "mcp")
- [x] Implementar FastEmbed real (`dogma-vdb-embed-fastembed`)
- [x] Workspace multi-crate (root Cargo.toml)
- [ ] Ejemplos completos en `examples/`

---

*Ultima actualizacion: 2026-05-16*
