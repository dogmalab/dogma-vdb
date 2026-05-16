# 🤖 AGENTS.md — Reglas para Implementar dogma-vdb

## Filosofía del Proyecto

```
dogma-vdb = Rust + JSONL + serde_json
            (1 dep core, portable, sin servidor)
```

Cada línea de código debe justificar su existencia. Preferimos **50 líneas claras** a 200 líneas "arquitectónicamente flexibles".

---

## ✅ Lo Que SÍ Hacemos

### 1. Rust idiomático — sin rodeos

- **Ownership ante todo**. Tomar prestado (`&`) por defecto, owned (`T`) solo cuando el callee necesita ser dueño.
- **`Into<T>` en constructores** para flexibilidad sin costo.
- **`impl Trait` en parámetros** (monomorfización) en lugar de `Box<dyn Trait>` a menos que necesites dynamic dispatch real.
- **`sort_unstable`** sobre `sort`. No necesitamos estabilidad.
- **`#[inline]`** solo en funciones de 1-3 líneas que están en hotspots (distances, dot product).
- **`debug_assert_eq!`** para precondiciones que solo deben chequearse en debug.

### 2. Código pequeño — cada archivo < 200 líneas

Máximo 200 líneas por archivo. Si un módulo crece más, se divide:

```rust
// Bien: 150 líneas, una responsabilidad
src/storage.rs

// Mal: 500 líneas mezclando storage + index + collection
```

### 3. Dependencias mínimas — preguntar antes de añadir

Cada dependencia externa debe pasar esta prueba:

1. **¿Realmente la necesito?** — ¿Puedo hacerlo con stdlib?
2. **¿Cuánto aporta?** — Si ahorra 10 líneas pero añade 40 crates transitorios, no.
3. **¿Es optional?** — Debe ir detrás de un feature flag si no es crítica para el core.

**Regla de oro**: el core (`dogma-vdb`) solo tiene `serde` + `serde_json` + `thiserror`.  
Todo lo demás (notify, tokio, rmcp) es opcional por features.

### 4. Pruebas desde el principio

- Cada módulo tiene `#[cfg(test)] mod tests` al final.
- Tests de integración en `tests/` usan archivos temporales reales.
- Los tests deben pasar **sin red** ni servicios externos.
- Un test que falla por "not yet implemented" está bien — **mientras compile**.

### 5. Formato JSONL — el centro del diseño

```
Archivo .vdb
├── Línea 1: {"id":"doc-1","text":"...","embedding":[0.1,...],"metadata":{...}}
├── Línea 2: {"id":"doc-2","text":"...","embedding":[...],"metadata":{...}}
└── ...
```

- **Cada línea es independiente** — se puede hacer `grep`, `sed`, `head`.
- **Append-only** por diseño — añadir es O(1). Actualizar requiere reescribir (pero es poco frecuente en RAG).
- **`serde_json::from_str`** linea por linea (streaming con BufReader).

### 6. Traits pequeños y enfocados

```rust
// Bien: 1 método necesario, default para batch
pub trait Embedder: Send + Sync {
    fn embed(&self, text: &str) -> Result<Vec<f32>>;
    fn dimension(&self) -> usize;
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> { /* default */ }
}

// Mal: trait con 8 métodos, 3 genéricos, 2 lifetimes, y un associated type
```

### 7. Documentación útil

- `///` con ejemplos de uso en toda la API pública.
- `# Examples` en docstrings (que se ejecutan con `cargo test --doc`).
- `#[must_use]` en funciones cuyo resultado no debería ignorarse.

---

## ❌ Lo Que NO Hacemos

### 1. NO añadir dependencias al core sin discutirlo

```toml
# MAL — esto va al core y lo arrastran todos los usuarios
[dependencies]
tokio = "1"
reqwest = "0.12"
anyhow = "1"
tracing = "0.1"

# BIEN — las dependencias pesadas van detrás de features
[dependencies]
tokio = { version = "1", optional = true }
```

Si alguien quiere solo leer archivos `.vdb` sin async ni HTTP, que pueda hacerlo con **1 sola dependencia**.

### 2. NO premature abstraction

```rust
// MAL — abstraer por abstraer
trait DistanceCalculator {
    fn compute(&self, a: &[f32], b: &[f32]) -> f32;
}
struct CosineCalculator;
impl DistanceCalculator for CosineCalculator { ... }

// BIEN — una función, concreta, reutilizable
pub fn cosine(a: &[f32], b: &[f32]) -> f32;
```

No necesitamos un trait `DistanceCalculator` ni un `VectorIndexFactory`.  
Empezamos con `BruteForceIndex` y si hace falta HNSW luego, se añade como otro implementador del trait `Index`.

### 3. NO clonar sin necesidad

```rust
// MAL — clonar todo para "seguridad"
fn search(&self, query: Vec<f32>) -> Vec<Document> {
    let query = query.clone();
    // ...
}

// BIEN — prestar lo que se pueda
fn search(&self, query: &[f32]) -> Vec<ScoredDocument>;
```

### 4. NO unwrap() en producción

```rust
// MAL — paniquea silenciosamente
let doc = docs.iter().find(|d| d.id == "x").unwrap();

// BIEN — error manejable
let doc = docs.iter().find(|d| d.id == "x")
    .ok_or_else(|| Error::DocumentNotFound("x".into()))?;
```

`unwrap()` solo en tests y ejemplos.

### 5. NO estructuras sobreingeniería

- Sin `async` en el core. Si se necesita async, va en el crate `mcp` o `cli`.
- Sin macros procedurales.
- Sin `unsafe` a menos que sea estrictamente necesario y medido.
- Sin genéricos innecesarios. Un `Vec<f32>` es un `Vec<f32>`, no un `Vector<T: Numeric + Clone>`.

### 6. NO ignorar los warnings de clippy

El CI falla con `-D warnings`. Silenciar warnings con `#[allow(...)]` solo si hay una razón justificada y documentada.

---

## 🛠️ Herramientas Que Tenemos

### Del core (siempre disponibles)

| Herramienta | Para qué |
|---|---|
| `std::fs` | Leer/escribir archivos .vdb |
| `std::io::{BufReader, BufWriter}` | Streaming línea por línea |
| `std::collections::HashMap` | Metadatos de documentos |
| `serde_json` | Serializar/deserializar JSONL |
| `thiserror` | Errores tipados |

### De la stdlib de Rust (sin dependencias extra)

```rust
// Matemáticas
f32::sqrt()          // → magnitud de vectores
f32::powi()          // → distancia euclideana
f32::abs()           // → tolerancias

// Iteradores
.iter().zip()        // → dot product
.map().sum()         // → suma de productos
.sort_unstable_by()  // → ordenar por score

// Archivos
File::open()         // → leer .vdb
File::create()       // → escribir .vdb
OpenOptions::append()// → append al .vdb

// Paths
Path::exists()       // → ¿existe el archivo?
Path::extension()    // → filtrar por extensión
Path::file_stem()    // → nombre de colección
```

### Con features opcionales

| Feature | Herramientas extra |
|---|---|
| `watch` | `notify` (inotify/kqueue), `crossbeam-channel` |
| `mcp` | `rmcp`, `tokio`, `tracing`, `clap` |

---

## 📐 Estructura Típica de un Módulo

```rust
//! 1. Docstring de una línea con el propósito.

// 2. Imports agrupados: stdlib, externos, crate
use std::path::PathBuf;
use crate::error::Result;

// 3. Tipos públicos (struct, enum, trait)
pub struct Foo { ... }
pub trait Bar { ... }

// 4. Implementaciones
impl Foo { ... }
impl Bar for Foo { ... }

// 5. Funciones públicas helpers (si aplica)
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

## 🧪 Cómo Evaluamos Código Nuevo

1. **Compila con `cargo check --all-features`** ✅
2. **Sin errores de clippy** (`cargo clippy --all-features -- -D warnings`) ✅
3. **Tests pasan** (`cargo test --all-features`) ✅
4. **Sin dependencias nuevas** en el core (o justificadas) ✅
5. **Formato correcto** (`cargo fmt --all -- --check`) ✅

Si cumple todo, el código puede mergearse.

---

*Última actualización: 2026-05-16*
