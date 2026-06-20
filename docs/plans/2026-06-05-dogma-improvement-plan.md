# Plan de Mejora Integral — dogma-vdb

> **Basado en:** Análisis externo + verificación vía RAG (dogma-vdb-rag) + auditoría estructural
> **Fecha:** 2026-06-05
> **LOC base:** 9,509 Rust (54 archivos) · **Tests:** 192 pasan, 0 fallan · **Clippy:** 21 warnings

---

## Filosofía Guía

1. **Menos es más** — simplificar antes de expandir, eliminar redundancias, consolidar
2. **Correctness > Speed > Features** — certificar robustez antes de añadir
3. **Decisiones basadas en datos** — benchmarks reales (ONNX), no sintéticos
4. **Sin regresiones** — 0 warnings nuevos, 0 tests rotos, tracking de LOC ahorradas

---

## Mapa de Ruta

```
Fase 0: Auditoría y Baseline ────→ Fase 1: Corrección y Seguridad
         (semana 1)                         (semana 2)
              │                                    │
              ↓                                    ↓
         Fase 2: Rendimiento y Robustez ──→ Fase 3: Documentación y Calidad
              (semana 3-4)                       (semana 4-5)
                                                    │
                                                    ↓
                                              Fase 4: Features Estratégicas
                                                    (semana 6-8)
```

---

## Fase 0: Auditoría Estructural Profunda (Estimado: 2-3 días)

### 0.1 Auditoría de Safety y Concurrencia

**Hallazgo:** 5 bloques `unsafe` en el código base.

| # | Archivo | Línea | Propósito | Riesgo | 
|---|---------|-------|-----------|--------|
| U1 | `src/storage/traits.rs` | 59 | `from_raw_parts()` conversión u8→f32 | Moderado — sin verificación `len % 4 == 0` |
| U2 | `src/storage/traits.rs` | 153 | `Mmap::map()` full file | Bajo — sin protección SIGBUS |
| U3 | `src/storage/traits.rs` | 177 | `MmapOptions::new().offset()` | Bajo — sin protección SIGBUS |
| U4 | `src/index/ivf_pq_persistence.rs` | 215 | `Mmap::map()` en load | Bajo — sin protección SIGBUS |
| U5 | `src/index/ivf_pq_persistence.rs` | 351 | `Mmap::map()` en compact | Bajo — sin protección SIGBUS |

**Tareas:**

- [ ] **0.1.1** Verificar SAFETY comments en cada bloque — añadir donde falten
- [ ] **0.1.2** Añadir `debug_assert_eq!(data.len() % 4, 0)` antes de `from_raw_parts()` en `traits.rs:59`
- [ ] **0.1.3** Evaluar si `Mmap::map()` necesita `mmap.advise(Advice::Random)` para reducir page faults
- [ ] **0.1.4** Verificar que ningún archivo mmap'd se accede después de `drop` del `File` handle
- [ ] **0.1.5** Considerar `MmapMut` vs `Mmap` para storage de escritura

### 0.2 Auditoría de CONFIG Global

**Hallazgo:** Uso de `once_cell` para CONFIG global — limitación para multi-tenancy.

- [ ] **0.2.1** Mapear todos los puntos de acceso a `CONFIG` en el código
- [ ] **0.2.2** Evaluar reemplazo por `Collection::open_with()` como API primaria
- [ ] **0.2.3** Documentar que `CONFIG` global es para defaults, no para multi-tenancy

### 0.3 Mapeo de Dependencias y Features

**Hallazgo:** 3 dependencias pesadas (fastembed, ort, tree-sitter) opcionales pero con costos.

```toml
# Estado actual de dependencias opcionales
dogma-vdb (core):
  - tree-sitter* (feature: chunker-syntax) → ~15MB

dogma-vdb-embed-fastembed:
  - fastembed = "4" → ~90MB modelo ONNX

dogma-vdb-rerank:
  - ort = "=2.0.0-rc.9" → ~50MB runtime ONNX
```

- [ ] **0.3.1** Verificar que `ort = "=2.0.0-rc.9"` tiene versión estable ya disponible
- [ ] **0.3.2** Confirmar que la feature `chunker-syntax` en dogma-vdb core es realmente opcional y no tira del tree-sitter si no se activa

---

## Fase 1: Corrección y Seguridad (Estimado: 4-5 días)

### 1.1 Consolidación de Bloques Unsafe (P0)

**Objetivo:** Reducir de 5 a ≤ 3 bloques unsafe moviendo lógica de mmap a un helper centralizado.

**Archivos a modificar:**
- `src/storage/traits.rs`
- `src/index/ivf_pq_persistence.rs`

**Estrategia:**
Crear un helper `fn mmap_file(path: &Path) -> Result<Mmap>` en `src/storage/mod.rs` que centralice:
- `Mmap::map()` con manejo de errores
- `mmap.advise(Advice::Random)` para reducir page faults
- SAFETY comment único y verificable

```rust
/// Helper seguro para mapear archivos a memoria.
///
/// # Safety
/// El caller debe garantizar que el archivo no sea modificado mientras
/// el mmap esté vivo. Esta función no añade safety propia — el mmap
/// es inherentemente unsafe porque permite acceso a memoria compartida
/// con el sistema de archivos.
pub(crate) fn mmap_file(path: &Path) -> Result<Mmap> {
    let file = File::open(path)?;
    let mmap = unsafe { Mmap::map(&file) }
        .map_err(|e| Error::Io { path: path.to_owned(), source: e })?;
    let _ = mmap.advise(memmap2::Advice::Random);
    Ok(mmap)
}
```

**Impacto:** 
- 5 unsafe → 3 (los mmap se consolidan en 1 helper, `from_raw_parts` sigue en traits.rs)
- LOC estimado: +30 (helper) -30 (dispersión) = ~0 neto
- Tests: 192 deben seguir pasando

### 1.2 Eliminación de unwrap() en Código de Producción (P1)

**Hallazgo:** ~100 llamadas `.unwrap()` — la mayoría en tests, pero hay en producción:

| Archivo | Línea | Severidad |
|---------|-------|-----------|
| `src/memory.rs:166` | `result.unwrap()` en producción | **Alta** — paniquea si falla la memoria |
| `src/smart_chunker/paragraph.rs:119` | `Regex::new().unwrap()` | **Baja** — regex estático, nunca falla |
| `src/doc.rs:183-184` | `serde_json::to_string().unwrap()` | **Media** — paniquea si documento no serializable |
| `src/rerank.rs:56,66` | `reranker.rerank().unwrap()` en test | Baja — en `#[cfg(test)]` |

**Estrategia:**
- Reemplazar con `?` o `expect("contexto")` donde sea posible
- Dejar `unwrap()` solo en regex estáticos y código de test

- [ ] **1.2.1** Reemplazar `result.unwrap()` en `memory.rs:166` con `?` o `expect()`
- [ ] **1.2.2** Reemplazar `serde_json::to_string().unwrap()` en `doc.rs:183` con `expect()`
- [ ] **1.2.3** Verificar que no haya unwrap() en hot paths de búsqueda (collection.rs)

**Impacto:** 0 LOC ahorrados, pero reducción de superficie de pánico.

### 1.3 Añadir Tests para dogma-vdb-rag (P1)

**Hallazgo:** 0 tests en el crate `dogma-vdb-rag`.

**Archivos a crear:**
- `dogma-vdb-rag/tests/integration.rs`

**Tests a escribir:**
1. **test_ingest_empty_dir** — ingest con directorio vacío → 0 chunks, no panic
2. **test_ingest_single_file** — ingest 1 archivo → verificar chunks > 0
3. **test_ingest_output_exists** — verificar que el archivo .vdb se crea
4. **test_query_basic** — ingest → query → resultados no vacíos
5. **test_query_hybrid** — ingest → query --hybrid → resultados con BM25 + Vector
6. **test_watch_file_change** — watch detecta cambio → re-index correcto
7. **test_info_metadata** — info muestra metadata correcta
8. **test_skip_hidden** — archivos ocultos se saltan
9. **test_extension_filter** — filtro por extensión funciona
10. **test_batch_insert** — insertar + search + delete en colección

```rust
#[test]
fn test_ingest_single_file() {
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("src");
    std::fs::create_dir(&src).unwrap();
    std::fs::write(src.join("test.rs"), "fn main() { println!(\"hello\"); }").unwrap();
    let out = tmp.path().join("out.vdb");
    
    let exit_code = std::process::Command::new(env!("CARGO_BIN_EXE_dogma-vdb-rag"))
        .args(["ingest", src.to_str().unwrap(), "--output", out.to_str().unwrap(),
               "--hash", "--dim", "64"])
        .status().unwrap();
    
    assert!(exit_code.success());
    assert!(out.exists());
}
```

**Impacto:**
- +1 archivo, +~200 LOC
- +10 tests funcionales
- Cobertura mínima del pipeline RAG

---

## Fase 2: Rendimiento y Robustez (Estimado: 4-6 días)

### 2.1 Mejora de Recall en HNSW para Datasets Grandes (P1)

**Hallazgo:** HNSW recall cae a 30-70% con datasets >100K vectores o embeddings aleatorios.

**Contexto:** El `audit-1-fix-report.md` documenta la mejora de 0-60% → 90% para HNSW+SQ+Rescore, pero los benchmarks con datos aleatorios muestran 30-70%.

**Estrategia:**
1. Investigar si la métrica `cosine` vs `dot` afecta el recall (HNSW asume distancia L2 internamente)
2. Verificar `ef_construction` y `M` parámetros — ¿están optimizados para datasets grandes?
3. Añadir benchmark con dataset real (embeddings ONNX) y registrar recall

- [ ] **2.1.1** Leer `src/index/hnsw.rs` y entender parámetros actuales (`M=16`, `ef_construction=200`)
- [ ] **2.1.2** Ejecutar benchmark con 10K vectores reales (FastEmbed) midiendo recall vs brute-force
- [ ] **2.1.3** Si recall < 90%, ajustar `ef_construction` y `M`
- [ ] **2.1.4** Documentar parámetros óptimos en `TUNING_REPORT.md`

### 2.2 Persistencia de BM25 entre Queries (P2)

**Hallazgo:** BM25 se reconstruye desde cero en cada `query --hybrid`.

**Archivos a modificar:**
- `dogma-vdb-rag/src/query.rs`
- `dogma-vdb/src/bm25.rs` (si se añade persistencia)

**Estrategia:** Cachear el índice BM25 en disco como archivo adyacente al .vdb.

```rust
// En bm25.rs — nuevo método save/load
impl Bm25Index {
    pub fn save(&self, path: &Path) -> Result<()> { ... }
    pub fn load(path: &Path) -> Result<Self> { ... }
}
```

**Impacto:**
- +~80 LOC en bm25.rs para persistencia
- -X ms por query híbrida (ahorra reconstrucción de BM25)
- Cambio localizado, no afecta API pública

### 2.3 Benchmark Reproductible con Embeddings Reales (P1)

**Hallazgo:** Los benchmarks actuales son agresivos (77μs vs ChromaDB 4000μs) y no especifican siempre las condiciones exactas.

**Estrategia:**
1. Estandarizar benchmark con FastEmbed (all-MiniLM-L6-v2, 384-dim)
2. Publicar metodología exacta: hardware, dataset, flags, parámetros
3. Incluir 3 escalas: 5K, 50K, 100K vectores

**Archivos a modificar:**
- `benchmarks/src/main.rs` — añadir benchmark estandarizado
- `benchmarks/BENCHMARK.md` — documentar metodología

- [ ] **2.3.1** Crear benchmark reproducible en `benchmarks/src/standard.rs`
- [ ] **2.3.2** Ejecutar y capturar resultados en tabla markdown
- [ ] **2.3.3** Actualizar `benchmarks_comparison.md` con números verificables

---

## Fase 3: Documentación y Calidad de Código (Estimado: 2-3 días)

### 3.1 Estandarización de Idioma en Código (P3)

**Hallazgo:** Código mixto ES/EN — ~30% español en comentarios.

**Estrategia:** Convertir comentarios en español a inglés de una sola pasada.

- [ ] **3.1.1** Buscar todos los comentarios en español con regex `//.*[áéíóúñ]`
- [ ] **3.1.2** Traducir comentarios a inglés manteniendo el significado técnico
- [ ] **3.1.3** NO traducir nombres de variables o funciones (breaking change innecesario)

**Impacto:** ~17 líneas modificadas, 0 LOC neto, puramente cosmético.

### 3.2 Actualización de SPEC.md y ARCH-SPEC.md (P2)

**Hallazgo:** SPEC.md menciona 7 crates, 192 tests — verificar sincronización exacta.

**Estrategia:** 
1. Leer every module's doc comment + public API
2. Diff contra SPEC.md actual
3. Actualizar: conteo de tests, nuevas features, cambios de API

- [ ] **3.2.1** Extraer feature inventory del código (no de la spec)
- [ ] **3.2.2** Verificar cada claim de SPEC.md contra el código real
- [ ] **3.2.3** Actualizar SPEC.md con información actualizada
- [ ] **3.2.4** Actualizar ARCH-SPEC.md si hay cambios en la estructura de workspace
- [ ] **3.2.5** Actualizar README.md con benchmarks verificables

### 3.3 Eliminación de Clippy Warnings (P2)

**Hallazgo:** 21 warnings de clippy.

**Estrategia:** Seguir el skill `rust-clippy-warning-cleanup` para eliminarlos uno por uno.

```bash
# Clasificar warnings
cargo clippy --workspace 2>&1 | grep 'warning:' | sort | uniq -c | sort -rn
```

- [ ] **3.3.1** Clasificar los 21 warnings por tipo
- [ ] **3.3.2** Corregir cada warning (orden: dead_code → style → complexity → pedantic)
- [ ] **3.3.3** Verificar 0 warnings post-fix

**Impacto:** 21 warnings → 0, sin cambio de LOC funcional.

---

## Fase 4: Features Estratégicas (Estimado: 5-7 días)

### 4.1 MCP con Transporte HTTP/SSE (P2)

**Hallazgo:** MCP solo soporta stdio. Roadmap menciona gRPC y HTTP/WS.

**Archivos a modificar:**
- `dogma-vdb-mcp/src/main.rs` — añadir bandera `--transport http` + configuración de puerto
- `dogma-vdb-mcp/Cargo.toml` — añadir `axum` o `warp` como dependencia opcional

**Estrategia:**
1. Añadir flag `--transport [stdio|http]` al CLI del MCP
2. HTTP transport usa `axum` para servir SSE + JSON-RPC
3. Mantener stdio como default (backwards compatible)

```bash
# Modo stdio (default, existente)
dogma-vdb-mcp

# Modo HTTP (nuevo)
dogma-vdb-mcp --transport http --port 8080
```

**Impacto:**
- +~150 LOC para handler HTTP
- +1 dependencia opcional (axum)
- Compila opcionalmente via feature flag

### 4.2 Sistema de Plugins para Embedders (P3)

**Hallazgo:** El trait `Embedder` existe pero FastEmbedAdapter está hardcodeado en dogma-vdb-rag.

**Estrategia:**
1. Mover `FastEmbedAdapter` a `dogma-vdb-embed-fastembed` como export público
2. Permitir que usuarios registren embedders custom vía trait

- [ ] **4.2.1** Exportar `FastEmbedAdapter` desde `dogma-vdb-embed-fastembed`
- [ ] **4.2.2** Documentar cómo implementar un Embedder custom

### 4.3 Sistema de Dogfooding Robusto (P1)

**Hallazgo:** El RAG existe pero no se usa activamente para documentar el propio proyecto.

**Estrategia:**
1. Script CI que ejecute `dogma-vdb-rag ingest` sobre el código fuente
2. Query predefinidas que verifiquen información clave
3. Generar reporte de salud del proyecto automáticamente

```bash
#!/bin/bash
# dogfood.sh — evaluar salud del proyecto vía RAG
dogma-vdb-rag ingest ./src --output .dogfood/health.vdb --hash --dim 64
dogma-vdb-rag query .dogfood/health.vdb "unsafe blocks" --hash --dim 64 -k 3
dogma-vdb-rag query .dogfood/health.vdb "recall benchmark performance" --hash --dim 64 -k 3
```

- [ ] **4.2.1** Crear `scripts/dogfood.sh`
- [ ] **4.2.2** Documentar en README cómo usar dogma-vdb-rag para auto-documentación

---

## Tabla de Prioridades Consolidada

| # | Tarea | Fase | Severidad | Esfuerzo | LOC impacto | Prioridad |
|---|-------|------|-----------|----------|-------------|-----------|
| 1.1.1-5 | Consolidación unsafe blocks | 1 | **Crítica** | 4h | ~0 | **P0** |
| 1.2.1-3 | Eliminar unwrap() en producción | 1 | **Alta** | 2h | ~0 | **P0** |
| 1.3.1-10 | Tests dogma-vdb-rag | 1 | **Alta** | 6h | +200 | **P0** |
| 3.3.1-3 | Eliminar clippy warnings | 3 | **Alta** | 4h | ~-20 | **P0** |
| 2.2.1-4 | Persistencia BM25 | 2 | **Media** | 4h | +80 | **P1** |
| 2.3.1-3 | Benchmark reproducible | 2 | **Media** | 6h | +150 | **P1** |
| 3.2.1-5 | Actualizar SPEC.md | 3 | **Media** | 4h | ~0 | **P1** |
| 0.1.1-5 | Auditoría unsafe | 0 | **Media** | 3h | ~0 | **P2** |
| 0.2.1-3 | Auditoría CONFIG | 0 | **Baja** | 2h | ~0 | **P2** |
| 2.1.1-4 | Mejora HNSW large-scale | 2 | **Media** | 8h | +100 | **P2** |
| 4.1.1-3 | MCP HTTP/SSE | 4 | **Media** | 8h | +150 | **P2** |
| 3.1.1-3 | Estandarizar idioma | 3 | **Cosmética** | 1h | ~0 | **P3** |
| 4.2.1-2 | Plugin Embedders | 4 | **Baja** | 6h | +80 | **P3** |
| 4.3.1-2 | Dogfooding automático | 4 | **Baja** | 3h | +30 | **P3** |

---

## Seguimiento de Progreso

### LOC Tracker

| Fase | LOC inicial | LOC final | Δ |
|------|------------|-----------|---|
| 0 | 9,509 | 9,509 | 0 |
| 1 | 9,509 | 9,689 | +200 (tests) -20 (clippy) = +180 |
| 2 | 9,689 | 9,919 | +230 |
| 3 | 9,919 | 9,899 | -20 (clippy fixes) |
| 4 | 9,899 | 10,129 | +230 |
| **Total** | **9,509** | **10,129** | **+620 (6.5%)** |

### Test Tracker

| Hito | Tests | Estado |
|------|-------|--------|
| Baseline | 192 | ✅ |
| + tests dogma-vdb-rag | 202 | ⏳ |
| + benchmark tests | 205 | ⏳ |
| Final | 205+ | 🎯 |

### Reglas de Verificación

**Cada tarea completada DEBE verificar:**
```bash
cargo check --workspace --all-features      # 0 errores
cargo test --workspace                      # 0 fallos, mismo número o mayor
cargo clippy --workspace                    # 0 nuevos warnings
```

**No comprometer:**
- API pública (no cambiar firmas sin migración)
- Formato de archivo .vdb (backwards compatible)
- Tests existentes (no modificar, solo añadir)

---

## Documentos Relacionados

- [SPEC.md](../SPEC.md) — Especificación del workspace
- [ARCH-SPEC.md](../ARCH-SPEC.md) — Arquitectura multi-crate
- [AGENTS.md](../AGENTS.md) — Guía para agentes IA
- [RCA_GUIDE.md](../RCA_GUIDE.md) — Diagnóstico de recall
- [benchmarks/BENCHMARK.md](../benchmarks/BENCHMARK.md) — Benchmarks actuales

---

## Notas Técnicas

### Sobre FastEmbedAdapter (descubrimiento durante verificación RAG)

Para usar FastEmbedder real desde dogma-vdb-rag:
```rust
// Import necesario — el trait Embedder de fastembed está en dogma_vdb_embed
use dogma_vdb_embed::Embedder as FastEmbedTrait;
// El wrapper FastEmbedAdapter (en ingest.rs) adapta al CoreEmbedder de dogma_vdb
```

Este detalle no está documentado en README.md — añadirlo a la documentación técnica.

### Sobre el formato .vdb

El análisis RAG confirmó que el formato binario v2 usa mmap zero-copy. El archivo
es un JSONL de documentos + metadata binaria de embeddings. La estructura:
```
.vdb/
  ├── meta.json        — metadatos de la colección
  ├── doc.jl           — documentos en JSONL
  └── emb.bin          — embeddings en f32[]
```

---

> **Siguiente paso:** ¿Procedemos con la Fase 0 (auditoría) o prefieres que empecemos por la Fase 1 (correcciones)?
