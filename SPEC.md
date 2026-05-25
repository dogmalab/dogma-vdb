# dogma-vdb — Functional Specification

> Basado en auditoría de código fuente (2026-05-25). Cubre crate principal
> `dogma-vdb` + 6 crates workspace: CLI, MCP, embed, fastembed, rerank, benchmarks.

---

## 1. Resumen

Base de datos vectorial portable en Rust. Formato binario `.vdb` con
auto-detección de formato JSONL legacy. Zero async en core, sin servidor,
config-driven.

**Problema**: ChromaDB es pesado (300 MB pip), LanceDB es complejo
(50K LOC, 200+ deps). Se necesita algo tiny, portable, debugeable
con `cat`/`grep`/`sed`, que corra en cualquier lado con un solo binario.

**Usuario objetivo**: Desarrolladores que necesitan ANN local para RAG
o datasets < 100K vectores, sin querer levantar servidores ni instalar
Python.

---

## 2. Index Backends

Cada backend implementa el trait `Index`:

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
- **Descripción**: Búsqueda exacta O(n·d) por scan lineal.
- **SQ opcional**: comprime f32→i8 para ~4× menos RAM, ~2× más rápido.
- **Rescore opcional**: recalcula top-k*2 con f32 para recuperar recall.
- **Pre-filtering**: aplica filtro ANTES de calcular distancia (eficiente).
- **Estado**: IMPLEMENTADO (~517 LOC).

### RF-02: HnswIndex
- **Descripción**: Búsqueda aproximada O(log n) via grafo jerárquico (Malkov & Yashunin 2016).
- **Config**: `M` (conexiones por nodo), `ef_construction`, `ef_search`.
- **Nivel aleatorio deterministico**: SplitMix64 (sin dependencia `rand`).
- **flat_embeddings**: Vec<f32> contiguo en vez de Vec<Vec<f32>> (reduce cache misses).
- **SQ opcional**: grafo construido con f32 exacto, búsqueda con i8.
- **Rescore opcional**: top-k*2 candidatos → rescore f32 (recall 90%+).
- **Heurístico de diversidad**: evita ciclos en capa 0.
- **Estado**: IMPLEMENTADO (~1,061 LOC).

### RF-03: IvfPqIndex
- **Descripción**: IVF + Product Quantization. Particiona espacio con K-Means, comprime subvectores a u8.
- **Config**: `n_list`, `n_probe`, `m_subspaces` (múltiplo de 8), `metric`, `rerank_enabled`.
- **Build** (batch):
  1. K-Means sobre embeddings (k-means++ init, max 20 iteraciones).
  2. Asignación a centroide + partición en subvectores.
  3. PQ codebook: 256 centroides por subespacio.
  4. Códigos u8 por documento.
- **Búsqueda**: distancia query→centroides, LUTs asimétricas, scan de códigos u8.
- **Auto-tuning**: si `rerank_enabled=true`, n_probe efectivo se reduce a la mitad.
- **Insert/Delete**: rebuild completo del índice.
- **Persistencia**: atómica (write-tmp + rename), soft-delete, compactación.
- **Validación**: `m_subspaces % 8 == 0` (alineación SIMD).
- **Estado**: IMPLEMENTADO (~921 LOC index + 724 LOC persistence).

### RF-04: SQ — Scalar Quantization
- **Descripción**: Capa de optimización ortogonal. Comprime f32→i8. 4× menos RAM, 2× más rápido.
- **Algoritmo**: `scale = (max - min) / 255.0`, `bias = (max + min) / 2.0`, `i8 = clamp((f32 - bias) / scale, -128, 127)`.
- **Funciones**: `quantize()`, `quantize_query()`, `dot_i8()`, `score_i8()`, `rescore()`.
- **Ortogonal**: funciona con BruteForce, HNSW e IVF-PQ.
- **Document siempre almacena f32**: SQ es solo para búsqueda en memoria.
- **Estado**: IMPLEMENTADO (~294 LOC).

### RF-05: BM25 Text Index
- **Descripción**: Índice de texto invertido liviano para búsqueda híbrida.
- **Fórmula**: BM25Okapi estándar (k₁=1.2, b=0.75).
- **Tokenización**: split en no-alfanuméricos + lowercase.
- **API**: `search(text, k) -> Vec<(doc_index, score)>`.
- **Estado**: IMPLEMENTADO (~194 LOC).

### RF-06: RRF — Reciprocal Rank Fusion
- **Descripción**: Combina dos listas rankeadas (vector + BM25) usando RRF estándar.
- **Fórmula**: `score(d) = Σ(1 / (k + rank_i(d)))` donde k=60.
- **Estado**: IMPLEMENTADO (~122 LOC).

---

## 3. Almacenamiento

### RF-07: BinStorage v2
- **Formato binario** (`.vdb`):
  ```
  Offset  Size  Field
  0       4     magic: "DVDB"
  4       4     version: u32 LE (2)
  8       4     dim: u32 LE
  12      4     count: u32 LE
  16      8     emb_offset: u64 LE
  24      —     metadata blocks (id, text, k-v metadata)
  emb_offset  —  embeddings f32 LE contiguos (32-byte aligned)
  ```
- **Padding**: alineación a 32 bytes para AVX2.
- **Auto-detección**: Collection.open() lee magic bytes `DVDB` — si no coincide, devuelve error.
- **embedding_region()**: lee solo header (24 bytes) para obtener offset/dim/count sin cargar metadata a RAM.
- **Estado**: IMPLEMENTADO (~563 LOC).

### RF-08: Export JSONL (vía Collection)
- **Formato**: JSONL (una línea por documento) — auto-descriptivo, debugeable con `cat`/`grep`.
- **Uso**: `collection.export_jsonl(path)` exporta documentos a JSONL para debug.
- **Nota**: el core ya no carga JSONL. La exportación es inline vía `serde_json::to_string`.
- **Estado**: IMPLEMENTADO (en collection.rs).

### RF-09: VectorStorage Trait
- **Propósito**: abstraer almacenamiento contiguo de embeddings (RAM o mmap).
- **API**:
  ```rust
  pub trait VectorStorage: Send + Sync {
      fn as_bytes(&self) -> &[u8];
      fn as_embeddings(&self) -> &[f32];  // unsafe: u8→f32 reinterpret (aislado, auditado)
      fn flush(&self) -> Result<()>;
      fn len(&self) -> usize;
      fn is_empty(&self) -> bool;
  }
  ```
- **MemoryBackedStorage**: Vec<u8> respaldo. Para tests, pipelines volátiles.
- **MmapBackedStorage**: memmap2 (~0ms load, OS pagea bajo demanda).
  - `open(path)` — mapea embedding region completa.
  - Incluye `advise(memmap2::Advice::Random)` para reducir page faults en lectura secuencial.
- **Estado**: IMPLEMENTADO (~297 LOC trait + 563 LOC storage).

---

## 4. Colección — API de Alto Nivel

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
Collection::export_jsonl(path) -> Result<()>                  // export a JSONL
Collection::documents() -> Iterator<Item = &Document>
Collection::embedding_storage() -> Option<&Arc<dyn VectorStorage>>
Collection::len() / is_empty() / name() / path()
```
- **Auto-detección**: formato binario vs JSONL legacy en `open()`.
- **3 backends**: configurable via `index_type` (bruteforce|hnsw|ivf_pq).
- **Estado**: IMPLEMENTADO (~808 LOC).

### RF-11: Hybrid Search Pipeline
1. **Extract**: `candidate_multiplier × top_k` de cada motor activo (vector + BM25).
2. **Fuse**: RRF si ambos motores activos, mantiene `2 × top_k`.
3. **Rerank**: si `PerformanceProfile` lo habilita y hay reranker, reordena con Cross-Encoder.
4. **Performance Profiles**: `PrecisionLocal`, `ProduccionHibrido`, `VelocidadExtrema`.
   - `use_bm25()`, `use_reranker()`, `candidate_multiplier()`.
- **Estado**: IMPLEMENTADO (integración en Collection + config perfil).

---

## 5. Distancias (SIMD)

### RF-12: Distance Metrics
- **Cosine** [−1, 1]: similitud coseno normalizada.
- **Dot**: producto punto directo.
- **Euclidean** [0, ∞): distancia euclidiana (negada internamente).
- **SIMD**: `wide` crate (f32x8 = SSE/AVX2 en x86, NEON en ARM).
- **Fallback**: graceful para elementos < 8. Sin `unsafe`.
- **score_i8**: distancia en aritmética entera para SQ.
- **Estado**: IMPLEMENTADO (~277 LOC).

---

## 6. Document Model

### RF-13: Document
- `id: String` — identificador único.
- `text: String` — contenido textual.
- `embedding: Vec<f32>` — vector de embedding (puede estar vacío).
- `metadata: HashMap<String, String>` — pares clave-valor arbitrarios.
- **Fluent Builder**: `Document::builder(id, text).embedding(vec).metadata(k, v).build()`.
- **Serde**: Serialize + Deserialize (JSONL-ready).
- **Estado**: IMPLEMENTADO (~205 LOC).

---

## 7. Filtrado de Metadatos

### RF-14: Filter API
- `Filter` = `Box<dyn Fn(&Document) -> bool>`.
- `metadata_eq(key, value)` — igualdad exacta.
- `metadata_contains(key, substr)` — substring match.
- `metadata_exists(key)` — clave presente.
- `all_of(filters)` — AND lógico.
- Closures inline: `\|doc\| doc.metadata_val("lang") == Some("en")`.
- **Comportamiento por backend**:
  - BruteForce: pre-filter (antes de distancia).
  - HNSW/IVF-PQ: post-filter con multiplicador k×3–5.
- **Estado**: IMPLEMENTADO (~122 LOC).

---

## 8. Smart Chunker

### RF-15: SmartChunker
- Auto-detecta `ChunkStrategy` por extensión:
  - `.rs`, `.py`, `.js`, `.ts`, `.go` → `Code`.
  - `.txt`, `.md`, `.jsonl`, `.json`, `.yaml`, `.toml`, `.sh` → `FixedWindow`.
  - Cualquier otra → `FixedWindow`.
  - Se puede asignar `Paragraph` explícitamente para chunking semántico.
- Cada chunk: `SmartChunk { text, structure: Option<String>, level: usize, start_line, end_line }`.
- **Estado**: IMPLEMENTADO (~682 LOC módulo principal).

### RF-16: CodeChunker (regex)
- **Descripción**: Divide código fuente por definiciones de nivel superior.
- **Patterns pre-compilados**: Rust (`fn`, `impl`, `struct`, `enum`, `trait`, `mod`),
  Python (`def`, `class`), JS/TS (`function`, `class`, `const`, `interface`), Go (`func`, `type`).
- **Dispatch automático**: detecta el lenguaje por contenido (keywords) si la extensión es ambigua.
- **Fallback**: Si ningún pattern matchea, subdivision por líneas.
- **Estado**: IMPLEMENTADO (~153 LOC).

### RF-17: ParagraphChunker (Semántico integrado)
- **Descripción**: Divide texto genérico por `\n\n` con overlap configurable.
- **Fix de seguridad**: start nunca retrocede + char boundaries garantizados (previene infinite loops y panics UTF-8).
- **Chunking semántico**: método `chunk_semantic()` que usa `Embedder` para dividir por similitud coseno entre oraciones adyacentes (threshold 0.35).
  - Fallback: sin embedder o si falla → chunking por párrafos.
- **Estado**: IMPLEMENTADO (~208 LOC).

### RF-18: FixedWindowChunker
- **Descripción**: Divide cualquier texto en ventanas de tamaño fijo con overlap.
- **Reemplaza**: las estrategias anteriores de Markdown, JSONL, y texto plano.
- **Seguridad UTF-8**: todos los cortes usan `.is_char_boundary()` — cero pánicos.
- **Subdivisión**: si un chunk excede `max_size`, se subdivide por líneas.
- **Estado**: IMPLEMENTADO (~120 LOC).

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
- **Implementaciones**: `dogma-vdb-embed-fastembed` (ONNX via fastembed.rs).
- **Estado**: IMPLEMENTADO (trait: ~28 LOC, fastembed: ~80 LOC).

---

## 10. Reranker Trait

### RF-23: Reranker
```rust
pub trait Reranker: Send + Sync {
    fn rerank(&self, query: &str, documents: &mut Vec<Document>) -> Result<()>;
}
```
- **NoRerank**: implementación default no-op (deja orden intacto).
- **Integración**: usado por `hybrid_search()` cuando el perfil lo habilita.
- **Implementación ONNX**: en `dogma-vdb-rerank` (Cross-Encoder via ort + tokenizers).
- **Estado**: IMPLEMENTADO (trait: ~69 LOC, ONNX runtime: ~177 LOC).

---

## 11. Watch Mode

### RF-24: File Watcher (feature = "watch")
- **Dependencias**: `notify` v8 + `crossbeam-channel`.
- **Eventos**: `WatchEvent::Modified(path)`, `WatchEvent::FileError(path, error)`.
- **start_watching()**: spawn hilo, retorna `Receiver<WatchEvent>`.
- **Config**: source_dirs, extensions, debounce_ms (default 500).
- **walkdir()**: escaneo recursivo con filtro de extensiones.
- **Estado**: IMPLEMENTADO (~316 LOC).

---

## 12. Configuración

### RF-25: Sistema de Config
- **Fuentes** (primera coincidencia gana):
  1. `$XDG_CONFIG_HOME/dogma-vdb/config.toml` (~/.config/dogma-vdb/config.toml)
  2. `./config.toml` en directorio de trabajo
  3. Variables de entorno `DOGMA_VDB_` (ej. `DOGMA_VDB_DEBUG=true`)
  4. Valores por defecto hardcodeados
- **Global**: `lazy_static CONFIG` (OnceCell/Lazy).
- **Secciones**:
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
- **PerformanceProfile**: `PrecisionLocal` (5x, rerank sí), `ProduccionHibrido` (3x, rerank sí), `VelocidadExtrema` (2x, rerank no).
- **QueryPipelineConfig**: `profile` + `top_k`.
- **Estado**: IMPLEMENTADO (~397 LOC).

---

## 13. Memoria

### RF-26: Memory Guard
- **Propósito**: evitar OOM en operaciones grandes (insert, build_index, chunking).
- **Fuente**: `/proc/meminfo` en Linux.
- **Niveles**: `Normal`, `Low` (free < 15%), `Critical` (free < 5%).
- **ensure_memory()**: aborta operación si `Critical`, advierte si `Low`.
- **Estado**: IMPLEMENTADO (~170 LOC).

---

## 14. Crate `dogma-vdb-cli` — Interfaz de Línea de Comandos

### RF-27: CLI
- **Dependencias**: `clap`, `dogma-vdb`, `serde_json`, `anyhow`.
- **Comandos**:
  - `query <path> <k> <query_vec...>` — búsqueda vectorial.
  - `ingest <path> <jsonl>` — insertar documentos desde JSONL.
  - `delete <path> <id>` — eliminar por ID.
  - `list <path>` — listar documentos.
  - `info <path>` — estadísticas de la colección.
- **Estado**: IMPLEMENTADO (~335 LOC).

---

## 15. Crate `dogma-vdb-mcp` — MCP Server

### RF-28: MCP Server
- **Dependencias**: `rmcp`, `tokio`, `serde`, `tracing`, `dogma-vdb`, `dogma-vdb-rerank`.
- **Transporte**: stdio (por ahora).
- **Herramientas**:
  - `vecdb_query` — buscar vectores.
  - `vecdb_ingest` — insertar documentos.
  - `vecdb_delete` — eliminar por ID.
  - `vecdb_list` — listar documentos.
  - `vecdb_info` — estadísticas.
- **Reranker**: integra `OnnxReranker` cuando `DOGMA_RERANK=1`.
- **Estado**: IMPLEMENTADO (~406 LOC server + ~109 LOC rerank adapter).

---

## 16. Crate `dogma-vdb-rerank` — Cross-Encoder Reranker

### RF-29: ONNX Reranker
- **Dependencias**: `ort` (ONNX Runtime), `tokenizers`, `rayon`, `ndarray`.
- **Modelo**: Cross-Encoder (MiniLM-L6-v2) descargado de HuggingFace.
- **API**: `compute_scores(query, texts) -> Vec<(usize, f32)>`.
- **Batch**: paralelismo con rayon.
- **Estado**: IMPLEMENTADO (~177 LOC).

---

## 17. Crate `dogma-vdb-benchmarks` — Grid Benchmark

### RF-30: Benchmark Grid
- **Propósito**: encontrar sweet-spots de configuración automáticamente.
- **Grid**: variaciones de tamaño (10K–100K), dimensión (128–768), HNSW (M, ef), IVF-PQ (nlist, M_sub).
- **Métricas**: Recall@1/10/100, QPS, latencia (mean/p50/p95/p99), RAM, build time.
- **Score**: `QPS / RAM_MB` para configs con recall ≥ 90%.
- **Reportes**: `BENCHMARK.md` + `TUNING_REPORT.md` con top-3 sweet spots.
- **Estado**: IMPLEMENTADO (~1,171 LOC).

---

## 18. IVF-PQ Persistencia

### RF-31: Persistencia Atómica
- **Protocolo**: escribir a `.tmp`, `sync_all()`, `rename()` → el archivo final solo aparece cuando la escritura está completa.
- **Formato**: metadata JSON + embeddings JSONL + mmap.
- **Soft-delete**: marcar documentos como eliminados en lugar de reescribir todo.
- **Compactación**: reescribe el archivo sin documentos eliminados, libera espacio.
- **Estado**: IMPLEMENTADO (~724 LOC).

---

## 19. Modelo de Seguridad

dogma-vdb es una herramienta **CLI local / librería embebible**:

| Componente | Exposición | Riesgo |
|-----------|-----------|--------|
| Core library | Ninguna (solo código usuario) | 0 |
| CLI | Local, usuario invoca explícitamente | 0 |
| MCP stdio | Procesos locales que el usuario autoriza | Bajo |
| Watcher | Directorios que el usuario configura | Bajo |
| fastembed | Descarga modelos de HuggingFace | Bajo |

**Principios**:
- Sin `unsafe` en código de producción (aislado a `as_embeddings()` en VectorStorage).
- MmapBackedStorage incluye documentación defensiva contra SIGBUS.
- Sin ejecución de comandos del sistema.
- Sin secretos hardcodeados.
- Sin red en el core (MCP server es binario separado y stdio por defecto).
- Sin dependencias externas para algoritmos core (HNSW, IVF-PQ, SQ — Rust puro).

---

## 20. Feature Flags

| Flag | Dependencias | Propósito |
|------|-------------|-----------|
| `watch` | notify, crossbeam-channel | File watcher |
| *(default)* | serde, serde_json, thiserror, rayon, wide, bytemuck, memmap2, once_cell, toml, log, regex-lite | Core mínimo |

---

## 21. Tests y Cobertura

- **Tests unitarios**: en cada módulo (`#[cfg(test)]`).
- **Tests de integración**: `tests/integration.rs`.
- **Tests documentación**: doc-tests en toda la API pública.
- **Total**: 192 tests (192 pasan, clippy clean).
- **Benchmarks**: benchmark grid exhaustivo + benchmark con embeddings reales ONNX.

---

## 22. Hoja de Ruta — Funcionalidades Futuras

Ver `ARCH-SPEC.md` sección 12 para items post-beta detallados. Prioridades:

| Prioridad | Feature | Impacto |
|-----------|---------|:-------:|
| Alta | Parallel IVF-PQ build (rayon) | Alto |
| Alta | Python bindings (PyO3) | Alto |
| Alta | LangChain VectorStore nativo | Alto |
| Media | SIMD para PQ lookup (wide) | Medio |
| Media | Fuzz testing | Medio |
| Media | CRUD update eficiente (in-place) | Medio |
| Media | Multi-index search | Medio |
| Media | Embedding models adicionales | Medio |
| Media | File locking (fs2) anti-SIGBUS | Medio |
| Baja | Formato Parquet export | Bajo |
| Baja | CLI modo REPL | Bajo |
| Baja | Benchmarks en CI | Bajo |
