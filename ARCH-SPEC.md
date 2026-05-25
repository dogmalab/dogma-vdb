# Architecture — dogma-vdb

## 1. Principios Arquitectonicos

1. **1 archivo Rust = 1 componente**. Cada componente tiene responsabilidad unica.
2. **Index trait como frontera**. Backends son intercambiables via Box<dyn Index>.
3. **SQ es ortogonal**. No cambia la API, solo el storage/distance.
4. **VectorStorage desacopla vectores de indices**. Los backends reciben
   embeddings contiguos inyectados, sin saber su origen (RAM o mmap).
5. **Sin dependencias externas para algoritmos core**. HNSW, IVF-PQ, SQ son Rust puro.
6. **Config-driven**. Todo parametro via config.toml, no hardcode.

---

## 2. Diagrama de Arquitectura

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
        Arc<dyn VectorStorage> (compartido)
          /               \
   MemoryBacked      MmapBacked
   (Vec<f32>)        (memmap2)

  BinStorage (persistencia JSONL + binario v2)
```

**SQ**: cuando `sq=true`, cada backend usa `score_i8()` con scale/bias
por documento. El grafo/topologia se construye con f32 original, la
busqueda puede usar i8 con rescore opcional.

---

## 3. Estructura de Archivos

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

### 4.1. Algoritmo de Cuantizacion (corregido)

Para cada embedding `v` de dimension `d`:

1. Calcular `min_d` y `max_d` **por documento** (no global).
2. `midpoint = (max_d + min_d) / 2.0` (bias centrado en el rango).
3. `scale = (max_d - min_d) / 255.0`.
4. `v_i8[i] = clamp(round((v[i] - bias) / scale), -128, 127)`.

El midpoint como bias reemplaza al `min` anterior, garantizando que
valores cercanos a 0 se mapeen correctamente al rango i8 simetrico.

### 4.2. Distancia en i8

```
dot_i8(a_i8, b_i8) = sum_i(a_i8[i] * b_i8[i])  // escala lineal
```

Para busqueda ANN donde solo importa el ranking, los factores constantes
de escala no afectan el orden.

### 4.3. Rescoring (opcional)

Para recuperar precision, despues de obtener top-k con i8, rescorear
los k*2 con f32 original. Esto anade ~20% overhead pero mejora recall
de 40% → 90% en HNSW+SQ.

### 4.4. Integracion por Backend

**BruteForce + SQ**: iterar embedding_i8, compute dot_i8, ordenar.
Si rescore=true, tomar top-k*2, rescore con f32.

**HNSW + SQ**: El grafo se construye con distancia f32 original
(garantiza topologia correcta). `search_layer()` usa `score_i8()`.
Con rescore: top-k*2 candidatos → rescore f32. Recall: 90%.

**IVF-PQ + SQ**: Los centroides K-Means se calculan en f32. La
asignacion a clusters y la distancia asimetrica se hace en f32.
SQ es ortogonal adicional sobre los codigos PQ.

### 4.5. Donde vive SQ

En `src/index/sq.rs`:

```rust
/// Cuantizar un embedding f32 a i8 con escala por documento.
pub fn quantize(embedding: &[f32], scale: f32, bias: f32) -> Vec<i8>;

/// Cuantizar el query para busqueda con i8.
pub fn quantize_query(query: &[f32], scale: f32, bias: f32) -> Vec<i8>;

/// Producto punto en i8 (SIMD-friendly).
pub fn dot_i8(a: &[i8], b: &[i8]) -> i32;

/// Score i8 convertido a f32.
pub fn score_i8(query_i8: &[i8], doc_i8: &[i8], scale: f32, bias: f32) -> f32;

/// Recalcular score exacto con f32 para rescoring.
pub fn rescore(query: &[f32], docs: &[&Document], metric: Metric) -> Vec<ScoredDocument>;
```

No es un index ni un wrapper — es un modulo de utilidades. Cada backend
lo usa cuando `sq=true`.

---

## 5. IVF-PQ — Inverted File + Product Quantization

### 5.1. Estructura de Datos

```rust
pub struct IvfPqIndex {
    documents: Vec<Document>,       // metadata, text, embedding (f32)
    centroids: Vec<Vec<f32>>,       // n_list centroides K-Means (f32)
    pq_codebook: Vec<Vec<Vec<f32>>>, // m_subspaces sub-codebooks (256 x (d/m) c/u)
    codes: Vec<Vec<u8>>,            // codigos PQ por documento (m_subspaces bytes c/u)
    assignments: Vec<usize>,        // asignacion cluster por documento
    config: IvfPqConfig,
    storage: Arc<dyn VectorStorage>, // embeddings contiguos compartidos
}

pub struct IvfPqConfig {
    pub n_list: usize,               // numero de centroides (default: 100)
    pub m_subspaces: usize,          // numero de subvectores (default: 32, multiplo de 8)
    pub n_probe: usize,              // clusters a explorar (default: 5)
    pub metric: Metric,
    pub rerank_enabled: bool,        // auto-tuning: reduce n_probe a la mitad
}
```

### 5.2. Algoritmo de Build (batch)

```
fn build(docs: &[Document]) -> IvfPqIndex:
    1. K-Means sobre todos los embeddings (max 20 iteraciones):
       a. Inicializar nlist centroides con k-means++
       b. Asignar cada embedding al centroide mas cercano
       c. Recalcular centroides como promedio de sus puntos
       d. Repetir hasta convergencia o max iter

    2. Product Quantization:
       a. Para cada dimension del embedding, dividir en m subvectores
          de tamano d/m
       b. Para cada subespacio, ejecutar K-Means con 256 centroides
          (codebook del subvector)
       c. Para cada documento, codificar cada subvector como el indice
          u8 del centroide mas cercano en ese subespacio

    3. Almacenar: centroides (f32), pq_codebook (f32), codes (u8),
       assignments (usize)
```

### 5.3. Busqueda

```
fn search(query, k) -> Vec<ScoredDocument>:
    1. Calcular distancia del query a todos los nlist centroides.
       Seleccionar los nprobe mas cercanos.

    2. Para cada uno de los nprobe clusters:
       a. Precomputar tabla de distancias (LUT) entre query y los
          256 centroides de cada subespacio PQ.
       b. Escanear los codigos u8 de los documentos en ese cluster:
          distancia_aprox = sum_m(LUT[m][code[m]])
       c. Mantener top-k global con min-heap.

    3. Ordenar candidatos por distancia, devolver top-k.
```

### 5.4. Complejidad

| Operacion | Complejidad |
|-----------|-------------|
| Build (K-Means) | O(n_list · n · d · iter) |
| Build (PQ) | O(256 · m_subspaces · (d/m_subspaces) · n) = O(256 · d · n) |
| Search | O(n_list · d + effective_probe · (n/n_list) · m_subspaces) |

donde `effective_probe = n_probe` si `rerank_enabled=false`, o
`(n_probe / 2).max(2)` si `rerank_enabled=true`.

### 5.5. Memoria

Para 5K docs 128-dim, n_list=100, m_subspaces=32:
- Centroides: 100 × 128 × 4B = 51 KB
- PQ codebook: 32 × 256 × (128/32) × 4B = 128 KB
- Codes: 5K × 32B = 160 KB
- Asignaciones: 5K × 8B = 40 KB
- **Total: ~380 KB** (~8× menos que HNSW/BF)

---

## 6. VectorStorage Trait

### 6.1. Definicion

```rust
pub trait VectorStorage: Send + Sync {
    fn as_bytes(&self) -> &[u8];
    fn as_embeddings(&self) -> &[f32];
    fn flush(&self) -> Result<()>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
}
```

### 6.2. Implementaciones

**MemoryBackedStorage**:
```rust
pub struct MemoryBackedStorage {
    data: Vec<u8>,
}
```
- Vec<u8> contiguo en RAM (embeddings f32 reinterpretados vía `as_embeddings()`).
- `as_embeddings()` devuelve `&[f32]` via `unsafe { from_raw_parts() }` (aislado, auditado).
- `from_f32_slice()`: construye desde `&[f32]` (copia).
- Para tests, pipelines volátiles, warm cache.

**MmapBackedStorage**:
```rust
pub struct MmapBackedStorage {
    _file: std::fs::File,    // mantiene fd vivo
    mmap: memmap2::Mmap,     // mapping de todo el archivo
}
```
- Carga ~0ms: el OS pagea bajo demanda.
- Formato binario v2 con padding 32-byte.
- `as_embeddings()` reinterpreta los bytes mapeados como `&[f32]`.
- `advise(Advice::Random)` para eliminar readahead page faults.
- ⚠️ SIGBUS: si un proceso externo trunca el archivo, el kernel mata
  el proceso. Documentado en el código.

### 6.3. Integracion en Collection

```rust
pub struct Collection {
    name: String,
    documents: Vec<Document>,
    index: Box<dyn Index>,
    storage: BinStorage,
    emb_storage: Arc<dyn VectorStorage>,
}

fn open(path) -> Result<Self> {
    let storage = BinStorage::load(path)?;       // leer metadata + docs
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

## 7. Flat Embeddings en HNSW

### 7.1. Storage

```rust
pub struct HnswIndex {
    documents: Vec<Document>,       // metadata, text
    embeddings_flat: Vec<f32>,      // solo si flat_embeddings=true
    dim: usize,                     // solo si flat_embeddings=true
    storage: Arc<dyn VectorStorage>, // embeddings contiguos (siempre presente)
    // ... resto igual
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

### 7.3. Insercion

Cuando `flat_embeddings=true`, insert_one() hace:
1. Extiende `embeddings_flat` con el nuevo embedding.
2. El embedding tambien vive en `storage` (VectorStorage compartido).

Decision de diseno: flat_embeddings es solo para busqueda en memoria.
El formato binario siempre guarda embedding f32 completo (portabilidad).

### 7.4. Delete con Flat

Cuando se elimina un documento con flat, hay que reconstruir
`embeddings_flat` desde los documents restantes (coste O(n·d) una vez,
equivalente a lo que ya hace el rebuild del grafo en delete).

---

## 8. Estrategia de Factory

En `index/mod.rs`:

```rust
fn build_index(cfg: &CollectionConfig, storage: Arc<dyn VectorStorage>) -> Box<dyn Index> {
    let mut index: Box<dyn Index> = match cfg.index_type {
        "hnsw" => Box::new(HnswIndex::new(HnswConfig { ... }, storage)),
        "ivf_pq" => Box::new(IvfPqIndex::new(IvfPqConfig { ... }, storage)),
        _ => Box::new(BruteForceIndex::new(metric, storage)),
    };

    // SQ no es un wrapper — cada backend recibe el flag sq
    // y actua en consecuencia en sus metodos search/insert.
}
```

---

## 9. Dependencias

### Actuales
- serde, serde_json, thiserror — core
- rayon — parallel BruteForce
- toml, once_cell, log — config
- memmap2 — MmapBackedStorage
- bytemuck — safe f32↔[u8] reinterpret
- wide — SIMD-accelerated distance functions
- regex-lite — smart chunker patterns
- notify, crossbeam-channel — watcher (feature)

### Sin dependencias externas para algoritmos core
- HNSW: SplitMix64 (ya implementado en core)
- IVF-PQ: K-Means y PQ son Rust puro (stdlib)
- SQ: Rust puro (stdlib)

### Opcionales
- `rand` (dev-dependency)

---

## 10. Metricas Objetivo

| Backend | 5K docs 128-dim | 50K docs 768-dim | 100K docs 384-dim |
|---------|:---------------:|:----------------:|:-----------------:|
| BruteForce | 1,460 us | ~200 ms | ~400 ms |
| HNSW | 77 us | ~500 us | ~1 ms |
| IVF-PQ | 128 us | ~2 ms | ~4 ms |
| HNSW+SQ+Rescore | 73 us | ~350 us | ~700 us |

RAM estimada para 100K docs 384-dim:
- f32 embeddings: 100K × 384 × 4 = ~153 MB
- HNSW graphs: ~200 MB adicional (conexiones)
- IVF-PQ: ~1.5 MB (centroides + codebook + codes)
- SQ i8: ~38 MB (solo i8, sin graphs)

---

## 11. Prioridad de Implementacion Completada

1. ~~**HNSW + flat_embeddings**~~ (completado)
2. ~~**SQ module**~~ (completado)
3. ~~**SQ integration** en BruteForce y HNSW~~ (completado, recall 90%)
4. ~~**Annoy**~~ (reemplazado por IVF-PQ)
5. ~~**IVF-PQ backend**~~ (completado, ~8× ahorro RAM)
6. ~~**VectorStorage trait**~~ (completado, mmap ~0ms load)
7. ~~**Benchmarks**~~ (actualizados)

---

## 12. Enriquecimiento Futuro (Post-Beta)

### 12.1. Seguridad

| Item | Prioridad | Descripcion |
|------|:---------:|-------------|
| MCP HTTP auth | Media | Seguridad para futura implementación HTTP del MCP server en crate separado |
| File locking (fs2) | Media | Bloqueo OS-level para evitar SIGBUS por escritura concurrente |
| Watcher path sandbox | Baja | Validar que `source_dirs` este dentro de un directorio base configurado |
| Model checksum verification | Baja | Verificar checksum SHA256 de modelos ONNX descargados |
| Audit CI hardening | Baja | Configurar `cargo audit` para fallar solo en vulnerabilidades reales |

### 12.2. Rendimiento y Escalabilidad

| Item | Impacto | Descripcion |
|------|:-------:|-------------|
| **Parallel IVF-PQ build** | Alto | Paralelizar K-Means y PQ build con rayon. ~1 sesion |
| **SIMD para PQ lookup** | Medio | Acelerar distancia asimetrica con SIMD (wide crate) |
| **HNSW parallel insert** | Medio | Batch insert con lock-free grafo. ~2 sesiones |
| **Multi-index search** | Bajo | Buscar en HNSW + IVF-PQ y fusionar resultados |

### 12.3. Formatos y Portabilidad

| Item | Impacto | Descripcion |
|------|:-------:|-------------|
| **Formato Parquet** | Medio | Export a Apache Parquet para interoperabilidad con data science |
| **Import desde ChromaDB/LanceDB** | Medio | Script de migracion desde otros formatos de vectores |
| **Compresion zstd en binario** | Bajo | Compresion opcional zstd para el formato binario v3 |

### 12.4. Integraciones

| Item | Impacto | Descripcion |
|------|:-------:|-------------|
| **Python bindings (PyO3)** | Alto | `pip install dogma-vdb` con API Python completa. ~3-4 sesiones |
| **LangChain VectorStore nativo** | Alto | Provider Python que implementa VectorStore de LangChain usando MCP subprocess |
| **Embedding models adicionales** | Medio | Soportar modelos ONNX distintos a MiniLM-L6-v2 (BGE, GTE, etc.) |
| **Llamarada / mistral.rs** | Medio | Embedding via llama.cpp para modelos locales |

### 12.5. Operaciones

| Item | Impacto | Descripcion |
|------|:-------:|-------------|
| **CRUD update eficiente** | Medio | Actualmente delete+insert reescribe todo. Hacer update in-place |
| **Snapshot / versionado** | Bajo | Mantener N versiones anteriores del .vdb para rollback |
| **CLI en REPL** | Bajo | Modo interactivo para explorar colecciones desde terminal |
| **Estadisticas de coleccion** | Bajo | Reportar distribucion de vectores, outliers, clustering |

### 12.6. Testing y CI

| Item | Impacto | Descripcion |
|------|:-------:|-------------|
| **Fuzz testing** | Medio | Fuzzing de entrada de datos (embeddings malformados, metadata corrupta) |
| **Benchmarks en CI** | Bajo | Ejecutar bench.rs en CI y comparar con commit anterior |
| **Test de integracion MCP** | Bajo | Test E2E que inicia MCP server, conecta, hace queries |
| **Proptest para indices** | Bajo | Test propiedad: search(k) siempre devuelve <= k resultados |
