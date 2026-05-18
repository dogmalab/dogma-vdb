# dogma-vdb — Functional Specification

## 1. Resumen

Base de datos vectorial portable en Rust. Formato JSONL, zero async en
core, 0-5 dependencias directas, config-driven.

**Problema**: ChromaDB es pesado (300 MB pip), LanceDB es complejo
(50K LOC, 200+ deps). Se necesita algo tiny, portable, debugeable
con `cat`/`grep`/`sed`, que corra en cualquier lado con un solo binario.

**Usuario objetivo**: Desarrolladores que necesitan ANN local para RAG
o datasets < 100K vectores, sin querer levantar servidores ni instalar
Python.

---

## 2. Index Backends

Cada backend implementa el trait `Index` definido en `src/index/mod.rs`:

```rust
pub trait Index: Send + Sync {
    fn insert(&mut self, docs: &[Document]);
    fn delete(&mut self, ids: &[&str]) -> usize;
    fn search(&self, query: &[f32], k: usize) -> Vec<ScoredDocument>;
    fn search_filtered(&self, query: &[f32], k: usize, filter: &(dyn Fn(&Document) -> bool + Sync)) -> Vec<ScoredDocument>;
    fn documents(&self) -> &[Document];
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool { self.len() == 0 }
}
```

### RF-01: BruteForceIndex

- **Descripcion**: Busqueda exacta O(n·d) por scan lineal.
- **Entrada**: Documentos con embedding Vec<f32>.
- **Salida**: Top-k por similitud descendente.
- **Comportamiento**:
  1. Iterar todos los documentos con embedding no-vacio.
  2. Calcular distancia segun metric configurada (cosine/dot/euclidean).
  3. Ordenar por score descendente, truncar a k.
- **Condiciones**: Recomendado para < 10K documentos.
- **Estado**: IMPLEMENTADO.

### RF-02: HnswIndex

- **Descripcion**: Busqueda aproximada O(log n) via grafo jerarquico
  (Malkov & Yashunin 2016).
- **Entrada**: Documentos con embedding + HnswConfig { m, ef_construction, ef_search, metric, flat_embeddings }.
- **Salida**: Top-k aproximado.
- **Comportamiento**:
  1. Cada nodo obtiene un nivel aleatorio deterministico (SplitMix64).
  2. Insercion: busca capa por capa, conecta a vecinos mas cercanos,
     aplica heuristico de diversidad en capa 0.
  3. Busqueda: desciende por capas superiores con ef=1, explora capa 0
     con ef=ef_search.
- **flat_embeddings**: Cuando true, almacena todos los embeddings en
  un solo Vec<f32> contiguo en vez de Vec<Vec<f32>> (reduce cache misses,
  TLB pressure, alloc overhead).
- **Estado**: IMPLEMENTADO.

### RF-03: IvfPqIndex

- **Descripcion**: Busqueda aproximada via archivo invertido (IVF) +
  cuantizacion de producto (PQ). Particiona el espacio con K-Means,
  comprime subvectores a u8 para minimizar memoria.
- **Entrada**: Documentos con embedding + IvfPqConfig { n_list, n_probe, m_subspaces, metric, rerank_enabled }.
- **Salida**: Top-k aproximado.
- **Comportamiento**:
  1. **Build** (batch): ejecuta K-Means sobre todos los embeddings
     para construir `n_list` centroides.
  2. Asigna cada embedding a su centroide mas cercano y lo particiona
     en `m_subspaces` subvectores. Cada subvector se cuantiza a u8.
  3. Almacena las tablas PQ (centroides de subvectores) y los codigos
     cuantizados (u8) por documento.
  4. **Busqueda**: calcula distancia del query a los `n_probe` centroides
     mas cercanos. Para cada cluster, construye lookup tables (LUTs)
     de distancias asimetricas y escanea los codigos u8.
  5. **Auto-tuning con rerank**: si `rerank_enabled=true`, el `n_probe`
     efectivo se reduce a la mitad (mínimo 2) para priorizar velocidad.
     El recall perdido se recupera en el paso de Cross-Encoder posterior.
  6. `insert()` / `delete()`: rebuild completo del indice.
- **Validacion**: `m_subspaces` debe ser multiplo de 8 (alineación SIMD
  para AVX2/NEON). `IvfPqConfig::validate()` retorna `Error::InvalidConfig`
  si no se cumple.
- **Condiciones**: Ideal para datasets estaticos o semi-estaticos donde
  se prioriza el ahorro de memoria (~8× menos RAM que HNSW/BF).
- **Estado**: IMPLEMENTADO.

### RF-04: Scalar Quantization (SQ)

- **Descripcion**: Capa de optimización ortogonal que comprime
  embeddings f32 a i8 (1 byte por valor) para reducir memoria ~4x
  y acelerar calculo de distancias ~2x.
- **Entrada**: Se activa via flag `sq: bool` en config.
- **Comportamiento**:
  1. En **insercion**: calcular `scale` y `bias` usando midpoint global
     (`midpoint = (max + min) / 2`, `scale = (max - min) / 255.0`).
     Cuantizar: `i8 = clamp((f32 - bias) / scale, -128, 127)`.
  2. En **busqueda**: cuantizar el query, calcular distancias con
     aritmetica entera (dot_i8). Rescore opcional con f32 los top-k*2.
  3. El Document almacena `embedding: Vec<f32>` siempre (para persistencia
     y debug). El backend usa `embedding_i8: Vec<Vec<i8>>` en memoria
     solo cuando SQ esta activo.
- **Estado**: IMPLEMENTADO.
- **Combinacion**: SQ funciona con cualquier backend (BruteForce, HNSW,
  IVF-PQ). Es ortogonal — no cambia el algoritmo, solo el storage y la
  funcion de distancia.

---

## 3. Metrica de Distancia

Soportadas por todos los backends:

| Metrica | Rango | Mayor = mejor? | Notas |
|---------|-------|:--------------:|-------|
| Cosine | [-1, 1] | si | Dot producto normalizado por magnitud |
| Dot | (-inf, inf) | si | Producto punto directo |
| Euclidean | [0, inf) | si | Negado internamente para mantener consistencia |

---

## 4. VectorStorage Trait

### RF-05: VectorStorage

- **Descripcion**: Abstraccion que desacopla el almacenamiento de vectores
  del ciclo de vida de los indices. Permite inyectar embeddings contiguos
  desde distintas fuentes (RAM, mmap) sin que los backends lo sepan.

```rust
pub trait VectorStorage: Send + Sync {
    fn len(&self) -> usize;
    fn dim(&self) -> usize;
    fn get(&self, idx: usize) -> &[f32];
    fn as_slice(&self) -> Option<&[f32]>;
}
```

- **Implementaciones**:
  - `MemoryBackedStorage`: Vec<f32> contiguo en RAM. Para tests y pipelines volatiles.
  - `MmapBackedStorage`: Archivo mapeado a memoria via `memmap2`. Carga ~0ms.
    Incluye documentacion defensiva contra SIGBUS.

---

## 5. Filtrado de Metadatos

### RF-06: Filter API

- `metadata_eq(key, value)` → igualdad exacta de string.
- `metadata_contains(key, substr)` → substring match.
- `metadata_exists(key)` → clave presente.
- `all_of(filters)` → AND logico.
- Closures inline: `|doc| doc.metadata_val("lang") == Some("en")`.

**Comportamiento por backend**:
- BruteForce: pre-filter (filtra antes de calcular distancia).
- HNSW/IVF-PQ: post-filter con multiplicador k*5.

**Limite**: No hay filtros numericos (range), ni OR, ni full-text search.

---

## 6. API de Alto Nivel

### Collection (implementado)

```rust
Collection::open(path) -> Result<Self>                     // config-driven
Collection::open_with(path, index_type, metric) -> Result<Self>
Collection::insert(doc) -> Result<()>
Collection::insert_batch(docs) -> Result<()>
Collection::delete(ids) -> Result<usize>
Collection::update(doc) -> Result<()>
Collection::search(query, k) -> Vec<ScoredDocument>
Collection::search_filtered(query, k, filter) -> Vec<ScoredDocument>
Collection::documents() -> Iterator<Item = &Document>
```

El tipo de indice se elige desde config (`index_type: bruteforce|hnsw|ivf_pq`).

---

## 7. Configuracion

### config.toml (CollectionConfig)

```toml
[collection]
index_type = "hnsw"           # bruteforce | hnsw | ivf_pq
index_metric = "cosine"       # cosine | dot | euclidean

# HNSW
hnsw_m = 16
hnsw_ef_construction = 200
hnsw_ef_search = 50
hnsw_flat_embeddings = false

# IVF-PQ
ivf_pq_n_clusters = 100          # n_list — centroides K-Means
ivf_pq_n_subvectors = 32         # m_subspaces — subvectores PQ (multiplo de 8)
ivf_pq_n_probe = 5               # clusters a explorar en busqueda

# SQ (ortogonal, aplica a cualquier backend)
sq = false
sq_rescore = false
```

---

## 8. Almacenamiento Binario (v2)

### RF-07: BinStorage v2

- **Formato**: Header JSON/TOML + padding 32-byte + vectores f32 contiguos.
- **Padding**: Bytes de relleno dinamico post-header para alinear la seccion
  de vectores a 32 bytes (AVX2-ready).
- **Carga**: Via MemoryBackedStorage (RAM) o MmapBackedStorage (~0ms).
- **Migracion**: Auto-deteccion de formato v1 → v2 en apertura.
- **Escritura**: Append-only O(1) para nuevos vectores.

---

## 9. Modelo de Seguridad

dogma-vdb es una herramienta **CLI local / libreria embebible**:

| Componente | Exposicion | Riesgo |
|-----------|-----------|--------|
| Core library | Ninguna (solo codigo usuario) | 0 |
| CLI | Local, usuario invoca explicitamente | 0 |
| MCP stdio | Procesos locales que el usuario autoriza | Bajo |
| MCP HTTP | **No implementado** — skeleton con `todo!()` | N/A |
| Watcher | Directorios que el usuario configura | Bajo |
| fastembed | Descarga modelos de HuggingFace | Bajo (sin verificar checksum) |

**Principios**:
- No `unsafe` en codigo de produccion (aislado a conversion de bytes en VectorStorage)
- MmapBackedStorage incluye documentacion defensiva contra SIGBUS
- Sin ejecucion de comandos del sistema
- Sin secretos hardcodeados
- Sin red en el core (el MCP server es opcional y stdio por defecto)

Si en el futuro se implementa `serve_http`, se anadiran:
- Path whitelist para operaciones de filesystem
- Autenticacion basica o bearer token
- Rate limiting


## Scenarios de Testing

| Escenario | Backend | Given | When | Then |
|-----------|---------|-------|------|------|
| Exactitud | BruteForce | 100 docs 128-dim | search k=5 | resultados exactos (100% recall) |
| Aprox | HNSW | 5000 docs 128-dim | search k=10 | recall >= 90% vs BF |
| Aprox | IVF-PQ | 5000 docs 128-dim | search k=10 | recall >= 85% vs BF |
| Flat | HNSW | flat_embeddings=true | insert 5000 | misma recall, ~2x velocidad |
| SQ | HNSW+SQ | sq=true, rescore=true | search | ~2x velocidad, recall >= 90% |
| SQ | BF+SQ | sq=true | search | ~2x velocidad, misma recall |
| SQ | IVF-PQ+SQ | sq=true | search | memoria ~32× vs BF |
| Filtro | BruteForce | docs con metadata | search_filtered | solo docs que matchean |
| Build | IVF-PQ | 5000 docs | build | < 50ms |
| Memoria | HNSW+SQ | 100K docs 768-dim | insert | < 150 MB RAM |
| Carga fria | MmapBackedStorage | 5K docs 384-dim | open | < 1ms |
| Validacion SIMD | IVF-PQ | m_subspaces=13 | validate() | Error::InvalidConfig |
| Auto-tuning | IVF-PQ | rerank_enabled=true, n_probe=10 | effective_probe() | 5 |
| Auto-tuning (min) | IVF-PQ | rerank_enabled=true, n_probe=3 | effective_probe() | 2 |
