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
- **Estado**: PARCIAL (implementado sin flat_embeddings).

### RF-03: AnnoyIndex

- **Descripcion**: Random Projection Forest (Spotify Annoy).
  Divide el espacio recursivamente con hiperplanos aleatorios.
  Construye `n_trees` arboles; busca atravesando todos y recolectando
  candidatos.
- **Entrada**: Documentos con embedding + AnnoyConfig { n_trees, search_k }.
- **Salida**: Top-k aproximado.
- **Comportamiento**:
  1. **Build**: batch desde slice de Documentos. No incremental.
  2. Por cada arbol: selecciona 2 puntos aleatorios para definir un
     hiperplano, divide el conjunto, recursivo hasta < `k` puntos por nodo.
  3. Busqueda: atraviesa los `n_trees` arboles con una priority queue
     compartida (prioridad = distancia al hiperplano). Recolecta candidatos,
     scores exactos al final.
  4. `insert()` / `delete()`: rebuild completo (panic o recrear).
- **Condiciones**: Ideal para datasets staticos. Build rapido, sin
  entrenamiento, sin parametros por dimension.
- **Estado**: NO IMPLEMENTADO.

### RF-04: Scalar Quantization (SQ)

- **Descripcion**: Capa de optimización ortogonal que comprime
  embeddings f32 a i8 (1 byte por valor) para reducir memoria ~4x
  y acelerar calculo de distancias ~2x.
- **Entrada**: Se activa via flag `sq: bool` en config.
- **Comportamiento**:
  1. En **insercion**: calcular `scale` y `bias` globales por dimension
     (o mejor, por todo el dataset con min/max por dimension).
     Cuantizar: `i8 = clamp((f32 - bias) / scale, -128, 127)`.
  2. En **busqueda**: cuantizar el query, calcular distancias con
     aritmetica entera (dot_i8 = suma de productos i8, mucho mas rapida).
     Opcional: rescore con f32 los top-`k*2` para recuperar precision.
  3. El Document almacena `embedding: Vec<f32>` siempre (para persistencia
     y debug). El backend almacena `embedding_i8: Vec<Vec<i8>>` en memoria
     solo cuando SQ esta activo.
- **Estado**: NO IMPLEMENTADO.
- **Combinacion**: SQ funciona con cualquier backend (BruteForce, HNSW,
  Annoy). Es ortogonal — no cambia el algoritmo, solo el storage y la
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

## 4. Filtrado de Metadatos

### RF-05: Filter API

- `metadata_eq(key, value)` → igualdad exacta de string.
- `metadata_contains(key, substr)` → substring match.
- `metadata_exists(key)` → clave presente.
- `all_of(filters)` → AND logico.
- Closures inline: `|doc| doc.metadata_val("lang") == Some("en")`.

**Comportamiento por backend**:
- BruteForce: pre-filter (filtra antes de calcular distancia).
- HNSW/Annoy: post-filter con multiplicador k*5.

**Limite**: No hay filtros numericos (range), ni OR, ni full-text search.

---

## 5. API de Alto Nivel

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

El tipo de indice se elige desde config (`index_type: bruteforce|hnsw|annoy`).

---

## 6. Configuracion

### config.toml (CollectionConfig)

```toml
[collection]
index_type = "hnsw"           # bruteforce | hnsw | annoy
index_metric = "cosine"       # cosine | dot | euclidean

# HNSW
hnsw_m = 16
hnsw_ef_construction = 200
hnsw_ef_search = 50
hnsw_flat_embeddings = false

# Annoy
annoy_n_trees = 10
annoy_search_k = -1           # -1 = auto (n_trees * k)

# SQ (ortogonal, aplica a cualquier backend)
sq = false
```

---

## 7. Criterios de Aceptacion

| Escenario | Backend | Given | When | Then |
|-----------|---------|-------|------|------|
| Exactitud | BruteForce | 100 docs 128-dim | search k=5 | resultados exactos (100% recall) |
| Aprox | HNSW | 5000 docs 128-dim | search k=10 | recall >= 90% vs BF |
| Aprox | Annoy | 5000 docs 128-dim | search k=10 | recall >= 85% vs BF |
| Flat | HNSW | flat_embeddings=true | insert 5000 | misma recall, ~2x velocidad |
| SQ | HNSW+SQ | sq=true | search | ~2x velocidad, recall >= 95% |
| SQ | BF+SQ | sq=true | search | ~2x velocidad, misma recall |
| Filtro | BruteForce | docs con metadata | search_filtered | solo docs que matchean |
| Build | Annoy | 5000 docs | build | < 500ms |
| Memoria | HNSW+SQ | 100K docs 768-dim | insert | < 150 MB RAM |
