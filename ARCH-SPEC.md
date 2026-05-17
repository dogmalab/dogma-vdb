# Architecture — dogma-vdb

## 1. Principios Arquitectonicos

1. **1 archivo Rust = 1 componente**. Cada componente tiene responsabilidad unica.
2. **Index trait como frontera**. Backends son intercambiables via Box<dyn Index>.
3. **SQ es ortogonal**. No cambia la API, solo el storage/distance.
4. **Sin dependencias externas para algoritmos core**. HNSW, Annoy, SQ son Rust puro.
5. **Config-driven**. Todo parametro via config.toml, no hardcode.

---

## 2. Diagrama de Arquitectura

```
                         Collection
                             |
                       Box<dyn Index>
                        /    |    \
                 /          |          \
         BruteForce      HnswIndex    AnnoyIndex
              |               |            |
              +-- SQ? --------+-- SQ? -----+-- SQ?
              |               |            |
         Vec<Document>    Vec<Document>  Vec<Document>
         metric: Metric   graphs[][]    trees: Vec<Tree>
                          node_layers[]
                          flat: Vec<f32>
```

**SQ**: cuando `sq=true`, cada backend almacena `embedding_i8: Vec<Vec<i8>>`
adicional y usa `score_i8()` para distancias. El flag es parte de la config
de cada backend (o global).

---

## 3. Estructura de Archivos

```
src/
  lib.rs                  # Mod declarations + prelude
  doc.rs                  # Document struct + builder
  distance.rs             # Metric, score(), dot(), cosine(), euclidean(), score_i8()
  error.rs                # Error types
  storage.rs              # JSONL read/write (load/store/append)
  collection.rs           # Collection API (open, insert, search, etc.)
  filter.rs               # Metadata filter helpers
  config.rs               # Global config from toml + env vars
  embedding.rs            # Embedder trait (for text→vec)
  watch.rs                # File watcher (notify v8)
  mcp.rs                  # MCP server (stdio)
  index/
    mod.rs                # Index trait + re-exports
    brute_force.rs        # BruteForceIndex
    hnsw.rs               # HnswIndex + HnswConfig
    annoy.rs              # AnnoyIndex + AnnoyConfig
    sq.rs                 # SQ helpers: quantize(), score_i8(), rescore()
```

---

## 4. SQ — Scalar Quantization

### 4.1. Algoritmo de Cuantizacion

Para cada embedding `v` de dimension `d`:

1. Calcular `min_d` y `max_d` por dimension sobre todo el dataset (o
   usar estadisticas globales). Alternativa mas simple: escala global
   unica basada en el rango total del dataset.
2. `scale = (max - min) / 255.0` (donde max/min son del dataset completo,
   no por dimension).
3. `bias = min`.
4. `v_i8[i] = clamp(round((v[i] - bias) / scale), -128, 127)`.

Algoritmo mas fino: min/max **por dimension**, almacenando `d` pares
de scale/bias. Mas memoria pero mejor precision.

### 4.2. Distancia en i8

```
dot_i8(a_i8, b_i8) = sum_i(a_i8[i] * b_i8[i]) * scale^2 + d * bias^2 + ...
```

Simplificacion: para busqueda ANN donde solo importa el ranking,
el factor constante no importa:

```
dot_i8_approx(a_i8, b_i8) = sum_i(a_i8[i] * b_i8[i])  // escala lineal
```

### 4.3. Rescoring (opcional)

Para recuperar precision, despues de obtener top-k con i8, rescorear
los k*2 con f32 original. Esto anade ~20% overhead pero mejora recall.

### 4.4. Integracion por Backend

**BruteForce + SQ**: iterar embedding_i8, compute dot_i8, ordenar.
Si rescore=true, tomar top-k*2, rescore con f32.

**HNSW + SQ**: `search_layer()` usa `score_i8()` en vez de `score()`.
Las conexiones del grafo se construyen con distancia f32 (una vez en
insert), pero la busqueda usa i8. Esto es asimetrico pero funciona
porque el grafo define la topologia, no los scores exactos.

**Annoy + SQ**: las divisiones del arbol se hacen con f32 (build),
pero la recoleccion de candidatos usa distancia i8. O mas simple:
Annoy recolecta candidatos sin distancia (solo atravesando arboles),
luego scoring con i8.

### 4.5. Donde vive SQ

En `src/index/sq.rs`:

```rust
/// Cuantizar un embedding f32 a i8 con escala global.
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

## 5. Annoy — Random Projection Forest

### 5.1. Estructura de Datos

```rust
pub struct AnnoyIndex {
    documents: Vec<Document>,   // metadata + f32 embeddings
    trees: Vec<Tree>,
    config: AnnoyConfig,
}

pub struct AnnoyConfig {
    pub n_trees: usize,         // default: 10
    pub search_k: i32,          // -1 = auto: n_trees * k
}

struct Tree {
    nodes: Vec<TreeNode>,
}

enum TreeNode {
    Leaf { indices: Vec<usize> },        // indices into documents[]
    Split { left: usize, right: usize, n: Vec<f32>, d: f32 },
    // n = normal vector, d = threshold for split
}
```

### 5.2. Algoritmo de Build

```
fn build(docs: &[Document]) -> AnnoyIndex:
    for tree in 0..n_trees:
        tree = build_tree(0..docs.len(), depth=0)

fn build_tree(indices: &[usize], depth: usize) -> TreeNode:
    if indices.len() <= k || depth >= max_depth:
        return Leaf { indices }

    a = random_point(docs)
    b = random_point(docs)
    n = a - b                          // normal vector
    d = dot(n, a)                      // threshold

    left = [i for i in indices if dot(n, docs[i].embedding) <= d]
    right = [i for i in indices if dot(n, docs[i].embedding) > d]

    if left.is_empty() || right.is_empty():
        return Leaf { indices }

    return Split {
        left: build_tree(left, depth+1),
        right: build_tree(right, depth+1),
        n, d,
    }
```

### 5.3. Algoritmo de Busqueda

```
fn search(query, k) -> Vec<ScoredDocument>:
    // Priority queue: prioridad = distancia absoluta al hiperplano
    // (menos distancia = mas probable que este cerca)
    heap = MaxHeap()   // pero ordenado por |dot(query, n) - d|

    for tree in trees:
        heap.push((abs_dist, tree.root))

    candidates = Set<usize>()

    while heap.not_empty() && candidates.len() < search_k:
        (dist, node) = heap.pop()

        match node:
            Leaf { indices } => candidates.extend(indices)
            Split { left, right, n, d } =>
                side = if dot(query, n) <= d then left else right
                other = if side == left then right else left
                dist_to_other = abs(dot(query, n) - d)

                heap.push((0, side))         // explore preferred side first
                heap.push((dist_to_other, other))

    // Score candidates
    candidates.sort_by(|id| score(query, docs[id].embedding))
    return candidates[..k]
```

### 5.4. SQ + Annoy

Con SQ activo, las divisiones del arbol se hacen con f32 original
(pues el build es batch y no necesita ser rapido), pero el scoring
de candidatos usa `score_i8()`.

---

## 6. Flat Embeddings en HNSW

### 6.1. Storage

```rust
pub struct HnswIndex {
    documents: Vec<Document>,       // metadata, text, embedding (f32)
    embeddings_flat: Vec<f32>,      // solo si flat_embeddings=true
    dim: usize,                     // solo si flat_embeddings=true
    // ... resto igual
}
```

### 6.2. Helper

```rust
fn embedding(&self, node_id: usize) -> &[f32] {
    if self.config.flat_embeddings {
        let start = node_id * self.dim;
        &self.embeddings_flat[start..start + self.dim]
    } else {
        &self.documents[node_id].embedding
    }
}
```

### 6.3. Insercion

Cuando `flat_embeddings=true`, insert_one() hace:
1. Extiende `embeddings_flat` con el nuevo embedding.
2. Limpia `doc.embedding` (libera memoria, el doc se persiste sin
   embedding — se reconstruye al cargar).

O alternativamente: mantiene ambos para no modificar Document.

Decision de diseno: flat_embeddings es solo para busqueda en memoria.
El JSONL siempre guarda embedding f32 completo (para portabilidad y debug).

### 6.4. Delete con Flat

Cuando se elimina un documento con flat, hay que reconstruir
`embeddings_flat` desde los documents restantes (coste O(n·d) una vez,
equivalente a lo que ya hace el rebuild del grafo en delete).

---

## 7. Estrategia de Factory

En `collection.rs`:

```rust
fn build(path, index) -> Collection {
    let mut index: Box<dyn Index> = match cfg.index_type {
        "hnsw" => Box::new(HnswIndex::new(HnswConfig { ... })),
        "annoy" => Box::new(AnnoyIndex::new(AnnoyConfig { ... })),
        _ => Box::new(BruteForceIndex::new(metric)),
    };

    // SQ envoltura? No: SQ es interno del backend.
    // Cada backend recibe el flag sq y actua en consecuencia.
}
```

---

## 8. Dependencias

### Actuales
- serde, serde_json, thiserror — core
- rayon — parallel BruteForce
- toml, once_cell, log — config
- notify, crossbeam-channel — watcher (feature)
- rmcp, tokio, tracing, clap — MCP (feature)

### Nuevas (para Annoy, SQ)
- **Ninguna**. Pure Rust stdlib. Annoy solo necesita generacion
  de numeros aleatorios: usar `SplitMix64` (ya usado en HNSW) o
  `rand` como dev-dependency.

### Opcionales
- `rand` (dev-dependency sola, o feature opcional si se necesita
  no-determinismo real)

---

## 9. Metricas Objetivo

| Backend | 5K docs 128-dim | 50K docs 768-dim | 100K docs 384-dim |
|---------|:---------------:|:----------------:|:-----------------:|
| BruteForce | 1,700 us | ~200 ms | ~400 ms |
| HNSW | 44 us | ~500 us | ~1 ms |
| Annoy | ~80 us | ~1 ms | ~2 ms |
| BruteForce+SQ | ~800 us | ~80 ms | ~150 ms |
| HNSW+SQ | ~25 us | ~300 us | ~600 us |

RAM estimada para 100K docs 384-dim:
- f32: 100K * 384 * 4 = ~153 MB (solo embeddings)
- HNSW graphs: ~200 MB adicional (conexiones)
- SQ i8: 100K * 384 * 1 = ~38 MB + graphs ~200 MB

---

## 10. Prioridad de Implementacion

1. **HNSW + flat_embeddings** (ya empezado, ~1 sesion)
2. **SQ module** (distance.rs + sq.rs, ~1 sesion)
3. **SQ integration** en BruteForce y HNSW (~1 sesion)
4. **Annoy** (nuevo backend completo, ~2 sesiones)
5. **Benchmarks** actualizados con todos los backends (~0.5 sesion)
