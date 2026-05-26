# 🗺️ Roadmap de Arquitectura: dogma-vdb (Próximos Pasos)

> Línea base estable: 8,594 LOC, 192 tests, IVF-PQ Recall@10 = 74% con K-Means++
> Fecha: 2026-05-25 | Commit: 0e118c50

---

## 1. 🧠 Smart-Tuning Autónomo (`src/tuning/agent.rs`)

**Objetivo:** Eliminar la necesidad de configuración manual de parámetros
vectoriales, indexando dinámicamente según el hardware y volumen de datos
disponibles.

### Mecanismo de Decisión (SLM / Heurística en Rust)

El tunear detecta el entorno en tiempo de ejecución y selecciona la
configuración óptima sin intervención del usuario.

#### Inputs del Sistema

```rust
pub enum TargetPriority {
    HighRecall,      // Máxima precisión (default para MCP / queries interactivas)
    UltraLowLatency, // Mínima latencia (batch / pipeline CI)
    Balanced,        // Trade-off por defecto
}

pub struct SystemProfile {
    pub free_ram_mb: u64,         // De MemoryGuard::check_memory()
    pub total_ram_mb: u64,        // De /proc/meminfo
    pub cpu_cores: usize,         // De available_parallelism()
    pub dataset_size: usize,      // Número de documentos
    pub dimension: usize,         // Dimensionalidad de embeddings
}
```

#### Matriz de Control Dinámico

| Condición | Backend | Config | Rationale |
|-----------|---------|--------|-----------|
| RAM < 8GB || dataset > 50K | `IVF-PQ` | `n_list=256, M=32, n_probe=8, sq=true` | Proteger OS de OOM. PQ + SQ = ~32× compresión vs f32 |
| RAM > 16GB || dataset < 10K | `HNSW` | `M=64, ef_construction=200, ef_search=100, f32` | Prioriza recall. Memoria no es limitación |
| 8GB < RAM < 16GB | `HNSW+SQ` | `M=32, ef=100, sq=true, sq_rescore=true` | Balance. SQ rescate recupera recall |
| dataset < 1K (cualquier RAM) | `BruteForce` | — | Exacto, O(n·d) trivial. Sin overhead de build |

#### Ajuste en Caliente (Hot Tuning)

Una vez el índice está construido, el sistema monitorea la latencia p99
de las últimas N consultas y ajusta parámetros de búsqueda dinámicamente:

```rust
struct HotTuningState {
    recent_latencies_p99: Vec<f64>,   // rolling window (últimas 100 queries)
    current_ef_search: usize,         // para HNSW
    current_n_probe: usize,           // para IVF-PQ
    target_latency_us: f64,           // ej: 500 μs
}

impl HotTuningState {
    /// Si p99 > target_latency, reduce ef_search / n_probe.
    /// Si p99 << target_latency y recall es suficiente, aumenta parámetros.
    fn tick(&mut self, measured_p99_us: f64) -> TuningAdjustment;
}
```

#### Integración

```rust
// En collection.rs
pub fn open_with_tuning(path, priority: TargetPriority) -> Result<Self>;

// O vía config.toml
// [tuning]
// priority = "high_recall"
// auto_tune = true
```

### Archivos a modificar/crear

| Archivo | Acción | LOC estimado |
|---------|--------|:------------:|
| `src/tuning/mod.rs` | Nuevo — SystemProfile, TargetPriority, matriz de decisión | ~120 |
| `src/tuning/agent.rs` | Nuevo — HotTuningState, rolling window, ajuste dinámico | ~80 |
| `src/config.rs` | Modificar — añadir `[tuning]` sección | ~20 |
| `src/collection.rs` | Modificar — integrar `open_with_tuning()` y `auto_tune` | ~40 |
| **Total** | | **~260 LOC** |

### Dependencias nuevas

- Ninguna. Todo es Rust puro + stdlib (cálculos de percentiles, rolling window).

### Criterio de éxito

```
Sin tuning manual, en una máquina con 8 GB RAM y 100K docs 384-dim:
  - IVF-PQ autoseleccionado
  - Build < 30s
  - Query p50 < 500 μs
  - Sin OOM
```

---

## 2. 🔀 Watcher Concurrente Asíncrono v2 (`src/watch/actor.rs`)

**Objetivo:** Elevar el sistema de monitoreo en tiempo real al estándar
de bajo nivel del resto del motor, eliminando el cuello de botella de
apertura/cierre de archivos en cada evento.

### Problema del watcher actual (v1)

```
Evento notify → open collection → insert docs → store → close
                   ↑ cada evento re-abre y persiste el .vdb completo
```

Para 100 archivos modificados simultáneamente, esto produce **100 aperturas,
100 inserciones, 100 escrituras completas a disco**. Con `debounce_ms=500`,
la ventana de coalescencia es fija, no adaptativa.

### Arquitectura Basada en Actores (v2)

#### 1. Instancia Viva Co-compartida

El watcher ya no recibe un `PathBuf` para re-abrir el archivo `.vdb`.
Recibe un `Arc<RwLock<Collection>>` para interactuar con la base de datos
abierta en RAM (mmap).

```rust
pub struct LiveCollection {
    collection: Arc<RwLock<Collection>>,
    /// Canal para notificar al actor cuándo hacer flush
    flush_tx: Sender<()>,
}

pub fn start_watching_v2(
    config: WatchConfig,
    collection: LiveCollection,
) -> Receiver<WatchEvent>;
```

#### 2. Tabla de Coalescencia (Debounce Avanzado)

```rust
struct CoalescenceTable {
    /// Mapa de path → última hora del evento
    pending: HashMap<PathBuf, Instant>,
    /// Duración de la ventana (configurable por tipo de evento)
    window: Duration,
}

impl CoalescenceTable {
    /// Registra un evento. Si ya existe uno para el mismo path en la
    /// ventana actual, se sobreescribe (coalesce).
    fn push(&mut self, path: PathBuf) -> bool;  // true = nuevo, false = coalescido

    /// Retorna los paths que han superado la ventana de coalescencia.
    fn drain_ready(&mut self) -> Vec<PathBuf>;
}
```

#### 3. Pipeline Flushed Decoupled

```
[notify events] → CoalescenceTable → inject en caliente (Collection in RAM)
                                           ↓
                              ¿silencio > 2s? → collection.flush() a disco
```

- Los chunks modificados se inyectan **en caliente** sobre la colección
  viva en memoria (sin re-abrir el archivo).
- El volcado a disco (`flush()`) se ejecuta de manera **perezosa (lazy)**
  únicamente cuando el canal de eventos de notify entra en silencio por
  más de 2 segundos, evitando tormentas de I/O.
- `LiveCollection::flush()` usa `BinStorage::store()` atómico (tmp + rename).

### Diagrama de flujo

```
                    ┌──────────────────────┐
                    │   notify::Watcher     │
                    │  (crossbeam-channel)  │
                    └──────────┬───────────┘
                               │ eventos crudos
                               ▼
                    ┌──────────────────────┐
                    │  CoalescenceTable    │
                    │  (ventana 500ms)     │
                    │  HashSet<PathBuf>    │
                    └──────────┬───────────┘
                               │ paths coalescedos
                               ▼
                    ┌──────────────────────┐
                    │  inject en caliente  │
                    │  Arc<RwLock<Coll>>   │
                    │  chunker + insert    │
                    └──────────┬───────────┘
                               │
                    ┌──────────┴───────────┐
                    │  Timer de silencio    │
                    │  ¿> 2s sin eventos?   │
                    └──────────┬───────────┘
                               │ sí
                               ▼
                    ┌──────────────────────┐
                    │  flush() a disco     │
                    │  (tmp + rename)      │
                    └──────────────────────┘
```

### Archivos a modificar/crear

| Archivo | Acción | LOC estimado |
|---------|--------|:------------:|
| `src/watch/actor.rs` | Nuevo — actor loop, CoalescenceTable, flush decoupled | ~150 |
| `src/watch/mod.rs` | Modificar — re-exportar `start_watching_v2`, mantener v1 | ~30 |
| `src/watch.rs` | Eliminar o convertir en delegación a `watch/actor.rs` | — |
| **Total** | | **~180 LOC** |

### Dependencias nuevas

- Ninguna. `crossbeam-channel` y `notify` ya están en el feature `watch`.
- `Arc<RwLock>` viene de stdlib.

### Criterio de éxito

```
Con 500 archivos .rs:
  - Carga inicial (initial_scan): < 2s
  - Modificación de 50 archivos simultáneos: 1 flush a disco, no 50
  - Latencia de inyección en caliente: < 10 ms por archivo
  - Colección siempre responde a queries durante la ingesta
```

---

## 3. 🌐 Capa de Red Industrial & Transporte (`src/network/`)

**Objetivo:** Exponer el motor vectorial a microservicios e integraciones
remotas empresariales mediante protocolos de alto rendimiento, bajo el
feature-gate `#[cfg(feature = "net")]`.

### 3.1. gRPC Server (`src/network/grpc.rs`)

Implementación nativa usando el crate [`tonic`](https://crates.io/crates/tonic).
Definición de un esquema `.proto` optimizado para el streaming bidireccional
de vectores densos (`f32` / `i8`) y payloads de metadatos.

```protobuf
service DogmaVdb {
  // Búsqueda vectorial simple
  rpc Search(SearchRequest) returns (SearchResponse);

  // Búsqueda híbrida (vector + BM25 + reranker)
  rpc HybridSearch(HybridSearchRequest) returns (SearchResponse);

  // Ingesta por streaming de documentos
  rpc Ingest(stream Document) returns (IngestResponse);

  // Eliminación por ID
  rpc Delete(DeleteRequest) returns (DeleteResponse);
}

message SearchRequest {
  string collection_path = 1;
  repeated float embedding = 2;
  uint32 k = 3;
  string index_type = 4;   // bruteforce | hnsw | ivf_pq
  string metric = 5;       // cosine | dot | euclidean
  bool rerank = 6;
  string query_text = 7;   // requerido si rerank = true
}
```

#### Arquitectura

```
                    ┌─────────────────────────────┐
                    │      tonic gRPC Server       │
                    │   (http2, multiplexed)       │
                    └──┬──────────┬───────────┬────┘
                       │          │           │
              ┌────────┴──┐ ┌─────┴──────┐ ┌──┴─────────┐
              │ Search    │ │ Hybrid     │ │ Ingest     │
              │ Handler   │ │ Handler    │ │ Handler    │
              └────┬──────┘ └─────┬──────┘ └─────┬──────┘
                   │              │               │
                   └──────┬──────┴───────────────┘
                          │
                    ┌─────┴──────┐
                    │ Collection │
                    │ (mmap)     │
                    └────────────┘
```

#### Cliente Python

```python
import grpc
from dogmavdb_pb2 import SearchRequest
from dogmavdb_pb2_grpc import DogmaVdbStub

channel = grpc.insecure_channel("localhost:50051")
client = DogmaVdbStub(channel)
response = client.Search(SearchRequest(
    collection_path="data/docs.vdb",
    embedding=[0.1, 0.2, 0.3],
    k=10,
    index_type="hnsw",
    metric="cosine",
))
```

### 3.2. MCP vía HTTP / WebSockets (`src/network/mcp_http.rs`)

Evolucionar el transporte del MCP server actual (stdio) a SSE
(Server-Sent Events) o WebSockets sobre HTTP. Permite que múltiples
agentes remotos (Claude Desktop, Cursor, opencode, scripts) se conecten
simultáneamente sin estar amarrados al ciclo de vida de un solo proceso
de terminal.

#### Estado actual del MCP server

```
hoy:   serve_server(server, (stdin(), stdout()).into_transport())
mañana: serve_server(server, http_into_transport("0.0.0.0:5000"))
```

El crate `rmcp` que ya usamos soporta transporte HTTP/SSE. Lo que falta
es añadir `axum` como servidor HTTP y elegir el transporte por flag CLI.

```bash
# Hoy (solo stdio):
dogma-vdb-mcp

# Mañana (HTTP / WebSocket):
dogma-vdb-mcp --transport http --port 5000
dogma-vdb-mcp --transport ws --port 5000
```

#### Diagrama de conexiones simultáneas

```
        ┌──────────────┐
        │ Claude       │────┐
        │ Desktop      │    │
        └──────────────┘    │   ┌────────────────────┐
                            ├──►│  dogma-vdb MCP      │
        ┌──────────────┐    │   │  Server (HTTP/WS)   │
        │ Cursor       │────┘   │  :5000              │
        └──────────────┘        │                     │
                                │  Collection (mmap)  │
        ┌──────────────┐        └────────────────────┘
        │ opencode     │────┐
        └──────────────┘    │
                            │
        ┌──────────────┐    │
        │ Python SDK   │────┘
        └──────────────┘
```

### 3.3. Feature Gate

```toml
[features]
net = ["dep:tonic", "dep:axum", "dep:rmcp"]
```

### Archivos a modificar/crear

| Archivo | Acción | LOC estimado |
|---------|--------|:------------:|
| `proto/dogmavdb.proto` | Nuevo — definición protobuf | ~80 |
| `src/network/mod.rs` | Nuevo — feature gate + re-exports | ~10 |
| `src/network/grpc.rs` | Nuevo — tonic server handlers | ~200 |
| `src/network/mcp_http.rs` | Nuevo — HTTP/WS transport para MCP | ~100 |
| `dogma-vdb-mcp/src/main.rs` | Modificar — flag `--transport` + `--port` | ~50 |
| `Cargo.toml` | Modificar — feature `net` + deps opcionales | ~10 |
| `build.rs` | Nuevo — compilación del .proto | ~30 |
| **Total** | | **~480 LOC** |

### Dependencias nuevas

| Crate | Versión | Propósito |
|-------|:-------:|-----------|
| `tonic` | 0.12+ | Servidor/framework gRPC |
| `prost` | 0.13+ | Compilación de `.proto` a Rust |
| `axum` | 0.8+ | Servidor HTTP para MCP transport |
| `rmcp` | 1.x | *(ya en el workspace, ampliar transporte)* |

### Criterio de éxito

```
- gRPC: 10,000 queries/s en single core, colección de 10K docs 384-dim
- MCP HTTP: 5 agentes simultáneos, sin interferencia entre conexiones
- MCP WebSocket: latencia < 100 μs overhead sobre el tiempo de query real
- net feature off por defecto: 0 impacto en binarios sin red
```

---

## Resumen de esfuerzo estimado

| Feature | LOC | Dependencias nuevas | Prioridad |
|---------|:---:|:-------------------:|:---------:|
| Smart-Tuning Autónomo | ~260 | 0 | Alta |
| Watcher Concurrente v2 | ~180 | 0 | Media |
| Capa de Red (gRPC + MCP HTTP) | ~480 | 4 (tonic, prost, axum, rmcp) | Media |
| **Total** | **~920** | | |

---

## Línea base actual (pre-requisito para ambas features)

| Métrica | Valor |
|---------|:-----:|
| LOC total | 8,594 |
| Tests | 192 pasan, 0 fallan |
| Compilación | 0 errors, 0 warnings |
| IVF-PQ Recall@10 (embeddings reales) | 74.0% |
| IVF-PQ Latencia p50 | 344 μs |
| K-Means++ | Implementado (D² weighting) |
| Feature flags | `watch` (off by default) |
| Formato almacenamiento | Binario v2 (DVDB), sin JSONL |
| Chunker estrategias | 3: Code, Paragraph, FixedWindow |
