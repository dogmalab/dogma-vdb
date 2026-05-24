# dogma-vdb — Benchmark Grid Results

> Generado automaticamente | Vectores 128-dim | Cosine | k=10 | 100 queries/config

## Parametros del Grid

- Tamaños: [100000]
- Dimensiones: [384]
- Metricas: ["Cosine"]
- HNSW grid: M∈[16], ef∈[50, 200]
- IVF-PQ grid: nlist∈[256], M_sub∈[16]
- Queries por configuracion: 100

---
## 100000 docs, 384 dim, Cosine


### Construccion: Build Time / Throughput / RAM

| Index | Build | vec/s | RAM (MB) |
|-------|-------|-------|----------|
| BF | 98 ms | 1M | 165.6 |
| HNSW M=16 ef=50 | 2.7 min | 614 | 0.0 |
| HNSW M=16 ef=200 | 4.4 min | 375 | 0.0 |
| IVF-PQ nlist=256 M=16 | 3.7 min | 453 | 0.0 |


### Precision: Recall@K (vs BruteForce)

| Index | Recall@1 | Recall@10 | Recall@100 |
|-------|----------|-----------|------------|
| BF | 100% | 100% | 100% |
| HNSW M=16 ef=50 | 0% | 60% | 19% |
| HNSW M=16 ef=200 | 0% | 80% | 38% |
| IVF-PQ nlist=256 M=16 | 0% | 40% | 16% |


### Rendimiento: Latencia de Consulta

| Index | Mean | p50 | p95 | p99 |
|-------|------|-----|-----|-----|
| BF | 186.4 ms | 186.0 ms | 192.7 ms | 204.4 ms |
| HNSW M=16 ef=50 | 993 us | 976 us | 1.2 ms | 1.2 ms |
| HNSW M=16 ef=200 | 1.9 ms | 1.9 ms | 2.1 ms | 2.4 ms |
| IVF-PQ nlist=256 M=16 | 11.5 ms | 11.5 ms | 12.2 ms | 13.8 ms |


### Sweet Spot: Recall@10 vs QPS vs RAM

| Index | Recall@10 | QPS | xBF | RAM (MB) |
|-------|-----------|-----|-----|----------|
| HNSW M=16 ef=50 | 60% | 1K | 188x | 0.0 |
| HNSW M=16 ef=200 | 80% | 531 | 99x | 0.0 |
| IVF-PQ nlist=256 M=16 | 40% | 87 | 16x | 0.0 |

#### Sweet Spot

- Mas rapido (Recall≥50%): **HNSW M=16 ef=50** — 993 us us, Recall@10=60%
- Menor RAM (Recall≥50%): **HNSW M=16 ef=50** — 0.0 MB, Recall@10=60%

---
*Benchmark generado con dogma-vdb grid benchmark*
