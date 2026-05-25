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
| BF | 38 ms | 3M | 11.5 |
| HNSW M=16 ef=50 | 2.4 min | 692 | 66.4 |
| HNSW M=16 ef=200 | 3.9 min | 433 | 48.9 |
| IVF-PQ nlist=256 M=16 | 1.9 min | 885 | 164.5 |


### Precision: Recall@K (vs BruteForce)

| Index | Recall@1 | Recall@10 | Recall@100 |
|-------|----------|-----------|------------|
| BF | 100% | 100% | 100% |
| HNSW M=16 ef=50 | 0% | 30% | 15% |
| HNSW M=16 ef=200 | 0% | 70% | 28% |
| IVF-PQ nlist=256 M=16 | 0% | 20% | 7% |


### Rendimiento: Latencia de Consulta

| Index | Mean | p50 | p95 | p99 |
|-------|------|-----|-----|-----|
| BF | 12.4 ms | 12.3 ms | 13.2 ms | 14.9 ms |
| HNSW M=16 ef=50 | 1.1 ms | 1.1 ms | 1.3 ms | 1.4 ms |
| HNSW M=16 ef=200 | 2.1 ms | 2.1 ms | 2.3 ms | 2.7 ms |
| IVF-PQ nlist=256 M=16 | 1.2 ms | 1.0 ms | 1.9 ms | 3.1 ms |


### Sweet Spot: Recall@10 vs QPS vs RAM

| Index | Recall@10 | QPS | xBF | RAM (MB) |
|-------|-----------|-----|-----|----------|
| HNSW M=16 ef=50 | 30% | 900 | 11x | 66.4 |
| HNSW M=16 ef=200 | 70% | 479 | 6x | 48.9 |
| IVF-PQ nlist=256 M=16 | 20% | 857 | 11x | 164.5 |

#### Sweet Spot

- Mas rapido (Recall≥50%): **HNSW M=16 ef=200** — 2.1 ms us, Recall@10=70%
- Menor RAM (Recall≥50%): **BF** — 11.5 MB, Recall@10=100%

---
*Benchmark generado con dogma-vdb grid benchmark*
