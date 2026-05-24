# dogma-vdb — Benchmark Grid Results

> Generado automaticamente | Vectores 128-dim | Cosine | k=10 | 50 queries/config

## Parametros del Grid

- Tamaños: [10000]
- Dimensiones: [128]
- Metricas: ["Cosine"]
- HNSW grid: M∈[16, 32], ef∈[50, 100, 150, 200]
- IVF-PQ grid: nlist∈[256], M_sub∈[8]
- Queries por configuracion: 50

---
## 10000 docs, 128 dim, Cosine


### Construccion: Build Time / Throughput / RAM

| Index | Build | vec/s | RAM (MB) |
|-------|-------|-------|----------|
| BF | 3 ms | 3M | 6.7 |
| HNSW M=16 ef=50 | 5.6s | 2K | 8.9 |
| HNSW M=16 ef=100 | 5.7s | 2K | 0.2 |
| HNSW M=16 ef=150 | 7.2s | 1K | 0.0 |
| HNSW M=16 ef=200 | 8.7s | 1K | 0.0 |
| HNSW M=32 ef=50 | 16.1s | 622 | 0.0 |
| HNSW M=32 ef=100 | 16.3s | 613 | 3.9 |
| HNSW M=32 ef=150 | 19.0s | 528 | 0.0 |
| HNSW M=32 ef=200 | 21.1s | 473 | 0.0 |
| IVF-PQ nlist=256 M=8 probe=1 | 8.7s | 1K | 0.0 |
| IVF-PQ nlist=256 M=8 probe=2 | 8.8s | 1K | 0.0 |
| IVF-PQ nlist=256 M=8 probe=4 | 8.7s | 1K | 0.0 |
| IVF-PQ nlist=256 M=8 probe=8 | 8.6s | 1K | 0.0 |
| IVF-PQ nlist=256 M=8 probe=16 | 8.5s | 1K | 0.0 |
| IVF-PQ nlist=256 M=8 probe=32 | 9.1s | 1K | 0.0 |


### Precision: Recall@K (vs BruteForce)

| Index | Recall@1 | Recall@10 | Recall@100 |
|-------|----------|-----------|------------|
| BF | 100% | 100% | 100% |
| HNSW M=16 ef=50 | 100% | 100% | 100% |
| HNSW M=16 ef=100 | 100% | 100% | 100% |
| HNSW M=16 ef=150 | 100% | 100% | 100% |
| HNSW M=16 ef=200 | 100% | 100% | 100% |
| HNSW M=32 ef=50 | 100% | 100% | 100% |
| HNSW M=32 ef=100 | 100% | 100% | 100% |
| HNSW M=32 ef=150 | 100% | 100% | 100% |
| HNSW M=32 ef=200 | 100% | 100% | 100% |
| IVF-PQ nlist=256 M=8 probe=1 | 0% | 0% | 4% |
| IVF-PQ nlist=256 M=8 probe=2 | 0% | 20% | 18% |
| IVF-PQ nlist=256 M=8 probe=4 | 0% | 0% | 26% |
| IVF-PQ nlist=256 M=8 probe=8 | 0% | 0% | 20% |
| IVF-PQ nlist=256 M=8 probe=16 | 0% | 0% | 26% |
| IVF-PQ nlist=256 M=8 probe=32 | 0% | 0% | 22% |


### Rendimiento: Latencia de Consulta

| Index | Mean | p50 | p95 | p99 |
|-------|------|-----|-----|-----|
| BF | 1.5 ms | 1.4 ms | 1.9 ms | 2.0 ms |
| HNSW M=16 ef=50 | 364 us | 359 us | 421 us | 427 us |
| HNSW M=16 ef=100 | 386 us | 380 us | 469 us | 515 us |
| HNSW M=16 ef=150 | 555 us | 548 us | 668 us | 727 us |
| HNSW M=16 ef=200 | 689 us | 689 us | 744 us | 765 us |
| HNSW M=32 ef=50 | 653 us | 649 us | 752 us | 781 us |
| HNSW M=32 ef=100 | 672 us | 678 us | 724 us | 767 us |
| HNSW M=32 ef=150 | 911 us | 919 us | 958 us | 994 us |
| HNSW M=32 ef=200 | 1.2 ms | 1.2 ms | 1.3 ms | 1.3 ms |
| IVF-PQ nlist=256 M=8 probe=1 | 124 us | 122 us | 183 us | 192 us |
| IVF-PQ nlist=256 M=8 probe=2 | 144 us | 136 us | 162 us | 381 us |
| IVF-PQ nlist=256 M=8 probe=4 | 159 us | 154 us | 189 us | 202 us |
| IVF-PQ nlist=256 M=8 probe=8 | 177 us | 171 us | 228 us | 259 us |
| IVF-PQ nlist=256 M=8 probe=16 | 261 us | 228 us | 406 us | 1.1 ms |
| IVF-PQ nlist=256 M=8 probe=32 | 305 us | 295 us | 367 us | 416 us |


### Sweet Spot: Recall@10 vs QPS vs RAM

| Index | Recall@10 | QPS | xBF | RAM (MB) |
|-------|-----------|-----|-----|----------|
| HNSW M=16 ef=50 | 100% | 3K | 4x | 8.9 |
| HNSW M=16 ef=100 | 100% | 3K | 4x | 0.2 |
| HNSW M=16 ef=150 | 100% | 2K | 3x | 0.0 |
| HNSW M=16 ef=200 | 100% | 1K | 2x | 0.0 |
| HNSW M=32 ef=50 | 100% | 2K | 2x | 0.0 |
| HNSW M=32 ef=100 | 100% | 1K | 2x | 3.9 |
| HNSW M=32 ef=150 | 100% | 1K | 2x | 0.0 |
| HNSW M=32 ef=200 | 100% | 851 | 1x | 0.0 |
| IVF-PQ nlist=256 M=8 probe=1 | 0% | 8K | 12x | 0.0 |
| IVF-PQ nlist=256 M=8 probe=2 | 20% | 7K | 10x | 0.0 |
| IVF-PQ nlist=256 M=8 probe=4 | 0% | 6K | 9x | 0.0 |
| IVF-PQ nlist=256 M=8 probe=8 | 0% | 6K | 8x | 0.0 |
| IVF-PQ nlist=256 M=8 probe=16 | 0% | 4K | 6x | 0.0 |
| IVF-PQ nlist=256 M=8 probe=32 | 0% | 3K | 5x | 0.0 |

#### Sweet Spot

- Mejor configuracion (Recall≥85%): **HNSW M=16 ef=50** — QPS=3K, Latencia=364 us, RAM=8.9 MB
- Mas rapido (Recall≥50%): **HNSW M=16 ef=50** — 364 us us, Recall@10=100%
- Menor RAM (Recall≥50%): **HNSW M=16 ef=150** — 0.0 MB, Recall@10=100%

---
*Benchmark generado con dogma-vdb grid benchmark*
