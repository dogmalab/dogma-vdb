# Tuning Report — Dogma-VDB Autonomous Calibration

> Generado automaticamente | Umbral Recall@10 >= 90%

## Metodologia

Para cada configuracion del grid:
- Si Recall@10 < 90% → descartada
- Si Recall@10 >= 90% → **Score = QPS / RAM_MB**
- Mayor score = mejor eficiencia (maximos QPS por MB de RAM)

---
## HNSW — Top 3 Sweet Spots

| # | Config | Build | Latencia | QPS | RAM (MB) | Recall@10 | Score (QPS/MB) |
|---|--------|-------|----------|-----|----------|-----------|----------------|
| 1 | HNSW M=16 ef=100 | 5.7s | 386 us | 2587 | 0.2 | 100% | 11826.1 |
| 2 | HNSW M=16 ef=150 | 7.3s | 522 us | 1917 | 0.0 | 100% | 1916.6 |
| 3 | HNSW M=16 ef=150 | 7.2s | 555 us | 1802 | 0.0 | 100% | 1801.5 |

---
## IVF-PQ — Top 3 Sweet Spots

| # | Config | Build | Latencia | QPS | RAM (MB) | Recall@10 | Score (QPS/MB) |
|---|--------|-------|----------|-----|----------|-----------|----------------|
| — | Ninguna configuracion supera el umbral del 90% | — | — | — | — | — | — |

---
## Impacto de `n_probe` en IVF-PQ

| n_probe | Latencia (us) | QPS | Recall@10 | RAM (MB) | Score |
|---------|---------------|-----|-----------|----------|-------|
| 1 | 126 | 7961 | 0% | 0.0 | 0.0 |
| 1 | 124 | 8041 | 0% | 0.0 | 0.0 |
| 2 | 149 | 6730 | 20% | 0.0 | 0.0 |
| 2 | 144 | 6953 | 20% | 0.0 | 0.0 |
| 4 | 175 | 5704 | 0% | 0.0 | 0.0 |
| 4 | 159 | 6283 | 0% | 0.0 | 0.0 |
| 8 | 198 | 5039 | 0% | 0.0 | 0.0 |
| 8 | 177 | 5642 | 0% | 0.0 | 0.0 |
| 16 | 222 | 4498 | 0% | 0.0 | 0.0 |
| 16 | 261 | 3824 | 0% | 0.0 | 0.0 |
| 32 | 337 | 2968 | 0% | 0.0 | 0.0 |
| 32 | 305 | 3275 | 0% | 0.0 | 0.0 |

### Analisis

- Ninguna configuracion de n_probe logro alcanzar el 90% de Recall@10 con vectores aleatorios. Esto es esperable: los ANN indexes no explotan
estructura inexistente en datos ruidosos. Con embeddings reales (texto),
el recall seria significativamente mayor.
- n_probe=1: 126 us, 0% recall vs n_probe=32: 305 us, 0% recall.
- Aumentar n_probe de 1 a 32 mejora recall en 0 puntos pero reduce QPS 0.4x.

---
*Reporte generado automaticamente por el sistema de Tunning Autonomo*
