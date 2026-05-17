# Informe: Fix HNSW+SQ Recall

**Fecha:** 2026-05-17
**Auditoría:** audit-1-temp.md
**Plan:** .hermes/plans/2026-05-17_145000-fix-hnsw-sq-recall.md

---

## Bugs Corregidos

### Bug #1 — Scale/bias de 1 documento en HNSW (src/index/hnsw.rs)

**Síntoma:** HNSW con `sq=true` cuantizaba usando scale/bias calculados del **primer documento solamente**.

**Causa:** En `insert_one()`, el bloque SQ usaba `let fake_docs = [doc.clone()]` para calcular scale/bias, ignorando el resto del dataset. Los documentos siguientes se cuantizaban con estos valores incorrectos y nunca se recalculaban.

**Fix:** Se eliminó la cuantización de `insert_one()`. Se añadió post-cuantización en `insert()` que:
1. Calcula scale/bias de **todos** los documentos (existentes + nuevos)
2. Re-cuantiza todo el dataset con `par_iter()` (paralelo)
3. Idéntico al approach de `BruteForceIndex` (probado)

**Archivo:** `src/index/hnsw.rs`
- Eliminadas líneas 372-383 (bloque SQ en `insert_one`)
- Añadidas ~15 líneas en `insert()` (post-cuantización)
- Añadido `use rayon::prelude::*`

### Bug #2 — Fórmula de cuantización centrada (src/index/sq.rs)

**Síntoma:** Incluso con scale/bias correctos, el recall de SQ era bajo (~12% en test).

**Causa raíz:** La fórmula `bias = min` mapea el rango `[min, max] → [0, 255]`. Pero `i8` solo almacena `[-128, 127]`. Los valores > 127 se saturan a 127. Para embeddings normalizados (rango ~[-1, 1] o [-3, 3]), **todo valor por encima del punto medio se cuantiza al mismo valor máximo**, perdiendo media dimensión de información.

```
Antes:  q = (v - min) * 255/(max-min)    → [0, 255] → clamp a i8 → pérdida masiva
Después: q = (v - mid) * 255/(max-min)    → [-128, 127] → encaja perfecto en i8
         mid = (min + max) / 2
```

**Fix:** `compute_scale_bias_per_dim` ahora retorna el punto medio como bias.

**Archivo:** `src/index/sq.rs`
- Línea 61: `(scales, mins)` → `(scales, biases)` con `bias = (min+max)/2`
- Tests actualizados para nuevos valores de bias

### Arreglos menores
- Warning `unused variable: biases` eliminado (test en sq.rs)
- `AGENTS.md` actualizado: 152 → 156 tests

---

## Resultados de Benchmark

Ejecutado con `cargo run --release --example bench` (vectores 128-dim aleatorios [-3, 3], 100 queries).

### Velocidad (5K docs, 128-dim)

| Backend | us/query | vs Antes |
|---------|:--------:|:--------:|
| **HNSW (ef=50)** | **85** | igual |
| **HNSW+SQ** | **73** | ~14% más rápido |
| HNSW+SQ+Rescore | 79 | leve costo |
| HNSW+Flat | 86 | igual |
| BruteForce | 1,505 | igual |
| BF+SQ | 1,529 | igual |
| Annoy | 3,258 | igual |

### Recall (5K docs, 128-dim, contra BruteForce exacto)

| Backend | Recall Antes | Recall **Después** | Mejora |
|---------|:------------:|:-------------------:|:------:|
| **HNSW** | 100% | 100% | — |
| **HNSW+SQ** | **0-60%** | **80%** | 🔥 |
| **HNSW+SQ+Rescore** | 0-60% | **90%** | 🔥 |
| HNSW+Flat | 100% | 100% | — |
| BF+SQ | ~40% | 60% | ✅ |
| BF+SQ+Rescore | ~90% | 90% | — |

### Recall por tamaño de dataset (HNSW+SQ+Rescore)

| Docs | Antes | **Después** |
|:----:|:-----:|:-----------:|
| 100 | ~0% | **100%** |
| 500 | ~0% | **100%** |
| 1,000 | ~0% | 40%* |
| 5,000 | ~60% | **90%** |

\* El caso 1K docs es un outlier: HNSW f32 sin SQ también da solo 80% con ef=50.
   Con ef=200 el recall sube a ~99% en f32 y ~85% en SQ+Rescore.

---

## Estado Final

```
156 tests, 0 fallos, 0 warnings
```

| Métrica | Antes | Después |
|---------|:-----:|:-------:|
| Tests | 152 | **156** |
| Warnings | 1 | **0** |
| HNSW+SQ recall (5K docs) | 0-60% | **80%** |
| HNSW+SQ+Rescore (5K docs) | 0-60% | **90%** |
| BF+SQ recall (5K docs) | ~40% | **60%** |
| Líneas de código | ~30 | ~40 (+10) |
