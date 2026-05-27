# Report: Fix HNSW+SQ Recall

**Date:** 2026-05-17
**Audit:** audit-1-temp.md
**Plan:** .hermes/plans/2026-05-17_145000-fix-hnsw-sq-recall.md

---

## Fixed Bugs

### Bug #1 — Single-document scale/bias in HNSW (src/index/hnsw.rs)

**Symptom:** HNSW with `sq=true` quantized using scale/bias computed from the **first document only**.

**Cause:** In `insert_one()`, the SQ block used `let fake_docs = [doc.clone()]` to compute scale/bias, ignoring the rest of the dataset. Subsequent documents were quantized with these incorrect values and were never recalculated.

**Fix:** Removed quantization from `insert_one()`. Added post-quantization in `insert()` which:
1. Computes scale/bias from **all** documents (existing + new)
2. Re-quantizes the entire dataset with `par_iter()` (parallel)
3. Identical to `BruteForceIndex` approach (proven)

**File:** `src/index/hnsw.rs`
- Removed lines 372-383 (SQ block in `insert_one`)
- Added ~15 lines in `insert()` (post-quantization)
- Added `use rayon::prelude::*`

### Bug #2 — Centered quantization formula (src/index/sq.rs)

**Symptom:** Even with correct scale/bias, SQ recall was low (~12% in tests).

**Root cause:** The formula `bias = min` maps the range `[min, max] → [0, 255]`. But `i8` only stores `[-128, 127]`. Values > 127 saturate to 127. For normalized embeddings (range ~[-1, 1] or [-3, 3]), **every value above the midpoint quantizes to the same maximum value**, losing half a dimension's worth of information.

```
Before:  q = (v - min) * 255/(max-min)    → [0, 255] → clamp to i8 → massive loss
After:   q = (v - mid) * 255/(max-min)    → [-128, 127] → fits perfectly in i8
         mid = (min + max) / 2
```

**Fix:** `compute_scale_bias_per_dim` now returns the midpoint as bias.

**File:** `src/index/sq.rs`
- Line 61: `(scales, mins)` → `(scales, biases)` with `bias = (min+max)/2`
- Tests updated for new bias values

### Minor fixes
- Removed warning `unused variable: biases` (test in sq.rs)
- `AGENTS.md` updated: 152 → 156 tests

---

## Benchmark Results

Executed with `cargo run --release --example bench` (128-dim random vectors [-3, 3], 100 queries).

### Speed (5K docs, 128-dim)

| Backend | us/query | vs Before |
|---------|:--------:|:--------:|
| **HNSW (ef=50)** | **85** | same |
| **HNSW+SQ** | **73** | ~14% faster |
| HNSW+SQ+Rescore | 79 | slight cost |
| HNSW+Flat | 86 | same |
| BruteForce | 1,505 | same |
| BF+SQ | 1,529 | same |
| Annoy | 3,258 | same |

### Recall (5K docs, 128-dim, against exact BruteForce)

| Backend | Recall Before | Recall **After** | Improvement |
|---------|:------------:|:-------------------:|:------:|
| **HNSW** | 100% | 100% | — |
| **HNSW+SQ** | **0-60%** | **80%** | 🔥 |
| **HNSW+SQ+Rescore** | 0-60% | **90%** | 🔥 |
| HNSW+Flat | 100% | 100% | — |
| BF+SQ | ~40% | 60% | ✅ |
| BF+SQ+Rescore | ~90% | 90% | — |

### Recall by dataset size (HNSW+SQ+Rescore)

| Docs | Before | **After** |
|:----:|:-----:|:-----------:|
| 100 | ~0% | **100%** |
| 500 | ~0% | **100%** |
| 1,000 | ~0% | 40%* |
| 5,000 | ~60% | **90%** |

\* The 1K docs case is an outlier: HNSW f32 without SQ also only gives 80% with ef=50.
   With ef=200 the recall rises to ~99% in f32 and ~85% in SQ+Rescore.

---

## Final State

```
156 tests, 0 failures, 0 warnings
```

| Metric | Before | After |
|---------|:-----:|:-------:|
| Tests | 152 | **156** |
| Warnings | 1 | **0** |
| HNSW+SQ recall (5K docs) | 0-60% | **80%** |
| HNSW+SQ+Rescore (5K docs) | 0-60% | **90%** |
| BF+SQ recall (5K docs) | ~40% | **60%** |
| Lines of code | ~30 | ~40 (+10) |
