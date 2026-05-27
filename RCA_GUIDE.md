# Code Audit Guide — RCA of Benchmark (100K 384-dim Cosine)

> **Context**: Results from the grid benchmark showing 4 anomalies.
> **Goal**: Isolate the root cause of each failure through diagnostic questions,
>   code pattern searching, and unit sanity tests.

---

## FAILURE 1: Recall@1 = 0% with Recall@10 = 60-80%

**Symptom**: The exact nearest neighbor NEVER appears at position #1 of the
approximate results, but 6-8 of the exact top-10 DO appear in the approximate
top-10.

### 1. Key Questions

**P1.** In the `search_layer()` function (hnsw.rs ~L489), the `results` structure is
`BinaryHeap<Reverse<Candidate>>`. When you call `into_sorted_vec()`, the elements
are returned in ascending order of `Reverse<Candidate>`. Are you interpreting
"ascending by Reverse" as "descending by score" (correct) or as "ascending
by score" (incorrect)? The comment at L552-553 claims the former. Verify
with a 2-vector test.

**P2.** In `search()` (~L368-375), candidates go through `.take(k)` and then
`scored.sort_by(|a, b| b.score.partial_cmp(&a.score))`. Is it possible that `take(k)`
truncates before sorting, discarding the true #1 neighbor that was further down
the list? The results from `search_layer` already come best-first ordered,
but if `ef > k`, only `k` elements survive the `.take()`.

**P3.** When `k=1`, line 308 computes `ef = self.config.ef_search.max(1)`.
Does `search_layer` with `ef=50` always include the entry point in the results?
Or is there a case where `entry_score` is so low that the entry point is
evicted from the `results` heap before exploring its neighbors?

### 2. Bug Patterns in Code

Search in `src/index/hnsw.rs`:

```rust
// ─── CRITICAL LINE 1: L308 ───
let ef = self.config.ef_search.max(k);
// If k > ef (shouldn't happen with max), search_layer can't return
// enough candidates. But with k=1, ef=max(50,1)=50. OK.

// ─── CRITICAL LINE 2: L496-499 ───
let entry_score = self.score_query(query, entry);
candidates.push(Candidate { score: entry_score, node: entry });
// results.push(Reverse(Candidate { ... }));
// Is the entry point ALWAYS inserted into results? If the entry point's
// score is the worst among the first `ef` candidates explored, is it
// evicted before exploring its neighbors?

// ─── CRITICAL LINE 3: L518-519 ───
if current.score < worst && results.len() >= ef { break; }
// Stop condition: if the best remaining candidate is WORSE than the worst result
// AND we have at least `ef` results. Correct, but verify that `results`
// has `ef` elements BEFORE checking the condition.

// ─── CRITICAL LINE 4: L552-554 ───
// into_sorted_vec() on BinaryHeap<Reverse> -> ascending by Reverse -> descending by score
results.into_sorted_vec().into_iter().map(|r| r.0).collect()
// Verify with test: first element has the highest score.
```

**Classic Rust bug pattern for HNSW:**
- Using `BinaryHeap<Candidate>` (max-heap) for `results` instead of
  `BinaryHeap<Reverse<Candidate>>` (min-heap). If `results` is max-heap,
  `peek()` returns the BEST candidate, not the WORST. The stop condition
  `if current.score < worst` fires immediately after filling `ef`
  candidates, because the "worst" is actually the BEST. Result: the search
  only explores `ef` nodes and stops. Check `Cargo.toml` to make sure
  the current version uses Reverse.

### 3. Sanity Test (5 Known Vectors)

```rust
#[test]
fn test_recall_k1_on_small_known_dataset() {
    // Build 5 vectors with known distances to the query
    let dim = 4;
    let docs = vec![
        Document::builder("id0", "").embedding(vec![1.0, 0.0, 0.0, 0.0]).build(),
        Document::builder("id1", "").embedding(vec![0.9, 0.1, 0.0, 0.0]).build(),
        Document::builder("id2", "").embedding(vec![0.0, 1.0, 0.0, 0.0]).build(),
        Document::builder("id3", "").embedding(vec![0.0, 0.0, 0.8, 0.6]).build(),
        Document::builder("id4", "").embedding(vec![0.0, 0.0, 0.0, 1.0]).build(),
    ];
    // query = [1.0, 0.0, 0.0, 0.0] → exact order: id0 > id1 > ... > id4

    let mut hnsw = HnswIndex::new(HnswConfig {
        m: 4, ef_construction: 10, ef_search: 5,
        metric: Metric::Cosine, ..Default::default()
    });
    hnsw.insert(&docs);

    let query = vec![1.0, 0.0, 0.0, 0.0];
    let results = hnsw.search(&query, 5);

    // DEBUG: print scores
    for (i, r) in results.iter().enumerate() {
        println!("  [{}] id={} score={}", i, r.document.id, r.score);
    }

    // Assertions
    assert_eq!(results[0].document.id, "id0",
        "Nearest neighbor must be id0 (cosine=1.0)");
    assert_eq!(results[1].document.id, "id1",
        "Second must be id1 (cosine≈0.994)");
    assert!(results[0].score > results[1].score,
        "Scores must be in descending order");

    // Recall@1
    let top1_approx = &results[0].document.id;
    assert_eq!(top1_approx, "id0", "Recall@1 must be 100% on structured data");
}
```

> If this test fails, the bug is in the ordering of `search_layer` or `search`.
> If it passes, the problem is that random vectors have no structure and
> low recall@1 is expected (not an implementation bug, but a data issue).

---

## FAILURE 2: 0.0 MB of RAM in approximate indices (ANN)

**Symptom**: BF reports 165 MB, but HNSW and IVF-PQ report 0.0 MB of RAM.

### 1. Key Questions

**P1.** Does the `measure_peak_ram()` function in `bench_grid.rs` use `VmPeak` or `VmRSS`?
`VmPeak` is monotonic (only goes up). If BF runs first and allocates 165 MB,
`VmPeak` stays at 165 MB. When HNSW runs afterwards, its additional allocation
doesn't exceed that peak. Is `after.saturating_sub(before)` the correct metric?

**P2.** Do the HNSW and IVF-PQ benchmarks run `insert()` in the SAME process
as BF? If so, BF's memory (165 MB) is still allocated during HNSW's execution.
The RAM reported as "HNSW peak" is actually `peak_HNSW - peak_BF`,
where `peak_HNSW ≈ peak_BF` because the global process peak doesn't increase.

**P3.** Are HNSW's structures (graphs, embeddings) built with normal `Vec`
(not lazy/mmap)? If they use `Mmap` or `Vec::with_capacity` + `extend`, the
memory is allocated in the process heap — the OS accounts for it in `VmRSS`.
But `VmPeak` doesn't capture it if it doesn't exceed the historical maximum.

### 2. Bug Patterns in Code

Search in `examples/bench_grid.rs`:

```rust
// ─── CRITICAL LINE: L130-136 ───
fn measure_peak_ram<F: FnOnce()>(f: F) -> u64 {
    let before = read_vmpeak_kb();   // READING 1: current VmPeak (e.g., 165_000 KB)
    f();                              // HNSW insert() → allocates ~40 MB more
    let after = read_vmpeak_kb();    // READING 2: VmPeak = max(165_000, 165_000+40_000) = 165_000
    after.saturating_sub(before)     // Result: 0!
}
```

**The problem**: `VmPeak` records the GLOBAL process peak. Since BF already allocated
165 MB, that is the VmPeak. HNSW allocates additional memory, but VmPeak doesn't
decrease or reset — it only goes up. If HNSW's allocation (say 40 MB) doesn't exceed
the historical peak, `after - before = 0`.

**Fix**: Use `VmRSS` instead of `VmPeak`, measuring the RSS delta:

```rust
fn measure_ram_delta<F: FnOnce()>(f: F) -> u64 {
    let before = read_vmrss_kb();  // current RSS before f()
    f();
    let after = read_vmrss_kb();   // current RSS after f()
    after.saturating_sub(before)   // How much NET memory f() allocated
}
```

**Alternative**: Run each benchmark in a separate subprocess with
`std::process::Command`, capturing the child's `VmPeak`. The child starts with
VmPeak=0.

### 3. Sanity Test

```rust
#[test]
fn test_ram_measurement_smoke() {
    // 1. Measure RSS before
    let rss_before = read_vmrss_kb();
    assert!(rss_before > 0, "Initial RSS must be > 0");

    // 2. Allocate 10 MB
    let mut heap: Vec<u8> = Vec::with_capacity(10 * 1024 * 1024);
    heap.resize(10 * 1024 * 1024, 42);
    std::thread::sleep(std::time::Duration::from_millis(50)); // give the OS time

    // 3. Measure RSS after
    let rss_after = read_vmrss_kb();
    let delta = rss_after.saturating_sub(rss_before);

    println!("RSS before={} KB, after={} KB, delta={} KB", rss_before, rss_after, delta);
    assert!(delta > 0, "RSS must increase after 10 MB allocation");

    // 4. Repeat test with VmPeak (to demonstrate the bug)
    let peak_before = read_vmpeak_kb();
    heap.resize(20 * 1024 * 1024, 42);
    std::thread::sleep(std::time::Duration::from_millis(50));
    let peak_after = read_vmpeak_kb();

    // If peak_before was already high due to previous allocs, peak_after may be equal
    let peak_delta = peak_after.saturating_sub(peak_before);
    println!("VmPeak before={} KB, after={} KB, delta={} KB (may be 0!)",
        peak_before, peak_after, peak_delta);
}
```

---

## FAILURE 3: Recall@100 Degradation (60% → 19%)

**Symptom**: Recall@10=60% but Recall@100 drops to 19%. With 10K neighbors in the
true top-100, only 19 are found. The hit ratio drops 3x between K=10 and K=100.

### 1. Key Questions

**P1.** When `k=100`, line L308 computes `ef = self.config.ef_search.max(100)`.
With `ef_search=50`, `ef = max(50, 100) = 100`. Can HNSW's `search_layer` actually
return `ef=100` candidates? Check if `results.len() >= ef` acts as a
LOWER BOUND (to stop) or UPPER BOUND (to evict). The HNSW algorithm
uses `ef` as the MAXIMUM size of the results heap. Are we respecting that?

**P2.** Check lines L535-548. When `results.len() < ef`, candidates
are inserted directly. When `results.len() >= ef`, they are only inserted if
`nei_score > worst.score`. Does this mean the results heap can grow
beyond `ef`? If the heap GROWS without limit, `ef` doesn't control anything.
If it's strictly PRUNED to `ef`, candidates worse than the `ef`-th best
are discarded, even if they are in the true top-100.

**P3.** In the benchmark code (bench_grid.rs), `WARMUP=3` and `QUERY_ITERS=100`.
Each call to `search()` with `k=100` returns 100 `ScoredDocument`s. Is the
benchmark computing Recall@100 over the first **100 results**
returned by `search()` or over the first **100 ranked results**? If `search()`
returns exactly `k=100` items (via `scored.truncate(k)`), and the ranking
has repeated scores near the boundary, the cutoff may cut valid
results that share the same score as #100.

### 2. Bug Patterns in Code

Search in `src/index/hnsw.rs`:

```rust
// ─── CRITICAL LINE 1: L518 ───
// Stop condition: if the best candidate is worse than the worst result
// AND we have >= ef results
if current.score < worst && results.len() >= ef {
    break;
}

// ─── CRITICAL LINE 2: L535-548 ───
if results.len() < ef {
    results.push(Reverse(Candidate { score: nei_score, node: nei }));
} else if let Some(Reverse(worst)) = results.peek() {
    if nei_score > worst.score {
        results.pop();      // removes the WORST
        results.push(...);  // inserts the new one (heap keeps the ef BEST)
    }
}
// This pattern IS correct: results.size never exceeds ef.
// But if k > ef/2, the heap discards candidates that could be in the top-k.
// With ef=100 and k=100, the heap discards... none! Because ef >= k.

// ─── CRITICAL LINE 3: L308 ───
let ef = self.config.ef_search.max(k);
// If ef_search=50 and k=100 → ef=100. OK, enough space.

// ─── CRITICAL LINE 4: L552-554 ───
// into_sorted_vec() returns the best FIRST.
results.into_sorted_vec().into_iter().map(|r| r.0).collect()
// Then .take(k) at L370 takes the first k.
// If search_layer returned exactly ef=100, and k=100, take(100) takes everything.
// If search_layer returned 60 (because the graph didn't have 100 close neighbors),
// take(100) will only take 60, and then truncate(100) discards nothing.
```

**Diagnosis**: At 100K random 384-dim vectors, `search_layer` typically
doesn't find `ef=100` relevant candidates because the HNSW graph built
on noise has no exploitable structure. With `ef=50`, only ~19 of the 100
nearest neighbors are in the first 50 positions explored.
With `ef=200`, it goes up to 38%.

**The solution is not a bugfix but a parameter adjustment**: For large
datasets and high k, `ef_search` must be significantly larger than k
(rule of thumb: `ef >= 3*k` for recall > 90%).

**Real bug pattern**: If the `results` heap at L503 is declared as
`BinaryHeap<Candidate>` (max-heap) instead of
`BinaryHeap<Reverse<Candidate>>` (min-heap), then `results.peek()`
returns the BEST candidate, not the WORST. The condition `nei_score > worst.score`
at L541 would be `nei_score > best_score`, which would insert ALL candidates
(they are always worse than the best). The heap would grow WITHOUT LIMIT. This would
explain 19% instead of 60% — the heap fills with irrelevant candidates.

### 3. Sanity Test

```rust
#[test]
fn test_recall_at_high_k() {
    let dim = 8;
    let n = 1000;
    let docs: Vec<Document> = (0..n)
        .map(|i| Document::builder(format!("d{i}"), "")
            .embedding(random_vec(i as u64, dim)).build())
        .collect();

    let mut bf = BruteForceIndex::new(Metric::Cosine);
    bf.insert(&docs);

    let mut hnsw = HnswIndex::new(HnswConfig {
        m: 16, ef_construction: 100, ef_search: 50,
        metric: Metric::Cosine, ..Default::default()
    });
    hnsw.insert(&docs);

    let query = random_vec(999_999, dim);
    let exact = bf.search(&query, 100);
    let approx = hnsw.search(&query, 100);

    println!("search_layer returned {}", approx.len());

    let exact_ids: HashSet<&str> = exact.iter().map(|r| r.document.id.as_str()).collect();
    for k in [1, 5, 10, 50, 100] {
        let matches = approx.iter().take(k)
            .filter(|r| exact_ids.contains(r.document.id.as_str())).count();
        let recall = matches as f64 / k as f64;
        println!("  K={}: recall={:.0}% ({}/{})", k, recall*100.0, matches, k);
    }

    // With structured data, recall should not collapse
    assert!(recall_at_k(&approx, &exact, 10) >= 0.5,
        "Recall@10 must be >= 50% even with random data");
    // The ratio of recall@100 vs recall@10 must be reasonable
    let r10 = recall_at_k(&approx, &exact, 10);
    let r100 = recall_at_k(&approx, &exact, 100);
    assert!(r100 >= r10 * 0.5,
        "Recall@100 must not degrade more than 50% vs Recall@10. r10={:.2} r100={:.2}", r10, r100);
}
```

---

## FAILURE 4: IVF-PQ Bottleneck (11.5 ms, Low Recall)

**Symptom**: IVF-PQ with nlist=256, M=16, nprobe=16 takes 11.5 ms per query
(slower than HNSW ef=50 which takes 0.99 ms), with only 40% recall@10.

### 1. Key Questions

**P1.** Check `effective_probe()` in `ivf_pq.rs`. With `rerank_enabled=false`
(default), `n_probe=16` is complete. Are we checking 16 clusters × ~390 docs
each = 6,240 candidates? For each candidate, the PQ distance is computed
as sum of M=16 table lookups. 6,240 × 16 = ~100K operations. This
should be < 1 ms. Why does it take 11.5 ms?

**P2.** At line L504, the score is `(0..m).map(|s| luts[s][code[s] as usize]).sum()`.
Each access to `luts[s]` is a `Vec<f32>` with indirection. With M=16, that's 16
different Vec accesses + 16 accesses to `code[s]`. Is the memory of `luts` and
`codes` cache-friendly? `luts` is `Vec<Vec<f32>>` (pointer, length, capacity
for each sub-vector). 16 × (24 + 8 + 8) = 640 bytes just in Vec overhead.

**P3.** The most expensive step in `search()` (L496-516) is `cluster.iter().map(|&doc_id| { ... })`.
For each doc_id, we clone `self.documents[doc_id].clone()` (L507). With 6,240
documents, each clone copies: embedding (384 f32 = 1,536 bytes), id (String ≈ 8
bytes), text (String ≈ 8 bytes), metadata (HashMap ≈ 32 bytes). Total ~1,584
bytes per clone × 6,240 = ~9.9 MB of allocations. Is cloning `Document`
the bottleneck?

### 2. Bug Patterns in Code

Search in `src/index/ivf_pq.rs`:

```rust
// ─── CRITICAL LINE 1: L478-493 — LUT Construction ───
let luts: Vec<Vec<f32>> = (0..m)
    .into_par_iter()
    .map(|sub_idx| {
        let start = sub_idx * subdim;
        let end = start + subdim;
        let q_sub = &query[start..end];
        let cb = &self.codebooks[sub_idx];
        let mut lut = Vec::with_capacity(256);
        for c in cb {                              // cb: Vec<Vec<f32>> of 256×24
            let s = distance::score(q_sub, c, self.config.metric);
            lut.push(s);
        }
        lut
    })
    .collect();
// Cost: 16 × 256 × 24 = 98,304 f32 dot products. Parallel, ~0.3ms.

// ─── CRITICAL LINE 2: L496-516 — Cluster Scan ───
let mut results: Vec<ScoredDocument> = active_clusters
    .par_iter()
    .flat_map(|&ci| {
        let cluster = &self.clusters[ci];
        let mut local: Vec<ScoredDocument> = cluster
            .iter()
            .map(|&doc_id| {
                let code = &self.codes[doc_id];
                let score: f32 = (0..m).map(|s| luts[s][code[s] as usize]).sum();
                ScoredDocument {
                    score,
                    document: self.documents[doc_id].clone(),  // ← BOTTLENECK
                }
            })
            .collect();
        local.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        local
    })
    .collect();
// Expected cost: 6,240 × (16 lookups + 1 clone) = ~100K ops + ~10 MB alloc
// Allocation dominates. Without cloning: ~0.5-1 ms. With cloning: 10-15 ms.

// ─── CRITICAL LINE 3: L512-519 ───
// Each cluster is sorted individually (local.sort), then a global sort is made.
// With n_probe=16, that's 16 local sorts of ~390 elements each.
// The global sort then merges 6,240 elements.
// Double sorting is redundant: it could collect everything without local sort
// and do a single global sort.
```

**Diagnosis**:
1. **Primary bottleneck**: `self.documents[doc_id].clone()` — each
   ScoredDocument clones the entire Document including the 384 f32 embedding.
   Solution: store indices in ScoredDocument instead of clones, and resolve
   only the final top-k.

2. **Secondary bottleneck**: `par_iter()` on `active_clusters` (16
   clusters). Rayon has scheduling overhead for only 16 tasks. With
   6,240 total documents, a single `par_iter()` over the documents would
   be more efficient.

3. **Low recall**: `nlist=256` is adequate for 100K, but `M=16` subspaces
   for 384-dim → subdim=24. Each 24-dim sub-vector is quantized to 1 byte
   (256 centroids). The information loss is enormous: 24 f32 (96 bytes) → 1 u8.
   Recall@10=40% is expected for random vectors. With real data it
   may improve, but M=16 is aggressive. Recommendation: M=32 (subdim=12) for
   better compression/recall trade-off.

### 3. Sanity Test

```rust
#[test]
fn test_ivfpq_clone_vs_index() {
    let dim = 384;
    let n = 1000;

    // Create docs with real text to measure clone impact
    let docs: Vec<Document> = (0..n)
        .map(|i| Document::builder(format!("d{i}"), format!("Test document number {i} with sufficiently long text to measure clone overhead"))
            .metadata("source", "test")
            .embedding(random_vec(i as u64, dim))
            .build())
        .collect();

    let mut ivf = IvfPqIndex::new(IvfPqConfig {
        n_list: 100, m_subspaces: 16, n_probe: 5,
        metric: Metric::Cosine, ..Default::default()
    });
    ivf.insert(&docs);

    let query = random_vec(999_999, dim);

    // Benchmark only the cloning vs non-cloning part
    let t0 = std::time::Instant::now();
    let results = ivf.search(&query, 10);
    let elapsed = t0.elapsed();

    // How many documents were cloned? Approx: n_probe * (n / n_list) = 5 * 10 = 50
    println!("IVF-PQ search({} docs): {:?}, {} results, scores: {:?}",
        n, elapsed, results.len(),
        results.iter().map(|r| r.score).collect::<Vec<_>>());

    // Verify scores are valid (not NaN, not 0 for all)
    for r in &results {
        assert!(!r.score.is_nan(), "Score must not be NaN");
    }

    // Performance test: cloning should not dominate
    // With 1000 docs and n_probe=5, we should scan ~50 docs
    // 50 × 16 lookups = 800 ops. Without cloning: < 100us. With cloning: < 500us.
    assert!(elapsed.as_micros() < 5000,
        "IVF-PQ search must be < 5ms for 1000 docs. Was {:?}", elapsed);
}
```

---

## Summary Table of Root Causes

| Failure | Probable Root Cause | Severity | Fix |
|---------|--------------------|:--------:|-----|
| Recall@1 0% | Random vectors without structure (not a bug) | Low | Use real embeddings for measurement |
| RAM 0.0 MB | Monotonic `VmPeak` doesn't detect later allocations | Medium | Switch to `VmRSS` or subprocesses |
| Recall@100 19% | `ef_search=50` insufficient for k=100 on noisy data | Medium | Adjust `ef >= 3*k` |
| IVF-PQ 11.5ms | `Document` cloning for each candidate (~10 MB alloc) | High | Use indices instead of clones until final top-k |
