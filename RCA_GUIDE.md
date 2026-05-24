# Guia de Auditoria de Codigo — RCA de Benchmark (100K 384-dim Cosine)

> **Contexto**: Resultados del grid benchmark que muestran 4 anomalias.
> **Objetivo**: Aislar la causa raiz de cada falla mediante preguntas de diagnostico,
>   busqueda de patrones en codigo, y pruebas unitarias de sanidad.

---

## FALLA 1: Recall@1 = 0% con Recall@10 = 60-80%

**Sintoma**: El vecino mas cercano exacto NUNCA aparece en la posicion #1 de los
resultados aproximados, pero 6-8 de los top-10 exactos SI aparecen en el top-10
aproximado.

### 1. Preguntas Clave

**P1.** En la funcion `search_layer()` (hnsw.rs ~L489), la estructura `results` es
`BinaryHeap<Reverse<Candidate>>`. Cuando llamas a `into_sorted_vec()`, los elementos
se devuelven en orden ascendente de `Reverse<Candidate>`. ¿Estas interpretando
"ascendente por Reverse" como "descendente por score" (correcto) o como "ascendente
por score" (incorrecto)? El comentario en L552-553 afirma lo primero. Verifica
con un test de 2 vectores.

**P2.** En `search()` (~L368-375), los candidatos pasan por `.take(k)` y luego
`scored.sort_by(|a, b| b.score.partial_cmp(&a.score))`. ¿Es posible que `take(k)`
trunque antes de ordenar, descartando el verdadero vecino #1 que estaba mas abajo
en la lista? Los resultados de `search_layer` ya vienen ordenados best-first,
pero si `ef > k`, solo `k` elementos sobreviven al `.take()`.

**P3.** Cuando `k=1`, la linea 308 calcula `ef = self.config.ef_search.max(1)`.
¿`search_layer` con `ef=50` siempre incluye al entry point en los resultados?
¿O existe un caso donde `entry_score` es tan baja que el entry point es
expulsado del heap de `results` antes de explorar sus vecinos?

### 2. Patrones de Bug en Codigo

Busca en `src/index/hnsw.rs`:

```rust
// ─── LINEA CRITICA 1: L308 ───
let ef = self.config.ef_search.max(k);
// Si k > ef (no deberia pasar con max), search_layer no puede devolver
// suficientes candidatos. Pero con k=1, ef=max(50,1)=50. OK.

// ─── LINEA CRITICA 2: L496-499 ───
let entry_score = self.score_query(query, entry);
candidates.push(Candidate { score: entry_score, node: entry });
// results.push(Reverse(Candidate { ... }));
// ¿El entry point SIEMPRE se inserta en results? Si la puntuacion del entry
// point es la peor entre los primeros `ef` candidatos explorados, ¿se
// expulsa antes de explorar sus vecinos?

// ─── LINEA CRITICA 3: L518-519 ───
if current.score < worst && results.len() >= ef { break; }
// Stop condition: si el mejor candidato restante es PEOR que el peor resultado
// Y tenemos al menos `ef` resultados. Es correcto, pero verifica que `results`
// tenga `ef` elementos ANTES de checkear la condicion.

// ─── LINEA CRITICA 4: L552-554 ───
// into_sorted_vec() on BinaryHeap<Reverse> -> ascending by Reverse -> descending by score
results.into_sorted_vec().into_iter().map(|r| r.0).collect()
// Verificar con test: primer elemento tiene mayor score.
```

**Patron de bug clasico en Rust para HNSW:**
- Usar `BinaryHeap<Candidate>` (max-heap) para `results` en lugar de
  `BinaryHeap<Reverse<Candidate>>` (min-heap). Si `results` es max-heap,
  `peek()` devuelve el MEjor candidato, no el PEOR. La condicion de parada
  `if current.score < worst` se dispara inmediatamente despues de llenar `ef`
  candidatos, porque el "worst" es en realidad el BEST. Resultado: la busqueda
  solo explora `ef` nodos y se detiene. Mira `Cargo.toml` para asegurarte
  de que la version actual usa Reverse.

### 3. Prueba de Sanidad (5 Vectores Conocidos)

```rust
#[test]
fn test_recall_k1_on_small_known_dataset() {
    // Construir 5 vectores con distancias conocidas al query
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

    // DEBUG: imprimir scores
    for (i, r) in results.iter().enumerate() {
        println!("  [{}] id={} score={}", i, r.document.id, r.score);
    }

    // Afirmaciones
    assert_eq!(results[0].document.id, "id0",
        "El vecino mas cercano debe ser id0 (cosine=1.0)");
    assert_eq!(results[1].document.id, "id1",
        "El segundo debe ser id1 (cosine≈0.994)");
    assert!(results[0].score > results[1].score,
        "Scores deben estar en orden descendente");

    // Recall@1
    let top1_approx = &results[0].document.id;
    assert_eq!(top1_approx, "id0", "Recall@1 debe ser 100% en datos estructurados");
}
```

> Si este test falla, el bug esta en el ordenamiento de `search_layer` o `search`.
> Si pasa, el problema es que los vectores aleatorios no tienen estructura y el
> recall@1 bajo es esperable (no es bug de implementacion, sino de datos).

---

## FALLA 2: 0.0 MB de RAM en indices aproximados (ANN)

**Sintoma**: BF reporta 165 MB, pero HNSW e IVF-PQ reportan 0.0 MB de RAM.

### 1. Preguntas Clave

**P1.** ¿La funcion `measure_peak_ram()` en `bench_grid.rs` usa `VmPeak` o `VmRSS`?
`VmPeak` es monotonico (solo sube). Si BF se ejecuta primero y asigna 165 MB,
`VmPeak` queda en 165 MB. Cuando HNSW se ejecuta despues, su asignacion adicional
no supera ese pico. ¿Es `after.saturating_sub(before)` la metrica correcta?

**P2.** ¿Los benchmarks de HNSW e IVF-PQ ejecutan `insert()` en el MISMO proceso
que BF? Si es asi, la memoria de BF (165 MB) sigue asignada durante la ejecucion
de HNSW. La RAM reportada como "pico de HNSW" es realmente `pico_HNSW - pico_BF`,
donde `pico_HNSW ≈ pico_BF` porque el pico global del proceso no sube.

**P3.** ¿Las estructuras de HNSW (grafos, embeddings) se construyen con `Vec` normal
(no lazy/mmap)? Si usan `Mmap` o `Vec::with_capacity` + `extend`, la memoria
esta asignada en el monticulo del proceso — el SO la contabiliza en `VmRSS`.
Pero `VmPeak` no la captura si no supera el maximo historico.

### 2. Patrones de Bug en Codigo

Busca en `examples/bench_grid.rs`:

```rust
// ─── LINEA CRITICA: L130-136 ───
fn measure_peak_ram<F: FnOnce()>(f: F) -> u64 {
    let before = read_vmpeak_kb();   // LECTURA 1: VmPeak actual (p.ej. 165_000 KB)
    f();                              // HNSW insert() → asigna ~40 MB mas
    let after = read_vmpeak_kb();    // LECTURA 2: VmPeak = max(165_000, 165_000+40_000) = 165_000
    after.saturating_sub(before)     // Resultado: 0!
}
```

**El problema**: `VmPeak` registra el pico GLOBAL del proceso. Como BF ya asigno
165 MB, ese es el VmPeak. HNSW asigna memoria adicional, pero VmPeak no baja
ni se resetea — solo sube. Si la asignacion de HNSW (digamos 40 MB) no supera
el pico historico, `after - before = 0`.

**Fix**: Usar `VmRSS` en lugar de `VmPeak`, midiendo el delta de RSS:

```rust
fn measure_ram_delta<F: FnOnce()>(f: F) -> u64 {
    let before = read_vmrss_kb();  // RSS actual antes de f()
    f();
    let after = read_vmrss_kb();   // RSS actual despues de f()
    after.saturating_sub(before)   // Cuanto asigno NETO f()
}
```

**Alternativa**: Ejecutar cada benchmark en un subproceso separado con
`std::process::Command`, capturando `VmPeak` del child. El child empieza con
VmPeak=0.

### 3. Prueba de Sanidad

```rust
#[test]
fn test_ram_measurement_smoke() {
    // 1. Mide RSS antes
    let rss_before = read_vmrss_kb();
    assert!(rss_before > 0, "RSS inicial debe ser > 0");

    // 2. Asigna 10 MB
    let mut heap: Vec<u8> = Vec::with_capacity(10 * 1024 * 1024);
    heap.resize(10 * 1024 * 1024, 42);
    std::thread::sleep(std::time::Duration::from_millis(50)); // dar tiempo al SO

    // 3. Mide RSS despues
    let rss_after = read_vmrss_kb();
    let delta = rss_after.saturating_sub(rss_before);

    println!("RSS before={} KB, after={} KB, delta={} KB", rss_before, rss_after, delta);
    assert!(delta > 0, "RSS debe aumentar despues de alloc de 10 MB");

    // 4. Repite la prueba con VmPeak (para demostrar el bug)
    let peak_before = read_vmpeak_kb();
    heap.resize(20 * 1024 * 1024, 42);
    std::thread::sleep(std::time::Duration::from_millis(50));
    let peak_after = read_vmpeak_kb();

    // Si peak_before ya era alto por allocs anteriores, peak_after puede ser igual
    let peak_delta = peak_after.saturating_sub(peak_before);
    println!("VmPeak before={} KB, after={} KB, delta={} KB (puede ser 0!)",
        peak_before, peak_after, peak_delta);
}
```

---

## FALLA 3: Degradacion de Recall@100 (60% → 19%)

**Sintoma**: Recall@10=60% pero Recall@100 cae a 19%. Con 10K vecinos en el
top-100 verdadero, solo se encuentran 19. La proporcion de aciertos
cae 3x entre K=10 y K=100.

### 1. Preguntas Clave

**P1.** Cuando `k=100`, la linea L308 calcula `ef = self.config.ef_search.max(100)`.
Con `ef_search=50`, `ef = max(50, 100) = 100`. ¿El `search_layer` de HNSW realmente
puede devolver `ef=100` candidatos? Revisa si `results.len() >= ef` actua como
LIMITE INFERIOR (para detener) o SUPERIOR (para expulsar). El algoritmo HNSW
usa `ef` como el tamano MAXIMO del heap de resultados. ¿Lo estamos respetando?

**P2.** Revisa las lineas L535-548. Cuando `results.len() < ef`, los candidatos
se insertan directamente. Cuando `results.len() >= ef`, solo se insertan si
`nei_score > worst.score`. ¿Esto significa que el heap de resultados puede
crecer mas alla de `ef`? Si el heap CRECE sin limite, `ef` no controla nada.
Si se PODA estrictamente a `ef`, los candidatos peores que el `ef`-esimo mejor
se descartan, incluso si estan en el top-100 verdadero.

**P3.** En el codigo de benchmark (bench_grid.rs), `WARMUP=3` y `QUERY_ITERS=100`.
Cada llamada a `search()` con `k=100` devuelve 100 `ScoredDocument`s. ¿El
benchmark esta computando Recall@100 sobre los primeros **100 resultados**
devolvuelto por `search()` o sobre los primeros **100 del ranking**? Si `search()`
devuelve exactamente `k=100` items (por `scored.truncate(k)`), y el ranking
tiene scores repetidos cerca del borde, el cutoff puede cortar resultados
validos que comparten el mismo score que el #100.

### 2. Patrones de Bug en Codigo

Busca en `src/index/hnsw.rs`:

```rust
// ─── LINEA CRITICA 1: L518 ───
// Stop condition: si el mejor candidato es peor que el peor resultado
// Y tenemos >= ef resultados
if current.score < worst && results.len() >= ef {
    break;
}

// ─── LINEA CRITICA 2: L535-548 ───
if results.len() < ef {
    results.push(Reverse(Candidate { score: nei_score, node: nei }));
} else if let Some(Reverse(worst)) = results.peek() {
    if nei_score > worst.score {
        results.pop();      // elimina el PEOR
        results.push(...);  // inserta el nuevo (el heap mantiene los ef MEJORES)
    }
}
// Este patron ES correcto: results.size nunca excede ef.
// Pero si k > ef/2, el heap descarta candidatos que podrian estar en el top-k.
// Con ef=100 y k=100, el heap descarta... ninguno! Porque ef >= k.

// ─── LINEA CRITICA 3: L308 ───
let ef = self.config.ef_search.max(k);
// Si ef_search=50 y k=100 → ef=100. OK, suficiente espacio.

// ─── LINEA CRITICA 4: L552-554 ───
// into_sorted_vec() devuelve los mejores PRIMERO.
results.into_sorted_vec().into_iter().map(|r| r.0).collect()
// Luego .take(k) en L370 toma los primeros k.
// Si search_layer devolvio exactamente ef=100, y k=100, take(100) toma todo.
// Si search_layer devolvio 60 (porque el grafo no tenia 100 vecinos cerca),
// take(100) solo tomara 60, y luego truncate(100) no descarta nada.
```

**Diagnostico**: A 100K vectores aleatorios 384-dim, el `search_layer` tipicamente
no encuentra `ef=100` candidatos relevantes porque el grafo HNSW construido
sobre ruido no tiene estructura explotable. Con `ef=50`, solo ~19 de los 100
vecinos mas cercanos estan en las primeras 50 posiciones exploradas.
Con `ef=200`, sube a 38%.

**La solucion no es un bugfix sino un ajuste de parametro**: Para datasets
grandes y k alto, `ef_search` debe ser significativamente mayor que k
(regla practica: `ef >= 3*k` para recall > 90%).

**Patron de bug real**: Si el heap `results` en L503 se declara como
`BinaryHeap<Candidate>` (max-heap) en lugar de
`BinaryHeap<Reverse<Candidate>>` (min-heap), entonces `results.peek()`
devuelve el MEJOR candidato, no el PEOR. La condicion `nei_score > worst.score`
en L541 seria `nei_score > best_score`, que insertaria TODOS los candidatos
(siempre son peores que el mejor). El heap creceria SIN LIMITE. Esto explicaria
19% en lugar de 60% — el heap se llena de candidato irrelevantes.

### 3. Prueba de Sanidad

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

    // Con datos estructurados, recall no debe colapsar
    assert!(recall_at_k(&approx, &exact, 10) >= 0.5,
        "Recall@10 debe ser >= 50% incluso con datos aleatorios");
    // La proporcion de recall@100 vs recall@10 debe ser razonable
    let r10 = recall_at_k(&approx, &exact, 10);
    let r100 = recall_at_k(&approx, &exact, 100);
    assert!(r100 >= r10 * 0.5,
        "Recall@100 no debe degradarse mas de 50% vs Recall@10. r10={:.2} r100={:.2}", r10, r100);
}
```

---

## FALLA 4: IVF-PQ Cuello de Botella (11.5 ms, bajo Recall)

**Sintoma**: IVF-PQ con nlist=256, M=16, nprobe=16 toma 11.5 ms por query
(mas lento que HNSW ef=50 que toma 0.99 ms), con solo 40% recall@10.

### 1. Preguntas Clave

**P1.** Revisa `effective_probe()` en `ivf_pq.rs`. Con `rerank_enabled=false`
(default), `n_probe=16` completa. ¿Estamos revisando 16 clusters × ~390 docs
cada uno = 6,240 candidatos? Para cada candidato, la distancia PQ se calcula
como suma de M=16 lookups en tabla. 6,240 × 16 = ~100K operaciones. Esto
deberia ser < 1 ms. ¿Por que toma 11.5 ms?

**P2.** En la linea L504, el score es `(0..m).map(|s| luts[s][code[s] as usize]).sum()`.
Cada acceso a `luts[s]` es un `Vec<f32>` con indireccion. Con M=16, son 16
accesos a Vec distintos + 16 accesos a `code[s]`. ¿La memoria de `luts` y
`codes` es cache-friendly? `luts` es `Vec<Vec<f32>>` (puntero, length, capacity
para cada sub-vector). 16 × (24 + 8 + 8) = 640 bytes solo de overhead de Vec.

**P3.** El paso mas costoso en `search()` (L496-516) es `cluster.iter().map(|&doc_id| { ... })`.
Por cada doc_id, clonamos `self.documents[doc_id].clone()` (L507). Con 6,240
documentos, cada clon copia: embedding (384 f32 = 1,536 bytes), id (String ≈ 8
bytes), texto (String ≈ 8 bytes), metadata (HashMap ≈ 32 bytes). Total ~1,584
bytes por clon × 6,240 = ~9.9 MB de allocaciones. ¿Es la clonacion de Document
el cuello de botella?

### 2. Patrones de Bug en Codigo

Busca en `src/index/ivf_pq.rs`:

```rust
// ─── LINEA CRITICA 1: L478-493 — Construccion de LUT ───
let luts: Vec<Vec<f32>> = (0..m)
    .into_par_iter()
    .map(|sub_idx| {
        let start = sub_idx * subdim;
        let end = start + subdim;
        let q_sub = &query[start..end];
        let cb = &self.codebooks[sub_idx];
        let mut lut = Vec::with_capacity(256);
        for c in cb {                              // cb: Vec<Vec<f32>> de 256×24
            let s = distance::score(q_sub, c, self.config.metric);
            lut.push(s);
        }
        lut
    })
    .collect();
// Costo: 16 × 256 × 24 = 98,304 productos punto f32. Paralelo, ~0.3ms.

// ─── LINEA CRITICA 2: L496-516 — Escaneo de clusters ───
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
// Costo esperado: 6,240 × (16 lookups + 1 clone) = ~100K op + ~10 MB alloc
// La allocacion domina. Sin clonacion: ~0.5-1 ms. Con clonacion: 10-15 ms.

// ─── LINEA CRITICA 3: L512-519 ───
// Se ordena CADA cluster individualmente (local.sort), luego se hace global sort.
// Con n_probe=16, son 16 sorts locales de ~390 elementos cada uno.
// El global sort luego mezcla 6,240 elementos.
// El sorting doble es redundante: podria recogerse todo sin orden local
// y hacer un unico global sort.
```

**Diagnostico**:
1. **Cuello de botella primario**: `self.documents[doc_id].clone()` — cada
   ScoredDocument clona el Document completo incluyendo embedding de 384 f32.
   Solucion: almacenar indices en ScoredDocument en lugar de clones, y resolver
   solo los top-k finales.

2. **Cuello de botella secundario**: `par_iter()` en `active_clusters` (16
   clusters). Rayon tiene overhead de scheduling para solo 16 tareas. Con
   6,240 documentos total, un solo `par_iter()` sobre los documentos seria
   mas eficiente.

3. **Bajo recall**: `nlist=256` es adecuado para 100K, pero `M=16` subespacios
   para 384-dim → subdim=24. Cada subvector de 24 dims se cuantiza a 1 byte
   (256 centroides). La perdida de informacion es enorme: 24 f32 (96 bytes) → 1 u8.
   El recall@10=40% es esperable para vectores aleatorios. Con datos reales
   puede mejorar, pero M=16 es agresivo. Recomendacion: M=32 (subdim=12) para
   mejor relacion compression/recall.

### 3. Prueba de Sanidad

```rust
#[test]
fn test_ivfpq_clone_vs_index() {
    let dim = 384;
    let n = 1000;

    // Crear docs con texto real para medir impacto de clone
    let docs: Vec<Document> = (0..n)
        .map(|i| Document::builder(format!("d{i}"), format!("Documento de prueba numero {i} con texto suficientemente largo para medir overhead de clone"))
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

    // Benchmark solo la parte de clonacion vs no clonacion
    let t0 = std::time::Instant::now();
    let results = ivf.search(&query, 10);
    let elapsed = t0.elapsed();

    // Cuantos documentos se clonaron? Aprox: n_probe * (n / n_list) = 5 * 10 = 50
    println!("IVF-PQ search({} docs): {:?}, {} results, scores: {:?}",
        n, elapsed, results.len(),
        results.iter().map(|r| r.score).collect::<Vec<_>>());

    // Verificar scores son validos (no NaN, no 0 para todos)
    for r in &results {
        assert!(!r.score.is_nan(), "Score no debe ser NaN");
    }

    // Test de rendimiento: la clonacion no debe dominar
    // Con 1000 docs y n_probe=5, debemos escanear ~50 docs
    // 50 × 16 lookups = 800 op. Sin clonacion: < 100us. Con clonacion: < 500us.
    assert!(elapsed.as_micros() < 5000,
        "IVF-PQ search debe ser < 5ms para 1000 docs. Fue {:?}", elapsed);
}
```

---

## Tabla Resumen de Causas Raiz

| Falla | Causa Raiz Probable | Severidad | Fix |
|-------|--------------------|-----------|-----|
| Recall@1 0% | Vectores aleatorios sin estructura (no bug) | Baja | Usar embeddings reales para medir |
| RAM 0.0 MB | `VmPeak` monotonico no detecta allocs posteriores | Media | Cambiar a `VmRSS` o subprocesos |
| Recall@100 19% | `ef_search=50` insuficiente para k=100 en datos ruidosos | Media | Ajustar `ef >= 3*k` |
| IVF-PQ 11.5ms | Clonacion de `Document` en cada candidato (~10 MB alloc) | Alta | Usar indices en lugar de clones hasta top-k final |
