# Plan: Arreglar Recall de HNSW+SQ

**Fecha:** 2026-05-17
**Objetivo:** Diagnosticar y corregir el recall 0-60% de HNSW cuando se combina con Scalar Quantization (SQ).

---

## 1. Diagnóstico (Completado)

### Causa Raíz

El scale/bias de cuantización se calcula a partir de **un solo documento** (el primero insertado), en lugar de todo el dataset. Esto produce:

- Primer documento → todo ceros en i8 (cada dimensión: min==max → scale=1.0 → cuantización = 0)
- Documentos siguientes → cuantización con rango pésimo (scale=1.0 sin importar la distribución real)
- Scale/bias nunca se recalculan al añadir más datos

**Contraste con BruteForce:** BF calcula scale/bias de TODOS los docs en el primer batch y re-cuantiza todo, manteniendo recall aceptable.

### Impacto

- `search_layer_i8` navega con distancias i8 corruptas → caminos equivocados en el grafo
- `sq_rescore` no puede recuperar el recall porque los candidatos top ya son incorrectos
- Benchmarks muestran recall 0-60% en HNSW+SQ vs ~95-100% en BF+SQ

---

## 2. Estrategia de Corrección

### Enfoque elegido: Reflejar la estrategia de BruteForce

Mover la cuantización SQ de `insert_one()` a `insert()` como paso posterior al batch, calculando scale/bias de todos los documentos.

**Por qué este enfoque:**
- Ya funciona correctamente en BruteForceIndex (probado)
- No requiere cambiar la topología del grafo (se construye con f32, igual que ahora)
- Mínimo impacto en el resto del código
- Consistente entre backends

---

## 3. Pasos de Implementación

### Paso 1: Mover cuantización SQ de `insert_one()` a `insert()`

**Archivo:** `src/index/hnsw.rs`

**Cambios en `insert()` (método público, ~linea 235):**
- Después del bucle `for doc in docs { self.insert_one(doc.clone()); }`
- Si `sq=true`, calcular scale/bias del total de documentos y re-cuantizar
- Usar `rayon::prelude::par_iter()` para re-cuantización paralela

**Cambios en `insert_one()` (~lineas 372-383):**
- Eliminar el bloque `if self.config.sq { ... }` que calcula scale/bias del primer doc
- Eliminar `self.embedding_i8.push(...)` aquí
- Mantener todo lo demás intacto (construcción del grafo con f32)

### Paso 2: Añadir import de `rayon::prelude::*` en hnsw.rs

### Paso 3: Corregir fórmula de cuantización en `sq.rs` (hallazgo durante impl)

**Archivo:** `src/index/sq.rs`

**Problema:** La función `compute_scale_bias_per_dim` retornaba `(scales, mins)` donde
`bias = min`. Esto mapea `[min, max] → [0, 255]`, pero `i8` solo almacena `[-128, 127]`.
Los valores >127 se saturaban a 127, perdiendo toda la información de valores ≥ punto medio.
Para datos centrados (embeddings normalizados en círculo unitario), esto causaba que
**cualquier valor ≥ 0 se cuantizara como 127**, reduciendo efectivamente la precisión
a 1 bit en la mitad del rango.

**Solución:** Usar el punto medio como bias:
```
bias[d] = (min[d] + max[d]) / 2.0
```
Esto mapea `[min, max] → [-127.5, 127.5]` que encaja perfectamente en `[-128, 127]`.

### Paso 4: Actualizar test de bias en sq.rs

El test `test_compute_per_dim_basic` verificaba `bias == min`. Ahora verifica `bias == mid`.

### Paso 5: Arreglar warning de variable no usada en sq.rs test

`_biases` en `test_compute_per_dim_only_one_doc_has_embedding`.

### Paso 6: Añadir test de recall para HNSW+SQ

**Archivo:** `src/index/hnsw.rs` (sección de tests)

Test que verifique:
1. Insertar N documentos con embeddings distribuidos
2. HNSW+SQ search vs BruteForce search (f32 exacto)
3. Recall > 90% con `sq_rescore=true`

---

## 4. Archivos a Modificar

| Archivo | Cambio |
|---------|--------|
| `src/index/hnsw.rs` | Mover cuantización de `insert_one()` a `insert()` + test nuevo |

Solo un archivo — el cambio es localizado.

---

## 5. Validación

### Tests existentes que deben seguir pasando
```bash
cargo test --lib hnsw  # Tests de HNSW (21 tests)
cargo test --lib sq    # Tests de SQ module (8 tests)
cargo test --lib brute_force  # Tests de BF (incluye SQ)
cargo test             # Full suite (155 tests)
```

### Nuevo test específico
```rust
#[test]
fn test_sq_recall_high() {
    // 1. Crear HNSW con sq=true, sq_rescore=true
    // 2. Insertar 200+ docs con embeddings aleatorios
    // 3. Hacer queries y comparar top-10 con BF exacto
    // 4. Assert recall > 85%
}
```

### Benchmark manual
```bash
cargo run --release --example bench
# Verificar que HNSW+SQ recall sube de 0-60% a >85%
```

---

## 6. Riesgos y Trade-offs

### Riesgo: Performance en inserts batch grandes
Re-cuantizar N documentos después de cada batch tiene costo O(N·d). Pero SQ ya es opcional y el usuario elige usarlo sabiendo el trade-off memoria↔CPU.

### Riesgo: Inserción incremental (un documento a la vez)
Si se insertan documentos de uno en uno, cada `insert()` re-cuantizaría todo el dataset. Esto es O(N·d) por inserción.
- **Mitigación:** La API pública de `Index::insert()` acepta `&[Document]` (batch). Inserción uno-a-uno es un batch de 1. El comportamiento es correcto, solo más lento con SQ activo. Esto es idéntico a BF.

### Trade-off: Consistencia con BF
BF usa escala incremental (primer batch → todos, batches siguientes → scale/bias existente). HNSW podría hacer lo mismo, pero es más complejo y menos correcto. El enfoque de re-cuantizar siempre es más simple y más correcto.

---

## 7. ¿Por qué no usar f32 para navegación y i8 solo para scoring?

El reviewer sugería: construir grafo con f32, navegar con f32, usar i8 solo para ranking final. Esto sería incluso más correcto, pero:
- Requiere mantener f32 + i8 en memoria (~5× en vez de ~4× de ahorro)
- La i8 navigation con scale/bias correcto debe dar recall >90%
- Si después de arreglar el bug el recall sigue bajo, esta es la siguiente optimización

**Decisión:** Primero arreglar el bug obvio, medir recall, y si sigue bajo aplicar esta estrategia.

---

## 8. Orden de Ejecución

1. ✅ Diagnosticar (completado)
2. Editar `insert()` y `insert_one()` en `src/index/hnsw.rs`
3. Añadir test de recall HNSW+SQ
4. `cargo test` — verificar que todo pasa
5. Benchmark opcional para medir mejora de recall
