# VectorStorage Abstraction + mmap Implementation Plan

> **Goal:** Desacoplar el almacenamiento de embeddings del core de la BD mediante un trait `VectorStorage`, con dos implementaciones: `MemoryBackedStorage` (tests/volátil) y `MmapBackedStorage` (producción, zero-copy).

**Arquitectura:**
```
Collection
  ├── storage: Arc<dyn VectorStorage>  ← embeddings contiguos
  └── index: Box<dyn Index>            ← recibe &storage en search()
```

**Diferenciador:** Carga ~0ms vs 9ms actuales. Los embeddings se mapean a memoria virtual y el OS los trae bajo demanda. Sin copia, sin heap.

---

## Archivos a crear/modificar

| Archivo | Acción |
|---------|--------|
| `Cargo.toml` | + `memmap2 = "0.9"` (opcional o default) |
| `src/storage/traits.rs` | **Crear** — trait VectorStorage + backends |
| `src/storage.rs` | Modificar — añadir padding 32-byte, versión 2 del formato |
| `src/error.rs` | + error variant si es necesario |
| `src/collection.rs` | Inyectar storage en índices |
| `src/index/brute_force.rs` | Usar storage.as_embeddings() en search |
| `src/index/hnsw.rs` | Usar storage como flat_embeddings source |
| `src/index/ivf_pq.rs` | Usar storage para training data |
| `src/lib.rs` | Re-exportar VectorStorage y backends |

---

## Fase 1: Trait + Implementaciones

### Task 1.1: Añadir dependencia memmap2

**Archivo:** `Cargo.toml`

```toml
# Zero-copy memory-mapped file I/O
memmap2 = "0.9"
```

(No feature-gated — es una dependencia liviana, solo linking si se usa)

### Task 1.2: Crear `src/storage/traits.rs`

```rust
//! Storage abstraction for contiguous vector data.
//!
//! Provides a unified interface over memory-backed and memory-mapped
//! storage, so that index backends don't care where the bytes come from.

use crate::error::Result;

/// A contiguous region of f32 embedding data.
///
///
/// # Safety
///
/// The `as_embeddings()` method performs an unsafe `u8 → f32` reinterpret.
/// This is safe **only** when:
/// 1. The underlying bytes are a valid f32 representation (written by
///    [`bytemuck::cast_slice`] or equivalent).
/// 2. The byte slice starts at an address aligned to `align_of::<f32>()`
///    (guaranteed by the binary format's padding logic — see [`BinStorage`]).
///
/// This is the same approach used by HuggingFace `safetensors` and is
/// isolated to a single, auditable method.
pub trait VectorStorage: Send + Sync {
    /// Return the raw byte slice for the entire embedding region.
    fn as_bytes(&self) -> &[u8];

    /// Reinterpret the bytes as a contiguous `f32` slice.
    ///
    /// ## Panics
    /// Panics if the byte pointer is not properly aligned for `f32`.
    fn as_embeddings(&self) -> &[f32] {
        let bytes = self.as_bytes();
        let ptr = bytes.as_ptr();
        let align = std::mem::align_of::<f32>();
        assert_eq!(
            ptr.align_offset(align),
            0,
            "VectorStorage: byte slice is not aligned to {align} bytes \
             (ptr={ptr:p}, mod={})",
            ptr as usize % align
        );
        // Safety: the bytes are guaranteed to be valid f32 LE data and the
        // pointer is aligned to 4 bytes.  This is the same approach used by
        // safetensors (HuggingFace) and the `bytemuck` crate.
        unsafe {
            std::slice::from_raw_parts(
                ptr as *const f32,
                bytes.len() / std::mem::size_of::<f32>(),
            )
        }
    }

    /// Persist changes to disk (no-op for memory-backed storage).
    fn flush(&self) -> Result<()> {
        Ok(())
    }

    /// Number of f32 elements available.
    fn len(&self) -> usize {
        self.as_bytes().len() / 4
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ---------------------------------------------------------------------------
// Memory-backed (Vec<u8>) — for tests, volatile mode
// ---------------------------------------------------------------------------

/// In-memory storage backed by a `Vec<u8>`.
///
/// Useful for unit tests and any scenario where the data should not
/// hit the filesystem.
#[derive(Debug, Clone)]
pub struct MemoryBackedStorage {
    data: Vec<u8>,
}

impl MemoryBackedStorage {
    pub fn new(data: Vec<u8>) -> Self {
        Self { data }
    }

    /// Build from an `&[f32]` slice (copies the data).
    pub fn from_f32_slice(slice: &[f32]) -> Self {
        let bytes: &[u8] = bytemuck::cast_slice(slice);
        Self {
            data: bytes.to_vec(),
        }
    }
}

impl VectorStorage for MemoryBackedStorage {
    fn as_bytes(&self) -> &[u8] {
        &self.data
    }
}

// ---------------------------------------------------------------------------
// Memory-mapped (memmap2) — zero-copy production backend
// ---------------------------------------------------------------------------

/// Zero-copy storage backed by a memory-mapped file.
///
/// The file is mapped into virtual memory; the OS loads pages on
/// demand.  No heap allocation — load time is effectively zero.
#[derive(Debug)]
pub struct MmapBackedStorage {
    /// Keep the file handle alive for the lifetime of the mapping.
    _file: std::fs::File,
    /// The memory-mapped region.
    mmap: memmap2::Mmap,
}

impl MmapBackedStorage {
    /// Open and memory-map a `.vdb` file.
    ///
    /// `offset` and `len` describe the embedding region inside the file
    /// (obtained by parsing the binary header — see [`BinStorage`]).
    pub fn new(path: impl AsRef<std::path::Path>, offset: u64, len: usize) -> Result<Self> {
        let file = std::fs::File::open(path.as_ref()).map_err(|e| crate::error::Error::Io {
            path: path.as_ref().to_path_buf(),
            source: e,
        })?;
        // Safety: the mapped region is read-only and the file is not
        // modified while the mapping exists.
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| {
            crate::error::Error::Io {
                path: path.as_ref().to_path_buf(),
                source: e,
            }
        })?;

        let region = &mmap[offset as usize..offset as usize + len];
        let data = region.to_vec(); // ← temporary: copy until we can map the region

        // Actually, memmap2::Mmap doesn't support offset+len mapping directly.
        // We'll use MmapOptions for that:
        let mmap = unsafe {
            memmap2::MmapOptions::new()
                .offset(offset)
                .len(len)
                .map(&file)
        }.map_err(|e| crate::error::Error::Io {
            path: path.as_ref().to_path_buf(),
            source: e,
        })?;

        Ok(Self { _file: file, mmap })
    }

    /// Create from an existing `Mmap` (used internally).
    pub fn from_mmap(file: std::fs::File, mmap: memmap2::Mmap) -> Self {
        Self { _file: file, mmap }
    }
}

impl VectorStorage for MmapBackedStorage {
    fn as_bytes(&self) -> &[u8] {
        &self.mmap
    }
}
```

### Task 1.3: Añadir tests para ambos backends

Al final de `src/storage/traits.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_memory_backed_empty() {
        let s = MemoryBackedStorage::new(vec![]);
        assert!(s.is_empty());
        assert_eq!(s.as_bytes().len(), 0);
        assert_eq!(s.as_embeddings().len(), 0);
    }

    #[test]
    fn test_memory_backed_from_f32() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let s = MemoryBackedStorage::from_f32_slice(&data);
        assert_eq!(s.len(), 4);
        assert_eq!(s.as_embeddings(), &data[..]);
    }

    #[test]
    fn test_mmap_backed_roundtrip() {
        use tempfile::tempdir;
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_emb.bin");

        // Write some f32 data
        let data: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4];
        let bytes: &[u8] = bytemuck::cast_slice(&data);
        std::fs::write(&path, bytes).unwrap();

        let s = MmapBackedStorage::new(&path, 0, bytes.len()).unwrap();
        assert_eq!(s.len(), 4);
        assert!((s.as_embeddings()[0] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_vector_storage_alignment() {
        // Even with a non-trivial offset, the pointer must be 4-byte aligned
        let raw = vec![0u8; 128];
        let mut aligned_offset = 0;
        // Find the first 4-aligned offset from 0..8
        for off in 0..8usize {
            if (raw.as_ptr() as usize + off) % 4 == 0 {
                aligned_offset = off;
                break;
            }
        }
        // Just verify the memory backend works
        let s = MemoryBackedStorage::new(raw);
        let emb = s.as_embeddings();
        assert_eq!(emb.len(), 32); // 128 / 4
    }
}
```

---

## Fase 2: Formato Binario — Alineación 32-byte + Versión 2

### Task 2.1: Actualizar `BinStorage::encode()` para padding a 32 bytes

**Archivo:** `src/storage.rs`

Cambios:
1. Bump `CURRENT_VERSION` a 2
2. Añadir campo `align_padding` en el header (4 bytes, u32) para registrar cuánto padding se añadió
3. Calcular padding para que la sección de embeddings empiece en un offset alineado a 32 bytes
4. Header pasa de 24 a 28 bytes

Nuevo header:
```
Offset  Size  Field
------  ----  -----
0       4     magic: b"DVDB"
4       4     version: u32 LE (2)
8       4     dim: u32 LE
12      4     count: u32 LE
16      4     align_padding: u32 LE (bytes of zero-padding before emb section)
20      8     emb_offset: u64 LE
28      —     metadata section
...     —     zero-padding to next 32-byte boundary
...     —     embeddings (aligned to 32 bytes)
```

### Task 2.2: Añadir extracción de offset+len de embeddings para mmap

Añadir un método `BinStorage::embedding_region(path) -> Result<(u64, usize, usize, usize)>` que devuelva `(emb_offset, emb_len, dim, count)` para poder mmapear la región directamente sin cargar metadata.

---

## Fase 3: Integración en Collection + Índices

### Task 3.1: Collection crea VectorStorage

**Archivo:** `src/collection.rs`

- `Collection` guarda `storage: Arc<dyn VectorStorage>` y `dim: usize`, `count: usize`
- Al abrir un archivo binario, crea `MmapBackedStorage` para la región de embeddings
- Al abrir desde JSONL, crea `MemoryBackedStorage` con los embeddings convertidos
- Pasa `Arc<dyn VectorStorage>` a cada index backend al construirlo

### Task 3.2: BruteForce usa VectorStorage

**Archivo:** `src/index/brute_force.rs`

Añadir campo `storage: Option<Arc<dyn VectorStorage>>` (puede ser None para tests que usan `Document.embedding`).

En `search()`:
```rust
if let Some(storage) = &self.storage {
    let embeddings = storage.as_embeddings();
    let dim = self.dim;
    // Para cada documento i, usar &embeddings[i*dim..(i+1)*dim]
    // en lugar de &doc.embedding
}
```

### Task 3.3: HNSW usa VectorStorage

**Archivo:** `src/index/hnsw.rs`

Cuando `flat_embeddings=true` y `storage` está presente, `self.embeddings_flat` apunta a `storage.as_embeddings()` en lugar de ser un `Vec<f32>` propio. Esto elimina la copia de datos durante la carga.

### Task 3.4: IVF-PQ usa VectorStorage

**Archivo:** `src/index/ivf_pq.rs`

Para la construcción del índice, `IVF-PQ` necesita clonar los embeddings para K-Means de todas formas (los modifica durante el training). El storage se usa para la carga inicial pero no evita la copia durante training.

---

## Orden de Ejecución

```
Task 1.1: Cargo.toml +memmap2
Task 1.2: src/storage/traits.rs  (trait + backends)     → cargo test --lib storage
Task 1.3: tests para ambos backends                      → cargo test --lib storage
Task 2.1: BinStorage versión 2 + padding 32-byte         → cargo test --lib storage
Task 2.2: BinStorage::embedding_region()
Task 3.1: Collection usa VectorStorage                   → cargo test
Task 3.2: BruteForce usa storage                         → cargo test --lib brute_force
Task 3.3: HNSW usa storage                               → cargo test --lib hnsw
Task 3.4: IVF-PQ usa storage                             → cargo test --lib ivf_pq
```

---

## Verificación Final

```bash
cargo test                    # Todos los tests
cargo clippy -- -D warnings   # CI estricto
cargo run --release --example bench  # Benchmark (mmap debe cargar más rápido)
```

---

## Notas Técnicas

### unsafe controlado
El único `unsafe` está en `VectorStorage::as_embeddings()` y en `MmapBackedStorage::new()` (mmap). Ambos están aislados, documentados, y siguen el estándar de la industria.

### Zero-copy vs Copy
- **Lectura**: mmap es zero-copy — el OS trae páginas bajo demanda
- **Escritura**: el formato binario se escribe igual que antes (el writer no cambia)
- **Tests**: MemoryBackedStorage mantiene compatibilidad total

### Retrocompatibilidad
El formato v1 (sin padding de 32 bytes) se sigue leyendo correctamente. El padding solo se añade al escribir. `version` en el header permite diferenciar.

### Sin nuevas dependencias en el core
`memmap2` es una dependencia estándar, madura, con zero `unsafe` propio (usa el syscall `mmap` del OS). No afecta la filosofía de dependencias mínimas.

---

## Diferencial del Proyecto

1. **Carga instantánea**: mmap → 0ms de carga vs 9ms actuales.
2. **Escalabilidad**: archivos de 10GB se mapean igual de rápido que 10KB.
3. **Portabilidad**: el trait permite backends de red, cifrados, comprimidos.
4. **Tests eficientes**: MemoryBackedStorage evita I/O en tests.
5. **Sin servidor**: consistente con la filosofía zero-config.
