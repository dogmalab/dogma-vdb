# Auditoría Arquitectónica de dogma-vdb

> Análisis técnico profundo de fortalezas, índices y oportunidades de mejora.

---

## 💎 Fortalezas de Arquitectura y Diseño

### 1. Ortogonalidad de SQ (Scalar Quantization)

Hacer que SQ (i8) sea un módulo ortogonal que se puede aplicar sobre *cualquier* backend es una decisión de diseño brillante. Reducir la RAM ~4× manteniendo la estructura del backend base (como HNSW) es exactamente como operan los motores comerciales. El hecho de que ya soporte *rescoring* en f32 demuestra madurez arquitectónica.

### 2. Dualidad JSONL / Binario (El balance perfecto)

- **JSONL** te da la mejor experiencia de desarrollador (*DX*): legibilidad, debuggability con grep/awk, compatibilidad con Git y persistencia *append-only* en O(1).
- **El formato binario** te da el rendimiento para producción (mapear embeddings contiguos a memoria). Que la migración sea automática es un gran acierto.

### 3. El ecosistema periférico (MCP + Chunker)

El servidor MCP nativo sobre stdio eleva tu proyecto de ser "una librería de vectores más" a ser una herramienta de productividad inmediata para el ecosistema de agentes (Claude Desktop, Cursor). Al incluir el *Smart Chunker* y el *File Watcher*, estás resolviendo el pipeline completo de RAG en un solo binario sin dependencias externas.

---

## 🔍 Diagnóstico de tus Índices (Analizando tus Benchmarks)

Mirando tu tabla de rendimiento (5K docs, 128-dim), hay datos muy reveladores que explican por qué sientes que HNSW es subóptimo en tu sistema:

- **HNSW (77 us) vs Annoy (3,216 us)**: Annoy está rindiendo peor que la fuerza bruta (1,460 us). Esto es un síntoma claro de que para datasets pequeños o medianos (<10K elementos), el costo de saltar entre árboles en Annoy o la sobrecarga de la abstracción supera al cómputo lineal.
- **HNSW + SQ (Recall 0-60%)**: Aquí está tu verdadero dolor de cabeza actual. Un recall del 0% al 60% es inutilizable para producción. Esto ocurre porque al pasar los vectores a i8 de forma lineal, la pérdida de información destruye la estructura geométrica del grafo de HNSW (los enlaces del grafo se calculan mal o las búsquedas se desvían prematuramente).

---

## 🛠️ ¿Cómo encaja ScaNN o las otras opciones en dogma-vdb?

### Opción A: Intentar implementar ScaNN puro (No recomendado)

ScaNN requiere implementar *Anisotropic Vector Quantization*. La matemática para resolver la función de pérdida que penaliza el error paralelo requiere una cantidad considerable de álgebra lineal compleja. Rompería tu regla de "módulos de <300 líneas" y "cero abstracciones complejas". Además, ScaNN brilla a partir de cientos de miles o millones de vectores; para el target actual de tu HNSW (<100K docs), sería añadir sobreingeniería.

### Opción B: Implementar IVF-PQ (La evolución natural de tu SQ)

Ya tienes el módulo SQ. Si creas un nuevo backend llamado `IVF_PQ`, puedes reutilizar conceptos:

1. Implementas un K-Means muy simple en `index/ivf.rs` para partir el espacio en listas invertidas.
2. En lugar de empaquetar todo el grafo de HNSW, los vectores se guardan comprimidos en sus respectivos buckets.
3. **Resultado**: Solucionas el problema del Recall bajo de tu HNSW+SQ actual, mantienes el ahorro de 4× de RAM (o más) y el código seguirá siendo limpio, lineal y vectorizable con tu crate `wide`.

### Opción C: Refactorizar HNSW usando un enfoque Compacto (Estilo USearch)

Dado que tu HNSW ya es ridículamente rápido (77 microsegundos por query), tu problema no es la velocidad, es la memoria y el recall con SQ. Si rediseñas la estructura de datos de tu grafo para que los nodos y sus vecinos estén contiguos en memoria (un solo `Vec<u8>` o `Vec<u32>` plano en lugar de nodos con punteros/IDs dispersos), reducirás drásticamente el consumo de RAM de HNSW sin perder el 100% de recall.

---

## 🚀 Puntos Críticos a Mejorar (Code Review Arquitectónico)

### 1. Arreglar el Recall de HNSW+SQ

Para usar SQ con HNSW con éxito, el grafo debe construirse utilizando los vectores originales (f32), y la cuantización i8 solo debe usarse en la fase de comparación a nivel SIMD, o bien aplicar un *Heuristic Routing* que soporte la pérdida de precisión. Si construyes el grafo directamente con las distancias cuantizadas, el grafo se rompe.

### 2. Aprovechar Memory Mapping (mmap)

Tu formato binario nativo tarda 9ms en cargar 5K documentos. Si cambias la lectura tradicional por mapeo de memoria (usando el crate `memmap2`, por ejemplo), el tiempo de carga será de ~0 milisegundos, ya que dejas que el sistema operativo cargue los vectores contiguos en la caché de la CPU a demanda desde el almacenamiento. Esto va de la mano con tu filosofía *zero-server*.

### 3. Sincronismo vs Escala

Tu diseño de "sin async por defecto" es excelente para la simplicidad. Sin embargo, para operaciones de `batch_insert` o la construcción de índices (Annoy o HNSW), asegúrate de usar `rayon` de manera interna (oculto tras el código síncrono) para paralelizar el uso de todos los cores de la CPU sin comprometer la API limpia que tienes.

---

## 🎯 Conclusión

dogma-vdb tiene un potencial enorme como la base de datos vectorial por excelencia para desarrollo local, CLI, herramientas internas y sistemas embebidos (Edge computing).

No necesitas la complejidad de ScaNN. Tu camino ideal para mantener la elegancia de tu proyecto es:

1. **Arreglar la implementación de SQ sobre HNSW** para recuperar el recall, o
2. **Añadir un backend IVF-PQ** que sustituya a Annoy (el cual claramente no te está aportando valor en rendimiento según tus benchmarks).
