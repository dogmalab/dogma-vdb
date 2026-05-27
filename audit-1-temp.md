# Architectural Audit of dogma-vdb

> Deep technical analysis of strengths, indexes, and improvement opportunities.

---

## 💎 Architecture and Design Strengths

### 1. SQ Orthogonality (Scalar Quantization)

Making SQ (i8) an orthogonal module that can be applied on top of *any* backend is a brilliant design decision. Reducing RAM ~4× while maintaining the base backend structure (like HNSW) is exactly how commercial engines operate. The fact that it already supports *rescoring* in f32 demonstrates architectural maturity.

### 2. JSONL / Binary Duality (The perfect balance)

- **JSONL** gives you the best developer experience (*DX*): readability, debuggability with grep/awk, Git compatibility, and O(1) append-only persistence.
- **The binary format** gives you production performance (memory-mappable contiguous embeddings). The fact that migration is automatic is a great achievement.

### 3. The peripheral ecosystem (MCP + Chunker)

The native MCP server over stdio elevates your project from being "just another vector library" to an immediate productivity tool for the agent ecosystem (Claude Desktop, Cursor). By including the *Smart Chunker* and the *File Watcher*, you are solving the complete RAG pipeline in a single binary with no external dependencies.

---

## 🔍 Index Diagnosis (Analyzing Your Benchmarks)

Looking at your performance table (5K docs, 128-dim), there are very revealing data points that explain why you feel HNSW is suboptimal in your system:

- **HNSW (77 us) vs Annoy (3,216 us)**: Annoy is performing worse than brute force (1,460 us). This is a clear symptom that for small or medium datasets (<10K elements), the cost of jumping between trees in Annoy or the abstraction overhead outweighs linear computation.
- **HNSW + SQ (Recall 0-60%)**: This is your current real headache. A recall of 0% to 60% is unusable for production. This occurs because when passing vectors to i8 linearly, the information loss destroys the geometric structure of the HNSW graph (graph links are computed incorrectly or searches diverge prematurely).

---

## 🛠️ How do ScaNN or other options fit into dogma-vdb?

### Option A: Attempting to implement pure ScaNN (Not recommended)

ScaNN requires implementing *Anisotropic Vector Quantization*. The mathematics to solve the loss function that penalizes parallel error requires a considerable amount of complex linear algebra. It would break your rule of "<300 line modules" and "zero complex abstractions". Furthermore, ScaNN shines starting at hundreds of thousands or millions of vectors; for your current HNSW target (<100K docs), it would be overengineering.

### Option B: Implementing IVF-PQ (The natural evolution of your SQ)

You already have the SQ module. If you create a new backend called `IVF_PQ`, you can reuse concepts:

1. Implement a very simple K-Means in `index/ivf.rs` to partition the space into inverted lists.
2. Instead of packaging the entire HNSW graph, vectors are stored compressed in their respective buckets.
3. **Result**: You solve the low Recall problem of your current HNSW+SQ, maintain the 4× (or more) RAM savings, and the code will remain clean, linear, and vectorizable with your `wide` crate.

### Option C: Refactoring HNSW using a Compact approach (USearch style)

Given that your HNSW is already ridiculously fast (77 microseconds per query), your problem is not speed, it's memory and recall with SQ. If you redesign your graph's data structure so that nodes and their neighbors are contiguous in memory (a single `Vec<u8>` or flat `Vec<u32>` instead of nodes with scattered pointers/IDs), you will drastically reduce HNSW's RAM consumption without losing 100% recall.

---

## 🚀 Critical Points to Improve (Architectural Code Review)

### 1. Fix HNSW+SQ Recall

To use SQ with HNSW successfully, the graph must be built using the original vectors (f32), and i8 quantization should only be used in the SIMD comparison phase, or alternatively apply *Heuristic Routing* that tolerates precision loss. If you build the graph directly with quantized distances, the graph breaks.

### 2. Leverage Memory Mapping (mmap)

Your native binary format takes 9ms to load 5K documents. If you switch from traditional reading to memory mapping (using the `memmap2` crate, for example), the load time will be ~0 milliseconds, as you let the operating system load the contiguous vectors into the CPU cache on demand from storage. This aligns with your *zero-server* philosophy.

### 3. Synchronicity vs Scale

Your "no async by default" design is excellent for simplicity. However, for `batch_insert` operations or index building (Annoy or HNSW), make sure to use `rayon` internally (hidden behind synchronous code) to parallelize usage of all CPU cores without compromising the clean API you have.

---

## 🎯 Conclusion

dogma-vdb has enormous potential as the go-to vector database for local development, CLI, internal tools, and embedded systems (Edge computing).

You do not need the complexity of ScaNN. Your ideal path to maintain the elegance of your project is:

1. **Fix the SQ implementation over HNSW** to recover recall, or
2. **Add an IVF-PQ backend** to replace Annoy (which clearly is not providing performance value according to your benchmarks).
