//! Criterion benchmarks for dogma-vdb.
//!
//! Run with: `cargo bench`

use criterion::{criterion_group, criterion_main, Criterion};

fn bench_search(_c: &mut Criterion) {}

fn bench_serialization(_c: &mut Criterion) {}

fn bench_chunking(_c: &mut Criterion) {}

criterion_group!(benches, bench_search, bench_serialization, bench_chunking);
criterion_main!(benches);
