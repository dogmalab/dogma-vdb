//! Scalar Quantization (SQ) helpers.
//!
//! Compresses `f32` embeddings to `i8` (1 byte per value) for ~4× less
//! memory and ~2× faster distance computation.
//!
//! # Usage
//! When the `sq` config flag is `true`, each index backend stores an
//! additional `embedding_i8: Vec<Vec<i8>>` and calls `score_i8()` instead
//! of `distance::score()` for search.
//!
//! Quantization is **global** — a single `(scale, bias)` pair is
//! computed from the entire dataset at insert time.

use crate::distance::{self, Metric};
use crate::doc::Document;

/// Quantize an `f32` embedding to `i8`.
///
/// `v_i8[i] = clamp(round((v[i] - bias) / scale), -128, 127)`
pub fn quantize(v: &[f32], scale: f32, bias: f32) -> Vec<i8> {
    let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
    v.iter()
        .map(|x| {
            let q = ((x - bias) * inv).round();
            q.clamp(-128.0, 127.0) as i8
        })
        .collect()
}

/// Quantize a query vector (same formula).
pub fn quantize_query(query: &[f32], scale: f32, bias: f32) -> Vec<i8> {
    quantize(query, scale, bias)
}

/// Dot product of two `i8` vectors, returned as `i32` (accumulator fits
/// easily: 128·dim·127² ≈ 2·10⁶ at dim=384 ≈ fits in i32).
#[inline]
pub fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    debug_assert_eq!(a.len(), b.len(), "dot_i8 requires equal-length slices");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| *x as i32 * *y as i32)
        .sum()
}

/// Approximate cosine similarity on quantised vectors.
///
/// Returns a score in the same order as `distance::score` for the same
/// metric (higher = more similar).  The values are **not** identical to
/// the `f32` version, but the ranking should be very close.
#[inline]
pub fn score_i8(query_i8: &[i8], doc_i8: &[i8], metric: Metric, _scale: f32, _bias: f32) -> f32 {
    match metric {
        Metric::Dot | Metric::Cosine => dot_i8(query_i8, doc_i8) as f32,
        Metric::Euclidean => {
            // Squared euclidean on i8 — note the missing sqrt, but
            // ranking is preserved.
            let sq_sum: i32 = query_i8
                .iter()
                .zip(doc_i8.iter())
                .map(|(a, b)| {
                    let d = *a as i32 - *b as i32;
                    d * d
                })
                .sum();
            -(sq_sum as f32)
        }
    }
}

/// Compute global scale/bias for a collection of embeddings.
///
/// `scale = (global_max - global_min) / 255.0`
/// `bias  = global_min`
///
/// Returns `(scale, bias)`.  If the dataset is empty or all values
/// identical, returns `(1.0, 0.0)` as a no-op fallback.
pub fn compute_scale_bias(docs: &[Document]) -> (f32, f32) {
    let mut global_min = f32::MAX;
    let mut global_max = f32::MIN;

    for doc in docs {
        if doc.embedding.is_empty() {
            continue;
        }
        for &v in &doc.embedding {
            if v < global_min {
                global_min = v;
            }
            if v > global_max {
                global_max = v;
            }
        }
    }

    if global_max <= global_min {
        return (1.0, 0.0);
    }

    let scale = (global_max - global_min) / 255.0;
    (scale, global_min)
}

/// Rescore candidates with exact `f32` distance.
///
/// Given the `docs` slice and candidate indices from an i8 search,
/// recomputes scores with full-precision embeddings and returns the
/// top `k` as `Vec<(f32, usize, &Document)>`.
pub fn rescore<'a>(
    query: &[f32],
    docs: &'a [Document],
    candidate_ids: &[usize],
    k: usize,
    metric: Metric,
) -> Vec<(f32, usize, &'a Document)> {
    let mut scored: Vec<(f32, usize, &'a Document)> = candidate_ids
        .iter()
        .filter(|&&id| !docs[id].embedding.is_empty())
        .map(|&id| {
            let score = distance::score(query, &docs[id].embedding, metric);
            (score, id, &docs[id])
        })
        .collect();

    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);
    scored
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_roundtrip() {
        let v = vec![0.0, 127.0, -128.0, 64.0, -64.0];
        // scale=1.0, bias=0.0 => identity mapping
        let q = quantize(&v, 1.0, 0.0);
        assert_eq!(q.len(), 5);
        // clamp rounds to nearest
        assert_eq!(q[0], 0);
        assert_eq!(q[1], 127);
        assert_eq!(q[2], -128);
    }

    #[test]
    fn test_quantize_clamp() {
        let v = vec![-200.0, 200.0];
        let q = quantize(&v, 1.0, 0.0);
        assert_eq!(q[0], -128);
        assert_eq!(q[1], 127);
    }

    #[test]
    fn test_dot_i8_basic() {
        let a = vec![1i8, 2, 3];
        let b = vec![4i8, 5, 6];
        assert_eq!(dot_i8(&a, &b), 1 * 4 + 2 * 5 + 3 * 6); // 32
    }

    #[test]
    fn test_dot_i8_empty() {
        assert_eq!(dot_i8(&[], &[]), 0);
    }

    #[test]
    fn test_compute_scale_bias() {
        let docs = vec![
            Document::builder("a", "")
                .embedding(vec![0.0, 10.0])
                .build(),
            Document::builder("b", "")
                .embedding(vec![5.0, 255.0])
                .build(),
        ];
        let (scale, bias) = compute_scale_bias(&docs);
        assert!((scale - (255.0 - 0.0) / 255.0).abs() < 1e-6);
        assert!((bias - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_scale_bias_empty() {
        let docs: Vec<Document> = vec![];
        let (scale, bias) = compute_scale_bias(&docs);
        assert!((scale - 1.0).abs() < 1e-6);
        assert!((bias - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_scale_bias_uniform() {
        let docs = vec![
            Document::builder("a", "").embedding(vec![5.0]).build(),
            Document::builder("b", "").embedding(vec![5.0]).build(),
        ];
        let (scale, bias) = compute_scale_bias(&docs);
        assert!((scale - 1.0).abs() < 1e-6);
        assert!((bias - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_score_i8_euclidean() {
        let a = vec![1i8, 2i8];
        let b = vec![1i8, 2i8];
        let s = score_i8(&a, &b, Metric::Euclidean, 1.0, 0.0);
        // squared diff = 0 → negated = 0
        assert!((s - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_rescore_basic() {
        let query = vec![1.0, 0.0];
        let doc_a = Document::builder("a", "").embedding(vec![1.0, 0.0]).build();
        let doc_b = Document::builder("b", "").embedding(vec![0.0, 1.0]).build();
        let docs = vec![doc_a, doc_b];

        let results = rescore(&query, &docs, &[1, 0], 2, Metric::Cosine);
        assert_eq!(results.len(), 2);
        assert!(results[0].0 > results[1].0); // a closer than b
        assert_eq!(results[0].2.id, "a");
    }
}
