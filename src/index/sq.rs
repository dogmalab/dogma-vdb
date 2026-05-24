//! Scalar Quantization (SQ) helpers.
//!
//! Compresses `f32` embeddings to `i8` (1 byte per value) for ~4× less
//! memory and ~2× faster distance computation.
//!
//! # Design
//!
//! Quantization is **per-dimension** — each dimension has its own
//! `(scale, bias)` pair computed from the min/max of that dimension
//! across the dataset.  This preserves vector shape much better than
//! global min/max and works across all metrics (cosine, dot, euclidean).

use crate::distance::{self, Metric};
use crate::doc::Document;

// ---------------------------------------------------------------------------
// Per‑dimension quantisation
// ---------------------------------------------------------------------------

/// Compute per‑dimension `(scales, biases)` for a collection of embeddings.
///
/// Returns `(Vec<f32>, Vec<f32>)` — one pair per dimension.
/// For each dimension `d`:
///   `scale[d] = (max[d] - min[d]) / 255.0`
///   `bias[d]  = (min[d] + max[d]) / 2.0`   (mid‑point, so the quantised
///     range `[min, max]` maps to roughly `[-128, 127]` instead of
///     `[0, 255]`, avoiding heavy clamping for centred data).
///
/// If the dataset is empty or has no embeddings, returns two empty `Vec`s.
pub fn compute_scale_bias_per_dim(docs: &[Document]) -> (Vec<f32>, Vec<f32>) {
    let dim = docs
        .iter()
        .find(|d| !d.embedding.is_empty())
        .map(|d| d.embedding.len());
    let dim = match dim {
        Some(d) if d > 0 => d,
        _ => return (Vec::new(), Vec::new()),
    };

    let mut mins = vec![f32::MAX; dim];
    let mut maxs = vec![f32::MIN; dim];

    for doc in docs {
        if doc.embedding.is_empty() {
            continue;
        }
        for (i, &v) in doc.embedding.iter().enumerate() {
            if v < mins[i] {
                mins[i] = v;
            }
            if v > maxs[i] {
                maxs[i] = v;
            }
        }
    }

    let scales: Vec<f32> = mins
        .iter()
        .zip(maxs.iter())
        .map(|(&mn, &mx)| if mx > mn { (mx - mn) / 255.0 } else { 1.0 })
        .collect();

    // Mid‑point bias so [min, max] maps to ~[-128, 127] instead of [0, 255].
    // This avoids catastrophic clamping when data is centred around zero
    // (e.g. normalised embeddings on a unit sphere).
    let biases: Vec<f32> = mins
        .iter()
        .zip(maxs.iter())
        .map(|(&mn, &mx)| (mn + mx) / 2.0)
        .collect();

    (scales, biases)
}

/// Quantize an `f32` embedding using per‑dimension `scales` and `biases`.
///
/// For each dimension `d`:
///   `i8[d] = clamp(round((v[d] - bias[d]) / scale[d]), -128, 127)`
pub fn quantize(v: &[f32], scales: &[f32], biases: &[f32]) -> Vec<i8> {
    debug_assert_eq!(v.len(), scales.len(), "quantize requires matching lengths");
    v.iter()
        .zip(scales.iter().zip(biases.iter()))
        .map(|(&x, (&s, &b))| {
            let inv = if s > 0.0 { 1.0 / s } else { 0.0 };
            let q = ((x - b) * inv).round();
            q.clamp(-128.0, 127.0) as i8
        })
        .collect()
}

/// Quantize a query vector (same per‑dimension formula).
pub fn quantize_query(query: &[f32], scales: &[f32], biases: &[f32]) -> Vec<i8> {
    quantize(query, scales, biases)
}

// ---------------------------------------------------------------------------
// i8 distance functions
// ---------------------------------------------------------------------------

/// Dot product of two `i8` vectors, returned as `i32`.
#[inline]
pub fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    debug_assert_eq!(a.len(), b.len(), "dot_i8 requires equal-length slices");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| *x as i32 * *y as i32)
        .sum()
}

/// Approximate similarity score on quantised vectors.
///
/// Higher score = more similar (same convention as `distance::score`).
/// The absolute values differ from the `f32` version but rankings
/// are preserved, especially when the metric is Cosine (vectors are
/// effectively normalised by per‑dim scaling).
#[inline]
pub fn score_i8(query_i8: &[i8], doc_i8: &[i8], metric: Metric) -> f32 {
    match metric {
        Metric::Dot | Metric::Cosine => dot_i8(query_i8, doc_i8) as f32,
        Metric::Euclidean => {
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

// ---------------------------------------------------------------------------
// Rescoring
// ---------------------------------------------------------------------------

/// Rescore candidates with exact `f32` distance.
///
/// Recomputes scores for the given candidate indices and returns
/// the top `k` as `Vec<(f32, usize)>` (score + doc index).
pub fn rescore(
    query: &[f32],
    docs: &[Document],
    candidate_ids: &[usize],
    k: usize,
    metric: Metric,
) -> Vec<(f32, usize)> {
    let mut scored: Vec<(f32, usize)> = candidate_ids
        .iter()
        .filter(|&&id| !docs[id].embedding.is_empty())
        .map(|&id| {
            let score = distance::score(query, &docs[id].embedding, metric);
            (score, id)
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
        let scales = vec![1.0; 5];
        let biases = vec![0.0; 5];
        let q = quantize(&v, &scales, &biases);
        assert_eq!(q.len(), 5);
        assert_eq!(q[0], 0);
        assert_eq!(q[1], 127);
        assert_eq!(q[2], -128);
    }

    #[test]
    fn test_quantize_clamp() {
        let v = vec![-200.0, 200.0];
        let scales = vec![1.0; 2];
        let biases = vec![0.0; 2];
        let q = quantize(&v, &scales, &biases);
        assert_eq!(q[0], -128);
        assert_eq!(q[1], 127);
    }

    #[test]
    fn test_dot_i8_basic() {
        let a = vec![1i8, 2, 3];
        let b = vec![4i8, 5, 6];
        assert_eq!(dot_i8(&a, &b), 4 + 10 + 18);
    }

    #[test]
    fn test_dot_i8_empty() {
        assert_eq!(dot_i8(&[], &[]), 0);
    }

    #[test]
    fn test_compute_per_dim_basic() {
        let docs = vec![
            Document::builder("a", "")
                .embedding(vec![0.0, 100.0])
                .build(),
            Document::builder("b", "")
                .embedding(vec![10.0, 200.0])
                .build(),
        ];
        let (scales, biases) = compute_scale_bias_per_dim(&docs);
        assert_eq!(scales.len(), 2);
        assert_eq!(biases.len(), 2);
        // Dim 0: range [0, 10] → scale = 10/255, bias = mid = 5
        assert!((scales[0] - 10.0 / 255.0).abs() < 1e-6);
        assert!((biases[0] - 5.0).abs() < 1e-6);
        // Dim 1: range [100, 200] → scale = 100/255, bias = mid = 150
        assert!((scales[1] - 100.0 / 255.0).abs() < 1e-6);
        assert!((biases[1] - 150.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_per_dim_only_one_doc_has_embedding() {
        // Edge case: some docs have no embedding
        let docs = vec![
            Document::new("empty", ""),
            Document::builder("full", "")
                .embedding(vec![5.0, 10.0])
                .build(),
        ];
        let (scales, _biases) = compute_scale_bias_per_dim(&docs);
        assert_eq!(scales.len(), 2);
        // Both dims have the same value, scale = 1.0 (fallback)
        assert!((scales[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_quantize_identity() {
        let v = vec![-128.0, 0.0, 127.0];
        let scales = vec![1.0; 3];
        let biases = vec![0.0; 3];
        let q = quantize(&v, &scales, &biases);
        assert_eq!(q, vec![-128, 0, 127]);
    }

    #[test]
    fn test_score_i8_euclidean() {
        let a = vec![1i8, 2i8];
        let b = vec![1i8, 2i8];
        let s = score_i8(&a, &b, Metric::Euclidean);
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
        assert!(results[0].0 > results[1].0);
        assert_eq!(results[0].1, 0); // doc_a is index 0
    }

    #[test]
    fn test_per_dim_preserves_ranking() {
        // Two similar vectors should have higher score than dissimilar ones,
        // even when dimensions have very different ranges.
        let docs = vec![
            Document::builder("close", "")
                .embedding(vec![1.0, 0.0])
                .build(),
            Document::builder("far", "")
                .embedding(vec![0.0, 1000.0])
                .build(),
        ];

        let (scales, biases) = compute_scale_bias_per_dim(&docs);
        let q_i8 = quantize_query(&[1.0, 0.0], &scales, &biases);
        let close_i8 = quantize(&[1.0, 0.0], &scales, &biases);
        let far_i8 = quantize(&[0.0, 1000.0], &scales, &biases);

        let score_close = score_i8(&q_i8, &close_i8, Metric::Cosine);
        let score_far = score_i8(&q_i8, &far_i8, Metric::Cosine);
        assert!(
            score_close > score_far,
            "close={} far={}",
            score_close,
            score_far
        );
    }
}
