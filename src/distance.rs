//! Pure distance‑metric functions.
//!
//! Dot product and Euclidean distance use SIMD acceleration via the
//! [`wide`] crate (SSE/AVX2 on x86, NEON on ARM) when processing
//! chunks of 8 floats at a time.
//!
//! Falls back gracefully for the remaining <8 elements — no `unsafe`,
//! no platform-specific code.

use wide::f32x8;

/// Supported similarity / distance metrics.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum Metric {
    /// Cosine similarity  [−1, 1].  1 = identical.
    Cosine,
    /// Dot product.
    Dot,
    /// Euclidean distance  [0, ∞).  0 = identical.
    Euclidean,
}

/// Dot product of two slices, **SIMD‑accelerated**.
///
/// Processes 8 floats per iteration via `f32x8`; the remaining
/// <8 elements are handled with scalar fallback.
///
/// # Panics
/// In debug mode if the slices have different lengths.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "dot product requires equal-length slices");

    let rem = a.len() % 8;
    let (a_simd, a_tail) = a.split_at(a.len() - rem);
    let (b_simd, b_tail) = b.split_at(b.len() - rem);

    // SIMD body (chunks of 8)
    let body: f32x8 = a_simd
        .chunks_exact(8)
        .zip(b_simd.chunks_exact(8))
        .fold(f32x8::ZERO, |acc, (av, bv)| {
            acc + f32x8::from(av) * f32x8::from(bv)
        });

    // Tail (<8)
    let tail: f32 = a_tail.iter().zip(b_tail.iter()).map(|(x, y)| x * y).sum();

    body.reduce_add() + tail
}

/// Magnitude (L2 norm) of a vector, **SIMD‑accelerated**.
#[inline]
pub fn magnitude(v: &[f32]) -> f32 {
    dot(v, v).sqrt()
}

/// Cosine similarity, **SIMD‑accelerated**.
///
/// Returns 0 when either vector has zero magnitude.
#[inline]
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mag_a = magnitude(a);
    let mag_b = magnitude(b);
    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }
    dot(a, b) / (mag_a * mag_b)
}

/// Euclidean distance, **SIMD‑accelerated**.
///
/// Uses `f32x8` to compute squared differences in parallel.
#[inline]
pub fn euclidean(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(
        a.len(),
        b.len(),
        "euclidean distance requires equal-length slices"
    );

    let rem = a.len() % 8;
    let (a_simd, a_tail) = a.split_at(a.len() - rem);
    let (b_simd, b_tail) = b.split_at(b.len() - rem);

    // SIMD body
    let body: f32x8 =
        a_simd
            .chunks_exact(8)
            .zip(b_simd.chunks_exact(8))
            .fold(f32x8::ZERO, |acc, (av, bv)| {
                let d = f32x8::from(av) - f32x8::from(bv);
                acc + d * d
            });

    // Tail (<8)
    let tail: f32 = a_tail
        .iter()
        .zip(b_tail.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum();

    (body.reduce_add() + tail).sqrt()
}

/// Convenience: applies the chosen metric.
///
/// For `Euclidean` the **negated** distance is returned so that a
/// higher value always means "more similar".
#[inline]
pub fn score(a: &[f32], b: &[f32], metric: Metric) -> f32 {
    match metric {
        Metric::Cosine => cosine(a, b),
        Metric::Dot => dot(a, b),
        Metric::Euclidean => -euclidean(a, b),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert!((dot(&a, &b) - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product_empty() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        assert!((dot(&a, &b) - 0.0).abs() < 1e-6);
    }

    #[test]
    #[should_panic = "dot product requires equal-length slices"]
    fn test_dot_product_mismatch_debug() {
        let a = vec![1.0, 2.0];
        let b = vec![3.0];
        dot(&a, &b);
    }

    #[test]
    fn test_dot_8_exact() {
        // Exactly 8 elements — exercises SIMD body path
        let a = vec![1.0; 8];
        let b = vec![1.0; 8];
        assert!((dot(&a, &b) - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_10_elements() {
        // 10 elements — exercises head + body + tail
        let a = vec![1.0; 10];
        let b = vec![1.0; 10];
        assert!((dot(&a, &b) - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_magnitude() {
        let v = vec![3.0, 4.0];
        assert!((magnitude(&v) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_magnitude_zero() {
        let v = vec![0.0, 0.0, 0.0];
        assert!((magnitude(&v) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_magnitude_empty() {
        let v: Vec<f32> = vec![];
        assert!((magnitude(&v) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_identical() {
        let v = vec![1.0, 2.0, 3.0];
        let s = cosine(&v, &v);
        assert!((s - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let s = cosine(&a, &b);
        assert!((s - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let s = cosine(&a, &b);
        assert!((s - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_zero_vector() {
        let a = vec![1.0, 2.0];
        let b = vec![0.0, 0.0];
        let s = cosine(&a, &b);
        assert!((s - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_both_zero() {
        let a = vec![0.0, 0.0];
        let b = vec![0.0, 0.0];
        let s = cosine(&a, &b);
        assert!((s - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_zero() {
        let v = vec![1.0, 2.0, 3.0];
        assert!((euclidean(&v, &v) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_known() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert!((euclidean(&a, &b) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_10_elements() {
        let a = vec![1.0; 10];
        let b = vec![0.0; 10];
        assert!((euclidean(&a, &b) - 3.162277).abs() < 0.001);
    }

    #[test]
    fn test_euclidean_empty() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        assert!((euclidean(&a, &b) - 0.0).abs() < 1e-6);
    }

    #[test]
    #[should_panic = "euclidean distance requires equal-length slices"]
    fn test_euclidean_mismatch_debug() {
        let a = vec![1.0, 2.0];
        let b = vec![3.0];
        euclidean(&a, &b);
    }

    #[test]
    fn test_score_cosine() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let s = score(&a, &b, Metric::Cosine);
        assert!((s - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_score_dot() {
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0];
        let s = score(&a, &b, Metric::Dot);
        assert!((s - 11.0).abs() < 1e-6);
    }

    #[test]
    fn test_score_euclidean_negated() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let s = score(&a, &b, Metric::Euclidean);
        assert!((s - (-5.0)).abs() < 1e-6);
    }
}
