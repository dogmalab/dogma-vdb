//! Pure distance‑metric functions.
//!
//! Every function is a pure `fn(&[f32], &[f32]) -> f32` — no state,
//! no allocations, no dependencies.

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

/// Dot product of two slices.
///
/// # Panics
/// In debug mode if the slices have different lengths.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "dot product requires equal-length slices");
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Magnitude (L2 norm) of a vector.
#[inline]
pub fn magnitude(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Cosine similarity.
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

/// Euclidean distance.
#[inline]
pub fn euclidean(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(
        a.len(),
        b.len(),
        "euclidean distance requires equal-length slices"
    );
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
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
        assert!((dot(&a, &b) - 32.0).abs() < 1e-6); // 1*4 + 2*5 + 3*6 = 32
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
    fn test_magnitude() {
        let v = vec![3.0, 4.0];
        assert!((magnitude(&v) - 5.0).abs() < 1e-6); // sqrt(9+16) = 5
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
        assert!((euclidean(&a, &b) - 5.0).abs() < 1e-6); // sqrt((3-0)^2 + (4-0)^2) = 5
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
        assert!((s - 11.0).abs() < 1e-6); // 1*3 + 2*4 = 11
    }

    #[test]
    fn test_score_euclidean_negated() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let s = score(&a, &b, Metric::Euclidean);
        assert!((s - (-5.0)).abs() < 1e-6); // negated distance
    }
}
