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
    todo!()
}

/// Magnitude (L2 norm) of a vector.
#[inline]
pub fn magnitude(v: &[f32]) -> f32 {
    todo!()
}

/// Cosine similarity.
///
/// Returns 0 when either vector has zero magnitude.
#[inline]
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    todo!()
}

/// Euclidean distance.
#[inline]
pub fn euclidean(a: &[f32], b: &[f32]) -> f32 {
    todo!()
}

/// Convenience: applies the chosen metric.
///
/// For `Euclidean` the **negated** distance is returned so that a
/// higher value always means "more similar".
#[inline]
pub fn score(a: &[f32], b: &[f32], metric: Metric) -> f32 {
    todo!()
}

#[cfg(test)]
mod tests;
