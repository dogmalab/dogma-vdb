//! K-Means clustering with K-Means++ initialisation.
//!
//! Used by IVF-PQ for vector partitioning and codebook training.

use crate::distance::{self, Metric};

/// K-Means clustering with **K-Means++ initialisation**.
///
/// Returns `k` centroids (each `D`-dimensional).  Initialisation uses the
/// standard K-Means++ D² weighting (Arthur & Vassilvitskii 2007):
///
/// 1. Pick the first centroid uniformly at random.
/// 2. For each remaining centroid, select a data point `x` with
///    probability proportional to D(x)² where D(x) is the **Euclidean
///    distance** to the nearest already-chosen centroid.
///
/// Runs up to `max_iter` iterations, early-stopping when no assignment
/// changes.  The iterative refinement uses the configured `metric`.
pub(crate) fn kmeans(
    data: &[Vec<f32>],
    k: usize,
    max_iter: usize,
    metric: Metric,
) -> Vec<Vec<f32>> {
    if data.is_empty() || k == 0 {
        return Vec::new();
    }
    let dim = data[0].len();
    let n = data.len();

    // 1. K-Means++ initialisation
    let mut rng = 42u64;
    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(k);
    let k_eff = k.min(n);

    fn next_splitmix(seed: &mut u64) -> u64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *seed >> 33
    }

    let idx0 = (next_splitmix(&mut rng) as usize) % n;
    centroids.push(data[idx0].clone());

    let mut min_d2 = vec![f64::MAX; n];
    for (i, v) in data.iter().enumerate() {
        min_d2[i] = squared_euclidean(v, &data[idx0]);
    }

    for _ in 1..k_eff {
        let total_w: f64 = min_d2.iter().sum();
        if total_w <= 0.0 {
            break;
        }

        let threshold = next_splitmix(&mut rng) as f64 / (u64::MAX as f64) * total_w;
        let mut cumulative = 0.0f64;
        let mut pick = 0;
        for (i, &d2) in min_d2.iter().enumerate() {
            cumulative += d2;
            if cumulative >= threshold {
                pick = i;
                break;
            }
        }

        centroids.push(data[pick].clone());

        for (i, v) in data.iter().enumerate() {
            let d2 = squared_euclidean(v, &data[pick]);
            if d2 < min_d2[i] {
                min_d2[i] = d2;
            }
        }
    }

    while centroids.len() < k {
        centroids.push(vec![0.0; dim]);
    }

    // 2. Iterate assignment -> update (Lloyd)
    let mut assignments = vec![0usize; n];
    for _iter in 0..max_iter {
        let mut changed = false;

        for (i, vec) in data.iter().enumerate() {
            let mut best = 0;
            let mut best_score = f32::NEG_INFINITY;
            for (j, c) in centroids.iter().enumerate() {
                let s = distance::score(vec, c, metric);
                if s > best_score {
                    best_score = s;
                    best = j;
                }
            }
            if assignments[i] != best {
                assignments[i] = best;
                changed = true;
            }
        }
        if !changed {
            break;
        }

        let mut sums: Vec<Vec<f32>> = vec![vec![0.0f32; dim]; k];
        let mut counts = vec![0usize; k];
        for (i, vec) in data.iter().enumerate() {
            let c = assignments[i];
            for (d, &v) in vec.iter().enumerate() {
                sums[c][d] += v;
            }
            counts[c] += 1;
        }
        for (j, sum) in sums.iter().enumerate() {
            if counts[j] > 0 {
                let inv = 1.0 / counts[j] as f32;
                for d in 0..dim {
                    centroids[j][d] = sum[d] * inv;
                }
            }
        }
    }

    centroids
}

/// Squared Euclidean distance between two f32 slices, returned as f64
/// (to avoid overflow when summing many dimensions).
#[inline]
pub(crate) fn squared_euclidean(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = *x as f64 - *y as f64;
            d * d
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmeans_basic() {
        let data: Vec<Vec<f32>> = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![10.0, 10.0],
            vec![10.1, 10.0],
        ];
        let centroids = kmeans(&data, 2, 30, Metric::Euclidean);
        assert_eq!(centroids.len(), 2);
        // Centroids should be near the two clusters
        let d00 = squared_euclidean(&centroids[0], &[0.0, 0.0]);
        let d01 = squared_euclidean(&centroids[0], &[10.0, 10.0]);
        let d10 = squared_euclidean(&centroids[1], &[0.0, 0.0]);
        let d11 = squared_euclidean(&centroids[1], &[10.0, 10.0]);
        // One centroid should be close to (0,0), the other to (10,10)
        let close_origin = d00 < 5.0 || d10 < 5.0;
        let close_far = d01 < 5.0 || d11 < 5.0;
        assert!(close_origin, "one centroid should be near origin");
        assert!(close_far, "one centroid should be near (10,10)");
    }

    #[test]
    fn test_kmeans_empty() {
        assert!(kmeans(&[], 5, 10, Metric::Cosine).is_empty());
    }

    #[test]
    fn test_kmeans_k_zero() {
        let data = vec![vec![1.0, 2.0]];
        assert!(kmeans(&data, 0, 10, Metric::Cosine).is_empty());
    }

    #[test]
    fn test_kmeans_single_point() {
        let data = vec![vec![5.0, 5.0]];
        let centroids = kmeans(&data, 1, 10, Metric::Cosine);
        assert_eq!(centroids.len(), 1);
        assert_eq!(centroids[0], vec![5.0, 5.0]);
    }

    #[test]
    fn test_kmeans_more_clusters_than_points() {
        let data = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let centroids = kmeans(&data, 10, 10, Metric::Cosine);
        // Should return k centroids, with zero-padded extras
        assert_eq!(centroids.len(), 10);
    }

    #[test]
    fn test_squared_euclidean() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 6.0, 3.0];
        let d2 = squared_euclidean(&a, &b);
        assert!((d2 - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_squared_euclidean_same() {
        let a = [1.0, 2.0];
        assert!(squared_euclidean(&a, &a).abs() < 1e-10);
    }
}
