#[cfg(feature = "ndarray_15")]
use crate::ndarray_15_extra::*;
use ndarray::{ArrayBase, Ix1};

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum EuclideanDistanceError {
    #[error(
        "Invalid vectors: Vectors must have the same length for similarity calculation. LHS: {lhs}, RHS: {rhs}"
    )]
    InvalidVectors { lhs: usize, rhs: usize },
}
pub trait EuclideanDistance<T, Rhs = Self> {
    /// Computes the euclidean distance between two vectors.
    ///
    /// A `Result` containing the euclidean distance as a `f64`, or an error if the vectors are invalid.
    fn euclidean_distance(&self, rhs: Rhs) -> Result<T, EuclideanDistanceError>;
}

impl<S1, S2, T> EuclideanDistance<T, ArrayBase<S2, Ix1>> for ArrayBase<S1, Ix1>
where
    S1: ndarray::Data<Elem = T>,
    S2: ndarray::Data<Elem = T>,
    T: num::Float + core::iter::Sum + 'static + Copy,
{
    fn euclidean_distance(&self, rhs: ArrayBase<S2, Ix1>) -> Result<T, EuclideanDistanceError> {
        if self.len() != rhs.len() {
            return Err(EuclideanDistanceError::InvalidVectors {
                lhs: self.len(),
                rhs: rhs.len(),
            });
        }
        debug_assert!(
            self.iter().all(|&x| x.is_finite()),
            "LHS vector contains non-finite values"
        );
        debug_assert!(
            rhs.iter().all(|&x| x.is_finite()),
            "RHS vector contains non-finite values"
        );
        // Ok(self
        //     .iter()
        //     .zip(rhs.iter())
        //     .map(|(lhs, rhs)| (*lhs - *rhs).powi(2))
        //     .sum::<T>()
        //     .sqrt())
        use core::ops::Sub;
        Ok(self.to_owned().sub(rhs).powi(2).sum().sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::*;

    #[test]
    fn test_same_vectors() {
        let a = array![1.0, 2.0, 3.0];
        let b = array![1.0, 2.0, 3.0];
        assert_eq!(a.euclidean_distance(b).unwrap(), 0.0);
    }

    #[test]
    fn test_orthogonal_vectors() {
        let a = array![1.0, 0.0, 0.0];
        let b = array![0.0, 1.0, 0.0];
        assert_eq!(a.euclidean_distance(b).unwrap(), 2.0_f64.sqrt());
    }
}
