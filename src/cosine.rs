#[cfg(feature = "ndarray_15")]
use crate::ndarray_15_extra::*;
use ndarray::{ArrayBase, Ix1};

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum CosineSimilarityError {
    #[error(
        "Invalid vectors: Vectors must have the same length for similarity calculation. LHS: {lhs}, RHS: {rhs}"
    )]
    InvalidVectors { lhs: usize, rhs: usize },
    // #[error(
    //     "Invalid matrices: Matrices must have the same shape for similarity calculation. LHS: {}x{}, RHS: {}x{}", lhs.0, lhs.1, rhs.0, rhs.1
    // )]
    // InvalidMatrices {
    //     lhs: (usize, usize),
    //     rhs: (usize, usize),
    // },
}
pub trait CosineSimilarity<T, Rhs = Self> {
    /// Computes the cosine similarity between two vectors.
    ///
    /// A `Result` containing the cosine similarity as a `f64`, or an error if the vectors are invalid.
    type Output;
    fn cosine_similarity(&self, rhs: Rhs) -> Result<Self::Output, CosineSimilarityError>;
}

impl<S1, S2, T> CosineSimilarity<T, ArrayBase<S2, Ix1>> for ArrayBase<S1, Ix1>
where
    S1: ndarray::Data<Elem = T>,
    S2: ndarray::Data<Elem = T>,
    T: num::traits::Float + 'static,
{
    type Output = T;
    fn cosine_similarity(&self, rhs: ArrayBase<S2, Ix1>) -> Result<T, CosineSimilarityError> {
        if self.len() != rhs.len() {
            return Err(CosineSimilarityError::InvalidVectors {
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
        let numerator = self.dot(&rhs);
        let denominator = self.powi(2).sum().sqrt() * rhs.powi(2).sum().sqrt();
        Ok(numerator / denominator)
    }
}

impl<S1, S2, T> CosineSimilarity<T, ArrayBase<S2, Ix1>> for &ArrayBase<S1, Ix1>
where
    S1: ndarray::Data<Elem = T>,
    S2: ndarray::Data<Elem = T>,
    T: num::traits::Float + 'static,
{
    type Output = T;
    fn cosine_similarity(&self, rhs: ArrayBase<S2, Ix1>) -> Result<T, CosineSimilarityError> {
        (*self).cosine_similarity(rhs)
    }
}

// impl<S1, S2, T> CosineSimilarity<T, ArrayBase<S2, Ix2>> for ArrayBase<S1, Ix2>
// where
//     S1: ndarray::Data<Elem = T>,
//     S2: ndarray::Data<Elem = T>,
//     T: num::traits::Float + 'static,
//     T: core::fmt::Debug,
// {
//     type Output = Array<T, Ix2>;
//     fn cosine_similarity(
//         &self,
//         rhs: ArrayBase<S2, Ix2>,
//     ) -> Result<Self::Output, CosineSimilarityError> {
//         if self.dim() != rhs.dim() {
//             return Err(CosineSimilarityError::InvalidMatrices {
//                 lhs: self.dim(),
//                 rhs: rhs.dim(),
//             });
//         }
//         debug_assert!(
//             self.iter().all(|&x| x.is_finite()),
//             "LHS matrix contains non-finite values"
//         );
//         debug_assert!(
//             rhs.iter().all(|&x| x.is_finite()),
//             "RHS matrix contains non-finite values"
//         );
//         let numerator = self.dot(&rhs.t());
//         let lhs_norm = self.powi(2).sum().sqrt();
//         let rhs_norm = rhs.powi(2).sum().sqrt();
//         dbg!(&lhs_norm, &rhs_norm);
//
//         let denominator = lhs_norm * rhs_norm.t();
//         Ok(numerator / denominator)
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::*;

    #[test]
    fn test_same_vectors() {
        let a = array![1.0, 2.0, 3.0];
        let b = array![1.0, 2.0, 3.0];
        assert_eq!(a.cosine_similarity(b).unwrap(), 1.0);
    }

    #[test]
    fn test_orthogonal_vectors() {
        let a = array![1.0, 0.0, 0.0];
        let b = array![0.0, 1.0, 0.0];
        assert_eq!(a.cosine_similarity(b).unwrap(), 0.0);
    }

    #[test]
    fn test_opposite_vectors() {
        let a = array![1.0, 2.0, 3.0];
        let b = array![-1.0, -2.0, -3.0];
        assert_eq!(a.cosine_similarity(b).unwrap(), -1.0);
    }

    #[test]
    fn test_invalid_vectors() {
        let a = array![1.0, 2.0];
        let b = array![1.0, 2.0, 3.0];
        assert!(matches!(
            a.cosine_similarity(b),
            Err(CosineSimilarityError::InvalidVectors { lhs: 2, rhs: 3 })
        ));
    }

    #[test]
    fn test_zero_vector() {
        let a = array![0.0, 0.0, 0.0];
        let b = array![1.0, 2.0, 3.0];
        let similarity = a.cosine_similarity(b);
        assert!(similarity.is_ok_and(|item: f64| item.is_nan()));
    }

    #[test]
    fn test_different_ndarray_types() {
        let a = array![1.0, 2.0, 3.0];
        let b = array![1.0, 2.0, 3.0];
        assert_eq!(a.cosine_similarity(b.view()).unwrap(), 1.0);
    }

    // #[test]
    // fn test_similarity_with_same_matrices() {
    //     let a = array![[1.0, 2.0], [3.0, 4.0]];
    //     let b = array![[1.0, 2.0], [3.0, 4.0]];
    //     assert_eq!(
    //         a.cosine_similarity(b).unwrap(),
    //         array![[1.0, 1.0], [1.0, 1.0]]
    //     );
    // }
    // #[test]
    // fn test_similarity_with_matrices() {
    //     let a = array![[1.0, 2.0], [3.0, 4.0]];
    //     let b = array![[5.0, 6.0], [7.0, 8.0]];
    //     assert_eq!(
    //         a.cosine_similarity(b).unwrap(),
    //         array![[0.2358, 0.3191], [0.5410, 0.7353]]
    //     );
    // }
}
