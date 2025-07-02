#[cfg(feature = "ndarray_15")]
extern crate ndarray_15 as ndarray;
#[cfg(feature = "ndarray_15")]
pub mod ndarray_15_extra;

mod cosine;
pub use cosine::{CosineSimilarity, CosineSimilarityError};
mod euclidean;
pub use euclidean::{EuclideanDistance, EuclideanDistanceError};
