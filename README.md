# ndarray-math

Additional math operations for ndarray

1. Cosine Similarity
```rust
use ndarray::*;
use ndarray_math::*;
fn test_orthogonal_vectors() {
    let a = array![1.0, 0.0, 0.0];
    let b = array![0.0, 1.0, 0.0];
    assert_eq!(a.cosine_similarity(b).unwrap(), 0.0);
}
```
