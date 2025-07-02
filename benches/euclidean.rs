#[divan::bench(sample_count = 1000, sample_size = 1000)]
fn euclidean_distance() {
    use ndarray::*;
    use ndarray_math::*;

    let a = Array::from_vec(vec![1.0; 512]);
    let b = Array::from_vec(vec![4.0; 512]);

    let result = divan::black_box((|| a.euclidean_distance(b))()).unwrap();
    assert_eq!(result, 67.88225099390856);
}
