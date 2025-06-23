pub trait Pow {
    type Output;
    fn powi(&self, rhs: i32) -> Self::Output;
}
impl<T, S, D> Pow for ndarray::ArrayBase<S, D>
where
    S: ndarray::Data<Elem = T>,
    T: num::Float,
    D: ndarray::Dimension,
{
    type Output = ndarray::Array<T, D>;
    fn powi(&self, rhs: i32) -> Self::Output {
        self.mapv(|x| x.powi(rhs))
    }
}
