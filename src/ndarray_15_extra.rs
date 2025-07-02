pub trait Pow {
    type Output;
    fn powi(&self, rhs: i32) -> Self::Output;
    fn powf(&self, rhs: Self) -> Self::Output;
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
    fn powf(&self, rhs: Self) -> Self::Output {
        self.mapv(|x| x.powf(rhs))
    }
}

pub trait Sqrt {
    type Output;
    fn sqrt(&self) -> Self::Output;
}

impl<T, S, D> Sqrt for ndarray::ArrayBase<S, D>
where
    S: ndarray::Data<Elem = T>,
    T: num::Float,
    D: ndarray::Dimension,
{
    type Output = ndarray::Array<T, D>;
    fn sqrt(&self) -> Self::Output {
        self.mapv(|x| x.sqrt())
    }
}
