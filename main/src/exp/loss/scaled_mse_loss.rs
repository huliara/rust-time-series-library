use burn::{
    nn::loss::{MseLoss, Reduction},
    prelude::*,
    tensor::backend::Backend,
};

pub struct ScaledMseLoss {
    pub scale: f64,
}

impl ScaledMseLoss {
    pub fn new(scale: f64) -> Self {
        Self { scale }
    }

    pub fn forward<const D: usize, B: Backend>(
        &self,
        predictions: Tensor<B, D>,
        targets: Tensor<B, D>,
        reduction: Reduction,
    ) -> Tensor<B, 1> {
        let loss = MseLoss::new()
            .forward(predictions, targets, Reduction::Mean)
            .mul_scalar((1.0 / self.scale).powf(2.0));
        match reduction {
            Reduction::Mean | Reduction::Auto => loss.mean(),
            Reduction::Sum => loss.sum(),
        }
    }
}
