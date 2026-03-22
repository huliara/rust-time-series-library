use burn::{
    nn::loss::{MseLoss, Reduction},
    prelude::*,
    tensor::backend::Backend,
};

pub struct WelschLoss {
    pub scale: f64,
}

impl WelschLoss {
    pub fn new(scale: f64) -> Self {
        Self { scale }
    }

    pub fn forward<const D: usize, B: Backend>(
        &self,
        predictions: Tensor<B, D>,
        targets: Tensor<B, D>,
        reduction: Reduction,
    ) -> Tensor<B, 1> {
        let loss = self.forward_no_reduction(predictions, targets);
        match reduction {
            Reduction::Mean | Reduction::Auto => loss.mean(),
            Reduction::Sum => loss.sum(),
        }
    }

    fn forward_no_reduction<const D: usize, B: Backend>(
        &self,
        predictions: Tensor<B, D>,
        targets: Tensor<B, D>,
    ) -> Tensor<B, D> {
        1.0 - (-((predictions - targets) / self.scale)
            .powf_scalar(2.0)
            .mul_scalar(0.5))
        .exp()
    }
}
