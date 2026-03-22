use burn::{nn::loss::Reduction, prelude::*, tensor::backend::Backend};

use crate::exp::loss::{
    cauchy_loss::CauchyLoss, scaled_mse_loss::ScaledMseLoss, welsch_loss::WelschLoss,
};

enum BarronLossAlpha {
    Two,
    Zero,
    NegativeInf,
    General(f64),
}

pub struct BarronLoss {
    alpha: BarronLossAlpha,
    pub scale: f64,
}

impl BarronLoss {
    pub fn new(alpha: f64, scale: f64) -> Self {
        let alpha_enum = match alpha {
            2.0 => BarronLossAlpha::Two,
            0.0 => BarronLossAlpha::Zero,
            f64::NEG_INFINITY => BarronLossAlpha::NegativeInf,
            a => BarronLossAlpha::General(a),
        };
        Self {
            alpha: alpha_enum,
            scale,
        }
    }

    pub fn forward<const D: usize, B: Backend>(
        &self,
        predictions: Tensor<B, D>,
        targets: Tensor<B, D>,
        reduction: Reduction,
    ) -> Tensor<B, 1> {
        match self.alpha {
            BarronLossAlpha::Two => {
                ScaledMseLoss::new(self.scale).forward(predictions, targets, reduction)
            }
            BarronLossAlpha::Zero => {
                CauchyLoss::new(self.scale).forward(predictions, targets, reduction)
            }
            BarronLossAlpha::NegativeInf => {
                WelschLoss::new(self.scale).forward(predictions, targets, reduction)
            }
            BarronLossAlpha::General(alpha) => {
                self._forward(predictions, targets, alpha, reduction)
            }
        }
    }

    fn _forward<const D: usize, B: Backend>(
        &self,
        predictions: Tensor<B, D>,
        targets: Tensor<B, D>,
        alpha: f64,
        reduction: Reduction,
    ) -> Tensor<B, 1> {
        let loss = self.forward_no_reduction(predictions, targets, alpha);

        match reduction {
            Reduction::Mean | Reduction::Auto => loss.mean(),
            Reduction::Sum => loss.sum(),
        }
    }

    fn forward_no_reduction<const D: usize, B: Backend>(
        &self,
        predictions: Tensor<B, D>,
        targets: Tensor<B, D>,
        alpha: f64,
    ) -> Tensor<B, D> {
        let residual = (predictions - targets) / self.scale;
        let residual_sq = residual.powf_scalar(2.0);

        let alpha_minus_two_abs = (alpha - 2.0).abs();
        let inner = residual_sq.div_scalar(alpha_minus_two_abs).add_scalar(1.0);
        let shape = inner.powf_scalar(alpha / 2.0).sub_scalar(1.0);
        shape.mul_scalar(alpha_minus_two_abs / alpha)
    }
}
