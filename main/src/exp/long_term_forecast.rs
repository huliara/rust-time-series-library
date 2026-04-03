mod forecast_output;
pub mod infer;
mod infer_step;
mod save_results;
pub mod train;
mod train_step;
use crate::{
    args::{
        data::DataCommand, exp::TaskName, model::gradient_model::GradientModelConfig,
        time_lengths::TimeLengths,
    },
    exp::long_term_forecast::train::ExpConfig,
    models::{gradient_model::GradientModel, traits::Forecast},
};
use burn::{prelude::*, tensor::backend::AutodiffBackend};

#[derive(Module, Debug)]
pub struct GradientForecastModel<B: Backend> {
    model: GradientModel<B>,
}

impl<B: Backend> GradientForecastModel<B> {
    pub fn new(
        model_config: GradientModelConfig,
        lengths: TimeLengths,
        device: &B::Device,
    ) -> Self {
        let model = model_config.init(TaskName::LongTermForecast, lengths, device);
        Self { model }
    }
}

impl<B: Backend> Forecast<B> for GradientForecastModel<B> {
    fn forecast(
        &self,
        x: Tensor<B, 3>,
        x_mark: Tensor<B, 3>,
        dec_input: Tensor<B, 3>,
        y_mark: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        match &self.model {
            GradientModel::PatchTST(model) => model.forecast(x, x_mark, dec_input, y_mark),
            GradientModel::DLinear(model) => model.forecast(x, x_mark, dec_input, y_mark),
            GradientModel::TimeXer(model) => model.forecast(x, x_mark, dec_input, y_mark),
        }
    }
}

struct LongTermForecastExp<B: AutodiffBackend> {
    result_path: String,
    exp_config: ExpConfig,
    data_config: DataCommand,
    lengths: TimeLengths,
    device: B::Device,
}
