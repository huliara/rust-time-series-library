use crate::args::exp::TaskName;
use crate::args::time_lengths::TimeLengths;
use crate::models::traits::Forecast;
use burn::nn::Initializer;
use burn::{
    config::Config,
    module::Module,
    nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig},
    tensor::{backend::Backend, Tensor},
};
use clap::Args;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Args)]
pub struct AttraosArgs {
    #[arg(long, default_value_t = 7)]
    pub enc_in: usize,

    #[arg(long, default_value_t = 96)]
    pub seq_len: usize,

    #[arg(long, default_value_t = 96)]
    pub patch_len: usize,

    #[arg(long, default_value_t = 3)]
    pub PSR_dim: usize,

    #[arg(long, default_value = "indep")]
    pub PSR_type: String,

    #[arg(long, default_value_t = 1)]
    pub PSR_delay: usize,

    #[arg(long, default_value_t = 128)]
    pub d_state: usize,

    #[arg(long, default_value_t = 16)]
    pub dt_rank: usize,

    #[arg(long, default_value_t = false)]
    pub FFT_evolve: bool,

    #[arg(long, default_value_t = false)]
    pub multi_res: bool,

    #[arg(long, default_value_t = 2)]
    pub e_layers: usize,

    #[arg(long, default_value_t = 10)]
    pub num_class: usize,
}

#[derive(Config, Debug)]
pub struct AttraosConfig {
    pub args: AttraosArgs,
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}"
    )]
    pub initializer: Initializer,
}

impl AttraosConfig {
    pub fn init<B: Backend>(
        self,
        _task_name: TaskName,
        lengths: TimeLengths,
        device: &B::Device,
    ) -> Attraos<B> {
        let config = &self.args;

        let d_inner = config.PSR_dim * config.patch_len;
        let layers = (0..config.e_layers)
            .map(|_| ResidualBlockConfig::new(d_inner).init(device))
            .collect();

        let out_layer = LinearConfig::new(d_inner * lengths.seq_len, lengths.pred_len)
            .with_initializer(self.initializer.clone())
            .init(device);

        Attraos {
            layers,
            out_layer,
            patch_len: config.patch_len,
            PSR_dim: config.PSR_dim,
            PSR_delay: config.PSR_delay,
            seq_len: lengths.seq_len,
            pad_len: (config.PSR_dim - 1) * config.PSR_delay,
            enc_in: config.enc_in,
            pred_len: lengths.pred_len,
        }
    }
}

#[derive(Module, Debug)]
pub struct Attraos<B: Backend> {
    layers: Vec<ResidualBlock<B>>,
    out_layer: Linear<B>,
    patch_len: usize,
    PSR_dim: usize,
    PSR_delay: usize,
    seq_len: usize,
    pad_len: usize,
    enc_in: usize,
    pred_len: usize,
}

impl<B: Backend> Attraos<B> {
    fn normalize(&self, x: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>) {
        let means = x.clone().mean_dim(1);
        let x_centered = x.sub(means.clone());

        let var = x_centered.clone().var(1);
        let stdev = (var + 1e-5).sqrt();
        let x_normalized = x_centered.div(stdev.clone());

        (x_normalized, means, stdev)
    }

    fn denormalize(
        &self,
        x: Tensor<B, 3>,
        means: Tensor<B, 3>,
        stdev: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        x.mul(stdev).add(means)
    }
}

impl<B: Backend> Forecast<B> for Attraos<B> {
    fn forecast(
        &self,
        x: Tensor<B, 3>,
        _x_mark: Tensor<B, 3>,
        _y: Tensor<B, 3>,
        _y_mark: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let [batch, seq_len, channels] = x.dims();

        // Normalization
        let (x_normalized, means, stdev) = self.normalize(x);

        // Swap dimensions: (B, L, C) -> (B, C, L)
        let x_swapped = x_normalized.swap_dims(1, 2);

        // Reshape: (B, C, L) -> (B*C, L, 1)
        let x_reshaped = x_swapped.reshape([batch * channels, seq_len, 1]);

        // Apply simple processing through layers
        let mut features = x_reshaped.clone();
        for layer in &self.layers {
            features = layer.forward(features);
        }

        // Prepare for output layer
        // Reshape: (B*C, L, D) -> (B*C, L*D)
        let [bc, l, d] = features.dims();
        let flattened = features.reshape([bc, l * d]);

        // Output layer: (B*C, pred_len)
        let predictions = self.out_layer.forward(flattened);

        // Reshape: (B*C, pred_len) -> (B, C, pred_len) -> (B, pred_len, C)
        let reshaped = predictions.reshape([batch, channels, self.pred_len]);
        let swapped = reshaped.swap_dims(1, 2);

        // Denormalization
        self.denormalize(swapped, means, stdev)
    }
}

// ============= Residual Block =============
#[derive(Config, Debug)]
pub struct ResidualBlockConfig {
    d_inner: usize,
}

impl ResidualBlockConfig {
    pub fn init<B: Backend>(self, device: &B::Device) -> ResidualBlock<B> {
        let layer_norm = LayerNormConfig::new(self.d_inner).init(device);
        let linear = LinearConfig::new(self.d_inner, self.d_inner)
            .with_initializer(Initializer::KaimingNormal {
                gain: 1.0,
                fan_out_only: false,
            })
            .init(device);

        ResidualBlock { layer_norm, linear }
    }
}

#[derive(Module, Debug)]
pub struct ResidualBlock<B: Backend> {
    layer_norm: LayerNorm<B>,
    linear: Linear<B>,
}

impl<B: Backend> ResidualBlock<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let normalized = self.layer_norm.forward(x.clone());
        let transformed = self.linear.forward(normalized);
        x + transformed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::args::exp::TaskName;
    use crate::args::time_lengths::TimeLengths;
    use burn::backend::Wgpu;
    use burn::nn::Initializer;
    use burn::Tensor;

    fn build_model<B: Backend>(device: &B::Device) -> Attraos<B> {
        let task_name = TaskName::LongTermForecast;

        let attraos_args = AttraosArgs {
            enc_in: 7,
            seq_len: 96,
            patch_len: 16,
            PSR_dim: 3,
            PSR_type: "indep".to_string(),
            PSR_delay: 1,
            d_state: 128,
            dt_rank: 16,
            FFT_evolve: false,
            multi_res: false,
            e_layers: 2,
            num_class: 10,
        };

        let lengths = TimeLengths {
            seq_len: 96,
            pred_len: 96,
            label_len: 48,
        };

        let initializer = Initializer::Constant { value: 0.1 };

        AttraosConfig::new(attraos_args)
            .with_initializer(initializer)
            .init(task_name, lengths, device)
    }

    #[test]
    fn test_attraos_forecast_output_shape_onedim() {
        type B = Wgpu;
        let device = Default::default();
        let model = build_model::<B>(&device);

        let batch = 2;
        let x = Tensor::<B, 3>::zeros([batch, 96, 1], &device);
        let x_mark = Tensor::<B, 3>::zeros([batch, 96, 1], &device);
        let y = Tensor::<B, 3>::zeros([batch, 96, 1], &device);
        let y_mark = Tensor::<B, 3>::zeros([batch, 96, 1], &device);

        let output = model.forecast(x, x_mark, y, y_mark);

        assert_eq!(output.dims(), [batch, 96, 1]);
    }

    #[test]
    fn test_attraos_forecast_output_shape_multidim() {
        type B = Wgpu;
        let device = Default::default();
        let model = build_model::<B>(&device);

        let batch = 2;
        let channels = 7;
        let x = Tensor::<B, 3>::zeros([batch, 96, channels], &device);
        let x_mark = Tensor::<B, 3>::zeros([batch, 96, channels], &device);
        let y = Tensor::<B, 3>::zeros([batch, 96, channels], &device);
        let y_mark = Tensor::<B, 3>::zeros([batch, 96, channels], &device);

        let output = model.forecast(x, x_mark, y, y_mark);

        assert_eq!(output.dims(), [batch, 96, channels]);
    }
}
