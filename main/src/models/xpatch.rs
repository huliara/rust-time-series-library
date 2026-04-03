use crate::args::exp::TaskName;
use crate::args::time_lengths::TimeLengths;
use crate::layers::decomposition::SeriesDecomp;
use crate::layers::replication_pad_1d::ReplicationPad1d;
use crate::models::traits::Forecast;
use burn::nn::Initializer;
use burn::{
    config::Config,
    module::Module,
    nn::{
        pool::{AvgPool1d, AvgPool1dConfig},
        BatchNorm, BatchNormConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig,
    },
    nn::conv::{Conv1d, Conv1dConfig},
    tensor::{backend::Backend, Tensor},
};
use clap::Args;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Args)]
pub struct XPatchArgs {
    #[arg(long, default_value_t = 7)]
    pub enc_in: usize,

    #[arg(long, default_value_t = 16)]
    pub patch_len: usize,

    #[arg(long, default_value_t = 8)]
    pub stride: usize,

    #[arg(long, default_value = "end")]
    pub padding_patch: String,

    #[arg(long, default_value_t = true)]
    pub revin: bool,

    #[arg(long, default_value = "reg")]
    pub ma_type: String,

    #[arg(long, default_value_t = 0.5)]
    pub alpha: f64,

    #[arg(long, default_value_t = 0.5)]
    pub beta: f64,

    #[arg(long, default_value_t = 10)]
    pub num_class: usize,
}

#[derive(Config, Debug)]
pub struct XPatchConfig {
    pub args: XPatchArgs,
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}"
    )]
    pub initializer: Initializer,
}

impl XPatchConfig {
    pub fn init<B: Backend>(
        self,
        _task_name: TaskName,
        lengths: TimeLengths,
        device: &B::Device,
    ) -> XPatch<B> {
        let config = &self.args;
        let seq_len = lengths.seq_len;
        let pred_len = lengths.pred_len;
        let patch_num = if config.padding_patch == "end" {
            (seq_len + config.stride - config.patch_len) / config.stride + 1
        } else {
            (seq_len - config.patch_len) / config.stride + 1
        };
        let patch_dim = config.patch_len * config.patch_len;

        let patch_embedding = LinearConfig::new(config.patch_len, patch_dim)
            .with_initializer(self.initializer.clone())
            .init(device);
        let patch_bn = BatchNormConfig::new(patch_num).init(device);
        let depthwise_conv = Conv1dConfig::new(patch_num, patch_num, 1)
            .with_initializer(self.initializer.clone())
            .init(device);
        let depthwise_bn = BatchNormConfig::new(patch_num).init(device);
        let residual_linear = LinearConfig::new(patch_dim, patch_dim)
            .with_initializer(self.initializer.clone())
            .init(device);
        let pointwise_conv = Conv1dConfig::new(patch_num, patch_num, 1)
            .with_initializer(self.initializer.clone())
            .init(device);
        let pointwise_bn = BatchNormConfig::new(patch_num).init(device);
        let head_1 = LinearConfig::new(patch_num * patch_dim, pred_len * 2)
            .with_initializer(self.initializer.clone())
            .init(device);
        let head_2 = LinearConfig::new(pred_len * 2, pred_len)
            .with_initializer(self.initializer.clone())
            .init(device);

        let trend_fc1 = LinearConfig::new(seq_len, pred_len * 4)
            .with_initializer(self.initializer.clone())
            .init(device);
        let trend_pool1 = AvgPool1dConfig::new(2).init();
        let trend_norm1 = LayerNormConfig::new(pred_len * 2).init(device);
        let trend_fc2 = LinearConfig::new(pred_len * 2, pred_len)
            .with_initializer(self.initializer.clone())
            .init(device);
        let trend_pool2 = AvgPool1dConfig::new(2).init();
        let trend_norm2 = LayerNormConfig::new(pred_len / 2).init(device);
        let trend_fc3 = LinearConfig::new(pred_len / 2, pred_len)
            .with_initializer(self.initializer.clone())
            .init(device);

        let fuse = LinearConfig::new(pred_len * 2, pred_len)
            .with_initializer(self.initializer.clone())
            .init(device);

        let decomposition = if config.ma_type == "reg" {
            None
        } else {
            let kernel_size = Self::decomposition_kernel(seq_len, config.patch_len, config.alpha, config.beta);
            Some(SeriesDecomp::<B>::new(kernel_size))
        };

        let padding_layer = if config.padding_patch == "end" {
            Some(ReplicationPad1d::new((0, config.stride)))
        } else {
            None
        };

        XPatch {
            patch_len: config.patch_len,
            stride: config.stride,
            revin: config.revin,
            use_decomposition: config.ma_type != "reg",
            decomposition,
            padding_layer,
            patch_embedding,
            patch_bn,
            depthwise_conv,
            depthwise_bn,
            residual_linear,
            pointwise_conv,
            pointwise_bn,
            head_1,
            head_2,
            trend_fc1,
            trend_pool1,
            trend_norm1,
            trend_fc2,
            trend_pool2,
            trend_norm2,
            trend_fc3,
            fuse,
            pred_len,
        }
    }

    fn decomposition_kernel(seq_len: usize, patch_len: usize, alpha: f64, beta: f64) -> usize {
        let base = if alpha >= beta {
            seq_len.min(patch_len.saturating_mul(2)).max(3)
        } else {
            seq_len.min(patch_len.saturating_add(4)).max(3)
        };

        if base % 2 == 0 { base + 1 } else { base }
    }
}

#[derive(Module, Debug)]
pub struct XPatch<B: Backend> {
    patch_len: usize,
    stride: usize,
    revin: bool,
    use_decomposition: bool,
    decomposition: Option<SeriesDecomp<B>>,
    padding_layer: Option<ReplicationPad1d>,
    patch_embedding: Linear<B>,
    patch_bn: BatchNorm<B>,
    depthwise_conv: Conv1d<B>,
    depthwise_bn: BatchNorm<B>,
    residual_linear: Linear<B>,
    pointwise_conv: Conv1d<B>,
    pointwise_bn: BatchNorm<B>,
    head_1: Linear<B>,
    head_2: Linear<B>,
    trend_fc1: Linear<B>,
    trend_pool1: AvgPool1d,
    trend_norm1: LayerNorm<B>,
    trend_fc2: Linear<B>,
    trend_pool2: AvgPool1d,
    trend_norm2: LayerNorm<B>,
    trend_fc3: Linear<B>,
    fuse: Linear<B>,
    pred_len: usize,
}

impl<B: Backend> XPatch<B> {
    fn normalize(&self, x: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>) {
        let means = x.clone().mean_dim(1);
        let x_centered = x.sub(means.clone());
        let var = x_centered.clone().var(1);
        let stdev = (var + 1e-5).sqrt();
        let x_normalized = x_centered.div(stdev.clone());
        (x_normalized, means, stdev)
    }

    fn denormalize(&self, x: Tensor<B, 3>, means: Tensor<B, 3>, stdev: Tensor<B, 3>) -> Tensor<B, 3> {
        x.mul(stdev).add(means)
    }

    fn preprocess(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        if let Some(padding_layer) = &self.padding_layer {
            padding_layer.forward(x)
        } else {
            x
        }
    }

    fn seasonal_branch(&self, seasonal: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch, channels, seq_len] = seasonal.dims();
        let mut s = seasonal.clone().reshape([batch * channels, seq_len]);
        s = self.preprocess(s);

        let s = s.unfold(-1, self.patch_len, self.stride);
        let [bc, patch_num, patch_len] = s.dims();
        let s = s.reshape([bc * patch_num, patch_len]);
        let s = self.patch_embedding.forward(s);
        let s = s.reshape([bc, patch_num, self.patch_len * self.patch_len]);
        let s = self.patch_bn.forward(s);

        let residual = self.residual_linear.forward(s.clone());
        let s = self.depthwise_conv.forward(s);
        let s = self.depthwise_bn.forward(s);
        let s = s + residual;

        let s = self.pointwise_conv.forward(s);
        let s = self.pointwise_bn.forward(s);

        let [bc, patch_num, dim] = s.dims();
        let s = s.reshape([bc, patch_num * dim]);
        let s = self.head_1.forward(s);
        self.head_2.forward(s)
    }

    fn trend_branch(&self, trend: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch, channels, seq_len] = trend.dims();
        let mut t = trend.reshape([batch * channels, seq_len]);
        t = self.trend_fc1.forward(t);
        let t = t.reshape([batch * channels, 1, self.pred_len * 4]);
        let t = self.trend_pool1.forward(t);
        let t = self.trend_norm1.forward(t);
        let t = self.trend_fc2.forward(t);
        let t = self.trend_pool2.forward(t);
        let t = self.trend_norm2.forward(t);
        let t = t.reshape([batch * channels, self.pred_len / 2]);
        self.trend_fc3.forward(t)
    }
}

impl<B: Backend> Forecast<B> for XPatch<B> {
    fn forecast(
        &self,
        x: Tensor<B, 3>,
        _x_mark: Tensor<B, 3>,
        _y: Tensor<B, 3>,
        _y_mark: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let [batch, _, channels] = x.dims();
        let x = if self.revin {
            let (x, means, stdev) = self.normalize(x);
            let x = x.swap_dims(1, 2);
            let x = self.forward_inner(x);
            return self.denormalize(x, means, stdev);
        } else {
            x.swap_dims(1, 2)
        };

        self.forward_inner(x).reshape([batch, self.pred_len, channels])
    }
}

impl<B: Backend> XPatch<B> {
    fn forward_inner(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, channels, _] = x.dims();
        let (seasonal_init, trend_init) = if self.use_decomposition {
            match &self.decomposition {
                Some(decomposition) => decomposition.forward(x),
                None => (x.clone(), x),
            }
        } else {
            (x.clone(), x)
        };

        let seasonal = self.seasonal_branch(seasonal_init);
        let trend = self.trend_branch(trend_init);
        let combined = Tensor::cat(vec![seasonal, trend], 1);
        let combined = self.fuse.forward(combined);
        combined.reshape([batch, channels, self.pred_len]).swap_dims(1, 2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Wgpu;
    use burn::tensor::Tensor;

    #[test]
    fn test_xpatch_forecast_onedim() {
        type B = Wgpu;
        let device = Default::default();

        let args = XPatchArgs {
            enc_in: 7,
            patch_len: 16,
            stride: 8,
            padding_patch: "end".to_string(),
            revin: true,
            ma_type: "reg".to_string(),
            alpha: 0.5,
            beta: 0.5,
            num_class: 10,
        };

        let lengths = TimeLengths {
            seq_len: 96,
            pred_len: 96,
            label_len: 48,
        };

        let model = XPatchConfig::new(args).init(TaskName::LongTermForecast, lengths, &device);

        let batch = 2;
        let x = Tensor::<B, 3>::zeros([batch, 96, 1], &device);
        let x_mark = Tensor::<B, 3>::zeros([batch, 96, 1], &device);
        let y = Tensor::<B, 3>::zeros([batch, 96, 1], &device);
        let y_mark = Tensor::<B, 3>::zeros([batch, 96, 1], &device);

        let output = model.forecast(x, x_mark, y, y_mark);

        assert_eq!(output.dims(), [batch, 96, 1]);
    }

    #[test]
    fn test_xpatch_forecast_multidim() {
        type B = Wgpu;
        let device = Default::default();

        let args = XPatchArgs {
            enc_in: 7,
            patch_len: 16,
            stride: 8,
            padding_patch: "end".to_string(),
            revin: true,
            ma_type: "reg".to_string(),
            alpha: 0.5,
            beta: 0.5,
            num_class: 10,
        };

        let lengths = TimeLengths {
            seq_len: 96,
            pred_len: 96,
            label_len: 48,
        };

        let model = XPatchConfig::new(args).init(TaskName::LongTermForecast, lengths, &device);

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