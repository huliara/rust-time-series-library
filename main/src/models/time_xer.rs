use super::traits::Forecast;
use crate::activation::Activation;
use crate::args::{activation::ActivationArg, exp::TaskName, time_lengths::TimeLengths};
use crate::layers::flatten_head::{FlattenHead, FlattenHeadConfig};
use crate::layers::{
    embed::{
        data_embedding_inverted::{DataEmbeddingInverted, DataEmbeddingInvertedConfig},
        en_embedding::{EnEmbedding, EnEmbeddingConfig},
    },
    self_attention_family::{
        attention_layer::{AttentionLayer, AttentionLayerConfig},
        full_attention::FullAttentionConfig,
    },
};
use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::{Conv1d, Conv1dConfig},
        Dropout, DropoutConfig, Initializer, LayerNorm, LayerNormConfig,
    },
    prelude::Bool,
    tensor::{backend::Backend, Tensor},
};
use clap::Args;
use serde::{Deserialize, Serialize};

#[derive(Config, Debug)]
pub struct EncoderLayerConfig {
    pub attention_config: AttentionLayerConfig,
    pub d_model: usize,
    pub dropout: f64,
    pub activation: ActivationArg,
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}"
    )]
    pub initializer: Initializer,
}

impl EncoderLayerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> EncoderLayer<B> {
        EncoderLayer {
            self_attention: self.attention_config.clone().init(device),
            cross_attention: self.attention_config.clone().init(device),
            conv1: Conv1dConfig::new(self.d_model, self.d_model * 4, 1)
                .with_initializer(self.initializer.clone())
                .init(device),
            conv2: Conv1dConfig::new(self.d_model * 4, self.d_model, 1)
                .with_initializer(self.initializer.clone())
                .init(device),
            norm1: LayerNormConfig::new(self.d_model).init(device),
            norm2: LayerNormConfig::new(self.d_model).init(device),
            norm3: LayerNormConfig::new(self.d_model).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
            activation: self.activation.init(),
        }
    }
}

#[derive(Module, Debug)]
pub struct EncoderLayer<B: Backend> {
    self_attention: AttentionLayer<B>,
    cross_attention: AttentionLayer<B>,
    conv1: Conv1d<B>,
    conv2: Conv1d<B>,
    norm1: LayerNorm<B>,
    norm2: LayerNorm<B>,
    norm3: LayerNorm<B>,
    dropout: Dropout,
    activation: Activation,
}

impl<B: Backend> EncoderLayer<B> {
    fn forward(
        &self,
        x: Tensor<B, 3>,
        cross: Tensor<B, 3>,
        x_mask: Option<Tensor<B, 4, Bool>>,
        cross_mask: Option<Tensor<B, 4, Bool>>,
    ) -> Tensor<B, 3> {
        let [b, _, d] = cross.dims();
        let (new_x, _) = self
            .self_attention
            .forward(x.clone(), x.clone(), x.clone(), x_mask);
        let x = self.norm1.forward(x + self.dropout.forward(new_x));

        let [bxn, seq, dxn] = x.dims();
        let x_glb_ori = x.clone().slice([0..bxn, (seq - 1)..seq, 0..dxn]);
        let x_glb = x_glb_ori.clone().reshape([b as isize, -1isize, d as isize]);
        let x_glb_attn = self.dropout.forward(
            self.cross_attention
                .forward(x_glb, cross.clone(), cross, cross_mask)
                .0,
        );
        let x_glb_attn_dims = x_glb_attn.dims();
        let x_glb = self.norm2.forward(
            x_glb_ori
                + x_glb_attn
                    .reshape([x_glb_attn_dims[0] * x_glb_attn_dims[1], x_glb_attn_dims[2]])
                    .unsqueeze_dim(1),
        );

        let x = Tensor::cat(vec![x.slice([0..bxn, 0..(seq - 1), 0..d]), x_glb], 1);
        let y = self.conv1.forward(x.clone().swap_dims(1, 2));
        let y = self.activation.forward(y);
        let y = self.dropout.forward(y);
        let y = self.conv2.forward(y).swap_dims(1, 2);
        self.norm3.forward(x + self.dropout.forward(y))
    }
}

#[derive(Module, Debug)]
pub struct Encoder<B: Backend> {
    layers: Vec<EncoderLayer<B>>,
    norm: LayerNorm<B>,
}

impl<B: Backend> Encoder<B> {
    fn forward(&self, mut x: Tensor<B, 3>, cross: Tensor<B, 3>) -> Tensor<B, 3> {
        for layer in &self.layers {
            x = layer.forward(x, cross.clone(), None, None);
        }
        self.norm.forward(x)
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, Args)]
pub struct TimeXerArgs {
    #[arg(long, default_value_t = 512)]
    pub d_model: usize,
    #[arg(long, default_value_t = 8)]
    pub n_heads: usize,
    #[arg(long, default_value_t = 2)]
    pub e_layers: usize,
    #[arg(long, default_value_t = 2048)]
    pub d_ff: usize,
    #[arg(long, default_value_t = 16)]
    pub patch_len: usize,
    #[arg(long, default_value_t = 0.1)]
    pub dropout: f64,
    #[arg(long, default_value_t = true)]
    pub use_norm: bool,
    #[arg(long, default_value = "timeF")]
    pub embed: String,
    #[arg(long, default_value = "h")]
    pub freq: String,
    #[arg(long, value_enum)]
    pub activation: ActivationArg,
}

#[derive(Config, Debug)]
pub struct TimeXerConfig {
    pub args: TimeXerArgs,
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}"
    )]
    pub initializer: Initializer,
}

impl TimeXerConfig {
    pub fn init<B: Backend>(
        &self,
        task_name: TaskName,
        lengths: TimeLengths,
        input_dim: usize,
        device: &B::Device,
    ) -> TimeXer<B> {
        let seq_len = lengths.seq_len;
        let pred_len = lengths.pred_len;
        let patch_num = seq_len / self.args.patch_len;
        let n_vars = input_dim;

        let en_embedding = EnEmbeddingConfig::new(
            n_vars,
            self.args.d_model,
            self.args.patch_len,
            self.args.dropout,
        )
        .with_initializer(self.initializer.clone())
        .init(device);

        let ex_embedding =
            DataEmbeddingInvertedConfig::new(seq_len, self.args.d_model, self.args.dropout)
                .init(device);

        let attention_cfg = AttentionLayerConfig {
            inner_attention: FullAttentionConfig {
                mask_flag: false,
                scale: None,
                attention_dropout: self.args.dropout,
                output_attention: false,
            },
            d_model: self.args.d_model,
            n_heads: self.args.n_heads,
            d_keys: None,
            d_values: None,
            initializer: self.initializer.clone(),
        };

        let layers = (0..self.args.e_layers)
            .map(|_| {
                EncoderLayerConfig::new(
                    attention_cfg.clone(),
                    self.args.d_model,
                    self.args.dropout,
                    self.args.activation.clone(),
                )
                .with_initializer(self.initializer.clone())
                .init(device)
            })
            .collect();

        let encoder = Encoder {
            layers,
            norm: LayerNormConfig::new(self.args.d_model).init(device),
        };

        let head_nf = self.args.d_model * (patch_num + 1);
        let head = FlattenHeadConfig::new(head_nf, pred_len, self.args.dropout)
            .with_initializer(self.initializer.clone())
            .init(device);

        TimeXer {
            is_forecast_task: matches!(
                task_name,
                TaskName::LongTermForecast | TaskName::ShortTermForecast
            ),
            pred_len,
            use_norm: self.args.use_norm,
            en_embedding,
            ex_embedding,
            encoder,
            head,
        }
    }
}

#[derive(Module, Debug)]
pub struct TimeXer<B: Backend> {
    is_forecast_task: bool,
    pred_len: usize,
    use_norm: bool,
    en_embedding: EnEmbedding<B>,
    ex_embedding: DataEmbeddingInverted<B>,
    encoder: Encoder<B>,
    head: FlattenHead<B>,
}

impl<B: Backend> TimeXer<B> {
    fn forecast(&self, x_enc: Tensor<B, 3>, x_mark_enc: Tensor<B, 3>) -> Tensor<B, 3> {
        let means = x_enc.clone().mean_dim(1);
        let centered = x_enc.clone().sub(means.clone());
        let var = centered.clone().mul(centered).mean_dim(1);
        let stdev = (var + 1e-5).sqrt();

        let x_enc = if self.use_norm {
            x_enc.sub(means.clone()).div(stdev.clone())
        } else {
            x_enc
        };

        let en_x = x_enc.clone().permute([0, 2, 1]);
        let (en_embed, n_vars) = self.en_embedding.forward(en_x);
        let ex_embed = self.ex_embedding.forward(x_enc.clone(), Some(x_mark_enc));

        let enc_out = self.encoder.forward(en_embed, ex_embed);
        let enc_dims = enc_out.dims();
        let enc_out = enc_out.reshape([
            -1isize,
            n_vars as isize,
            enc_dims[1] as isize,
            enc_dims[2] as isize,
        ]);
        let dec_out = self
            .head
            .forward(enc_out.permute([0, 1, 3, 2]))
            .swap_dims(1, 2);

        if !self.use_norm {
            return dec_out;
        }

        dec_out.mul(stdev).add(means)
    }
}

impl<B: Backend> Forecast<B> for TimeXer<B> {
    fn forecast(
        &self,
        x: Tensor<B, 3>,
        x_mark: Tensor<B, 3>,
        _y: Tensor<B, 3>,
        _y_mark: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let dec_out = self.forecast(x, x_mark);

        let [b, t, c] = dec_out.dims();
        let start = t - self.pred_len;
        dec_out.slice([0..b, start..t, 0..c])
    }
}
#[cfg(test)]
mod tests {
    use super::super::test::assert_module_forecast;
    use super::{TimeXer, TimeXerArgs, TimeXerConfig};
    use crate::args::activation::ActivationArg;
    use crate::args::exp::TaskName;
    use crate::args::time_lengths::TimeLengths;
    use crate::test_utils::dim::Dim;
    use burn::backend::Wgpu;
    use burn::nn::Initializer;

    #[test]
    fn test_time_xer_forecast_one_dim() {
        type B = Wgpu;
        let device = Default::default();
        let task_name = TaskName::LongTermForecast;
        let lengths = TimeLengths {
            seq_len: 96,
            pred_len: 96,
            label_len: 48,
        };

        let initializer = Initializer::Constant { value: (0.01) };

        let onedim_args = TimeXerArgs {
            d_model: 512,
            patch_len: 16,
            e_layers: 2,
            n_heads: 8,
            d_ff: 2048,
            dropout: 0.0,
            use_norm: true,
            embed: "timeF".to_string(),
            freq: "h".to_string(),
            activation: ActivationArg::Gelu,
        };

        let onedim_model = TimeXerConfig::new(onedim_args)
            .with_initializer(initializer.clone())
            .init(task_name.clone(), lengths.clone(), 1, &device);

        assert_module_forecast::<B, TimeXer<B>>(Dim::Onedim, onedim_model);
    }
    #[test]
    fn test_time_xer_forecast_multi_dim() {
        type B = Wgpu;
        let device = Default::default();

        let task_name = TaskName::LongTermForecast;
        let lengths = TimeLengths {
            seq_len: 96,
            pred_len: 96,
            label_len: 48,
        };

        let initializer = Initializer::Constant { value: (0.01) };
        let multidim_args = TimeXerArgs {
            d_model: 512,
            patch_len: 16,
            e_layers: 2,
            n_heads: 8,
            d_ff: 2048,
            dropout: 0.1,
            use_norm: true,
            embed: "timeF".to_string(),
            freq: "h".to_string(),
            activation: ActivationArg::Gelu,
        };
        let multidim_model = TimeXerConfig::new(multidim_args)
            .with_initializer(initializer)
            .init(task_name, lengths, 7, &device);
        assert_module_forecast::<B, TimeXer<B>>(Dim::Multidim, multidim_model);
    }
}
