use super::traits::Forecast;
use crate::activation::Activation;
use crate::args::{activation::ActivationArg, exp::TaskName, time_lengths::TimeLengths};
use crate::layers::{
    embed::{
        data_embedding::DataEmbeddingInverted,
        positional_embedding::{PositionalEmbedding, PositionalEmbeddingConfig},
    },
    self_attention_family::{
        attention_layer::{AttentionLayer, AttentionLayerConfig},
        full_attention::FullAttentionConfig,
    },
};
use burn::{
    config::Config,
    module::{Module, Param},
    nn::{
        conv::{Conv1d, Conv1dConfig},
        Dropout, DropoutConfig, Initializer, LayerNorm, LayerNormConfig, Linear, LinearConfig,
    },
    tensor::{backend::Backend, Bool, Distribution, Tensor},
};
use clap::Args;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize, Args)]
pub struct TimeXerArgs {
    #[arg(long, default_value = "M")]
    pub features: String,
    #[arg(long, default_value_t = 7)]
    pub enc_in: usize,
    #[arg(long, default_value_t = 512)]
    pub d_model: usize,
    #[arg(long, default_value_t = 8)]
    pub n_heads: usize,
    #[arg(long, default_value_t = 2)]
    pub e_layers: usize,
    #[arg(long, default_value_t = 2048)]
    pub d_ff: usize,
    #[arg(long, default_value_t = 1)]
    pub factor: usize,
    #[arg(long, default_value_t = 0.1)]
    pub dropout: f64,
    #[arg(long, default_value = "timeF")]
    pub embed: String,
    #[arg(long, default_value = "h")]
    pub freq: String,
    #[arg(long, default_value_t = ActivationArg::Gelu)]
    pub activation: ActivationArg,
    #[arg(long, default_value_t = true)]
    pub use_norm: bool,
    #[arg(long, default_value_t = 16)]
    pub patch_len: usize,
}

#[derive(Config, Debug)]
pub struct TimeXerConfig {
    model_args: TimeXerArgs,
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}"
    )]
    pub initializer: Initializer,
}

impl TimeXerConfig {
    pub fn init<B: Backend>(
        &self,
        _task_name: TaskName,
        lengths: TimeLengths,
        device: &B::Device,
    ) -> TimeXer<B> {
        let patch_num = lengths.seq_len / self.model_args.patch_len;
        let n_vars = if self.model_args.features == "MS" {
            1
        } else {
            self.model_args.enc_in
        };

        let en_embedding = EnEmbeddingConfig::new(
            n_vars,
            self.model_args.d_model,
            self.model_args.patch_len,
            self.model_args.dropout,
        )
        .with_initializer(self.initializer.clone())
        .init(device);

        let ex_embedding = DataEmbeddingInverted::new(
            lengths.seq_len,
            self.model_args.d_model,
            self.model_args.embed.clone(),
            self.model_args.freq.clone(),
            self.model_args.dropout,
            device,
        );

        let layer_config = TimeXerEncoderLayerConfig {
            self_attention_config: AttentionLayerConfig {
                inner_attention: FullAttentionConfig {
                    mask_flag: false,
                    scale: None,
                    attention_dropout: self.model_args.dropout,
                    output_attention: false,
                },
                d_model: self.model_args.d_model,
                n_heads: self.model_args.n_heads,
                d_keys: None,
                d_values: None,
            },
            cross_attention_config: AttentionLayerConfig {
                inner_attention: FullAttentionConfig {
                    mask_flag: false,
                    scale: None,
                    attention_dropout: self.model_args.dropout,
                    output_attention: false,
                },
                d_model: self.model_args.d_model,
                n_heads: self.model_args.n_heads,
                d_keys: None,
                d_values: None,
            },
            d_model: self.model_args.d_model,
            d_ff: Some(self.model_args.d_ff),
            dropout: self.model_args.dropout,
            activation: self.model_args.activation.clone(),
            initializer: self.initializer.clone(),
        };

        let encoder = TimeXerEncoderConfig::new(self.model_args.e_layers, layer_config)
            .with_norm_dim(Some(self.model_args.d_model))
            .with_initializer(self.initializer.clone())
            .init(device);

        let head_nf = self.model_args.d_model * (patch_num + 1);
        let head = FlattenHeadConfig::new(head_nf, lengths.pred_len, self.model_args.dropout)
            .with_initializer(self.initializer.clone())
            .init(device);

        TimeXer {
            pred_len: lengths.pred_len,
            use_norm: self.model_args.use_norm,
            multivariate: self.model_args.features == "M",
            en_embedding,
            ex_embedding,
            encoder,
            head,
        }
    }
}

#[derive(Config, Debug)]
pub struct FlattenHeadConfig {
    pub nf: usize,
    pub target_window: usize,
    pub head_dropout: f64,
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}"
    )]
    pub initializer: Initializer,
}

impl FlattenHeadConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> FlattenHead<B> {
        let linear = LinearConfig::new(self.nf, self.target_window)
            .with_initializer(self.initializer.clone())
            .init(device);
        let dropout = DropoutConfig::new(self.head_dropout).init();

        FlattenHead { linear, dropout }
    }
}

#[derive(Module, Debug)]
pub struct FlattenHead<B: Backend> {
    linear: Linear<B>,
    dropout: Dropout,
}

impl<B: Backend> FlattenHead<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        let x = x.flatten(-2, -1);
        let x = self.linear.forward(x);
        self.dropout.forward(x)
    }
}

#[derive(Config, Debug)]
pub struct EnEmbeddingConfig {
    pub n_vars: usize,
    pub d_model: usize,
    pub patch_len: usize,
    pub dropout: f64,
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}"
    )]
    pub initializer: Initializer,
}

impl EnEmbeddingConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> EnEmbedding<B> {
        let value_embedding = LinearConfig::new(self.patch_len, self.d_model)
            .with_bias(false)
            .with_initializer(self.initializer.clone())
            .init(device);
        let glb_token = Param::from_tensor(Tensor::random(
            [1, self.n_vars, 1, self.d_model],
            Distribution::Normal(0.0, 1.0),
            device,
        ));
        let position_embedding = PositionalEmbeddingConfig::new(self.d_model, 5000).init(device);
        let dropout = DropoutConfig::new(self.dropout).init();

        EnEmbedding {
            patch_len: self.patch_len,
            value_embedding,
            glb_token,
            position_embedding,
            dropout,
        }
    }
}

#[derive(Module, Debug)]
pub struct EnEmbedding<B: Backend> {
    patch_len: usize,
    value_embedding: Linear<B>,
    glb_token: Param<Tensor<B, 4>>,
    position_embedding: PositionalEmbedding<B>,
    dropout: Dropout,
}

impl<B: Backend> EnEmbedding<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> (Tensor<B, 3>, usize) {
        let [batch_size, n_vars, _seq_len] = x.dims();
        let glb = self.glb_token.val().repeat_dim(0, batch_size);

        let x: Tensor<B, 4> = x.unfold(-1, self.patch_len, self.patch_len);
        let [unfold_batch, unfold_vars, unfold_tokens, unfold_patch] = x.dims();
        let x = x.reshape([unfold_batch * unfold_vars, unfold_tokens, unfold_patch]);
        let x = self.value_embedding.forward(x);
        let [embedded_batch, embedded_tokens, embedded_model] = x.dims();
        let x = x.clone() + self.position_embedding.forward(&x);
        let x = x.reshape([
            embedded_batch as isize / n_vars as isize,
            n_vars as isize,
            embedded_tokens as isize,
            embedded_model as isize,
        ]);
        let x = Tensor::cat(vec![x, glb], 2);
        let [cat_batch, cat_vars, cat_tokens, cat_model] = x.dims();
        let x = x.reshape([cat_batch * cat_vars, cat_tokens, cat_model]);

        (self.dropout.forward(x), n_vars)
    }
}

#[derive(Config, Debug)]
pub struct TimeXerEncoderLayerConfig {
    pub self_attention_config: AttentionLayerConfig,
    pub cross_attention_config: AttentionLayerConfig,
    pub d_model: usize,
    pub d_ff: Option<usize>,
    pub dropout: f64,
    pub activation: ActivationArg,
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}"
    )]
    pub initializer: Initializer,
}

impl TimeXerEncoderLayerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TimeXerEncoderLayer<B> {
        let d_ff = self.d_ff.unwrap_or(4 * self.d_model);
        let self_attention = self.self_attention_config.clone().init(device);
        let cross_attention = self.cross_attention_config.clone().init(device);
        let conv1 = Conv1dConfig::new(self.d_model, d_ff, 1)
            .with_initializer(self.initializer.clone())
            .init(device);
        let conv2 = Conv1dConfig::new(d_ff, self.d_model, 1)
            .with_initializer(self.initializer.clone())
            .init(device);
        let norm1 = LayerNormConfig::new(self.d_model).init(device);
        let norm2 = LayerNormConfig::new(self.d_model).init(device);
        let norm3 = LayerNormConfig::new(self.d_model).init(device);
        let dropout = DropoutConfig::new(self.dropout).init();
        let activation = self.activation.init();

        TimeXerEncoderLayer {
            self_attention,
            cross_attention,
            conv1,
            conv2,
            norm1,
            norm2,
            norm3,
            dropout,
            activation,
        }
    }
}

#[derive(Module, Debug)]
pub struct TimeXerEncoderLayer<B: Backend> {
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

impl<B: Backend> TimeXerEncoderLayer<B> {
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        cross: Tensor<B, 3>,
        x_mask: Option<Tensor<B, 4, Bool>>,
        cross_mask: Option<Tensor<B, 4, Bool>>,
    ) -> Tensor<B, 3> {
        let [cross_batch, _, d_model] = cross.dims();

        let (self_attended, _) =
            self.self_attention
                .forward(x.clone(), x.clone(), x.clone(), x_mask);
        let x = x + self.dropout.forward(self_attended);
        let x = self.norm1.forward(x);

        let [batch_vars, tokens, _] = x.dims();
        let n_vars = batch_vars / cross_batch;

        let x_glb_ori = x
            .clone()
            .slice([0..batch_vars, (tokens - 1)..tokens, 0..d_model]);
        let x_glb = x_glb_ori.clone().reshape([cross_batch, n_vars, d_model]);
        let (x_glb_attn, _) = self
            .cross_attention
            .forward(x_glb, cross.clone(), cross, cross_mask);
        let x_glb_attn = self.dropout.forward(x_glb_attn);
        let x_glb_attn = x_glb_attn.reshape([batch_vars, 1, d_model]);
        let x_glb = self.norm2.forward(x_glb_ori + x_glb_attn);

        let x_patch = x.slice([0..batch_vars, 0..(tokens - 1), 0..d_model]);
        let x = Tensor::cat(vec![x_patch, x_glb], 1);

        let y = self.dropout.forward(
            self.activation
                .forward(self.conv1.forward(x.clone().transpose())),
        );
        let y = self.dropout.forward(self.conv2.forward(y).transpose());

        self.norm3.forward(x + y)
    }
}

#[derive(Config, Debug)]
pub struct TimeXerEncoderConfig {
    pub n_layers: usize,
    pub layer_config: TimeXerEncoderLayerConfig,
    pub norm_dim: Option<usize>,
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}"
    )]
    pub initializer: Initializer,
}

impl TimeXerEncoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TimeXerEncoder<B> {
        let layers = (0..self.n_layers)
            .map(|_| {
                self.layer_config
                    .clone()
                    .with_initializer(self.initializer.clone())
                    .init(device)
            })
            .collect();
        let norm = self
            .norm_dim
            .map(|dim| LayerNormConfig::new(dim).init(device));

        TimeXerEncoder { layers, norm }
    }
}

#[derive(Module, Debug)]
pub struct TimeXerEncoder<B: Backend> {
    layers: Vec<TimeXerEncoderLayer<B>>,
    norm: Option<LayerNorm<B>>,
}

impl<B: Backend> TimeXerEncoder<B> {
    pub fn forward(
        &self,
        mut x: Tensor<B, 3>,
        cross: Tensor<B, 3>,
        x_mask: Option<Tensor<B, 4, Bool>>,
        cross_mask: Option<Tensor<B, 4, Bool>>,
    ) -> Tensor<B, 3> {
        for layer in &self.layers {
            x = layer.forward(x, cross.clone(), x_mask.clone(), cross_mask.clone());
        }

        if let Some(norm) = &self.norm {
            x = norm.forward(x);
        }

        x
    }
}

#[derive(Module, Debug)]
pub struct TimeXer<B: Backend> {
    pred_len: usize,
    use_norm: bool,
    multivariate: bool,
    en_embedding: EnEmbedding<B>,
    ex_embedding: DataEmbeddingInverted<B>,
    encoder: TimeXerEncoder<B>,
    head: FlattenHead<B>,
}

impl<B: Backend> TimeXer<B> {
    fn decode(&self, enc_out: Tensor<B, 3>, n_vars: usize, batch_size: usize) -> Tensor<B, 3> {
        let [enc_tokens, enc_model] = [enc_out.dims()[1], enc_out.dims()[2]];
        let enc_out = enc_out.reshape([
            batch_size as isize,
            n_vars as isize,
            enc_tokens as isize,
            enc_model as isize,
        ]);
        let enc_out = enc_out.permute([0, 1, 3, 2]);
        self.head.forward(enc_out).swap_dims(1, 2)
    }

    fn forecast_single_target(
        &self,
        x_enc: Tensor<B, 3>,
        x_mark_enc: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len, channels] = x_enc.dims();

        let target = x_enc
            .clone()
            .slice([0..batch_size, 0..seq_len, (channels - 1)..channels])
            .swap_dims(1, 2);
        let cross = x_enc.slice([0..batch_size, 0..seq_len, 0..(channels - 1)]);

        let (en_embed, n_vars) = self.en_embedding.forward(target);
        let ex_embed = self.ex_embedding.forward(cross, Some(x_mark_enc));
        let enc_out = self.encoder.forward(en_embed, ex_embed, None, None);

        self.decode(enc_out, n_vars, batch_size)
    }

    fn forecast_multi_target(&self, x_enc: Tensor<B, 3>, x_mark_enc: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, _, _] = x_enc.dims();
        let (en_embed, n_vars) = self.en_embedding.forward(x_enc.clone().swap_dims(1, 2));
        let ex_embed = self.ex_embedding.forward(x_enc, Some(x_mark_enc));
        let enc_out = self.encoder.forward(en_embed, ex_embed, None, None);

        self.decode(enc_out, n_vars, batch_size)
    }
}

impl<B: Backend> Forecast<B> for TimeXer<B> {
    fn forecast(
        &self,
        x_enc: Tensor<B, 3>,
        x_mark_enc: Tensor<B, 3>,
        _x_dec: Tensor<B, 3>,
        _x_mark_dec: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        if self.use_norm {
            let means = x_enc.clone().mean_dim(1);
            let x_enc = x_enc.sub(means.clone());
            let stdev = (x_enc.clone().var(1) + 1e-5).sqrt();
            let x_enc = x_enc.div(stdev.clone());
            let [batch_size, _, channels] = x_enc.dims();

            let dec_out = if self.multivariate {
                self.forecast_multi_target(x_enc, x_mark_enc)
                    .mul(stdev.clone())
                    .add(means)
            } else {
                let scale = stdev.slice([0..batch_size, 0..1, (channels - 1)..channels]);
                let shift = means.slice([0..batch_size, 0..1, (channels - 1)..channels]);
                self.forecast_single_target(x_enc, x_mark_enc)
                    .mul(scale)
                    .add(shift)
            };

            let [batch_size, out_len, out_dim] = dec_out.dims();
            dec_out.slice([
                0..batch_size,
                (out_len - self.pred_len)..out_len,
                0..out_dim,
            ])
        } else {
            let dec_out = if self.multivariate {
                self.forecast_multi_target(x_enc, x_mark_enc)
            } else {
                self.forecast_single_target(x_enc, x_mark_enc)
            };
            let [batch_size, out_len, out_dim] = dec_out.dims();
            dec_out.slice([
                0..batch_size,
                (out_len - self.pred_len)..out_len,
                0..out_dim,
            ])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{TimeXerArgs, TimeXerConfig};
    use crate::args::{activation::ActivationArg, exp::TaskName, time_lengths::TimeLengths};
    use crate::models::traits::Forecast;
    use burn::{nn::Initializer, tensor::Tensor};
    use burn_ndarray::NdArray;

    #[test]
    fn test_time_xer_forecast_multivariate_shape() {
        type B = NdArray;
        let device = Default::default();

        let args = TimeXerArgs {
            features: "M".to_string(),
            enc_in: 7,
            d_model: 32,
            n_heads: 4,
            e_layers: 1,
            d_ff: 64,
            factor: 1,
            dropout: 0.1,
            embed: "timeF".to_string(),
            freq: "h".to_string(),
            activation: ActivationArg::Gelu,
            use_norm: true,
            patch_len: 16,
        };
        let lengths = TimeLengths {
            seq_len: 96,
            pred_len: 24,
            label_len: 48,
        };

        let model = TimeXerConfig::new(args)
            .with_initializer(Initializer::Constant { value: 0.01 })
            .init(TaskName::LongTermForecast, lengths, &device);

        let x_enc = Tensor::<B, 3>::zeros([2, 96, 7], &device);
        let x_mark_enc = Tensor::<B, 3>::zeros([2, 96, 4], &device);
        let x_dec = Tensor::<B, 3>::zeros([2, 72, 7], &device);
        let x_mark_dec = Tensor::<B, 3>::zeros([2, 72, 4], &device);

        let output = model.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec);
        assert_eq!(output.dims(), [2, 24, 7]);
    }

    #[test]
    fn test_time_xer_forecast_single_target_shape() {
        type B = NdArray;
        let device = Default::default();

        let args = TimeXerArgs {
            features: "S".to_string(),
            enc_in: 1,
            d_model: 32,
            n_heads: 4,
            e_layers: 1,
            d_ff: 64,
            factor: 1,
            dropout: 0.1,
            embed: "timeF".to_string(),
            freq: "h".to_string(),
            activation: ActivationArg::Gelu,
            use_norm: true,
            patch_len: 16,
        };
        let lengths = TimeLengths {
            seq_len: 96,
            pred_len: 24,
            label_len: 48,
        };

        let model = TimeXerConfig::new(args)
            .with_initializer(Initializer::Constant { value: 0.01 })
            .init(TaskName::LongTermForecast, lengths, &device);

        let x_enc = Tensor::<B, 3>::zeros([2, 96, 1], &device);
        let x_mark_enc = Tensor::<B, 3>::zeros([2, 96, 4], &device);
        let x_dec = Tensor::<B, 3>::zeros([2, 72, 1], &device);
        let x_mark_dec = Tensor::<B, 3>::zeros([2, 72, 4], &device);

        let output = model.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec);
        assert_eq!(output.dims(), [2, 24, 1]);
    }
}
