use crate::layers::self_attention_family::full_attention::FullAttentionConfig;

use super::full_attention::FullAttention;
use burn::config::Config;
use burn::module::Module;
use burn::nn::{Initializer, Linear, LinearConfig};
use burn::prelude::Bool;
use burn::tensor::{backend::Backend, Tensor};

#[derive(Config, Debug)]
pub struct AttentionLayerConfig {
    pub inner_attention: FullAttentionConfig,
    pub d_model: usize,
    pub n_heads: usize,
    pub d_keys: Option<usize>,
    pub d_values: Option<usize>,
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}"
    )]
    pub initializer: Initializer,
}

impl AttentionLayerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> AttentionLayer<B> {
        let d_keys = self.d_keys.unwrap_or(self.d_model / self.n_heads);
        let d_values = self.d_values.unwrap_or(self.d_model / self.n_heads);
        let inner_attention = self.inner_attention.init();

        let query_projection = LinearConfig::new(self.d_model, d_keys * self.n_heads)
            .with_initializer(self.initializer.clone())
            .init(device);
        let key_projection = LinearConfig::new(self.d_model, d_keys * self.n_heads)
            .with_initializer(self.initializer.clone())
            .init(device);
        let value_projection = LinearConfig::new(self.d_model, d_values * self.n_heads)
            .with_initializer(self.initializer.clone())
            .init(device);
        let out_projection = LinearConfig::new(d_values * self.n_heads, self.d_model)
            .with_initializer(self.initializer.clone())
            .init(device);

        AttentionLayer {
            inner_attention,
            query_projection,
            key_projection,
            value_projection,
            out_projection,
            n_heads: self.n_heads,
        }
    }
}

#[derive(Module, Debug)]
pub struct AttentionLayer<B: Backend> {
    inner_attention: FullAttention,
    query_projection: Linear<B>,
    key_projection: Linear<B>,
    value_projection: Linear<B>,
    out_projection: Linear<B>,
    n_heads: usize,
}

impl<B: Backend> AttentionLayer<B> {
    pub fn forward(
        &self,
        queries: Tensor<B, 3>,
        keys: Tensor<B, 3>,
        values: Tensor<B, 3>,
        attn_mask: Option<Tensor<B, 4, Bool>>,
    ) -> (Tensor<B, 3>, Option<Tensor<B, 4>>) {
        let [b, l, _] = queries.dims();
        let [_, s, _] = keys.dims();
        let h = self.n_heads;

        let queries = self.query_projection.forward(queries);
        let queries = queries.reshape([b as isize, l as isize, h as isize, -1isize]);

        let keys = self.key_projection.forward(keys);
        let keys = keys.reshape([b as isize, s as isize, h as isize, -1isize]);

        let values = self.value_projection.forward(values);
        let values = values.reshape([b as isize, s as isize, h as isize, -1isize]);

        let (out, attn) = self
            .inner_attention
            .forward(queries, keys, values, attn_mask);

        let out = out.reshape([b as isize, l as isize, -1isize]);
        let out = self.out_projection.forward(out);

        (out, attn)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use burn::tensor::Shape;
    use burn_ndarray::NdArray;

    #[test]
    fn test_attention_layer_forward() {
        type B = NdArray;
        let device = Default::default();

        let inner_config = FullAttentionConfig {
            mask_flag: false,
            scale: None,
            attention_dropout: 0.1,
            output_attention: true,
        };

        let config = AttentionLayerConfig {
            inner_attention: inner_config,
            d_model: 32,
            n_heads: 4,
            d_keys: None,
            d_values: None,
            initializer: Initializer::Uniform {
                min: -0.1,
                max: 0.1,
            },
        };
        let attention_layer = config.init::<B>(&device);

        let b_size = 2;
        let l = 8;
        let s = 8;
        let d_model = 32;

        let queries = Tensor::<B, 3>::zeros([b_size, l, d_model], &device);
        let keys = Tensor::<B, 3>::zeros([b_size, s, d_model], &device);
        let values = Tensor::<B, 3>::zeros([b_size, s, d_model], &device);

        let (out, attn) = attention_layer.forward(queries, keys, values, None);

        assert_eq!(out.shape(), Shape::new([b_size, l, d_model]));

        if let Some(attn_tensor) = attn {
            assert_eq!(
                attn_tensor.shape(),
                Shape::new([b_size, config.n_heads, l, s])
            );
        } else {
            panic!("Expected attention output");
        }
    }
}
