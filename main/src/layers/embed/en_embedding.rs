use crate::layers::embed::positional_embedding::{PositionalEmbedding, PositionalEmbeddingConfig};

use burn::{
    config::Config,
    module::{Module, Param},
    nn::{Dropout, DropoutConfig, Initializer, Linear, LinearConfig},
    tensor::{backend::Backend, Distribution, Tensor},
};

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
        let glb_token = match self.initializer {
            Initializer::Constant { value } => Param::from_tensor(Tensor::full(
                [1, self.n_vars, 1, self.d_model],
                value,
                device,
            )),
            _ => Param::from_tensor(Tensor::random(
                [1, self.n_vars, 1, self.d_model],
                Distribution::Normal(0.0, 1.0),
                device,
            )),
        };
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
        let [batch, n_vars, _] = x.dims();
        let glb = self.glb_token.val().repeat_dim(0, batch);

        let x: Tensor<B, 4> = x.unfold(-1, self.patch_len, self.patch_len);
        let x = x
            .clone()
            .reshape([x.dims()[0] * x.dims()[1], x.dims()[2], x.dims()[3]]);
        let x = self.value_embedding.forward(x);
        let x = x.clone() + self.position_embedding.forward(&x);
        let x_dims = x.dims();
        let x = x.reshape([
            -1isize,
            n_vars as isize,
            x_dims[1] as isize,
            x_dims[2] as isize,
        ]);
        let x = Tensor::cat(vec![x, glb], 2);
        let x = x
            .clone()
            .reshape([x.dims()[0] * x.dims()[1], x.dims()[2], x.dims()[3]]);
        (self.dropout.forward(x), n_vars)
    }
}
#[cfg(test)]
mod tests {
    use super::super::super::test::assert_layer_forward;
    use super::EnEmbeddingConfig;
    use crate::layers::Layer;
    use crate::test_utils::dim::Dim;
    use burn::backend::Wgpu;
    use burn::nn::Initializer;

    #[test]
    fn test_enembed_forward() {
        type B = Wgpu;
        let device = Default::default();

        let initializer = Initializer::Constant { value: (0.01) };
        let onedim_config = EnEmbeddingConfig {
            n_vars: 1,
            d_model: 512,
            patch_len: 16,
            dropout: 0.,
            initializer: initializer.clone(),
        };
        let onedim_model = onedim_config.init(&device);
        let multidim_config = EnEmbeddingConfig {
            n_vars: 7,
            d_model: 512,
            patch_len: 16,
            dropout: 0.,
            initializer,
        };
        let multidim_model = multidim_config.init(&device);

        assert_layer_forward::<B>(Dim::Onedim, Layer::EnEmbedding(onedim_model));
        assert_layer_forward::<B>(Dim::Multidim, Layer::EnEmbedding(multidim_model));
    }
}
