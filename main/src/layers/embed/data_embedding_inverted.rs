use burn::{
    config::Config,
    module::Module,
    nn::{Dropout, DropoutConfig, Initializer, Linear, LinearConfig},
    tensor::{backend::Backend, Tensor},
};

#[derive(Module, Debug)]
pub struct DataEmbeddingInverted<B: Backend> {
    value_embedding: Linear<B>,
    dropout: Dropout,
}

#[derive(Config, Debug)]
pub struct DataEmbeddingInvertedConfig {
    pub c_in: usize,
    pub d_model: usize,
    pub dropout: f64,
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}"
    )]
    pub initializer: Initializer,
}

impl DataEmbeddingInvertedConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DataEmbeddingInverted<B> {
        let value_embedding = LinearConfig::new(self.c_in, self.d_model)
            .with_initializer(self.initializer.clone())
            .init(device);
        let dropout = DropoutConfig::new(self.dropout).init();
        DataEmbeddingInverted {
            value_embedding,
            dropout,
        }
    }
}

impl<B: Backend> DataEmbeddingInverted<B> {
    pub fn forward(&self, x: Tensor<B, 3>, x_mark: Option<Tensor<B, 3>>) -> Tensor<B, 3> {
        // x: [Batch, Seq, Variate]
        let x = x.swap_dims(1, 2); // [Batch, Variate, Seq]

        let inp = if let Some(mark) = x_mark {
            // mark: [Batch, Seq, TimeFeatures]
            Tensor::cat(vec![x, mark.swap_dims(1, 2)], 1) // [Batch, Variate+TimeFeatures, Seq]
        } else {
            x
        };

        let out = self.value_embedding.forward(inp); // [Batch, Var, d_model]
        self.dropout.forward(out)
    }
}

#[cfg(test)]
mod tests {

    use super::DataEmbeddingInvertedConfig;
    use crate::layers::{test::assert_layer_forward, Layer};
    use crate::test_utils::dim::Dim;
    use burn::backend::Wgpu;
    use burn::nn::Initializer;

    #[test]

    fn test_data_embedding_inverted_forward() {
        type B = Wgpu;
        let device = Default::default();

        let initializer = Initializer::Constant { value: (0.01) };
        let config = DataEmbeddingInvertedConfig {
            c_in: 96,
            d_model: 512,
            dropout: 0.0,
            initializer,
        };
        let embedding = config.init(&device);

        assert_layer_forward::<B>(Dim::Onedim, Layer::DataEmbeddingInverted(embedding.clone()));
        assert_layer_forward::<B>(Dim::Multidim, Layer::DataEmbeddingInverted(embedding));
    }
}
