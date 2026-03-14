use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::{Conv1d, Conv1dConfig},
        Initializer, PaddingConfig1d,
    },
    tensor::{backend::Backend, Tensor},
};

#[derive(Module, Debug)]
pub struct TokenEmbedding<B: Backend> {
    conv: Conv1d<B>,
}

#[derive(Config, Debug)]
pub struct TokenEmbeddingConfig {
    pub c_in: usize,
    pub d_model: usize,
    #[config(default = "Initializer::KaimingNormal{gain:1.0, fan_out_only:false}")]
    pub initializer: Initializer,
}

impl TokenEmbeddingConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TokenEmbedding<B> {
        let conv = Conv1dConfig::new(self.c_in, self.d_model, 3)
            .with_padding(PaddingConfig1d::Explicit(0)) // We do manual circular padding
            .with_bias(false)
            .with_initializer(self.initializer.clone())
            .init(device);

        TokenEmbedding { conv }
    }
}

impl<B: Backend> TokenEmbedding<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // x: [batch, seq_len, c_in]
        let x = x.permute([0, 2, 1]); // [batch, c_in, seq_len]

        // Circular Padding (padding=1)
        let dims = x.dims();
        let seq_len = dims[2];

        // slice args: [start..end] for each dim
        let x_first = x.clone().slice([0..dims[0], 0..dims[1], 0..1]);
        // To avoid "attempt to subtract with overflow" or incorrect slicing for 0-len seq (though unlikely here)
        let x_last = if seq_len > 0 {
            x.clone()
                .slice([0..dims[0], 0..dims[1], seq_len - 1..seq_len])
        } else {
            // Handle edge case if needed, but usually seq_len > 0
            x.clone().slice([0..dims[0], 0..dims[1], 0..0])
        };

        let x_padded = Tensor::cat(vec![x_last, x, x_first], 2);

        let out = self.conv.forward(x_padded); // [batch, d_model, seq_len]

        out.permute([0, 2, 1]) // [batch, seq_len, d_model]
    }
}
