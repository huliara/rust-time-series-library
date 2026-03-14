use burn::{
    config::Config,
    module::Module,
    nn::{Initializer, Linear, LinearConfig},
    tensor::{backend::Backend, Tensor},
};

use crate::args::time_embed::TimeEmbed;

#[derive(Module, Debug)]
pub struct TimeFeatureEmbedding<B: Backend> {
    embed: Linear<B>,
}

#[derive(Config, Debug)]
pub struct TimeFeatureEmbeddingConfig {
    pub d_model: usize,
    pub embed_type: TimeEmbed,
    pub freq: String,
    #[config(default = "Initializer::KaimingNormal{gain:1.0, fan_out_only:false}")]
    pub initializer: Initializer,
}

impl TimeFeatureEmbeddingConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TimeFeatureEmbedding<B> {
        let _ = &self.embed_type;
        let d_inp = match self.freq.as_str() {
            "h" => 4,
            "t" => 5,
            "s" => 6,
            "m" | "a" => 1,
            "w" => 2,
            "d" | "b" => 3,
            _ => 4,
        };

        let embed = LinearConfig::new(d_inp, self.d_model)
            .with_bias(false)
            .with_initializer(self.initializer.clone())
            .init(device);

        TimeFeatureEmbedding { embed }
    }
}

impl<B: Backend> TimeFeatureEmbedding<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.embed.forward(x)
    }
}
