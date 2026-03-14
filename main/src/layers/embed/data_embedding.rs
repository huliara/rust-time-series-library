use crate::{
    args::time_embed::TimeEmbed,
    layers::embed::{
        positional_embedding::{PositionalEmbedding, PositionalEmbeddingConfig},
        temporal_embedding::TemporalEmbedding,
        time_feature_embedding::{TimeFeatureEmbedding, TimeFeatureEmbeddingConfig},
        token_embedding::{TokenEmbedding, TokenEmbeddingConfig},
    },
};
use burn::{
    config::Config,
    module::Module,
    nn::{Dropout, DropoutConfig, Initializer, Linear, LinearConfig},
    tensor::{backend::Backend, Tensor},
};

#[derive(Module, Debug)]
pub enum TemporalEmbed<B: Backend> {
    Temporal(TemporalEmbedding<B>),
    TimeFeature(TimeFeatureEmbedding<B>),
}

impl<B: Backend> TemporalEmbed<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        match self {
            Self::Temporal(m) => m.forward(x),
            Self::TimeFeature(m) => m.forward(x),
        }
    }
}

#[derive(Module, Debug)]
pub struct DataEmbedding<B: Backend> {
    value_embedding: TokenEmbedding<B>,
    position_embedding: PositionalEmbedding<B>,
    temporal_embedding: TemporalEmbed<B>,
    dropout: Dropout,
}

#[derive(Config, Debug)]
pub struct DataEmbeddingConfig {
    pub c_in: usize,
    pub d_model: usize,
    pub embed_type: TimeEmbed,
    pub freq: String,
    pub dropout: f64,
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}"
    )]
    pub initializer: Initializer,
}

impl DataEmbeddingConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DataEmbedding<B> {
        let value_embedding = TokenEmbeddingConfig::new(self.c_in, self.d_model)
            .with_initializer(self.initializer.clone())
            .init(device);
        let position_embedding = PositionalEmbeddingConfig::new(self.d_model, 5000).init(device);

        let temporal_embedding = if self.embed_type != TimeEmbed::TimeF {
            TemporalEmbed::Temporal(TemporalEmbedding::new(
                self.d_model,
                &self.embed_type,
                &self.freq,
                device,
            ))
        } else {
            TemporalEmbed::TimeFeature(
                TimeFeatureEmbeddingConfig::new(
                    self.d_model,
                    self.embed_type.clone(),
                    self.freq.clone(),
                )
                .with_initializer(self.initializer.clone())
                .init(device),
            )
        };

        let dropout = DropoutConfig::new(self.dropout).init();

        DataEmbedding {
            value_embedding,
            position_embedding,
            temporal_embedding,
            dropout,
        }
    }
}

impl<B: Backend> DataEmbedding<B> {
    pub fn new(
        c_in: usize,
        d_model: usize,
        embed_type: TimeEmbed,
        freq: String,
        dropout: f64,
        device: &B::Device,
    ) -> Self {
        DataEmbeddingConfig::new(c_in, d_model, embed_type, freq, dropout).init(device)
    }

    pub fn forward(&self, x: Tensor<B, 3>, x_mark: Option<Tensor<B, 3>>) -> Tensor<B, 3> {
        let val = self.value_embedding.forward(x);
        let pos = self.position_embedding.forward(&val);

        // val and pos Should be [Batch, Seq, d_model]

        let mut out = val + pos;

        if let Some(mark) = x_mark {
            // mark: [Batch, Seq, Features]
            out = out + self.temporal_embedding.forward(mark);
        }

        self.dropout.forward(out)
    }
}

#[derive(Module, Debug)]
pub struct DataEmbeddingInverted<B: Backend> {
    value_embedding: Linear<B>,
    dropout: Dropout,
}

#[derive(Config, Debug)]
pub struct DataEmbeddingInvertedConfig {
    pub c_in: usize,
    pub d_model: usize,
    pub embed_type: String,
    pub freq: String,
    pub dropout: f64,
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}"
    )]
    pub initializer: Initializer,
}

impl DataEmbeddingInvertedConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DataEmbeddingInverted<B> {
        let _ = (&self.embed_type, &self.freq);
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
        let x = x.permute([0, 2, 1]); // [Batch, Variate, Seq]

        let inp = if let Some(mark) = x_mark {
            // mark: [Batch, Seq, TimeFeatures]
            let mark = mark.permute([0, 2, 1]); // [Batch, TimeFeatures, Seq]
            Tensor::cat(vec![x, mark], 1) // [Batch, Variate+TimeFeatures, Seq]
        } else {
            x
        };

        let out = self.value_embedding.forward(inp); // [Batch, Var, d_model]
        self.dropout.forward(out)
    }
}

#[derive(Module, Debug)]
pub struct DataEmbeddingWoPos<B: Backend> {
    value_embedding: TokenEmbedding<B>,
    // position_embedding: PositionalEmbedding<B>, // Created but unused in Python? We skip it here to avoid unused field.
    temporal_embedding: TemporalEmbed<B>,
    dropout: Dropout,
}

#[derive(Config, Debug)]
pub struct DataEmbeddingWoPosConfig {
    pub c_in: usize,
    pub d_model: usize,
    pub embed_type: TimeEmbed,
    pub freq: String,
    pub dropout: f64,
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}"
    )]
    pub initializer: Initializer,
}

impl DataEmbeddingWoPosConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DataEmbeddingWoPos<B> {
        let value_embedding = TokenEmbeddingConfig::new(self.c_in, self.d_model)
            .with_initializer(self.initializer.clone())
            .init(device);

        let temporal_embedding = if self.embed_type != TimeEmbed::TimeF {
            TemporalEmbed::Temporal(TemporalEmbedding::new(
                self.d_model,
                &self.embed_type,
                &self.freq,
                device,
            ))
        } else {
            TemporalEmbed::TimeFeature(
                TimeFeatureEmbeddingConfig::new(
                    self.d_model,
                    self.embed_type.clone(),
                    self.freq.clone(),
                )
                .with_initializer(self.initializer.clone())
                .init(device),
            )
        };

        let dropout = DropoutConfig::new(self.dropout).init();

        DataEmbeddingWoPos {
            value_embedding,
            temporal_embedding,
            dropout,
        }
    }
}

impl<B: Backend> DataEmbeddingWoPos<B> {
    pub fn forward(&self, x: Tensor<B, 3>, x_mark: Option<Tensor<B, 3>>) -> Tensor<B, 3> {
        let val = self.value_embedding.forward(x);
        let mut out = val;

        if let Some(mark) = x_mark {
            out = out + self.temporal_embedding.forward(mark);
        }

        self.dropout.forward(out)
    }
}
