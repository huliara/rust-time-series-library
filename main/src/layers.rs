#![allow(dead_code)]
pub mod decomposition;
pub mod embed;
pub mod flatten_head;
pub mod replication_pad_1d;
pub mod self_attention_family;
#[cfg(test)]
pub mod test;
pub mod transformer_enc_dec;
use burn::prelude::Backend;
use burn::tensor::Tensor;

use crate::layers::self_attention_family::attention_layer::AttentionLayer;

use self::embed::{data_embedding_inverted::DataEmbeddingInverted, en_embedding::EnEmbedding};

#[derive(strum::Display)]
pub enum Layer<B: Backend> {
    EnEmbedding(EnEmbedding<B>),
    #[strum(serialize = "DataEmbedding_inverted")]
    DataEmbeddingInverted(DataEmbeddingInverted<B>),
    AttentionLayer(AttentionLayer<B>),
}

impl<B: Backend> Layer<B> {
    pub fn forward(&self, x: Tensor<B, 3>, x_mark: Tensor<B, 3>) -> burn::tensor::Tensor<B, 3> {
        match self {
            Layer::EnEmbedding(layer) => {
                let (tensor, _) = layer.forward(x.swap_dims(1, 2));
                tensor
            }
            Layer::DataEmbeddingInverted(layer) => layer.forward(x, Some(x_mark)),
            Layer::AttentionLayer(layer) => {
                let (tensor, _) = layer.forward(x.clone(), x.clone(), x, None);
                tensor
            }
        }
    }
}
