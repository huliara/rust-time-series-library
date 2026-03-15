#![allow(dead_code)]
pub mod decomposition;
pub mod embed;
pub mod flatten_head;
pub mod replication_pad_1d;
pub mod self_attention_family;
pub mod test;
pub mod transformer_enc_dec;
use burn::prelude::Backend;

use self::embed::en_embedding::EnEmbedding;

#[derive(strum::Display)]
pub enum Layer<B: Backend> {
    EnEmbedding(EnEmbedding<B>),
}

impl<B: Backend> Layer<B> {
    pub fn forward(&self, x: burn::tensor::Tensor<B, 3>) -> burn::tensor::Tensor<B, 3> {
        match self {
            Layer::EnEmbedding(layer) => {
                let (tensor, _) = layer.forward(x.swap_dims(1, 2));
                tensor
            }
        }
    }
}
