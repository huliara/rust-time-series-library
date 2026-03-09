/// PyTorch→Burn の変換テーブル。
/// `py_module`: 対応する Python モジュール名の部分 ("nn.Linear" → "Linear")
/// `burn_type`: Burn の型名 (ジェネリクスなし)
/// `burn_import`: Burn の use パス
/// `needs_device`: `.init(device)` が必要か
/// `needs_backend`: `<B>` が必要か
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct LayerInfo {
    pub burn_type: &'static str,
    pub burn_import: &'static str,
    pub config_type: &'static str,
    pub needs_device: bool,
    pub needs_backend: bool,
    /// forward を `.forward(...)` に変換すべきか (activation 関数なら false)
    pub is_module: bool,
}

/// "Linear" -> LayerInfo
pub fn layer_table() -> HashMap<&'static str, LayerInfo> {
    let mut m = HashMap::new();

    // ── Linear ──────────────────────────────────────────────────
    m.insert(
        "Linear",
        LayerInfo {
            burn_type: "Linear",
            burn_import: "burn::nn::Linear",
            config_type: "LinearConfig",
            needs_device: true,
            needs_backend: true,
            is_module: true,
        },
    );

    // ── Conv ────────────────────────────────────────────────────
    m.insert(
        "Conv1d",
        LayerInfo {
            burn_type: "Conv1d",
            burn_import: "burn::nn::conv::Conv1d",
            config_type: "Conv1dConfig",
            needs_device: true,
            needs_backend: true,
            is_module: true,
        },
    );
    m.insert(
        "Conv2d",
        LayerInfo {
            burn_type: "Conv2d",
            burn_import: "burn::nn::conv::Conv2d",
            config_type: "Conv2dConfig",
            needs_device: true,
            needs_backend: true,
            is_module: true,
        },
    );
    m.insert(
        "ConvTranspose1d",
        LayerInfo {
            burn_type: "ConvTranspose1d",
            burn_import: "burn::nn::conv::ConvTranspose1d",
            config_type: "ConvTranspose1dConfig",
            needs_device: true,
            needs_backend: true,
            is_module: true,
        },
    );
    m.insert(
        "ConvTranspose2d",
        LayerInfo {
            burn_type: "ConvTranspose2d",
            burn_import: "burn::nn::conv::ConvTranspose2d",
            config_type: "ConvTranspose2dConfig",
            needs_device: true,
            needs_backend: true,
            is_module: true,
        },
    );

    // ── Pooling ─────────────────────────────────────────────────
    m.insert(
        "AdaptiveAvgPool1d",
        LayerInfo {
            burn_type: "AdaptiveAvgPool1d",
            burn_import: "burn::nn::pool::AdaptiveAvgPool1d",
            config_type: "AdaptiveAvgPool1dConfig",
            needs_device: false,
            needs_backend: false,
            is_module: true,
        },
    );
    m.insert(
        "AdaptiveAvgPool2d",
        LayerInfo {
            burn_type: "AdaptiveAvgPool2d",
            burn_import: "burn::nn::pool::AdaptiveAvgPool2d",
            config_type: "AdaptiveAvgPool2dConfig",
            needs_device: false,
            needs_backend: false,
            is_module: true,
        },
    );
    m.insert(
        "MaxPool1d",
        LayerInfo {
            burn_type: "MaxPool1d",
            burn_import: "burn::nn::pool::MaxPool1d",
            config_type: "MaxPool1dConfig",
            needs_device: false,
            needs_backend: false,
            is_module: true,
        },
    );
    m.insert(
        "MaxPool2d",
        LayerInfo {
            burn_type: "MaxPool2d",
            burn_import: "burn::nn::pool::MaxPool2d",
            config_type: "MaxPool2dConfig",
            needs_device: false,
            needs_backend: false,
            is_module: true,
        },
    );
    m.insert(
        "AvgPool1d",
        LayerInfo {
            burn_type: "AvgPool1d",
            burn_import: "burn::nn::pool::AvgPool1d",
            config_type: "AvgPool1dConfig",
            needs_device: false,
            needs_backend: false,
            is_module: true,
        },
    );
    m.insert(
        "AvgPool2d",
        LayerInfo {
            burn_type: "AvgPool2d",
            burn_import: "burn::nn::pool::AvgPool2d",
            config_type: "AvgPool2dConfig",
            needs_device: false,
            needs_backend: false,
            is_module: true,
        },
    );

    // ── Norm ────────────────────────────────────────────────────
    m.insert(
        "BatchNorm1d",
        LayerInfo {
            burn_type: "BatchNorm",
            burn_import: "burn::nn::BatchNorm",
            config_type: "BatchNormConfig",
            needs_device: true,
            needs_backend: true,
            is_module: true,
        },
    );
    m.insert(
        "BatchNorm2d",
        LayerInfo {
            burn_type: "BatchNorm",
            burn_import: "burn::nn::BatchNorm",
            config_type: "BatchNormConfig",
            needs_device: true,
            needs_backend: true,
            is_module: true,
        },
    );
    m.insert(
        "LayerNorm",
        LayerInfo {
            burn_type: "LayerNorm",
            burn_import: "burn::nn::LayerNorm",
            config_type: "LayerNormConfig",
            needs_device: true,
            needs_backend: true,
            is_module: true,
        },
    );
    m.insert(
        "GroupNorm",
        LayerInfo {
            burn_type: "GroupNorm",
            burn_import: "burn::nn::GroupNorm",
            config_type: "GroupNormConfig",
            needs_device: true,
            needs_backend: true,
            is_module: true,
        },
    );
    m.insert(
        "InstanceNorm1d",
        LayerInfo {
            burn_type: "InstanceNorm",
            burn_import: "burn::nn::InstanceNorm",
            config_type: "InstanceNormConfig",
            needs_device: true,
            needs_backend: true,
            is_module: true,
        },
    );
    m.insert(
        "InstanceNorm2d",
        LayerInfo {
            burn_type: "InstanceNorm",
            burn_import: "burn::nn::InstanceNorm",
            config_type: "InstanceNormConfig",
            needs_device: true,
            needs_backend: true,
            is_module: true,
        },
    );
    m.insert(
        "RMSNorm",
        LayerInfo {
            burn_type: "RmsNorm",
            burn_import: "burn::nn::RmsNorm",
            config_type: "RmsNormConfig",
            needs_device: true,
            needs_backend: true,
            is_module: true,
        },
    );

    // ── Activation ──────────────────────────────────────────────
    m.insert(
        "ReLU",
        LayerInfo {
            burn_type: "Relu",
            burn_import: "burn::nn::Relu",
            config_type: "ReluConfig",
            needs_device: false,
            needs_backend: false,
            is_module: true,
        },
    );
    m.insert(
        "GELU",
        LayerInfo {
            burn_type: "Gelu",
            burn_import: "burn::nn::Gelu",
            config_type: "GeluConfig",
            needs_device: false,
            needs_backend: false,
            is_module: true,
        },
    );
    m.insert(
        "SiLU",
        LayerInfo {
            burn_type: "Silu",
            burn_import: "burn::nn::Silu",
            config_type: "SiluConfig",
            needs_device: false,
            needs_backend: false,
            is_module: true,
        },
    );
    m.insert(
        "Tanh",
        LayerInfo {
            burn_type: "Tanh",
            burn_import: "burn::nn::Tanh",
            config_type: "TanhConfig",
            needs_device: false,
            needs_backend: false,
            is_module: true,
        },
    );
    m.insert(
        "Sigmoid",
        LayerInfo {
            burn_type: "Sigmoid",
            burn_import: "burn::nn::Sigmoid",
            config_type: "SigmoidConfig",
            needs_device: false,
            needs_backend: false,
            is_module: true,
        },
    );
    m.insert(
        "LeakyReLU",
        LayerInfo {
            burn_type: "LeakyRelu",
            burn_import: "burn::nn::LeakyRelu",
            config_type: "LeakyReluConfig",
            needs_device: false,
            needs_backend: false,
            is_module: true,
        },
    );
    m.insert(
        "Mish",
        LayerInfo {
            burn_type: "Mish",
            burn_import: "burn::nn::Mish",
            config_type: "MishConfig",
            needs_device: false,
            needs_backend: false,
            is_module: true,
        },
    );

    // ── Dropout ─────────────────────────────────────────────────
    m.insert(
        "Dropout",
        LayerInfo {
            burn_type: "Dropout",
            burn_import: "burn::nn::Dropout",
            config_type: "DropoutConfig",
            needs_device: false,
            needs_backend: false,
            is_module: true,
        },
    );
    m.insert(
        "AlphaDropout",
        LayerInfo {
            burn_type: "Dropout",
            burn_import: "burn::nn::Dropout",
            config_type: "DropoutConfig",
            needs_device: false,
            needs_backend: false,
            is_module: true,
        },
    );

    // ── Embedding ───────────────────────────────────────────────
    m.insert(
        "Embedding",
        LayerInfo {
            burn_type: "Embedding",
            burn_import: "burn::nn::Embedding",
            config_type: "EmbeddingConfig",
            needs_device: true,
            needs_backend: true,
            is_module: true,
        },
    );

    // ── Recurrent ───────────────────────────────────────────────
    m.insert(
        "LSTM",
        LayerInfo {
            burn_type: "Lstm",
            burn_import: "burn::nn::lstm::Lstm",
            config_type: "LstmConfig",
            needs_device: true,
            needs_backend: true,
            is_module: true,
        },
    );
    m.insert(
        "GRU",
        LayerInfo {
            burn_type: "Gru",
            burn_import: "burn::nn::gru::Gru",
            config_type: "GruConfig",
            needs_device: true,
            needs_backend: true,
            is_module: true,
        },
    );

    // ── Attention ───────────────────────────────────────────────
    m.insert(
        "MultiheadAttention",
        LayerInfo {
            burn_type: "MultiHeadAttention",
            burn_import: "burn::nn::attention::MultiHeadAttention",
            config_type: "MultiHeadAttentionConfig",
            needs_device: true,
            needs_backend: true,
            is_module: true,
        },
    );

    m
}

/// torch / F の関数呼び出しを Burn の式に変換するテーブル
/// key: "torch.relu" / "F.relu" など
pub fn fn_table() -> HashMap<&'static str, &'static str> {
    let mut m = HashMap::new();
    // activation
    m.insert("torch.relu", "burn::tensor::activation::relu");
    m.insert("F.relu", "burn::tensor::activation::relu");
    m.insert("torch.sigmoid", "burn::tensor::activation::sigmoid");
    m.insert("F.sigmoid", "burn::tensor::activation::sigmoid");
    m.insert("torch.tanh", "burn::tensor::activation::tanh");
    m.insert("F.tanh", "burn::tensor::activation::tanh");
    m.insert("torch.gelu", "burn::tensor::activation::gelu");
    m.insert("F.gelu", "burn::tensor::activation::gelu");
    m.insert("torch.silu", "burn::tensor::activation::silu");
    m.insert("F.silu", "burn::tensor::activation::silu");
    m.insert("torch.mish", "burn::tensor::activation::mish");
    m.insert("F.mish", "burn::tensor::activation::mish");
    m.insert("F.leaky_relu", "burn::tensor::activation::leaky_relu");
    m.insert("F.softmax", "burn::tensor::activation::softmax");
    m.insert("torch.softmax", "burn::tensor::activation::softmax");
    m.insert("F.log_softmax", "burn::tensor::activation::log_softmax");
    m.insert("torch.log_softmax", "burn::tensor::activation::log_softmax");
    m
}
