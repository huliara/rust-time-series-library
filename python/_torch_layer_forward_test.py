# rustのモデルをテストする際、Rustから呼び出すコード
import inspect
import torch
import torch.nn as nn
from data_provider.data_factory import data_provider
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from _args_mock import Args_mock
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import (
    my_Layernorm,
    moving_avg,
    series_decomp,
    series_decomp_multi,
    EncoderLayer as Autoformer_EncoderLayer,
    Encoder as Autoformer_Encoder,
    DecoderLayer as Autoformer_DecoderLayer,
    Decoder as Autoformer_Decoder,
)
from layers.Conv_Blocks import Inception_Block_V1, Inception_Block_V2
from layers.Crossformer_EncDec import (
    SegMerging,
    scale_block,
    Encoder as Crossformer_Encoder,
    DecoderLayer as Crossformer_DecoderLayer,
    Decoder as Crossformer_Decoder,
)
from layers.DWT_Decomposition import DWT1DForward, DWT1DInverse
from layers.Embed import (
    PositionalEmbedding,
    TokenEmbedding,
    FixedEmbedding,
    TemporalEmbedding,
    TimeFeatureEmbedding,
    DataEmbedding,
    DataEmbedding_inverted,
    DataEmbedding_wo_pos,
    PatchEmbedding,
)
from layers.ETSformer_EncDec import (
    ExponentialSmoothing,
    Feedforward,
    GrowthLayer,
    FourierLayer,
    LevelLayer,
    EncoderLayer as ETSformer_EncoderLayer,
    Encoder as ETSformer_Encoder,
    DampingLayer,
    DecoderLayer as ETSformer_DecoderLayer,
    Decoder as ETSformer_Decoder,
)
from layers.FourierCorrelation import FourierBlock, FourierCrossAttention
from layers.MSGBlock import (
    Predict,
    Attention_Block,
    self_attention,
    FullAttention,
    GraphBlock,
    nconv,
    linear,
    mixprop,
    simpleVIT,
    MultiHeadAttention,
    FeedForward,
)
from layers.MultiWaveletCorrelation import (
    MultiWaveletTransform,
    MultiWaveletCross,
    FourierCrossAttentionW,
    sparseKernelFT1d,
    MWT_CZ1d,
)
from layers.Pyraformer_EncDec import (
    EncoderLayer as Pyraformer_EncoderLayer,
    Encoder as Pyraformer_Encoder,
    ConvLayer as Pyraformer_ConvLayer,
    Bottleneck_Construct,
    PositionwiseFeedForward,
)
from layers.SelfAttention_Family import (
    DSAttention,
    FullAttention,
    ProbAttention,
    AttentionLayer,
    ReformerLayer,
    TwoStageAttentionLayer,
)
from layers.StandardNorm import Normalize
from layers.TimeFilter_layers import (
    GCN,
    mask_moe,
    GraphLearner,
    GraphFilter,
    GraphBlock,
    TimeFilter_Backbone,
)
from layers.Transformer_EncDec import (
    ConvLayer as Transformer_ConvLayer,
    EncoderLayer as Transformer_EncoderLayer,
    Encoder as Transformer_Encoder,
    DecoderLayer as Transformer_DecoderLayer,
    Decoder as Transformer_Decoder,
)
from models.TimeXer import EnEmbedding as TimeXerEnEmbedding

layer_dict = {
    "AutoCorrelation": AutoCorrelation,
    "AutoCorrelationLayer": AutoCorrelationLayer,
    "Autoformer_EncoderLayer": Autoformer_EncoderLayer,
    "Autoformer_Encoder": Autoformer_Encoder,
    "Autoformer_DecoderLayer": Autoformer_DecoderLayer,
    "Autoformer_Decoder": Autoformer_Decoder,
    "scale_block": scale_block,
    "Crossformer_DecoderLayer": Crossformer_DecoderLayer,
    "Crossformer_Decoder": Crossformer_Decoder,
    "DWT1DInverse": DWT1DInverse,
    "DataEmbedding": DataEmbedding,
    "DataEmbedding_inverted": DataEmbedding_inverted,
    "DataEmbedding_wo_pos": DataEmbedding_wo_pos,
    "ExponentialSmoothing": ExponentialSmoothing,
    "GrowthLayer": GrowthLayer,
    "LevelLayer": LevelLayer,
    "ETSformer_EncoderLayer": ETSformer_EncoderLayer,
    "ETSformer_Encoder": ETSformer_Encoder,
    "ETSformer_DecoderLayer": ETSformer_DecoderLayer,
    "ETSformer_Decoder": ETSformer_Decoder,
    "FourierBlock": FourierBlock,
    "FourierCrossAttention": FourierCrossAttention,
    "Attention_Block": Attention_Block,
    "self_attention": self_attention,
    "FullAttention": FullAttention,
    "GraphBlock": GraphBlock,
    "nconv": nconv,
    "mixprop": mixprop,
    "MultiHeadAttention": MultiHeadAttention,
    "MultiWaveletTransform": MultiWaveletTransform,
    "MultiWaveletCross": MultiWaveletCross,
    "FourierCrossAttentionW": FourierCrossAttentionW,
    "Pyraformer_EncoderLayer": Pyraformer_EncoderLayer,
    "Pyraformer_Encoder": Pyraformer_Encoder,
    "Bottleneck_Construct": Bottleneck_Construct,
    "DSAttention": DSAttention,
    "ProbAttention": ProbAttention,
    "AttentionLayer": AttentionLayer,
    "ReformerLayer": ReformerLayer,
    "TwoStageAttentionLayer": TwoStageAttentionLayer,
    "Normalize": Normalize,
    "GCN": GCN,
    "mask_moe": mask_moe,
    "GraphLearner": GraphLearner,
    "GraphFilter": GraphFilter,
    "TimeFilter_Backbone": TimeFilter_Backbone,
    "Transformer_EncoderLayer": Transformer_EncoderLayer,
    "Transformer_Encoder": Transformer_Encoder,
    "Transformer_DecoderLayer": Transformer_DecoderLayer,
    "Transformer_Decoder": Transformer_Decoder,
    "EnEmbedding": TimeXerEnEmbedding,
    "my_Layernorm": my_Layernorm,
    "moving_avg": moving_avg,
    "series_decomp": series_decomp,
    "series_decomp_multi": series_decomp_multi,
    "Inception_Block_V1": Inception_Block_V1,
    "Inception_Block_V2": Inception_Block_V2,
    "SegMerging": SegMerging,
    "Crossformer_Encoder": Crossformer_Encoder,
    "DWT1DForward": DWT1DForward,
    "PositionalEmbedding": PositionalEmbedding,
    "TokenEmbedding": TokenEmbedding,
    "FixedEmbedding": FixedEmbedding,
    "TemporalEmbedding": TemporalEmbedding,
    "TimeFeatureEmbedding": TimeFeatureEmbedding,
    "PatchEmbedding": PatchEmbedding,
    "Feedforward": Feedforward,
    "FourierLayer": FourierLayer,
    "DampingLayer": DampingLayer,
    "Predict": Predict,
    "linear": linear,
    "simpleVIT": simpleVIT,
    "FeedForward": FeedForward,
    "sparseKernelFT1d": sparseKernelFT1d,
    "MWT_CZ1d": MWT_CZ1d,
    "Pyraformer_ConvLayer": Pyraformer_ConvLayer,
    "PositionwiseFeedForward": PositionwiseFeedForward,
    "Transformer_ConvLayer": Transformer_ConvLayer,
}


def _build_full_attention(args, mask_flag=False):
    return FullAttention(
        mask_flag=mask_flag,
        factor=args.factor,
        attention_dropout=args.dropout,
        output_attention=False,
    )


def _build_attention_layer(args, mask_flag=False):
    return AttentionLayer(
        _build_full_attention(args, mask_flag=mask_flag),
        args.d_model,
        args.n_heads,
    )


def _init_simple_layer(name, args):
    if name == "EnEmbedding":
        n_vars = 1 if args.features in ["S", "MS"] else args.enc_in
        return TimeXerEnEmbedding(n_vars, args.d_model, args.patch_len, args.dropout)
    elif name == "AutoCorrelation":
        return AutoCorrelation(
            mask_flag=False,
            factor=args.factor,
            attention_dropout=args.dropout,
            output_attention=False,
        )
    elif name == "AutoCorrelationLayer":
        return AutoCorrelationLayer(
            AutoCorrelation(
                mask_flag=False,
                factor=args.factor,
                attention_dropout=args.dropout,
                output_attention=False,
            ),
            args.d_model,
            args.n_heads,
        )
    elif name == "Autoformer_EncoderLayer":
        return Autoformer_EncoderLayer(
            AutoCorrelationLayer(
                AutoCorrelation(
                    mask_flag=False,
                    factor=args.factor,
                    attention_dropout=args.dropout,
                    output_attention=False,
                ),
                args.d_model,
                args.n_heads,
            ),
            args.d_model,
            d_ff=args.d_ff,
            moving_avg=args.moving_avg,
            dropout=args.dropout,
            activation=args.activation,
        )
    elif name == "Autoformer_Encoder":
        enc_layer = Autoformer_EncoderLayer(
            AutoCorrelationLayer(
                AutoCorrelation(
                    mask_flag=False,
                    factor=args.factor,
                    attention_dropout=args.dropout,
                    output_attention=False,
                ),
                args.d_model,
                args.n_heads,
            ),
            args.d_model,
            d_ff=args.d_ff,
            moving_avg=args.moving_avg,
            dropout=args.dropout,
            activation=args.activation,
        )
        return Autoformer_Encoder([enc_layer], norm_layer=nn.LayerNorm(args.d_model))
    elif name == "Autoformer_DecoderLayer":
        return Autoformer_DecoderLayer(
            _build_attention_layer(args, mask_flag=False),
            _build_attention_layer(args, mask_flag=False),
            args.d_model,
            args.c_out,
            d_ff=args.d_ff,
            moving_avg=args.moving_avg,
            dropout=args.dropout,
            activation=args.activation,
        )
    elif name == "Autoformer_Decoder":
        layer = Autoformer_DecoderLayer(
            _build_attention_layer(args, mask_flag=False),
            _build_attention_layer(args, mask_flag=False),
            args.d_model,
            args.c_out,
            d_ff=args.d_ff,
            moving_avg=args.moving_avg,
            dropout=args.dropout,
            activation=args.activation,
        )
        return Autoformer_Decoder(
            [layer],
            norm_layer=my_Layernorm(args.d_model),
            projection=nn.Linear(args.d_model, args.c_out),
        )
    elif name == "scale_block":
        return scale_block(
            args,
            args.down_sampling_window,
            args.d_model,
            args.n_heads,
            args.d_ff,
            args.e_layers,
            args.dropout,
            seg_num=max(1, args.seq_len // max(1, args.seg_len)),
            factor=args.factor,
        )
    elif name == "Crossformer_DecoderLayer":
        return Crossformer_DecoderLayer(
            _build_attention_layer(args, mask_flag=False),
            _build_attention_layer(args, mask_flag=False),
            args.seg_len,
            args.d_model,
            d_ff=args.d_ff,
            dropout=args.dropout,
        )
    elif name == "Crossformer_Decoder":
        layer = Crossformer_DecoderLayer(
            _build_attention_layer(args, mask_flag=False),
            _build_attention_layer(args, mask_flag=False),
            args.seg_len,
            args.d_model,
            d_ff=args.d_ff,
            dropout=args.dropout,
        )
        return Crossformer_Decoder([layer])
    elif name == "DWT1DInverse":
        return DWT1DInverse()
    elif name == "DataEmbedding":
        return DataEmbedding(
            args.enc_in, args.d_model, args.embed, args.freq, args.dropout
        )
    elif name == "DataEmbedding_inverted":
        return DataEmbedding_inverted(
            args.seq_len, args.d_model, args.embed, args.freq, args.dropout
        )
    elif name == "DataEmbedding_wo_pos":
        return DataEmbedding_wo_pos(
            args.enc_in, args.d_model, args.embed, args.freq, args.dropout
        )
    elif name == "ExponentialSmoothing":
        return ExponentialSmoothing(args.d_model, args.n_heads, args.dropout)
    elif name == "GrowthLayer":
        return GrowthLayer(args.d_model, args.n_heads, dropout=args.dropout)
    elif name == "LevelLayer":
        return LevelLayer(args.d_model, args.c_out, dropout=args.dropout)
    elif name == "ETSformer_EncoderLayer":
        return ETSformer_EncoderLayer(
            args.d_model,
            args.n_heads,
            args.c_out,
            args.seq_len,
            args.pred_len,
            args.top_k,
            dim_feedforward=args.d_ff,
            dropout=args.dropout,
            activation=args.activation,
        )
    elif name == "ETSformer_Encoder":
        layer = ETSformer_EncoderLayer(
            args.d_model,
            args.n_heads,
            args.c_out,
            args.seq_len,
            args.pred_len,
            args.top_k,
            dim_feedforward=args.d_ff,
            dropout=args.dropout,
            activation=args.activation,
        )
        return ETSformer_Encoder([layer])
    elif name == "ETSformer_DecoderLayer":
        return ETSformer_DecoderLayer(
            args.d_model, args.n_heads, args.c_out, args.pred_len, dropout=args.dropout
        )
    elif name == "ETSformer_Decoder":
        layer = ETSformer_DecoderLayer(
            args.d_model, args.n_heads, args.c_out, args.pred_len, dropout=args.dropout
        )
        return ETSformer_Decoder([layer])
    elif name == "FourierBlock":
        return FourierBlock(
            args.d_model,
            args.d_model,
            args.n_heads,
            args.seq_len,
            modes=args.top_k,
        )
    elif name == "FourierCrossAttention":
        return FourierCrossAttention(
            args.d_model,
            args.d_model,
            args.seq_len,
            args.seq_len,
            modes=args.top_k,
            num_heads=args.n_heads,
        )
    elif name == "Attention_Block":
        return Attention_Block(
            args.d_model,
            d_ff=args.d_ff,
            n_heads=args.n_heads,
            dropout=args.dropout,
            activation=args.activation,
        )
    elif name == "self_attention":
        return self_attention(FullAttention, args.d_model, args.n_heads)
    elif name == "FullAttention":
        return _build_full_attention(args, mask_flag=False)
    elif name == "GraphBlock":
        return GraphBlock(
            args.d_model,
            args.enc_in,
            d_ff=args.d_ff,
            n_heads=args.n_heads,
            top_p=args.top_p,
            dropout=args.dropout,
            in_dim=args.seq_len,
        )
    elif name == "nconv":
        return nconv()
    elif name == "mixprop":
        return mixprop(
            args.conv_channel,
            args.skip_channel,
            args.gcn_depth,
            args.gcn_dropout,
            args.propalpha,
        )
    elif name == "MultiHeadAttention":
        return MultiHeadAttention(args.d_model, args.n_heads, args.dropout)
    elif name == "MultiWaveletTransform":
        return MultiWaveletTransform(ich=args.d_model, k=args.top_k, alpha=args.d_model)
    elif name == "MultiWaveletCross":
        return MultiWaveletCross(
            args.d_model,
            args.d_model,
            args.seq_len,
            args.seq_len,
            modes=args.top_k,
            c=args.d_model,
            k=args.top_k,
            ich=args.d_model,
        )
    elif name == "FourierCrossAttentionW":
        return FourierCrossAttentionW(
            args.d_model,
            args.d_model,
            args.seq_len,
            args.seq_len,
            modes=args.top_k,
        )
    elif name == "Pyraformer_EncoderLayer":
        return Pyraformer_EncoderLayer(
            args.d_model,
            args.d_ff,
            args.n_heads,
            dropout=args.dropout,
        )
    elif name == "Pyraformer_Encoder":
        return Pyraformer_Encoder(args, window_size=[2, 2, 2], inner_size=3)
    elif name == "Bottleneck_Construct":
        return Bottleneck_Construct(args.d_model, [2, 2, 2], args.d_ff)
    elif name == "DSAttention":
        return DSAttention(
            mask_flag=False,
            factor=args.factor,
            attention_dropout=args.dropout,
            output_attention=False,
        )
    elif name == "ProbAttention":
        return ProbAttention(
            mask_flag=False,
            factor=args.factor,
            attention_dropout=args.dropout,
            output_attention=False,
        )
    elif name == "AttentionLayer":
        return _build_attention_layer(args, mask_flag=False)
    elif name == "ReformerLayer":
        return ReformerLayer(
            _build_attention_layer(args, mask_flag=False),
            args.d_model,
            args.n_heads,
            causal=False,
        )
    elif name == "TwoStageAttentionLayer":
        return TwoStageAttentionLayer(
            args,
            seg_num=max(1, args.seq_len // max(1, args.seg_len)),
            factor=args.factor,
            d_model=args.d_model,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            dropout=args.dropout,
        )
    elif name == "Normalize":
        return Normalize(args.enc_in)
    elif name == "GCN":
        return GCN(args.node_dim, args.n_heads)
    elif name == "mask_moe":
        return mask_moe(args.enc_in, top_p=args.top_p, in_dim=args.seq_len)
    elif name == "GraphLearner":
        return GraphLearner(
            args.d_model, args.enc_in, top_p=args.top_p, in_dim=args.seq_len
        )
    elif name == "GraphFilter":
        return GraphFilter(
            args.d_model,
            args.enc_in,
            n_heads=args.n_heads,
            top_p=args.top_p,
            dropout=args.dropout,
            in_dim=args.seq_len,
        )
    elif name == "TimeFilter_Backbone":
        return TimeFilter_Backbone(
            args.d_model,
            args.enc_in,
            d_ff=args.d_ff,
            n_heads=args.n_heads,
            n_blocks=max(1, args.e_layers),
            top_p=args.top_p,
            dropout=args.dropout,
            in_dim=args.seq_len,
        )
    elif name == "Transformer_EncoderLayer":
        return Transformer_EncoderLayer(
            _build_attention_layer(args, mask_flag=False),
            args.d_model,
            d_ff=args.d_ff,
            dropout=args.dropout,
            activation=args.activation,
        )
    elif name == "Transformer_Encoder":
        layer = Transformer_EncoderLayer(
            _build_attention_layer(args, mask_flag=False),
            args.d_model,
            d_ff=args.d_ff,
            dropout=args.dropout,
            activation=args.activation,
        )
        return Transformer_Encoder([layer], norm_layer=nn.LayerNorm(args.d_model))
    elif name == "Transformer_DecoderLayer":
        return Transformer_DecoderLayer(
            _build_attention_layer(args, mask_flag=False),
            _build_attention_layer(args, mask_flag=False),
            args.d_model,
            d_ff=args.d_ff,
            dropout=args.dropout,
            activation=args.activation,
        )
    elif name == "Transformer_Decoder":
        layer = Transformer_DecoderLayer(
            _build_attention_layer(args, mask_flag=False),
            _build_attention_layer(args, mask_flag=False),
            args.d_model,
            d_ff=args.d_ff,
            dropout=args.dropout,
            activation=args.activation,
        )
        return Transformer_Decoder(
            [layer],
            norm_layer=nn.LayerNorm(args.d_model),
            projection=nn.Linear(args.d_model, args.c_out),
        )
    elif name == "my_Layernorm":
        return my_Layernorm(args.d_model)
    elif name == "moving_avg":
        return moving_avg(args.moving_avg, 1)
    elif name == "series_decomp":
        return series_decomp(args.moving_avg)
    elif name == "series_decomp_multi":
        return series_decomp_multi([args.moving_avg])
    elif name == "Inception_Block_V1":
        return Inception_Block_V1(args.d_model, args.d_model, args.num_kernels)
    elif name == "Inception_Block_V2":
        return Inception_Block_V2(args.d_model, args.d_model, args.num_kernels)
    elif name == "SegMerging":
        return SegMerging(args.d_model, args.down_sampling_window)
    elif name == "Crossformer_Encoder":
        # attn_layers is constructed in the original model path; keep empty for layer-only init.
        return Crossformer_Encoder([])
    elif name == "DWT1DForward":
        return DWT1DForward()
    elif name == "PositionalEmbedding":
        return PositionalEmbedding(args.d_model)
    elif name == "TokenEmbedding":
        return TokenEmbedding(args.enc_in, args.d_model)
    elif name == "FixedEmbedding":
        return FixedEmbedding(args.enc_in, args.d_model)
    elif name == "TemporalEmbedding":
        return TemporalEmbedding(args.d_model, args.embed, args.freq)
    elif name == "TimeFeatureEmbedding":
        return TimeFeatureEmbedding(args.d_model, args.embed, args.freq)
    elif name == "PatchEmbedding":
        return PatchEmbedding(
            args.d_model,
            args.patch_len,
            args.patch_len,
            args.patch_len,
            args.dropout,
        )
    elif name == "Feedforward":
        return Feedforward(args.d_model, args.d_ff, args.dropout, args.activation)
    elif name == "FourierLayer":
        return FourierLayer(args.d_model, args.pred_len)
    elif name == "DampingLayer":
        return DampingLayer(args.pred_len, args.n_heads, args.dropout)
    elif name == "Predict":
        return Predict(
            args.individual, args.c_out, args.seq_len, args.pred_len, args.dropout
        )
    elif name == "linear":
        return linear(args.enc_in, args.c_out)
    elif name == "simpleVIT":
        return simpleVIT(args.enc_in, args.d_model, dropout=args.dropout)
    elif name == "FeedForward":
        return FeedForward(args.d_model, args.d_ff)
    elif name == "sparseKernelFT1d":
        return sparseKernelFT1d(args.top_k, args.d_model)
    elif name == "MWT_CZ1d":
        return MWT_CZ1d(k=args.top_k, alpha=args.d_model)
    elif name == "Pyraformer_ConvLayer":
        return Pyraformer_ConvLayer(args.d_model, args.d_conv)
    elif name == "PositionwiseFeedForward":
        return PositionwiseFeedForward(args.d_model, args.d_ff, args.dropout)
    elif name == "Transformer_ConvLayer":
        return Transformer_ConvLayer(args.d_model)
    else:
        raise ValueError(f"Unsupported simple layer: {name}")


def init_layer(name, args):
    if name in layer_dict:
        return _init_simple_layer(name, args)

    raise ValueError(f"Unsupported layer for now: {name}")


def torch_forward_test_multidim(name):
    args = Args_mock()
    return _torch_layer_forward_test(name, args)


def torch_forward_test_onedim(name):
    args = Args_mock()
    args.features = "S"
    return _torch_layer_forward_test(name, args)


def _torch_layer_forward_test(name, args):
    exp = Exp_Long_Term_Forecast(args)
    device = exp.device
    module: nn.Module = init_layer(name, args).float()
    module.to(device)
    module.eval()

    for param_name, param in module.named_parameters():
        print("param name:", param_name)
        if "weight" in param_name and module.__class__.__name__ == "EnEmbedding":
            nn.init.constant_(param, 0.01)
        elif "weight" in param_name:
            nn.init.constant_(param, 0.01)
        elif "bias" in param_name:
            nn.init.constant_(param, 0.0)
        elif "glb_token" in param_name:
            nn.init.constant_(param, 0.01)

    _, data_loader = data_provider(args, flag="test")
    all_outputs = []
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len :, :]).float()
            dec_inp = (
                torch.cat([batch_y[:, : args.label_len, :], dec_inp], dim=1)
                .float()
                .to(device)
            )
            if name == "EnEmbedding":
                outputs, _ = module(batch_x.permute(0, 2, 1))
            elif name in ["Crossformer_DecoderLayer", "Crossformer_Decoder"]:
                outputs = module(dec_inp, batch_x, batch_y_mark, batch_x_mark)
            elif name in [
                "DataEmbedding",
                "DataEmbedding_inverted",
                "DataEmbedding_wo_pos",
            ]:
                outputs = module(batch_x, batch_x_mark)
            else:
                outputs = module(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            if outputs.dim() == 3 and outputs.size(1) >= args.pred_len:
                f_dim = -1 if args.features == "MS" else 0
                outputs = outputs[:, -args.pred_len :, f_dim:]

            all_outputs.append(outputs)
    all_outputs = torch.cat(all_outputs, dim=0)
    print("all_outputs shape:", all_outputs.shape)
    return all_outputs


if __name__ == "__main__":
    args = Args_mock()
    args.features = "S"
    output = _torch_layer_forward_test("TimeXer", args)
    print("output shape:", output.shape)
