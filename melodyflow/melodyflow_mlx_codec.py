#!/usr/bin/env python3
"""MLX port of MelodyFlow compression model (SEANet + no-quant path)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch


def _to_mx(t: torch.Tensor, dtype: Any) -> mx.array:
    arr = mx.array(t.detach().cpu().float().numpy())
    if arr.dtype != dtype:
        arr = arr.astype(dtype)
    return arr


def _conv_weight_to_mx(weight_oik: torch.Tensor, dtype: Any) -> mx.array:
    # torch Conv1d: [out, in, kernel] -> mlx Conv1d: [out, kernel, in]
    w = np.moveaxis(weight_oik.detach().cpu().float().numpy(), 1, 2)
    arr = mx.array(w)
    if arr.dtype != dtype:
        arr = arr.astype(dtype)
    return arr


def _convtr_weight_to_mx(weight_iok: torch.Tensor, dtype: Any) -> mx.array:
    # torch ConvTranspose1d: [in, out, kernel] -> mlx ConvTranspose1d: [out, kernel, in]
    w = np.moveaxis(weight_iok.detach().cpu().float().numpy(), 0, 2)
    arr = mx.array(w)
    if arr.dtype != dtype:
        arr = arr.astype(dtype)
    return arr


def _get_extra_padding_for_conv1d(length: int, kernel_size: int, stride: int, padding_total: int = 0) -> int:
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return int(ideal_length - length)


def _pad1d(x: mx.array, padding_left: int, padding_right: int, mode: str = "constant") -> mx.array:
    if mode != "reflect":
        return mx.pad(x, ((0, 0), (padding_left, padding_right), (0, 0)))

    # x: [B, T, C]
    length = int(x.shape[1])
    left = int(padding_left)
    right = int(padding_right)
    if left == 0 and right == 0:
        return x

    if length <= 1:
        return mx.pad(x, ((0, 0), (left, right), (0, 0)))

    left_start = 1
    left_end = min(left + 1, length)
    left_ref = x[:, left_start:left_end, :]
    if left_ref.shape[1] > 0:
        left_ref = left_ref[:, ::-1, :]

    right_start = max(length - (right + 1), 0)
    right_ref = x[:, right_start:-1, :]
    if right_ref.shape[1] > 0:
        right_ref = right_ref[:, ::-1, :]

    parts = []
    if left > 0:
        if int(left_ref.shape[1]) < left:
            left_pad = mx.pad(left_ref, ((0, 0), (0, left - int(left_ref.shape[1])), (0, 0)))
            parts.append(left_pad)
        else:
            parts.append(left_ref[:, :left, :])
    parts.append(x)
    if right > 0:
        if int(right_ref.shape[1]) < right:
            right_pad = mx.pad(right_ref, ((0, 0), (0, right - int(right_ref.shape[1])), (0, 0)))
            parts.append(right_pad)
        else:
            parts.append(right_ref[:, :right, :])
    return mx.concatenate(parts, axis=1)


class MlxSnake1d:
    def __init__(self, alpha_1c1: mx.array):
        # torch shape is [1, C, 1]; convert to channel-last broadcast [1, 1, C]
        if alpha_1c1.ndim != 3:
            raise ValueError(f"Expected alpha with 3 dims, got shape={alpha_1c1.shape}")
        self.alpha = mx.transpose(alpha_1c1, (0, 2, 1))

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, T, C]
        a = self.alpha.astype(x.dtype)
        return x + (1.0 / (a + 1e-9)) * mx.square(mx.sin(a * x))


class MlxStreamableConv1d:
    def __init__(
        self,
        *,
        weight_oki: mx.array,
        bias_o: mx.array | None,
        stride: int,
        dilation: int,
        causal: bool,
        pad_mode: str,
    ):
        out_channels, kernel_size, in_channels = map(int, weight_oki.shape)
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=int(stride),
            dilation=int(dilation),
            bias=bias_o is not None,
        )
        self.conv.weight = weight_oki
        if bias_o is not None:
            self.conv.bias = bias_o
        self.causal = bool(causal)
        self.pad_mode = str(pad_mode)
        self.stride = int(stride)
        self.dilation = int(dilation)
        self.kernel_size = int(kernel_size)

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, T, C]
        kernel_eff = (self.kernel_size - 1) * self.dilation + 1
        padding_total = kernel_eff - self.stride
        extra_padding = _get_extra_padding_for_conv1d(int(x.shape[1]), kernel_eff, self.stride, padding_total)
        if self.causal:
            x = _pad1d(x, padding_total, extra_padding, mode=self.pad_mode)
        else:
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            x = _pad1d(x, padding_left, padding_right + extra_padding, mode=self.pad_mode)
        return self.conv(x)


class MlxStreamableConvTranspose1d:
    def __init__(
        self,
        *,
        weight_oki: mx.array,
        bias_o: mx.array | None,
        stride: int,
        causal: bool,
        trim_right_ratio: float,
    ):
        out_channels, kernel_size, in_channels = map(int, weight_oki.shape)
        self.convtr = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=int(stride),
            bias=bias_o is not None,
        )
        self.convtr.weight = weight_oki
        if bias_o is not None:
            self.convtr.bias = bias_o
        self.causal = bool(causal)
        self.trim_right_ratio = float(trim_right_ratio)
        self.stride = int(stride)
        self.kernel_size = int(kernel_size)

    def __call__(self, x: mx.array) -> mx.array:
        y = self.convtr(x)
        padding_total = self.kernel_size - self.stride
        if self.causal:
            padding_right = int(math.ceil(padding_total * self.trim_right_ratio))
            padding_left = int(padding_total - padding_right)
        else:
            padding_right = int(padding_total // 2)
            padding_left = int(padding_total - padding_right)
        end = int(y.shape[1]) - padding_right
        return y[:, padding_left:end, :]


# Adapted from mlx-examples/encodec.
_lstm_kernel = mx.fast.metal_kernel(
    name="melodyflow_codec_lstm",
    input_names=["x", "h_in", "cell", "hidden_size", "time_step", "num_time_steps"],
    output_names=["hidden_state", "cell_state"],
    header="""
    template <typename T>
    T sigmoid(T x) {
        auto y = 1 / (1 + metal::exp(-metal::abs(x)));
        return (x < 0) ? 1 - y : y;
    }
    """,
    source="""
        uint b = thread_position_in_grid.x;
        uint d = hidden_size * 4;

        uint elem = b * d + thread_position_in_grid.y;
        uint index = elem;
        uint x_index = b * num_time_steps * d + time_step * d + index;

        auto i = sigmoid(h_in[index] + x[x_index]);
        index += hidden_size;
        x_index += hidden_size;
        auto f = sigmoid(h_in[index] + x[x_index]);
        index += hidden_size;
        x_index += hidden_size;
        auto g = metal::precise::tanh(h_in[index] + x[x_index]);
        index += hidden_size;
        x_index += hidden_size;
        auto o = sigmoid(h_in[index] + x[x_index]);

        cell_state[elem] = f * cell[elem] + i * g;
        hidden_state[elem] = o * metal::precise::tanh(cell_state[elem]);
    """,
)


def _lstm_custom(x: mx.array, h_in: mx.array, cell: mx.array, time_step: int) -> tuple[mx.array, mx.array]:
    out_shape = cell.shape
    return _lstm_kernel(
        inputs=[x, h_in, cell, out_shape[-1], time_step, x.shape[-2]],
        output_shapes=[out_shape, out_shape],
        output_dtypes=[h_in.dtype, h_in.dtype],
        grid=(x.shape[0], h_in.size // 4, 1),
        threadgroup=(256, 1, 1),
    )


class MlxLSTMLayer:
    def __init__(self, wx: mx.array, wh: mx.array, bias: mx.array):
        self.Wx = wx
        self.Wh = wh
        self.bias = bias
        self.hidden_size = int(wh.shape[1])

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, T, D]
        x_proj = mx.addmm(self.bias, x, self.Wx.T)
        all_hidden = []
        hidden = None
        cell = mx.zeros((x.shape[0], self.hidden_size), dtype=x.dtype)
        for t in range(int(x.shape[-2])):
            if hidden is None:
                hidden_lin = mx.zeros((x.shape[0], self.hidden_size * 4), dtype=x.dtype)
            else:
                hidden_lin = hidden @ self.Wh.T
            hidden, cell = _lstm_custom(x_proj, hidden_lin, cell, t)
            all_hidden.append(hidden)
        return mx.stack(all_hidden, axis=-2)


class MlxStreamableLSTM:
    def __init__(self, layers: list[MlxLSTMLayer], skip: bool = True):
        self.layers = layers
        self.skip = bool(skip)

    def __call__(self, x: mx.array) -> mx.array:
        y = x
        for layer in self.layers:
            y = layer(y)
        if self.skip:
            y = y + x
        return y


class MlxResnetBlock:
    def __init__(self, layers: list[Any], shortcut: Any | None = None):
        self.layers = layers
        self.shortcut = shortcut

    def __call__(self, x: mx.array) -> mx.array:
        residual = x if self.shortcut is None else self.shortcut(x)
        y = x
        for layer in self.layers:
            y = layer(y)
        return residual + y


class MlxSequential:
    def __init__(self, layers: list[Any]):
        self.layers = layers

    def __call__(self, x: mx.array) -> mx.array:
        y = x
        for layer in self.layers:
            y = layer(y)
        return y


@dataclass
class MlxMelodyFlowCodec:
    encoder: MlxSequential
    decoder: MlxSequential
    dtype: Any

    def encode(self, x_bct: mx.array) -> tuple[mx.array, None]:
        # x_bct: [B, C, T] -> internal channel-last [B, T, C]
        x_btc = mx.transpose(x_bct, (0, 2, 1))
        emb_btc = self.encoder(x_btc)
        emb_bct = mx.transpose(emb_btc, (0, 2, 1))
        # Mirror DummyQuantizer behavior used in MelodyFlow no-quant path.
        codes = mx.expand_dims(emb_bct, axis=1)  # [B, 1, C, T]
        return codes, None

    def decode(self, codes: mx.array, scale: mx.array | None = None) -> mx.array:
        if scale is not None:
            raise ValueError("Scale is not supported for MelodyFlow no-quant codec path.")
        emb_bct = codes
        if emb_bct.ndim == 4:
            emb_bct = mx.squeeze(emb_bct, axis=1)
        if emb_bct.ndim != 3:
            raise ValueError(f"Expected codes with 3 or 4 dims, got shape={emb_bct.shape}")
        emb_btc = mx.transpose(emb_bct, (0, 2, 1))
        out_btc = self.decoder(emb_btc)
        return mx.transpose(out_btc, (0, 2, 1))


def _convert_snake(layer: Any, dtype: Any) -> MlxSnake1d:
    return MlxSnake1d(_to_mx(layer.alpha, dtype))


def _convert_streamable_conv1d(layer: Any, dtype: Any) -> MlxStreamableConv1d:
    conv = layer.conv.conv
    return MlxStreamableConv1d(
        weight_oki=_conv_weight_to_mx(conv.weight, dtype),
        bias_o=_to_mx(conv.bias, dtype) if conv.bias is not None else None,
        stride=int(conv.stride[0]),
        dilation=int(conv.dilation[0]),
        causal=bool(layer.causal),
        pad_mode=str(layer.pad_mode),
    )


def _convert_streamable_convtr1d(layer: Any, dtype: Any) -> MlxStreamableConvTranspose1d:
    convtr = layer.convtr.convtr
    return MlxStreamableConvTranspose1d(
        weight_oki=_convtr_weight_to_mx(convtr.weight, dtype),
        bias_o=_to_mx(convtr.bias, dtype) if convtr.bias is not None else None,
        stride=int(convtr.stride[0]),
        causal=bool(layer.causal),
        trim_right_ratio=float(layer.trim_right_ratio),
    )


def _convert_streamable_lstm(layer: Any, dtype: Any) -> MlxStreamableLSTM:
    lstm = layer.lstm
    n_layers = int(lstm.num_layers)
    layers = []
    for i in range(n_layers):
        wx = _to_mx(getattr(lstm, f"weight_ih_l{i}"), dtype)
        wh = _to_mx(getattr(lstm, f"weight_hh_l{i}"), dtype)
        bias = _to_mx(getattr(lstm, f"bias_ih_l{i}") + getattr(lstm, f"bias_hh_l{i}"), dtype)
        layers.append(MlxLSTMLayer(wx=wx, wh=wh, bias=bias))
    return MlxStreamableLSTM(layers=layers, skip=bool(layer.skip))


def _convert_resnet_block(layer: Any, dtype: Any) -> MlxResnetBlock:
    block_layers = []
    for sub in layer.block:
        if sub.__class__.__name__ == "Snake1d":
            block_layers.append(_convert_snake(sub, dtype))
        elif sub.__class__.__name__ == "StreamableConv1d":
            block_layers.append(_convert_streamable_conv1d(sub, dtype))
        else:
            raise TypeError(f"Unsupported SEANetResnetBlock sub-layer: {type(sub)}")
    shortcut = None
    if layer.shortcut.__class__.__name__ != "Identity":
        if layer.shortcut.__class__.__name__ == "StreamableConv1d":
            shortcut = _convert_streamable_conv1d(layer.shortcut, dtype)
        else:
            raise TypeError(f"Unsupported SEANetResnetBlock shortcut: {type(layer.shortcut)}")
    return MlxResnetBlock(layers=block_layers, shortcut=shortcut)


def _convert_layer(layer: Any, dtype: Any) -> Any:
    name = layer.__class__.__name__
    if name == "Snake1d":
        return _convert_snake(layer, dtype)
    if name == "StreamableConv1d":
        return _convert_streamable_conv1d(layer, dtype)
    if name == "StreamableConvTranspose1d":
        return _convert_streamable_convtr1d(layer, dtype)
    if name == "StreamableLSTM":
        return _convert_streamable_lstm(layer, dtype)
    if name == "SEANetResnetBlock":
        return _convert_resnet_block(layer, dtype)
    raise TypeError(f"Unsupported layer type for MLX codec conversion: {type(layer)}")


def build_mlx_melodyflow_codec(compression_model: Any, dtype: Any = mx.float32) -> MlxMelodyFlowCodec:
    if compression_model.__class__.__name__ != "EncodecModel":
        raise TypeError(
            "Expected MelodyFlow compression model of type EncodecModel, "
            f"got {compression_model.__class__.__name__}"
        )
    encoder_layers = [_convert_layer(layer, dtype) for layer in compression_model.encoder.model]
    decoder_layers = [_convert_layer(layer, dtype) for layer in compression_model.decoder.model]
    return MlxMelodyFlowCodec(
        encoder=MlxSequential(encoder_layers),
        decoder=MlxSequential(decoder_layers),
        dtype=dtype,
    )

