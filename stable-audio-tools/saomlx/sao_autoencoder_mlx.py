"""MLX implementation of SAO Oobleck/VAE autoencoder (inference path)."""

from __future__ import annotations

import math
import re
import typing as tp
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from .sao_autoencoder import TorchSAOAutoencoder


def _softplus(x: mx.array) -> mx.array:
    return mx.logaddexp(x, mx.zeros_like(x))


class Identity(nn.Module):
    def __call__(self, x):
        return x


class SnakeBetaMLX(nn.Module):
    """SnakeBeta activation used by SAO Oobleck encoder/decoder."""

    def __init__(self, channels: int, alpha_logscale: bool = True):
        super().__init__()
        self.alpha_logscale = alpha_logscale
        self.alpha = mx.zeros((channels,), dtype=mx.float32)
        self.beta = mx.zeros((channels,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        # x is [B, T, C]
        alpha = self.alpha[None, None, :]
        beta = self.beta[None, None, :]
        if self.alpha_logscale:
            alpha = mx.exp(alpha)
            beta = mx.exp(beta)
        return x + (1.0 / (beta + 1e-9)) * mx.sin(x * alpha) ** 2


def get_activation(use_snake: bool, channels: int):
    if use_snake:
        return SnakeBetaMLX(channels)
    return nn.elu


class ResidualUnitMLX(nn.Module):
    def __init__(self, channels: int, dilation: int, use_snake: bool):
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        act1 = get_activation(use_snake, channels)
        act2 = get_activation(use_snake, channels)
        self.layers = [
            act1,
            nn.Conv1d(channels, channels, kernel_size=7, stride=1, padding=padding, dilation=dilation),
            act2,
            nn.Conv1d(channels, channels, kernel_size=1, stride=1, padding=0),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        h = x
        for layer in self.layers:
            h = layer(h)
        return x + h


class EncoderBlockMLX(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, use_snake: bool):
        super().__init__()
        self.layers = [
            ResidualUnitMLX(in_channels, dilation=1, use_snake=use_snake),
            ResidualUnitMLX(in_channels, dilation=3, use_snake=use_snake),
            ResidualUnitMLX(in_channels, dilation=9, use_snake=use_snake),
            get_activation(use_snake, in_channels),
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        h = x
        for layer in self.layers:
            h = layer(h)
        return h


class DecoderBlockMLX(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, use_snake: bool):
        super().__init__()
        self.layers = [
            get_activation(use_snake, in_channels),
            nn.ConvTranspose1d(
                in_channels,
                out_channels,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
            ResidualUnitMLX(out_channels, dilation=1, use_snake=use_snake),
            ResidualUnitMLX(out_channels, dilation=3, use_snake=use_snake),
            ResidualUnitMLX(out_channels, dilation=9, use_snake=use_snake),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        h = x
        for layer in self.layers:
            h = layer(h)
        return h


class OobleckEncoderMLX(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        latent_dim: int,
        c_mults: list[int],
        strides: list[int],
        use_snake: bool,
    ):
        super().__init__()
        mults = [1] + list(c_mults)
        depth = len(mults)

        layers: list[tp.Any] = [
            nn.Conv1d(in_channels, mults[0] * channels, kernel_size=7, stride=1, padding=3)
        ]

        for i in range(depth - 1):
            layers.append(
                EncoderBlockMLX(
                    in_channels=mults[i] * channels,
                    out_channels=mults[i + 1] * channels,
                    stride=strides[i],
                    use_snake=use_snake,
                )
            )

        layers.extend(
            [
                get_activation(use_snake, mults[-1] * channels),
                nn.Conv1d(mults[-1] * channels, latent_dim, kernel_size=3, stride=1, padding=1),
            ]
        )
        self.layers = layers

    def __call__(self, x: mx.array) -> mx.array:
        h = x
        for layer in self.layers:
            h = layer(h)
        return h


class OobleckDecoderMLX(nn.Module):
    def __init__(
        self,
        out_channels: int,
        channels: int,
        latent_dim: int,
        c_mults: list[int],
        strides: list[int],
        use_snake: bool,
        final_tanh: bool,
    ):
        super().__init__()
        mults = [1] + list(c_mults)
        depth = len(mults)

        layers: list[tp.Any] = [
            nn.Conv1d(latent_dim, mults[-1] * channels, kernel_size=7, stride=1, padding=3),
        ]

        for i in range(depth - 1, 0, -1):
            layers.append(
                DecoderBlockMLX(
                    in_channels=mults[i] * channels,
                    out_channels=mults[i - 1] * channels,
                    stride=strides[i - 1],
                    use_snake=use_snake,
                )
            )

        layers.extend(
            [
                get_activation(use_snake, mults[0] * channels),
                nn.Conv1d(mults[0] * channels, out_channels, kernel_size=7, stride=1, padding=3, bias=False),
                nn.tanh if final_tanh else Identity(),
            ]
        )
        self.layers = layers

    def __call__(self, x: mx.array) -> mx.array:
        h = x
        for layer in self.layers:
            h = layer(h)
        return h


class VAEBottleneckMLX(nn.Module):
    def __call__(
        self,
        x: mx.array,
        *,
        sample: bool = True,
        return_info: bool = False,
    ):
        mean, scale = mx.split(x, 2, axis=-1)
        stdev = _softplus(scale) + 1e-4
        var = stdev * stdev
        logvar = mx.log(var)
        if sample:
            z = mx.random.normal(mean.shape, dtype=mean.dtype) * stdev + mean
        else:
            z = mean

        kl = mx.mean(mx.sum(mean * mean + var - logvar - 1, axis=-1))
        if return_info:
            return z, {"kl": kl}
        return z


@dataclass(frozen=True)
class ConversionReport:
    missing_keys: list[str]
    unexpected_keys: list[str]
    transposed_keys: list[str]


class SAOAutoencoderMLX(nn.Module):
    """MLX Oobleck/VAE autoencoder with torch-checkpoint conversion support."""

    def __init__(self, config: dict[str, tp.Any]):
        super().__init__()
        self.config = config
        ae = config["model"]
        enc = ae["encoder"]["config"]
        dec = ae["decoder"]["config"]
        bottleneck_type = ae["bottleneck"]["type"]
        if bottleneck_type != "vae":
            raise ValueError(f"Only VAE bottleneck is currently supported, got '{bottleneck_type}'")

        self.sample_rate = int(config["sample_rate"])
        self.downsampling_ratio = int(ae["downsampling_ratio"])
        self.io_channels = int(config["audio_channels"])
        self.latent_dim = int(ae["latent_dim"])

        self.encoder = OobleckEncoderMLX(
            in_channels=int(enc["in_channels"]),
            channels=int(enc["channels"]),
            latent_dim=int(enc["latent_dim"]),
            c_mults=list(enc["c_mults"]),
            strides=list(enc["strides"]),
            use_snake=bool(enc.get("use_snake", False)),
        )

        self.bottleneck = VAEBottleneckMLX()

        self.decoder = OobleckDecoderMLX(
            out_channels=int(dec["out_channels"]),
            channels=int(dec["channels"]),
            latent_dim=int(dec["latent_dim"]),
            c_mults=list(dec["c_mults"]),
            strides=list(dec["strides"]),
            use_snake=bool(dec.get("use_snake", False)),
            final_tanh=bool(dec.get("final_tanh", True)),
        )

    @staticmethod
    def _ncl_to_nlc(x: mx.array) -> mx.array:
        return mx.transpose(x, (0, 2, 1))

    @staticmethod
    def _nlc_to_ncl(x: mx.array) -> mx.array:
        return mx.transpose(x, (0, 2, 1))

    def encode(self, audio_ncl: mx.array, *, sample: bool = True, return_info: bool = False):
        x = self._ncl_to_nlc(audio_ncl)
        x = self.encoder(x)
        x, info = self.bottleneck(x, sample=sample, return_info=True)
        x = self._nlc_to_ncl(x)
        if return_info:
            return x, info
        return x

    def decode(self, latents_ncl: mx.array) -> mx.array:
        x = self._ncl_to_nlc(latents_ncl)
        x = self.decoder(x)
        return self._nlc_to_ncl(x)

    def roundtrip(self, audio_ncl: mx.array, *, sample: bool = True):
        z = self.encode(audio_ncl, sample=sample)
        y = self.decode(z)
        return y, z

    def decode_numpy(self, latents_ncl: np.ndarray) -> np.ndarray:
        x = mx.array(np.asarray(latents_ncl, dtype=np.float32))
        y = self.decode(x)
        return np.asarray(y)

    def roundtrip_numpy(self, audio_tc: np.ndarray, *, sample: bool = True) -> tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(audio_tc, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.ndim != 2:
            raise ValueError(f"Expected (samples, channels) array, got {arr.shape}")
        if arr.shape[1] != self.io_channels:
            raise ValueError(f"Expected {self.io_channels} channels, got {arr.shape[1]}")

        x = mx.array(arr.T[None, ...])  # [1, C, T]
        y, z = self.roundtrip(x, sample=sample)
        y_np = np.asarray(y[0].T)
        z_np = np.asarray(z[0].T)

        # Align output length to source length
        source_len = arr.shape[0]
        if y_np.shape[0] >= source_len:
            y_np = y_np[:source_len]
        else:
            pad = np.zeros((source_len - y_np.shape[0], y_np.shape[1]), dtype=y_np.dtype)
            y_np = np.concatenate([y_np, pad], axis=0)
        return y_np, z_np

    def load_torch_state_dict(self, torch_state_dict: dict[str, tp.Any]) -> ConversionReport:
        params = dict(tree_flatten(self.parameters()))
        missing: list[str] = []
        unexpected = sorted(k for k in torch_state_dict if k not in params)
        transposed: list[str] = []
        updates: list[tuple[str, mx.array]] = []

        for key, target in params.items():
            if key not in torch_state_dict:
                missing.append(key)
                continue
            src = torch_state_dict[key].detach().cpu().numpy()
            src, did_transpose = _convert_weight_to_mlx_shape(key, src, tuple(target.shape))
            if did_transpose:
                transposed.append(key)
            updates.append((key, mx.array(src.astype(np.float32, copy=False))))

        if missing:
            raise RuntimeError(f"Missing {len(missing)} keys for MLX model load, e.g. {missing[:5]}")

        self.update(tree_unflatten(updates))
        return ConversionReport(
            missing_keys=missing,
            unexpected_keys=unexpected,
            transposed_keys=transposed,
        )

    @classmethod
    def from_torch_autoencoder(cls, torch_ae: TorchSAOAutoencoder) -> tuple["SAOAutoencoderMLX", ConversionReport]:
        import torch
        from torch.nn.utils import remove_weight_norm

        model = torch_ae.model.cpu().eval()
        for module in model.modules():
            try:
                remove_weight_norm(module)
            except Exception:
                pass

        mlx_model = cls(torch_ae.config)
        report = mlx_model.load_torch_state_dict(model.state_dict())
        mx.eval(mlx_model.parameters())
        return mlx_model, report

    @classmethod
    def from_torch_model(
        cls,
        torch_model,
        config: dict[str, tp.Any],
    ) -> tuple["SAOAutoencoderMLX", ConversionReport]:
        from torch.nn.utils import remove_weight_norm

        model = torch_model.cpu().eval()
        for module in model.modules():
            try:
                remove_weight_norm(module)
            except Exception:
                pass

        mlx_model = cls(config)
        report = mlx_model.load_torch_state_dict(model.state_dict())
        mx.eval(mlx_model.parameters())
        return mlx_model, report

    @classmethod
    def from_torch_pretrained(
        cls,
        repo_id: str = "stabilityai/stable-audio-open-1.0",
        *,
        torch_device: str = "cpu",
        torch_dtype: str = "float32",
    ) -> tuple["SAOAutoencoderMLX", ConversionReport]:
        torch_ae = TorchSAOAutoencoder.from_pretrained(
            repo_id=repo_id,
            device=torch_device,
            dtype=torch_dtype,
        )
        return cls.from_torch_autoencoder(torch_ae)


_DECODER_DECONV_WEIGHT_RE = re.compile(r"^decoder\.layers\.\d+\.layers\.1\.weight$")


def _is_decoder_deconv_weight(key: str) -> bool:
    """Identify decoder upsample conv-transpose weights by parameter path."""
    return bool(_DECODER_DECONV_WEIGHT_RE.match(key))


def _convert_weight_to_mlx_shape(
    key: str,
    arr: np.ndarray,
    target_shape: tuple[int, ...],
) -> tuple[np.ndarray, bool]:
    """Convert torch tensor layout into MLX layout by shape matching."""
    if arr.shape == target_shape:
        return arr, False

    if arr.ndim == 3:
        # torch Conv1d: (out, in, k) -> MLX Conv1d: (out, k, in)
        cand_conv = np.transpose(arr, (0, 2, 1))
        # torch ConvTranspose1d: (in, out, k) -> MLX ConvTranspose1d: (out, k, in)
        cand_deconv = np.transpose(arr, (1, 2, 0))

        conv_match = cand_conv.shape == target_shape
        deconv_match = cand_deconv.shape == target_shape

        if conv_match and deconv_match:
            # Ambiguous shape case (e.g., in_channels == out_channels). Use key path
            # to select the intended mapping for decoder transposed conv layers.
            return (
                (cand_deconv, True)
                if _is_decoder_deconv_weight(key)
                else (cand_conv, True)
            )
        if conv_match:
            return cand_conv, True
        if deconv_match:
            return cand_deconv, True

    if arr.ndim == 2:
        cand = arr.T
        if cand.shape == target_shape:
            return cand, True

    raise ValueError(f"Unable to map tensor with shape {arr.shape} to target {target_shape}")
