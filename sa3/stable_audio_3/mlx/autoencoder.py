from __future__ import annotations

import typing as tp
from dataclasses import dataclass

import numpy as np

from stable_audio_3.mlx.dit_blocks import Identity, TransformerBlock
from stable_audio_3.mlx.runtime import import_mlx_core, import_mlx_nn

try:
    from mlx.utils import tree_flatten, tree_unflatten
except ImportError as exc:  # pragma: no cover - guarded by runtime import in normal use
    from stable_audio_3.mlx.runtime import MLXRuntimeUnavailableError

    raise MLXRuntimeUnavailableError("MLX is required for the SAME-L autoencoder port.") from exc

mx = import_mlx_core(required=True)
nn = import_mlx_nn(required=True)


@dataclass(frozen=True)
class AutoencoderConversionReport:
    missing_keys: list[str]
    unexpected_keys: list[str]
    transposed_keys: list[str]
    synthesized_keys: list[str]


def _autoencoder_config_from_model_config(model_config: dict[str, tp.Any]) -> dict[str, tp.Any]:
    model = model_config.get("model", model_config)
    if "pretransform" in model and "config" in model["pretransform"]:
        return model["pretransform"]["config"]
    return model


def _zero_pad_modulo_sequence(x, size: int, dim: int = -2):
    axis = dim if dim >= 0 else x.ndim + dim
    input_len = int(x.shape[axis])
    pad_len = (int(size) - input_len % int(size)) % int(size)
    if pad_len <= 0:
        return x

    pad_shape = list(x.shape)
    pad_shape[axis] = pad_len
    pad = mx.zeros(tuple(pad_shape), dtype=x.dtype)
    return mx.concatenate([x, pad], axis=axis)


def _apply_conv1d_ncl(conv: nn.Conv1d, x_ncl):
    x_nlc = mx.transpose(x_ncl, (0, 2, 1))
    y_nlc = conv(x_nlc)
    return mx.transpose(y_nlc, (0, 2, 1))


class MLXTranspose(nn.Module):
    def __call__(self, x):
        return mx.swapaxes(x, -1, -2)


class MLXPatchedPretransform(nn.Module):
    def __init__(self, *, channels: int, patch_size: int, oversampling: int = 1, **kwargs):
        super().__init__()
        if int(oversampling) != 1:
            raise NotImplementedError("MLX PatchedPretransform does not implement oversampling yet.")
        if kwargs.get("postfilter_channels", 0):
            raise NotImplementedError("MLX PatchedPretransform does not implement postfiltering yet.")

        self.channels = int(channels)
        self.patch_size = int(patch_size)
        self.downsampling_ratio = self.patch_size
        self.io_channels = self.channels
        self.encoded_channels = self.channels * self.patch_size

    def _pad(self, x):
        seq_len = int(x.shape[-1])
        pad_len = (self.patch_size - seq_len % self.patch_size) % self.patch_size
        if pad_len <= 0:
            return x
        return mx.concatenate([x, mx.zeros((*x.shape[:-1], pad_len), dtype=x.dtype)], axis=-1)

    def encode(self, x):
        x = self._pad(x)
        b, c, t = x.shape
        if c != self.channels:
            raise ValueError(f"Expected {self.channels} audio channels, got {c}.")
        x = x.reshape(b, c, t // self.patch_size, self.patch_size)
        x = mx.transpose(x, (0, 1, 3, 2))
        return x.reshape(b, c * self.patch_size, t // self.patch_size)

    def decode(self, x):
        b, ch, length = x.shape
        if ch != self.channels * self.patch_size:
            raise ValueError(
                f"Expected {self.channels * self.patch_size} patched channels, got {ch}."
            )
        x = x.reshape(b, self.channels, self.patch_size, length)
        x = mx.transpose(x, (0, 1, 3, 2))
        return x.reshape(b, self.channels, length * self.patch_size)


class MLXSoftNormBottleneck(nn.Module):
    def __init__(
        self,
        *,
        dim: int = 32,
        noise_augment_dim: int = 0,
        noise_regularize: bool = False,
        auto_scale: bool = False,
        freeze: bool = False,
    ):
        super().__init__()
        self.noise_augment_dim = int(noise_augment_dim)
        self.noise_regularize = bool(noise_regularize)
        self.freeze = bool(freeze)
        self.scaling_factor = mx.ones((1, int(dim), 1), dtype=mx.float32)
        self.bias = mx.zeros((1, int(dim), 1), dtype=mx.float32)
        self.noise_scaling_factor = mx.ones((1, self.noise_augment_dim, 1), dtype=mx.float32)
        if auto_scale:
            self.running_std = mx.ones((1,), dtype=mx.float32)

    def encode(self, x):
        x = x * self.scaling_factor + self.bias
        if hasattr(self, "running_std"):
            x = x / self.running_std
        return x

    def decode(self, x, *, add_noise: bool | None = None):
        if hasattr(self, "running_std"):
            x = x * self.running_std

        should_add_noise = self.noise_regularize if add_noise is None else bool(add_noise)
        if should_add_noise:
            scaling = self.running_std if hasattr(self, "running_std") else mx.std(x, axis=-1, keepdims=True)
            x = x + mx.random.normal(x.shape, dtype=x.dtype) * scaling * 1e-3

        if self.noise_augment_dim > 0:
            noise = self.noise_scaling_factor * mx.random.normal(
                (int(x.shape[0]), self.noise_augment_dim, int(x.shape[-1])),
                dtype=x.dtype,
            )
            x = mx.concatenate([x, noise], axis=1)

        return x


class MLXTransformerResamplingBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        stride: int,
        sliding_window=None,
        chunk_size: int = 128,
        chunk_midpoint_shift: bool = False,
        type: str = "encoder",
        transformer_depth: int = 3,
        conformer: bool = False,
        layer_scale: bool = False,
        dim_heads: int = 128,
        differential: bool = True,
        variable_stride: bool = False,
        feat_scale: bool = False,
        sinusoidal_blocks: int = 0,
        mask_noise: float = 0.0,
        ff_mult: int = 3,
        mapping_bias: bool = True,
        cross_attn: bool = False,
        dyt: bool = True,
        conv_mapping: bool = False,
        use_sliding_window: bool = True,
        param_dtype=mx.float32,
        **kwargs,
    ):
        super().__init__()
        if type not in {"encoder", "decoder"}:
            raise ValueError(f"type must be 'encoder' or 'decoder', got {type!r}.")
        if conformer:
            raise NotImplementedError("SAME-L MLX does not implement conformer blocks.")
        if cross_attn:
            raise NotImplementedError("SAME-L MLX does not implement cross-attention resampling.")
        if chunk_midpoint_shift:
            raise NotImplementedError("SAME-L MLX does not implement chunk_midpoint_shift yet.")
        if conv_mapping:
            raise NotImplementedError("SAME-L MLX does not implement conv_mapping=True yet.")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.variable_stride = bool(variable_stride)
        self.stride = int(stride)
        self.chunk_size = int(chunk_size)
        self.chunk_midpoint_shift = bool(chunk_midpoint_shift)
        self.type = str(type)
        self.mask_noise = float(mask_noise)
        self.sliding_window_latents = sliding_window
        self.use_sliding_window = bool(use_sliding_window)
        self.input_seg_size, self.output_seg_size, self.sub_chunk_size = self._get_seg_sizes(
            self.stride
        )

        transformer_dim = self.out_channels if self.type == "encoder" else self.in_channels
        if self.in_channels != self.out_channels:
            self.mapping = nn.Conv1d(
                self.in_channels,
                self.out_channels,
                kernel_size=3 if conv_mapping else 1,
                padding=0,
                bias=bool(mapping_bias),
            )
        else:
            self.mapping = Identity()

        new_token_dim = self.out_channels if self.type == "encoder" else self.in_channels
        new_token_len = self.output_seg_size if not self.variable_stride else 1
        self.new_tokens = (
            mx.random.normal((1, new_token_len, new_token_dim), dtype=mx.float32) * 1e-5
        ).astype(param_dtype)

        norm_type = "dyt" if dyt else "rms_norm"
        qk_norm = "dyt" if dyt else "rms"
        self.transformers = [
            TransformerBlock(
                transformer_dim,
                dim_heads=int(dim_heads),
                causal=False,
                zero_init_branch_outputs=not bool(layer_scale),
                conformer=False,
                layer_scale=bool(layer_scale),
                add_rope=True,
                norm_type=norm_type,
                attn_kwargs={
                    "qk_norm": qk_norm,
                    "qk_norm_eps": 1e-3,
                    "differential": bool(differential),
                    "feat_scale": bool(feat_scale),
                },
                ff_kwargs={
                    "mult": ff_mult,
                    "no_bias": False,
                    "sinusoidal": bool((int(transformer_depth) - i) < int(sinusoidal_blocks)),
                },
                norm_kwargs={"eps": 1e-3},
            )
            for i in range(int(transformer_depth))
        ]

    def _get_sliding_window_size(self, window, stride: int, prepend_cond_length: int = 0):
        if window is None:
            return None
        return tuple(int(win) * (int(stride) + 1 + int(prepend_cond_length)) for win in window)

    def _get_seg_sizes(self, stride: int, prepend_cond_length: int = 0):
        sub_chunk_size = int(stride) + 1 + int(prepend_cond_length)
        input_seg_size = int(stride) if self.type == "encoder" else 1
        output_seg_size = 1 if self.type == "encoder" else int(stride)
        return input_seg_size, output_seg_size, sub_chunk_size

    def _apply_mapping(self, x):
        if isinstance(self.mapping, nn.Conv1d):
            return _apply_conv1d_ncl(self.mapping, x)
        return self.mapping(x)

    def __call__(self, x, *, stride: int | None = None, override_new_tokens=None):
        batch_size = int(x.shape[0])

        if stride is None:
            input_seg_size = self.input_seg_size
            output_seg_size = self.output_seg_size
            sub_chunk_size = self.sub_chunk_size
            structure_sliding_window = self._get_sliding_window_size(
                self.sliding_window_latents,
                self.stride,
            )
        else:
            if not self.variable_stride:
                raise ValueError("Cannot override stride when variable_stride=False.")
            input_seg_size, output_seg_size, sub_chunk_size = self._get_seg_sizes(stride)
            structure_sliding_window = self._get_sliding_window_size(
                self.sliding_window_latents,
                stride,
            )

        if self.type == "encoder":
            if self.transformers:
                pad_modulo = input_seg_size if structure_sliding_window is not None else self.chunk_size
                x = _zero_pad_modulo_sequence(x, pad_modulo, dim=-1)
            x = self._apply_mapping(x)

        if self.transformers:
            x = mx.transpose(x, (0, 2, 1))

            if self.type != "encoder":
                if structure_sliding_window is None:
                    active_stride = int(stride) if stride is not None else self.stride
                    x = _zero_pad_modulo_sequence(x, self.chunk_size // active_stride, dim=-2)
                else:
                    x = _zero_pad_modulo_sequence(x, input_seg_size, dim=-2)

            b, seq_len, dim = x.shape
            if seq_len % input_seg_size != 0:
                raise ValueError(
                    f"Sequence length {seq_len} is not divisible by input segment size {input_seg_size}."
                )
            n = seq_len // input_seg_size
            x = x.reshape(b * n, input_seg_size, dim)

            new_token_len = output_seg_size if self.variable_stride else int(self.new_tokens.shape[1])
            new_tokens = mx.broadcast_to(self.new_tokens, (int(x.shape[0]), new_token_len, int(x.shape[-1])))
            if override_new_tokens is not None:
                override = override_new_tokens.reshape(b * n, output_seg_size, int(x.shape[-1]))
                new_tokens = new_tokens + override
            elif self.mask_noise > 0.0:
                new_tokens = new_tokens + mx.random.normal(new_tokens.shape, dtype=new_tokens.dtype) * self.mask_noise

            x = mx.concatenate([x, new_tokens], axis=-2)
            x = x.reshape(batch_size, n * sub_chunk_size, dim)

            if structure_sliding_window is None:
                active_stride = int(stride) if stride is not None else self.stride
                effective_chunk_size = self.chunk_size + self.chunk_size // active_stride
                if x.shape[1] % effective_chunk_size != 0:
                    x = _zero_pad_modulo_sequence(x, effective_chunk_size, dim=-2)
                nc = int(x.shape[1]) // effective_chunk_size
                x = x.reshape(batch_size * nc, effective_chunk_size, dim)

            attention_window = structure_sliding_window if self.use_sliding_window else None
            for layer in self.transformers:
                x = layer(x, self_attention_sliding_window=attention_window)

            if structure_sliding_window is None:
                x = x.reshape(batch_size, -1, dim)

            if int(x.shape[1]) % sub_chunk_size != 0:
                x = _zero_pad_modulo_sequence(x, sub_chunk_size, dim=-2)
            n = int(x.shape[1]) // sub_chunk_size
            x = x.reshape(batch_size * n, sub_chunk_size, dim)
            x = x[:, -output_seg_size:, :]
            x = x.reshape(batch_size, n * output_seg_size, dim)
            x = mx.transpose(x, (0, 2, 1))

        if self.type == "decoder":
            x = self._apply_mapping(x)

        return x


class MLXSAMEEncoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 2,
        channels: int = 128,
        latent_dim: int = 32,
        c_mults=(1, 2, 4, 8),
        strides=(2, 4, 8, 8),
        transformer_depths=(3, 3, 3, 3),
        use_sliding_window: bool = True,
        param_dtype=mx.float32,
        **kwargs,
    ):
        super().__init__()
        channel_dims = [int(c) * int(channels) for c in c_mults]
        channel_dims = [int(in_channels), *channel_dims]
        self.depth = len(c_mults)
        self.layers = []
        for i in range(self.depth):
            self.layers.append(
                MLXTransformerResamplingBlock(
                    in_channels=channel_dims[i],
                    out_channels=channel_dims[i + 1],
                    stride=int(strides[i]),
                    transformer_depth=int(transformer_depths[i]),
                    type="encoder",
                    use_sliding_window=use_sliding_window,
                    param_dtype=param_dtype,
                    **kwargs,
                )
            )
        self.layers.extend(
            [
                MLXTranspose(),
                nn.Linear(channel_dims[-1], int(latent_dim)),
                MLXTranspose(),
            ]
        )

    def __call__(self, x, *, override_stride: list[int] | None = None):
        transformer_layer_index = 0
        for layer in self.layers:
            if isinstance(layer, MLXTransformerResamplingBlock):
                stride = None if override_stride is None else override_stride[transformer_layer_index]
                x = layer(x, stride=stride)
                transformer_layer_index += 1
            else:
                x = layer(x)
        return x


class MLXSAMEDecoder(nn.Module):
    def __init__(
        self,
        *,
        out_channels: int = 2,
        channels: int = 128,
        latent_dim: int = 32,
        c_mults=(1, 2, 4, 8),
        strides=(2, 4, 8, 8),
        transformer_depths=(3, 3, 3, 3),
        sinusoidal_blocks=(0, 0, 0, 0),
        use_sliding_window: bool = True,
        param_dtype=mx.float32,
        **kwargs,
    ):
        super().__init__()
        channel_dims = [int(c) * int(channels) for c in c_mults]
        channel_dims = [int(out_channels), *channel_dims]
        self.depth = len(c_mults)
        self.layers = [
            MLXTranspose(),
            nn.Linear(int(latent_dim), channel_dims[-1]),
            MLXTranspose(),
        ]

        for i in range(self.depth, 0, -1):
            self.layers.append(
                MLXTransformerResamplingBlock(
                    in_channels=channel_dims[i],
                    out_channels=channel_dims[i - 1],
                    stride=int(strides[i - 1]),
                    type="decoder",
                    transformer_depth=int(transformer_depths[i - 1]),
                    sinusoidal_blocks=int(sinusoidal_blocks[i - 1]),
                    use_sliding_window=use_sliding_window,
                    param_dtype=param_dtype,
                    **kwargs,
                )
            )

    def __call__(self, x, *, override_stride: list[int] | None = None):
        transformer_layer_index = 0
        for layer in self.layers:
            if isinstance(layer, MLXTransformerResamplingBlock):
                stride = None if override_stride is None else override_stride[transformer_layer_index]
                x = layer(x, stride=stride)
                transformer_layer_index += 1
            else:
                x = layer(x)
        return x


class MLXAudioAutoencoder(nn.Module):
    def __init__(
        self,
        *,
        encoder: MLXSAMEEncoder,
        decoder: MLXSAMEDecoder,
        latent_dim: int,
        downsampling_ratio: int,
        sample_rate: int,
        io_channels: int = 2,
        bottleneck: MLXSoftNormBottleneck | None = None,
        pretransform: MLXPatchedPretransform | None = None,
        soft_clip: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.bottleneck = bottleneck
        self.pretransform = pretransform
        self.latent_dim = int(latent_dim)
        self.downsampling_ratio = int(downsampling_ratio)
        self.sample_rate = int(sample_rate)
        self.io_channels = int(io_channels)
        self.soft_clip = bool(soft_clip)

    @classmethod
    def from_config(
        cls,
        model_config: dict[str, tp.Any],
        *,
        sample_rate: int | None = None,
        use_sliding_window: bool = True,
        param_dtype=mx.float32,
    ) -> "MLXAudioAutoencoder":
        autoencoder_config = _autoencoder_config_from_model_config(model_config)
        sample_rate = int(sample_rate or model_config.get("sample_rate", 44100))

        encoder = MLXSAMEEncoder(
            **autoencoder_config["encoder"]["config"],
            use_sliding_window=use_sliding_window,
            param_dtype=param_dtype,
        )
        decoder = MLXSAMEDecoder(
            **autoencoder_config["decoder"]["config"],
            use_sliding_window=use_sliding_window,
            param_dtype=param_dtype,
        )
        pretransform = MLXPatchedPretransform(**autoencoder_config["pretransform"]["config"])
        bottleneck = MLXSoftNormBottleneck(**autoencoder_config["bottleneck"]["config"])
        return cls(
            encoder=encoder,
            decoder=decoder,
            latent_dim=int(autoencoder_config["latent_dim"]),
            downsampling_ratio=int(autoencoder_config["downsampling_ratio"]),
            sample_rate=sample_rate,
            io_channels=int(autoencoder_config["io_channels"]),
            bottleneck=bottleneck,
            pretransform=pretransform,
        )

    @classmethod
    def from_torch_autoencoder(
        cls,
        torch_autoencoder,
        model_config: dict[str, tp.Any],
        *,
        mlx_dtype=mx.float32,
        use_sliding_window: bool = True,
    ) -> tuple["MLXAudioAutoencoder", AutoencoderConversionReport]:
        source_autoencoder = torch_autoencoder
        wrapped_autoencoder = getattr(torch_autoencoder, "model", None)
        if (
            not hasattr(source_autoencoder, "sample_rate")
            and wrapped_autoencoder is not None
            and hasattr(wrapped_autoencoder, "sample_rate")
        ):
            source_autoencoder = wrapped_autoencoder

        sample_rate = int(getattr(source_autoencoder, "sample_rate", model_config.get("sample_rate", 44100)))
        model = cls.from_config(
            model_config,
            sample_rate=sample_rate,
            use_sliding_window=use_sliding_window,
            param_dtype=mlx_dtype,
        )
        report = model.load_torch_state_dict(
            source_autoencoder.state_dict(),
            torch_autoencoder=source_autoencoder,
            mlx_dtype=mlx_dtype,
        )
        mx.eval(model.parameters())
        return model, report

    def encode(self, audio):
        if self.pretransform is not None:
            audio = self.pretransform.encode(audio)
        latents = self.encoder(audio)
        if self.bottleneck is not None:
            latents = self.bottleneck.encode(latents)
        return latents

    def decode(self, latents, *, add_bottleneck_noise: bool | None = None):
        if self.bottleneck is not None:
            latents = self.bottleneck.decode(latents, add_noise=add_bottleneck_noise)
        decoded = self.decoder(latents)
        if self.pretransform is not None:
            decoded = self.pretransform.decode(decoded)
        if self.soft_clip:
            decoded = mx.tanh(decoded)
        return decoded

    def decode_audio(
        self,
        latents,
        *,
        chunked: bool = False,
        chunk_size: int = 128,
        overlap: int = 32,
        chunk_batch_size: int = 1,
        add_bottleneck_noise: bool | None = None,
    ):
        if not chunked or int(latents.shape[-1]) < int(chunk_size):
            return self.decode(latents, add_bottleneck_noise=add_bottleneck_noise)

        chunk_size = int(chunk_size)
        overlap = int(overlap)
        chunk_batch_size = int(chunk_batch_size)
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError(
                f"overlap must be >= 0 and smaller than chunk_size, got {overlap}"
            )
        if chunk_batch_size < 1:
            raise ValueError(f"chunk_batch_size must be positive, got {chunk_batch_size}")

        total_latents = int(latents.shape[-1])
        hop_latents = chunk_size - overlap
        chunk_starts = list(range(0, total_latents - chunk_size + 1, hop_latents))
        final_start = total_latents - chunk_size
        if chunk_starts[-1] != final_start:
            chunk_starts.append(final_start)

        batch_size = int(latents.shape[0])
        decoded_chunks = []
        for offset in range(0, len(chunk_starts), chunk_batch_size):
            batch_starts = chunk_starts[offset : offset + chunk_batch_size]
            chunk_latents = mx.concatenate(
                [latents[..., start : start + chunk_size] for start in batch_starts],
                axis=0,
            )
            decoded_batch = self.decode(
                chunk_latents,
                add_bottleneck_noise=add_bottleneck_noise,
            )
            mx.eval(decoded_batch)
            decoded_chunks.extend(
                decoded_batch[index * batch_size : (index + 1) * batch_size]
                for index in range(len(batch_starts))
            )

        samples_per_latent = int(self.downsampling_ratio)
        total_samples = total_latents * samples_per_latent
        chunk_size_samples = chunk_size * samples_per_latent
        half_overlap_samples = (overlap // 2) * samples_per_latent

        intervals = []
        num_chunks = len(chunk_starts)
        for index, (start_latent, chunk) in enumerate(zip(chunk_starts, decoded_chunks)):
            is_first = index == 0
            is_last = index == num_chunks - 1
            out_start = (
                total_samples - chunk_size_samples
                if is_last
                else int(start_latent) * samples_per_latent
            )
            left = 0 if is_first else half_overlap_samples
            right = chunk_size_samples if is_last else chunk_size_samples - half_overlap_samples
            intervals.append(
                {
                    "target_start": out_start + left,
                    "target_end": out_start + right,
                    "left": left,
                    "right": right,
                    "chunk": chunk,
                }
            )

        pieces = []
        cursor = 0
        output_shape_prefix = tuple(int(dim) for dim in decoded_chunks[0].shape[:-1])
        for index, interval in enumerate(intervals):
            next_start = (
                int(intervals[index + 1]["target_start"])
                if index + 1 < len(intervals)
                else int(interval["target_end"])
            )
            target_start = int(interval["target_start"])
            target_end = min(int(interval["target_end"]), next_start)
            clipped_start = max(target_start, cursor)
            if clipped_start > cursor:
                pieces.append(
                    mx.zeros(
                        (*output_shape_prefix, clipped_start - cursor),
                        dtype=decoded_chunks[0].dtype,
                    )
                )
                cursor = clipped_start
            if target_end <= clipped_start:
                continue

            left = int(interval["left"]) + (clipped_start - target_start)
            right = int(interval["left"]) + (target_end - target_start)
            pieces.append(interval["chunk"][..., left:right])
            cursor = target_end

        if cursor < total_samples:
            pieces.append(
                mx.zeros(
                    (*output_shape_prefix, total_samples - cursor),
                    dtype=decoded_chunks[0].dtype,
                )
            )

        return mx.concatenate(pieces, axis=-1)

    def load_torch_state_dict(
        self,
        torch_state_dict: dict[str, tp.Any],
        *,
        torch_autoencoder=None,
        mlx_dtype=mx.float32,
    ) -> AutoencoderConversionReport:
        params = dict(tree_flatten(self.parameters()))
        source_state = dict(torch_state_dict)
        synthesized = []

        if torch_autoencoder is not None:
            for name, module in torch_autoencoder.named_modules():
                if module.__class__.__name__ == "Conv1d" and hasattr(module, "weight"):
                    weight_key = f"{name}.weight"
                    weight_g_key = f"{name}.weight_g"
                    weight_v_key = f"{name}.weight_v"
                    if weight_g_key in torch_state_dict and weight_v_key in torch_state_dict:
                        weight_g = torch_state_dict[weight_g_key].detach().cpu().float().numpy()
                        weight_v = torch_state_dict[weight_v_key].detach().cpu().float().numpy()
                        denom = np.sqrt(np.sum(weight_v * weight_v, axis=(1, 2), keepdims=True))
                        source_state[weight_key] = weight_v * (weight_g / np.maximum(denom, 1e-12))
                    else:
                        source_state[weight_key] = module.weight.detach().cpu().float().numpy()
                    synthesized.append(weight_key)

        missing: list[str] = []
        used: set[str] = set()
        transposed: list[str] = []
        updates: list[tuple[str, tp.Any]] = []

        for key, target in params.items():
            if key.endswith(".rope.inv_freq"):
                continue
            if key not in source_state:
                missing.append(key)
                continue
            used.add(key)
            source = source_state[key]
            if hasattr(source, "detach"):
                source = source.detach().cpu().float().numpy()
            source = np.asarray(source, dtype=np.float32)
            source, did_transpose = _convert_weight_to_mlx_shape(source, tuple(target.shape))
            if did_transpose:
                transposed.append(key)
            arr = mx.array(source.astype(np.float32, copy=False))
            if arr.dtype != mlx_dtype:
                arr = arr.astype(mlx_dtype)
            updates.append((key, arr))

        if missing:
            raise RuntimeError(f"Missing {len(missing)} SAME-L MLX keys, e.g. {missing[:8]}")

        self.update(tree_unflatten(updates))
        ignored_suffixes = (".rope.inv_freq", ".weight_g", ".weight_v")
        unexpected = sorted(
            key for key in source_state if key not in used and not key.endswith(ignored_suffixes)
        )
        return AutoencoderConversionReport(
            missing_keys=missing,
            unexpected_keys=unexpected,
            transposed_keys=transposed,
            synthesized_keys=sorted(synthesized),
        )


def _convert_weight_to_mlx_shape(arr: np.ndarray, target_shape: tuple[int, ...]) -> tuple[np.ndarray, bool]:
    if arr.shape == target_shape:
        return arr, False

    if arr.ndim == 2:
        candidate = arr.T
        if candidate.shape == target_shape:
            return candidate, True

    if arr.ndim == 3:
        candidate = np.transpose(arr, (0, 2, 1))
        if candidate.shape == target_shape:
            return candidate, True

    raise ValueError(f"Unable to map tensor with shape {arr.shape} to target {target_shape}")
