from __future__ import annotations

import copy
import math
import typing as tp
from dataclasses import dataclass

import numpy as np

from stable_audio_3.mlx.dit_blocks import (
    ContinuousTransformer,
    ExpoFourierFeatures,
    FourierFeatures,
    run_layers,
)
from stable_audio_3.mlx.runtime import (
    MLXRuntimeUnavailableError,
    import_mlx_core,
    import_mlx_nn,
)

try:
    from mlx.utils import tree_flatten, tree_unflatten
except ImportError as exc:
    raise MLXRuntimeUnavailableError(
        "MLX is not installed in this environment. "
        "Install the Apple Silicon MLX runtime before attempting MLX inference."
    ) from exc

mx = import_mlx_core(required=True)
nn = import_mlx_nn(required=True)


def _as_scalar(value) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    return float(np.asarray(value.astype(mx.float32)).reshape(-1)[0])


@dataclass(frozen=True)
class ConversionReport:
    missing_keys: list[str]
    unexpected_keys: list[str]
    transposed_keys: list[str]


def extract_dit_config(model_config: dict[str, tp.Any]) -> dict[str, tp.Any]:
    diffusion = model_config.get("model", {}).get("diffusion", {})
    if diffusion.get("type") != "dit":
        raise ValueError(f"Expected diffusion.type='dit', got {diffusion.get('type')!r}")
    config = diffusion.get("config")
    if not isinstance(config, dict):
        raise ValueError("Missing `model.diffusion.config` in model config.")
    return copy.deepcopy(config)


def extract_diffusion_objective(model_config: dict[str, tp.Any]) -> str:
    diffusion = model_config.get("model", {}).get("diffusion", {})
    objective = diffusion.get("diffusion_objective")
    return objective if isinstance(objective, str) and objective else "v"


class StableAudioMLXDiT(nn.Module):
    _LOGSNR_MIN = -12.0
    _LOGSNR_MAX = 5.0
    _LOGSNR_RANGE = _LOGSNR_MAX - _LOGSNR_MIN

    def __init__(
        self,
        config: dict[str, tp.Any],
        *,
        diffusion_objective: str = "v",
        param_dtype=mx.float32,
    ):
        super().__init__()
        self.config = copy.deepcopy(config)
        self.diffusion_objective = diffusion_objective
        self.param_dtype = param_dtype

        self.cond_token_dim = int(config.get("cond_token_dim", 0))
        self.timestep_cond_type = config.get("timestep_cond_type", "global")
        self.timestep_features_logsnr = bool(config.get("timestep_features_logsnr", False))

        timestep_features_dim = int(config.get("timestep_features_dim", 256))
        timestep_features_type = config.get("timestep_features_type", "learned")
        if timestep_features_type == "expo":
            self.timestep_features = ExpoFourierFeatures(timestep_features_dim, 0.5, 10000.0)
        else:
            self.timestep_features = FourierFeatures(1, timestep_features_dim)

        embed_dim = int(config["embed_dim"])
        timestep_embed_dim = config.get("timestep_embed_dim")
        input_concat_dim = int(config.get("input_concat_dim", 0))

        if self.timestep_cond_type == "global":
            timestep_embed_dim = embed_dim
        elif self.timestep_cond_type == "input_concat":
            if timestep_embed_dim is None:
                raise ValueError(
                    "timestep_embed_dim is required when timestep_cond_type='input_concat'"
                )
            input_concat_dim += int(timestep_embed_dim)
        else:
            raise ValueError(f"Unsupported timestep_cond_type: {self.timestep_cond_type}")

        self.to_timestep_embed = [
            nn.Linear(timestep_features_dim, int(timestep_embed_dim), bias=True),
            nn.silu,
            nn.Linear(int(timestep_embed_dim), int(timestep_embed_dim), bias=True),
        ]

        if self.cond_token_dim > 0:
            project_cond_tokens = bool(config.get("project_cond_tokens", True))
            cond_embed_dim = self.cond_token_dim if not project_cond_tokens else embed_dim
            self.to_cond_embed = [
                nn.Linear(self.cond_token_dim, cond_embed_dim, bias=False),
                nn.silu,
                nn.Linear(cond_embed_dim, cond_embed_dim, bias=False),
            ]
        else:
            cond_embed_dim = 0
            self.to_cond_embed = None

        global_cond_dim = int(config.get("global_cond_dim", 0))
        if global_cond_dim > 0:
            project_global_cond = bool(config.get("project_global_cond", True))
            global_embed_dim = global_cond_dim if not project_global_cond else embed_dim
            self.to_global_embed = [
                nn.Linear(global_cond_dim, global_embed_dim, bias=False),
                nn.silu,
                nn.Linear(global_embed_dim, global_embed_dim, bias=False),
            ]
        else:
            self.to_global_embed = None

        prepend_cond_dim = int(config.get("prepend_cond_dim", 0))
        if prepend_cond_dim > 0:
            self.to_prepend_embed = [
                nn.Linear(prepend_cond_dim, embed_dim, bias=False),
                nn.silu,
                nn.Linear(embed_dim, embed_dim, bias=False),
            ]
        else:
            self.to_prepend_embed = None

        self.input_concat_dim = input_concat_dim
        self.patch_size = int(config.get("patch_size", 1))
        self.transformer_type = config.get("transformer_type", "continuous_transformer")
        if self.transformer_type != "continuous_transformer":
            raise ValueError(f"Unsupported transformer_type: {self.transformer_type}")

        self.global_cond_type = config.get("global_cond_type", "prepend")
        if self.global_cond_type not in {"prepend", "adaLN"}:
            raise ValueError(f"Unsupported global_cond_type: {self.global_cond_type}")

        depth = int(config["depth"])
        num_heads = int(config["num_heads"])
        io_channels = int(config["io_channels"])
        dim_in = io_channels + self.input_concat_dim

        transformer_global_dim = embed_dim if self.global_cond_type == "adaLN" else None
        self.transformer = ContinuousTransformer(
            dim=embed_dim,
            depth=depth,
            dim_heads=embed_dim // num_heads,
            dim_in=dim_in * self.patch_size,
            dim_out=io_channels * self.patch_size,
            cross_attend=self.cond_token_dim > 0,
            cond_token_dim=cond_embed_dim,
            global_cond_dim=transformer_global_dim,
            local_add_cond_dim=config.get("local_add_cond_dim"),
            num_memory_tokens=int(config.get("num_memory_tokens", 0)),
            norm_type=config.get("norm_type", "layer_norm"),
            attn_kwargs=config.get("attn_kwargs", {}),
            ff_kwargs=config.get("ff_kwargs", {}),
            norm_kwargs=config.get("norm_kwargs", {}),
        )

        self.preprocess_conv = nn.Conv1d(dim_in, dim_in, kernel_size=1, bias=False)
        self.preprocess_conv.weight = mx.zeros_like(self.preprocess_conv.weight)
        self.postprocess_conv = nn.Conv1d(io_channels, io_channels, kernel_size=1, bias=False)
        self.postprocess_conv.weight = mx.zeros_like(self.postprocess_conv.weight)

    @staticmethod
    def _model_dtype_from_params(module: "StableAudioMLXDiT"):
        params = tree_flatten(module.parameters())
        if not params:
            return mx.float32
        return params[0][1].dtype

    @staticmethod
    def _apply_conv1d_ncl(conv: nn.Conv1d, x_ncl):
        x_nlc = mx.transpose(x_ncl, (0, 2, 1))
        y_nlc = conv(x_nlc)
        return mx.transpose(y_nlc, (0, 2, 1))

    @staticmethod
    def _patchify_nlc(x_nlc, patch_size: int):
        if patch_size == 1:
            return x_nlc
        b, t, c = x_nlc.shape
        if t % patch_size != 0:
            raise ValueError(f"Sequence length {t} is not divisible by patch_size {patch_size}")
        x = x_nlc.reshape(b, t // patch_size, patch_size, c)
        x = mx.transpose(x, (0, 1, 3, 2))
        return x.reshape(b, t // patch_size, c * patch_size)

    @staticmethod
    def _unpatchify_ncl(x_ncl, patch_size: int):
        if patch_size == 1:
            return x_ncl
        b, cp, t = x_ncl.shape
        if cp % patch_size != 0:
            raise ValueError(f"Channel dim {cp} is not divisible by patch_size {patch_size}")
        c = cp // patch_size
        x = x_ncl.reshape(b, c, patch_size, t)
        x = mx.transpose(x, (0, 1, 3, 2))
        return x.reshape(b, c, t * patch_size)

    def _t_to_logsnr_cond(self, t):
        t_clamped = mx.clip(t.astype(mx.float32), 1e-7, 1.0 - 1e-7)
        logsnr = mx.log((1.0 - t_clamped) / t_clamped)
        logsnr = mx.clip(logsnr, self._LOGSNR_MIN, self._LOGSNR_MAX)
        return ((self._LOGSNR_MAX - logsnr) / self._LOGSNR_RANGE).astype(t.dtype)

    @staticmethod
    def _apg_project(v0, v1, padding_mask=None):
        dtype = v0.dtype
        v0 = v0.astype(mx.float32)
        v1 = v1.astype(mx.float32)

        if padding_mask is not None:
            mask = padding_mask[:, None, :].astype(mx.float32)
            v0_masked = v0 * mask
            v1_masked = v1 * mask
            v1_norm = mx.sqrt(mx.sum(v1_masked * v1_masked, axis=(-1, -2), keepdims=True))
            v1_normalized = v1_masked / mx.maximum(v1_norm, 1e-8)
            projection_scale = mx.sum(v0_masked * v1_normalized, axis=(-1, -2), keepdims=True)
            v0_parallel = projection_scale * v1_normalized
            orthogonal_scale = mx.sum(v0 * v1_normalized, axis=(-1, -2), keepdims=True)
            v0_orthogonal = (v0 - orthogonal_scale * v1_normalized) * mask
        else:
            v1_norm = mx.sqrt(mx.sum(v1 * v1, axis=(-1, -2), keepdims=True))
            v1_normalized = v1 / mx.maximum(v1_norm, 1e-8)
            projection_scale = mx.sum(v0 * v1_normalized, axis=(-1, -2), keepdims=True)
            v0_parallel = projection_scale * v1_normalized
            v0_orthogonal = v0 - v0_parallel

        return v0_parallel.astype(dtype), v0_orthogonal.astype(dtype)

    def _forward(
        self,
        x,
        t,
        *,
        cross_attn_cond=None,
        cross_attn_cond_mask=None,
        input_concat_cond=None,
        local_add_cond=None,
        modular_local_cond=None,
        global_embed=None,
        prepend_cond=None,
        prepend_cond_mask=None,
        padding_mask=None,
        return_info: bool = False,
        exit_layer_ix: int | None = None,
    ):
        del cross_attn_cond_mask
        del prepend_cond_mask
        if modular_local_cond is not None:
            raise NotImplementedError("modular_local_cond is not implemented in the MLX DiT.")

        if cross_attn_cond is not None and self.to_cond_embed is not None:
            cross_attn_cond = run_layers(self.to_cond_embed, cross_attn_cond)

        if global_embed is not None and self.to_global_embed is not None:
            global_embed = run_layers(self.to_global_embed, global_embed)

        prepend_inputs = None
        prepend_length = 0
        if prepend_cond is not None:
            if self.to_prepend_embed is None:
                raise ValueError("Received prepend_cond but the model has no prepend conditioning projection.")
            prepend_inputs = run_layers(self.to_prepend_embed, prepend_cond)
            prepend_length = int(prepend_inputs.shape[1])

        if input_concat_cond is not None:
            if int(input_concat_cond.shape[2]) != int(x.shape[2]):
                raise NotImplementedError(
                    "Interpolating input_concat_cond to a different latent length is not implemented in the MLX DiT."
                )
            x = mx.concatenate([x, input_concat_cond], axis=1)

        if local_add_cond is not None:
            local_add_cond = mx.transpose(local_add_cond, (0, 2, 1))

        t_cond = self._t_to_logsnr_cond(t) if self.timestep_features_logsnr else t
        timestep_embed = run_layers(
            self.to_timestep_embed,
            self.timestep_features(t_cond[:, None].astype(x.dtype)),
        )

        if self.timestep_cond_type == "global":
            if global_embed is not None:
                global_embed = global_embed + timestep_embed
            else:
                global_embed = timestep_embed
        elif self.timestep_cond_type == "input_concat":
            x = mx.concatenate(
                [
                    x,
                    mx.broadcast_to(
                        timestep_embed[:, :, None],
                        (x.shape[0], timestep_embed.shape[-1], x.shape[2]),
                    ),
                ],
                axis=1,
            )

        if self.global_cond_type == "prepend" and global_embed is not None:
            global_token = global_embed[:, None, :]
            prepend_inputs = (
                global_token
                if prepend_inputs is None
                else mx.concatenate([prepend_inputs, global_token], axis=1)
            )
            prepend_length = int(prepend_inputs.shape[1])

        x = self._apply_conv1d_ncl(self.preprocess_conv, x) + x
        x = mx.transpose(x, (0, 2, 1))
        x = self._patchify_nlc(x, self.patch_size)

        transformer_kwargs = {}
        if self.global_cond_type == "adaLN":
            transformer_kwargs["global_cond"] = global_embed

        out = self.transformer(
            x,
            prepend_embeds=prepend_inputs,
            context=cross_attn_cond,
            local_add_cond=local_add_cond,
            return_info=return_info,
            exit_layer_ix=exit_layer_ix,
            padding_mask=padding_mask,
            **transformer_kwargs,
        )
        if return_info:
            out, info = out

        if exit_layer_ix is not None:
            if return_info:
                return out, info
            return out

        out = mx.transpose(out, (0, 2, 1))
        if prepend_length > 0:
            out = out[:, :, prepend_length:]
        out = self._unpatchify_ncl(out, self.patch_size)
        out = self._apply_conv1d_ncl(self.postprocess_conv, out) + out

        if return_info:
            return out, info
        return out

    def __call__(
        self,
        x,
        t,
        *,
        cross_attn_cond=None,
        cross_attn_cond_mask=None,
        negative_cross_attn_cond=None,
        negative_cross_attn_mask=None,
        input_concat_cond=None,
        local_add_cond=None,
        modular_local_cond=None,
        global_embed=None,
        negative_global_embed=None,
        prepend_cond=None,
        prepend_cond_mask=None,
        padding_mask=None,
        cfg_scale: float = 1.0,
        cfg_dropout_prob: float = 0.0,
        cfg_interval: tuple[float, float] = (0.0, 1.0),
        causal: bool = False,
        scale_phi: float = 0.0,
        cfg_norm_threshold: float = 0.0,
        apg_scale: float = 0.0,
        return_info: bool = False,
        exit_layer_ix: int | None = None,
        **_: tp.Any,
    ):
        if causal:
            raise ValueError("Causal mode is not supported for StableAudioMLXDiT.")
        if modular_local_cond is not None:
            raise NotImplementedError("modular_local_cond is not implemented in the MLX DiT.")

        model_dtype = self._model_dtype_from_params(self)
        x = x.astype(model_dtype)
        t = t.astype(mx.float32)

        if cross_attn_cond is not None:
            cross_attn_cond = cross_attn_cond.astype(model_dtype)
        if negative_cross_attn_cond is not None:
            negative_cross_attn_cond = negative_cross_attn_cond.astype(model_dtype)
        if input_concat_cond is not None:
            input_concat_cond = input_concat_cond.astype(model_dtype)
        if local_add_cond is not None:
            local_add_cond = local_add_cond.astype(model_dtype)
        if global_embed is not None:
            global_embed = global_embed.astype(model_dtype)
        if negative_global_embed is not None:
            negative_global_embed = negative_global_embed.astype(model_dtype)
        if prepend_cond is not None:
            prepend_cond = prepend_cond.astype(model_dtype)

        if cross_attn_cond_mask is not None:
            cross_attn_cond_mask = None
        if prepend_cond_mask is not None:
            prepend_cond_mask = prepend_cond_mask.astype(mx.bool_)

        if exit_layer_ix is not None:
            return self._forward(
                x,
                t,
                cross_attn_cond=cross_attn_cond,
                cross_attn_cond_mask=cross_attn_cond_mask,
                input_concat_cond=input_concat_cond,
                local_add_cond=local_add_cond,
                modular_local_cond=modular_local_cond,
                global_embed=global_embed,
                prepend_cond=prepend_cond,
                prepend_cond_mask=prepend_cond_mask,
                padding_mask=padding_mask,
                return_info=return_info,
                exit_layer_ix=exit_layer_ix,
            )

        if cfg_dropout_prob > 0.0 and cfg_scale == 1.0:
            if cross_attn_cond is not None:
                null_embed = mx.zeros_like(cross_attn_cond)
                dropout = mx.random.uniform(
                    shape=(cross_attn_cond.shape[0], 1, 1)
                ) < (1.0 - float(cfg_dropout_prob))
                cross_attn_cond = mx.where(dropout, cross_attn_cond, null_embed)

            if prepend_cond is not None:
                null_embed = mx.zeros_like(prepend_cond)
                dropout = mx.random.uniform(
                    shape=(prepend_cond.shape[0], 1, 1)
                ) < (1.0 - float(cfg_dropout_prob))
                prepend_cond = mx.where(dropout, prepend_cond, null_embed)

        if self.diffusion_objective == "v":
            sigma = mx.sin(t * (math.pi / 2.0))
            alpha = mx.cos(t * (math.pi / 2.0))
        elif self.diffusion_objective in {"rectified_flow", "rf_denoiser"}:
            sigma = t
            alpha = None
        else:
            sigma = t
            alpha = None

        # Converting an MLX array to a Python scalar forces an eager evaluation,
        # which is illegal while the training step is being transformed by
        # mx.compile/value_and_grad. Training always uses cfg_scale=1, so only
        # inspect sigma when inference actually requests classifier-free guidance.
        should_cfg = False
        if cfg_scale != 1.0 and (
            cross_attn_cond is not None or prepend_cond is not None
        ):
            sigma0 = _as_scalar(sigma)
            should_cfg = cfg_interval[0] <= sigma0 <= cfg_interval[1]

        if should_cfg:
            batch_inputs = mx.concatenate([x, x], axis=0)
            batch_t = mx.concatenate([t, t], axis=0)
            batch_global = (
                mx.concatenate([global_embed, global_embed], axis=0)
                if global_embed is not None
                else None
            )
            batch_input_concat = (
                mx.concatenate([input_concat_cond, input_concat_cond], axis=0)
                if input_concat_cond is not None
                else None
            )
            batch_local_add = (
                mx.concatenate([local_add_cond, local_add_cond], axis=0)
                if local_add_cond is not None
                else None
            )
            batch_padding_mask = (
                mx.concatenate([padding_mask, padding_mask], axis=0)
                if padding_mask is not None
                else None
            )

            batch_cond = None
            batch_cond_mask = None
            if cross_attn_cond is not None:
                null_embed = mx.zeros_like(cross_attn_cond)
                if negative_cross_attn_cond is not None:
                    if negative_cross_attn_mask is not None:
                        mask = negative_cross_attn_mask.astype(mx.bool_)[:, :, None]
                        negative_cross_attn_cond = mx.where(mask, negative_cross_attn_cond, null_embed)
                    batch_cond = mx.concatenate([cross_attn_cond, negative_cross_attn_cond], axis=0)
                else:
                    batch_cond = mx.concatenate([cross_attn_cond, null_embed], axis=0)
                if cross_attn_cond_mask is not None:
                    batch_cond_mask = mx.concatenate([cross_attn_cond_mask, cross_attn_cond_mask], axis=0)

            batch_prepend = None
            batch_prepend_mask = None
            if prepend_cond is not None:
                null_embed = mx.zeros_like(prepend_cond)
                batch_prepend = mx.concatenate([prepend_cond, null_embed], axis=0)
                if prepend_cond_mask is not None:
                    batch_prepend_mask = mx.concatenate([prepend_cond_mask, prepend_cond_mask], axis=0)

            batch_out = self._forward(
                batch_inputs,
                batch_t,
                cross_attn_cond=batch_cond,
                cross_attn_cond_mask=batch_cond_mask,
                input_concat_cond=batch_input_concat,
                local_add_cond=batch_local_add,
                global_embed=batch_global,
                prepend_cond=batch_prepend,
                prepend_cond_mask=batch_prepend_mask,
                padding_mask=batch_padding_mask,
                return_info=return_info,
            )
            if return_info:
                batch_out, info = batch_out

            cond_out, uncond_out = mx.split(batch_out, 2, axis=0)
            if self.diffusion_objective == "v":
                cond_denoised = x * alpha[:, None, None] - cond_out * sigma[:, None, None]
                uncond_denoised = x * alpha[:, None, None] - uncond_out * sigma[:, None, None]
            else:
                cond_denoised = x - cond_out * sigma[:, None, None]
                uncond_denoised = x - uncond_out * sigma[:, None, None]

            diff = cond_denoised - uncond_denoised
            if cfg_norm_threshold > 0.0:
                if padding_mask is not None:
                    mask = padding_mask[:, None, :].astype(diff.dtype)
                    diff_norm = mx.sqrt(
                        mx.sum((diff * mask) * (diff * mask), axis=(-1, -2), keepdims=True)
                    )
                else:
                    diff_norm = mx.sqrt(mx.sum(diff * diff, axis=(-1, -2), keepdims=True))
                scale_factor = mx.minimum(
                    mx.ones_like(diff_norm),
                    float(cfg_norm_threshold) / mx.maximum(diff_norm, 1e-12),
                )
                diff = diff * scale_factor

            if apg_scale == 0.0:
                cfg_diff = diff
            elif apg_scale == 1.0:
                _, diff_orthogonal = self._apg_project(
                    diff,
                    cond_denoised,
                    padding_mask=padding_mask,
                )
                cfg_diff = diff_orthogonal
            else:
                _, diff_orthogonal = self._apg_project(
                    diff,
                    cond_denoised,
                    padding_mask=padding_mask,
                )
                cfg_diff = float(apg_scale) * diff_orthogonal + (1.0 - float(apg_scale)) * diff

            cfg_denoised = cond_denoised + (float(cfg_scale) - 1.0) * cfg_diff
            if self.diffusion_objective == "v":
                output = (x * alpha[:, None, None] - cfg_denoised) / sigma[:, None, None]
            else:
                output = (x - cfg_denoised) / sigma[:, None, None]

            if scale_phi != 0.0:
                cond_std = mx.std(cond_out, axis=1, keepdims=True)
                cfg_std = mx.std(output, axis=1, keepdims=True) + 1e-12
                output = float(scale_phi) * (output * (cond_std / cfg_std)) + (
                    1.0 - float(scale_phi)
                ) * output

            if return_info:
                info["uncond_output"] = uncond_out
                return output, info
            return output

        return self._forward(
            x,
            t,
            cross_attn_cond=cross_attn_cond,
            cross_attn_cond_mask=cross_attn_cond_mask,
            input_concat_cond=input_concat_cond,
            local_add_cond=local_add_cond,
            modular_local_cond=modular_local_cond,
            global_embed=global_embed,
            prepend_cond=prepend_cond,
            prepend_cond_mask=prepend_cond_mask,
            padding_mask=padding_mask,
            return_info=return_info,
        )

    def load_torch_state_dict(self, torch_state_dict: dict[str, tp.Any]) -> ConversionReport:
        params = dict(tree_flatten(self.parameters()))
        missing: list[str] = []
        used_keys: set[str] = set()
        transposed: list[str] = []
        updates: list[tuple[str, tp.Any]] = []

        for key, target in params.items():
            source_key = _resolve_torch_state_key(key, torch_state_dict)
            if source_key is None:
                missing.append(key)
                continue
            used_keys.add(source_key)
            src = torch_state_dict[source_key].detach().cpu().numpy()
            src, did_transpose = _convert_weight_to_mlx_shape(src, tuple(target.shape))
            if did_transpose:
                transposed.append(key)
            arr = mx.array(src.astype(np.float32, copy=False))
            if arr.dtype != self.param_dtype:
                arr = arr.astype(self.param_dtype)
            updates.append((key, arr))

        if missing:
            raise RuntimeError(f"Missing {len(missing)} keys for MLX DiT load, e.g. {missing[:5]}")

        self.update(tree_unflatten(updates))
        unexpected = sorted(k for k in torch_state_dict if k not in used_keys)
        return ConversionReport(
            missing_keys=missing,
            unexpected_keys=unexpected,
            transposed_keys=transposed,
        )

    @classmethod
    def from_sao_model_config(
        cls,
        model_config: dict[str, tp.Any],
        *,
        param_dtype=mx.float32,
    ) -> "StableAudioMLXDiT":
        return cls(
            config=extract_dit_config(model_config),
            diffusion_objective=extract_diffusion_objective(model_config),
            param_dtype=param_dtype,
        )

    @classmethod
    def from_hosted_medium_npz(
        cls,
        model_config: dict[str, tp.Any],
        weights_path: str,
        *,
        param_dtype=mx.float16,
    ) -> "StableAudioMLXDiT":
        """Load the official hosted medium weights into Gary's generic DiT.

        The optimized checkpoint has the same 522 DiT tensors as the generic
        model. Its only naming differences are MLX RMSNorm's ``weight`` versus
        Gary's ``gamma`` and the official local-embed wrapper's ``seq`` level.
        """

        model = cls.from_sao_model_config(model_config, param_dtype=param_dtype)
        hosted = {
            key: value
            for key, value in dict(mx.load(str(weights_path))).items()
            if not key.startswith("cond.")
        }
        updates = []
        missing = []
        for target_key, _ in tree_flatten(model.parameters()):
            source_key = _hosted_medium_source_key(target_key)
            source = hosted.get(source_key)
            if source is None:
                missing.append((target_key, source_key))
                continue
            updates.append((target_key, source.astype(param_dtype)))
        if missing:
            raise RuntimeError(
                "Hosted medium NPZ is incompatible with Gary's DiT; "
                f"missing {len(missing)} tensor(s), e.g. {missing[:3]}"
            )
        if len(updates) != len(hosted):
            used = {_hosted_medium_source_key(key) for key, _ in updates}
            unexpected = sorted(set(hosted) - used)
            raise RuntimeError(
                "Hosted medium NPZ has unexpected DiT tensors; "
                f"expected {len(updates)}, found {len(hosted)}, "
                f"e.g. {unexpected[:3]}"
            )
        model.load_weights(updates, strict=True)
        del hosted, updates
        mx.eval(model.parameters())
        return model

    @classmethod
    def from_torch_dit(
        cls,
        torch_dit_model,
        model_config: dict[str, tp.Any],
        *,
        mlx_dtype=mx.float32,
    ) -> tuple["StableAudioMLXDiT", ConversionReport]:
        mlx_model = cls.from_sao_model_config(model_config, param_dtype=mlx_dtype)
        report = mlx_model.load_torch_state_dict(torch_dit_model.state_dict())
        mx.eval(mlx_model.parameters())
        return mlx_model, report


def _resolve_torch_state_key(key: str, torch_state_dict: dict[str, tp.Any]) -> str | None:
    for candidate in (key, f"model.{key}", f"model.model.{key}"):
        if candidate in torch_state_dict:
            return candidate
    return None


def _hosted_medium_source_key(target_key: str) -> str:
    source_key = target_key
    if ".gamma" in source_key:
        source_key = source_key.replace(".gamma", ".weight")
    if ".to_local_embed." in source_key:
        source_key = source_key.replace(
            ".to_local_embed.",
            ".to_local_embed.seq.",
        )
    return source_key


def _convert_weight_to_mlx_shape(arr: np.ndarray, target_shape: tuple[int, ...]) -> tuple[np.ndarray, bool]:
    if arr.shape == target_shape:
        return arr, False

    if arr.ndim == 3:
        candidate = np.transpose(arr, (0, 2, 1))
        if candidate.shape == target_shape:
            return candidate, True

    if arr.ndim == 2:
        candidate = arr.T
        if candidate.shape == target_shape:
            return candidate, True

    raise ValueError(f"Unable to map tensor with shape {arr.shape} to target {target_shape}")
