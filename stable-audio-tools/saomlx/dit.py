"""MLX DiT model for Stable Audio Open diffusion checkpoints."""

from __future__ import annotations

import copy
import math
import sys
import typing as tp
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from .dit_blocks import ContinuousTransformer, FourierFeatures, Identity, run_layers


def _ensure_sat_import_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sat_dir = repo_root / "third_party" / "stable-audio-tools"
    if not sat_dir.exists():
        raise FileNotFoundError(
            f"Could not find stable-audio-tools at {sat_dir}. "
            "Run ./scripts/pin_deps.sh first."
        )
    sat_path = str(sat_dir)
    if sat_path not in sys.path:
        sys.path.insert(0, sat_path)


def _as_scalar(value: mx.array | float) -> float:
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
    cfg = diffusion.get("config")
    if not isinstance(cfg, dict):
        raise ValueError("Missing `model.diffusion.config` in model config.")
    return copy.deepcopy(cfg)


def extract_diffusion_objective(model_config: dict[str, tp.Any]) -> str:
    diffusion = model_config.get("model", {}).get("diffusion", {})
    objective = diffusion.get("diffusion_objective")
    return objective if isinstance(objective, str) and objective else "v"


class SAODiT(nn.Module):
    """MLX DiffusionTransformer subset used by Stable Audio Open checkpoints."""

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
        timestep_features_dim = 256
        self.timestep_features = FourierFeatures(1, timestep_features_dim)

        embed_dim = int(config["embed_dim"])
        timestep_embed_dim = config.get("timestep_embed_dim")
        input_concat_dim = int(config.get("input_concat_dim", 0))
        if self.timestep_cond_type == "global":
            timestep_embed_dim = embed_dim
        elif self.timestep_cond_type == "input_concat":
            if timestep_embed_dim is None:
                raise ValueError("timestep_embed_dim is required when timestep_cond_type='input_concat'")
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

        if self.global_cond_type == "adaLN":
            # SAT uses adaLN inside TransformerBlock; this path is intentionally deferred
            # until we need it for a checkpoint.
            raise NotImplementedError("global_cond_type='adaLN' is not implemented yet.")

        depth = int(config["depth"])
        num_heads = int(config["num_heads"])
        io_channels = int(config["io_channels"])
        dim_in = io_channels + self.input_concat_dim

        self.transformer = ContinuousTransformer(
            dim=embed_dim,
            depth=depth,
            dim_heads=embed_dim // num_heads,
            dim_in=dim_in * self.patch_size,
            dim_out=io_channels * self.patch_size,
            cross_attend=self.cond_token_dim > 0,
            cond_token_dim=cond_embed_dim,
            global_cond_dim=None,
            attn_kwargs=config.get("attn_kwargs", {}),
            ff_kwargs=config.get("ff_kwargs", {}),
            norm_kwargs=config.get("norm_kwargs", {}),
        )

        self.preprocess_conv = nn.Conv1d(dim_in, dim_in, kernel_size=1, bias=False)
        self.preprocess_conv.weight = mx.zeros_like(self.preprocess_conv.weight)

        self.postprocess_conv = nn.Conv1d(io_channels, io_channels, kernel_size=1, bias=False)
        self.postprocess_conv.weight = mx.zeros_like(self.postprocess_conv.weight)

    @staticmethod
    def _model_dtype_from_params(module: "SAODiT"):
        params = tree_flatten(module.parameters())
        if not params:
            return mx.float32
        return params[0][1].dtype

    @staticmethod
    def _apply_conv1d_ncl(conv: nn.Conv1d, x_ncl: mx.array) -> mx.array:
        x_nlc = mx.transpose(x_ncl, (0, 2, 1))
        y_nlc = conv(x_nlc)
        return mx.transpose(y_nlc, (0, 2, 1))

    @staticmethod
    def _patchify_nlc(x_nlc: mx.array, patch_size: int) -> mx.array:
        if patch_size == 1:
            return x_nlc
        b, t, c = x_nlc.shape
        if t % patch_size != 0:
            raise ValueError(f"Sequence length {t} not divisible by patch_size {patch_size}")
        x = x_nlc.reshape(b, t // patch_size, patch_size, c)
        x = mx.transpose(x, (0, 1, 3, 2))
        return x.reshape(b, t // patch_size, c * patch_size)

    @staticmethod
    def _unpatchify_ncl(x_ncl: mx.array, patch_size: int) -> mx.array:
        if patch_size == 1:
            return x_ncl
        b, cp, t = x_ncl.shape
        if cp % patch_size != 0:
            raise ValueError(f"Channel dim {cp} not divisible by patch_size {patch_size}")
        c = cp // patch_size
        x = x_ncl.reshape(b, c, patch_size, t)
        x = mx.transpose(x, (0, 1, 3, 2))
        return x.reshape(b, c, t * patch_size)

    def _forward(
        self,
        x: mx.array,
        t: mx.array,
        *,
        cross_attn_cond: mx.array | None = None,
        input_concat_cond: mx.array | None = None,
        global_embed: mx.array | None = None,
        prepend_cond: mx.array | None = None,
        prepend_cond_mask: mx.array | None = None,
        return_info: bool = False,
        exit_layer_ix: int | None = None,
    ):
        if input_concat_cond is not None:
            raise NotImplementedError("input_concat_cond path is not implemented in MLX DiT yet.")
        if prepend_cond_mask is not None:
            # SAT currently ignores this in its DiT forward path.
            prepend_cond_mask = None

        if cross_attn_cond is not None and self.to_cond_embed is not None:
            cross_attn_cond = run_layers(self.to_cond_embed, cross_attn_cond)

        if global_embed is not None and self.to_global_embed is not None:
            global_embed = run_layers(self.to_global_embed, global_embed)

        prepend_inputs = None
        prepend_length = 0
        if prepend_cond is not None:
            if self.to_prepend_embed is None:
                raise ValueError("Received prepend_cond but model has no prepend conditioning projection.")
            prepend_inputs = run_layers(self.to_prepend_embed, prepend_cond)
            prepend_length = int(prepend_inputs.shape[1])

        timestep_embed = run_layers(self.to_timestep_embed, self.timestep_features(t[:, None]))

        if self.timestep_cond_type == "global":
            if global_embed is not None:
                global_embed = global_embed + timestep_embed
            else:
                global_embed = timestep_embed
        elif self.timestep_cond_type == "input_concat":
            raise NotImplementedError("timestep_cond_type='input_concat' is not implemented in MLX DiT yet.")

        if self.global_cond_type == "prepend" and global_embed is not None:
            gtok = global_embed[:, None, :]
            if prepend_inputs is None:
                prepend_inputs = gtok
            else:
                prepend_inputs = mx.concatenate([prepend_inputs, gtok], axis=1)
            prepend_length = int(prepend_inputs.shape[1])

        x = self._apply_conv1d_ncl(self.preprocess_conv, x) + x
        x = mx.transpose(x, (0, 2, 1))  # [B, T, C]
        x = self._patchify_nlc(x, self.patch_size)

        out = self.transformer(
            x,
            prepend_embeds=prepend_inputs,
            context=cross_attn_cond,
            return_info=return_info,
            exit_layer_ix=exit_layer_ix,
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
        x: mx.array,
        t: mx.array,
        *,
        cross_attn_cond: mx.array | None = None,
        cross_attn_cond_mask: mx.array | None = None,
        negative_cross_attn_cond: mx.array | None = None,
        negative_cross_attn_mask: mx.array | None = None,
        input_concat_cond: mx.array | None = None,
        global_embed: mx.array | None = None,
        negative_global_embed: mx.array | None = None,
        prepend_cond: mx.array | None = None,
        prepend_cond_mask: mx.array | None = None,
        cfg_scale: float = 1.0,
        cfg_dropout_prob: float = 0.0,
        cfg_interval: tuple[float, float] = (0.0, 1.0),
        causal: bool = False,
        scale_phi: float = 0.0,
        return_info: bool = False,
        exit_layer_ix: int | None = None,
    ):
        if causal:
            raise ValueError("Causal mode is not supported for SAODiT.")
        if cfg_dropout_prob > 0.0 and cfg_scale == 1.0:
            raise NotImplementedError("cfg_dropout_prob path is not implemented yet.")

        model_dtype = self._model_dtype_from_params(self)
        x = x.astype(model_dtype)
        t = t.astype(model_dtype)

        if cross_attn_cond is not None:
            cross_attn_cond = cross_attn_cond.astype(model_dtype)
        if negative_cross_attn_cond is not None:
            negative_cross_attn_cond = negative_cross_attn_cond.astype(model_dtype)
        if input_concat_cond is not None:
            input_concat_cond = input_concat_cond.astype(model_dtype)
        if global_embed is not None:
            global_embed = global_embed.astype(model_dtype)
        if negative_global_embed is not None:
            negative_global_embed = negative_global_embed.astype(model_dtype)
        if prepend_cond is not None:
            prepend_cond = prepend_cond.astype(model_dtype)

        # SAT currently disables cross-attention masks in DiT due kernel issues.
        cross_attn_cond_mask = None
        if prepend_cond_mask is not None:
            prepend_cond_mask = prepend_cond_mask.astype(mx.bool_)

        if exit_layer_ix is not None:
            return self._forward(
                x,
                t,
                cross_attn_cond=cross_attn_cond,
                input_concat_cond=input_concat_cond,
                global_embed=global_embed,
                prepend_cond=prepend_cond,
                prepend_cond_mask=prepend_cond_mask,
                return_info=return_info,
                exit_layer_ix=exit_layer_ix,
            )

        if self.diffusion_objective == "v":
            sigma = mx.sin(t * (math.pi / 2.0))
        elif self.diffusion_objective in {"rectified_flow", "rf_denoiser"}:
            sigma = t
        else:
            sigma = t

        sigma0 = _as_scalar(sigma)
        should_cfg = (
            cfg_scale != 1.0
            and (cross_attn_cond is not None or prepend_cond is not None)
            and (cfg_interval[0] <= sigma0 <= cfg_interval[1])
        )

        if should_cfg:
            batch_inputs = mx.concatenate([x, x], axis=0)
            batch_t = mx.concatenate([t, t], axis=0)
            batch_global = mx.concatenate([global_embed, global_embed], axis=0) if global_embed is not None else None
            batch_input_concat = (
                mx.concatenate([input_concat_cond, input_concat_cond], axis=0)
                if input_concat_cond is not None
                else None
            )

            batch_cond = None
            if cross_attn_cond is not None:
                null_embed = mx.zeros_like(cross_attn_cond)
                if negative_cross_attn_cond is not None:
                    if negative_cross_attn_mask is not None:
                        mask = negative_cross_attn_mask.astype(mx.bool_)[:, :, None]
                        negative_cross_attn_cond = mx.where(mask, negative_cross_attn_cond, null_embed)
                    batch_cond = mx.concatenate([cross_attn_cond, negative_cross_attn_cond], axis=0)
                else:
                    batch_cond = mx.concatenate([cross_attn_cond, null_embed], axis=0)

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
                input_concat_cond=batch_input_concat,
                global_embed=batch_global,
                prepend_cond=batch_prepend,
                prepend_cond_mask=batch_prepend_mask,
                return_info=return_info,
            )
            if return_info:
                batch_out, info = batch_out

            cond_out, uncond_out = mx.split(batch_out, 2, axis=0)
            cfg_out = uncond_out + (cond_out - uncond_out) * cfg_scale

            if scale_phi != 0.0:
                cond_std = mx.std(cond_out, axis=1, keepdims=True)
                cfg_std = mx.std(cfg_out, axis=1, keepdims=True) + 1e-12
                out = scale_phi * (cfg_out * (cond_std / cfg_std)) + (1.0 - scale_phi) * cfg_out
            else:
                out = cfg_out

            if return_info:
                info["uncond_output"] = uncond_out
                return out, info
            return out

        return self._forward(
            x,
            t,
            cross_attn_cond=cross_attn_cond,
            input_concat_cond=input_concat_cond,
            global_embed=global_embed,
            prepend_cond=prepend_cond,
            prepend_cond_mask=prepend_cond_mask,
            return_info=return_info,
        )

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
    ) -> "SAODiT":
        return cls(
            config=extract_dit_config(model_config),
            diffusion_objective=extract_diffusion_objective(model_config),
            param_dtype=param_dtype,
        )

    @classmethod
    def from_torch_dit(
        cls,
        torch_dit_model,
        model_config: dict[str, tp.Any],
        *,
        mlx_dtype=mx.float32,
    ) -> tuple["SAODiT", ConversionReport]:
        mlx_model = cls.from_sao_model_config(model_config, param_dtype=mlx_dtype)
        report = mlx_model.load_torch_state_dict(torch_dit_model.state_dict())
        mx.eval(mlx_model.parameters())
        return mlx_model, report

    @classmethod
    def from_torch_pretrained(
        cls,
        repo_id: str = "stabilityai/stable-audio-open-small",
        *,
        torch_device: str = "cpu",
        torch_dtype: str = "float32",
        mlx_dtype=mx.float32,
    ) -> tuple["SAODiT", ConversionReport, dict[str, tp.Any]]:
        import torch

        _ensure_sat_import_path()
        from stable_audio_tools.models.pretrained import get_pretrained_model

        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        if torch_dtype not in dtype_map:
            raise ValueError(f"Unsupported torch_dtype '{torch_dtype}', expected one of {sorted(dtype_map)}")

        model, model_config = get_pretrained_model(repo_id)
        model = model.to(torch_device).eval()
        model = model.to(dtype_map[torch_dtype])
        torch_dit = model.model.model
        mlx_model, report = cls.from_torch_dit(torch_dit, model_config, mlx_dtype=mlx_dtype)
        return mlx_model, report, model_config


def _convert_weight_to_mlx_shape(arr: np.ndarray, target_shape: tuple[int, ...]) -> tuple[np.ndarray, bool]:
    if arr.shape == target_shape:
        return arr, False

    if arr.ndim == 3:
        # torch Conv1d: (out, in, k) -> MLX Conv1d: (out, k, in)
        cand = np.transpose(arr, (0, 2, 1))
        if cand.shape == target_shape:
            return cand, True

    if arr.ndim == 2:
        cand = arr.T
        if cand.shape == target_shape:
            return cand, True

    raise ValueError(f"Unable to map tensor with shape {arr.shape} to target {target_shape}")
