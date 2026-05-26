from __future__ import annotations

import typing as tp
from dataclasses import dataclass

from stable_audio_3.mlx.dit import StableAudioMLXDiT
from stable_audio_3.mlx.runtime import import_mlx_core
from stable_audio_3.mlx.spec import MLXPortRequirements, extract_mlx_port_requirements

mx = import_mlx_core(required=True)


@dataclass(frozen=True)
class MLXDiTSmokeInputs:
    x: tp.Any
    t: tp.Any
    cross_attn_cond: tp.Any | None
    cross_attn_cond_mask: tp.Any | None
    global_embed: tp.Any | None
    local_add_cond: tp.Any | None
    padding_mask: tp.Any | None


def _find_conditioner(requirements: MLXPortRequirements, cond_id: str):
    for conditioner in requirements.conditioners:
        if conditioner.id == cond_id:
            return conditioner
    return None


def build_dummy_dit_smoke_inputs(
    model_config: dict[str, tp.Any],
    *,
    batch_size: int = 1,
    latent_length: int = 64,
    seconds_total: float = 30.0,
    dtype_name: str = "float32",
) -> MLXDiTSmokeInputs:
    requirements = extract_mlx_port_requirements(model_config)
    dtype = getattr(mx, dtype_name)
    dit_config = model_config["model"]["diffusion"]["config"]
    io_channels = int(dit_config["io_channels"])
    cond_token_dim = int(dit_config.get("cond_token_dim", 0) or 0)
    global_cond_dim = int(dit_config.get("global_cond_dim", 0) or 0)
    local_add_cond_dim = int(dit_config.get("local_add_cond_dim", 0) or 0)

    x = mx.random.normal((batch_size, io_channels, latent_length), dtype=dtype)
    t = mx.full((batch_size,), 0.5, dtype=dtype)

    prompt_seq_len = 0
    for cond_id in requirements.diffusion.cross_attention_cond_ids:
        conditioner = _find_conditioner(requirements, cond_id)
        if conditioner is None:
            continue
        if conditioner.type == "t5gemma":
            prompt_seq_len += int(conditioner.config.get("max_length", 0) or 0)
        else:
            prompt_seq_len += 1

    cross_attn_cond = None
    cross_attn_cond_mask = None
    if cond_token_dim > 0 and prompt_seq_len > 0:
        cross_attn_cond = mx.random.normal(
            (batch_size, prompt_seq_len, cond_token_dim),
            dtype=dtype,
        )
        cross_attn_cond_mask = mx.ones((batch_size, prompt_seq_len), dtype=mx.bool_)

    global_embed = None
    if global_cond_dim > 0 and requirements.diffusion.global_cond_ids:
        values = mx.full(
            (batch_size, global_cond_dim),
            float(seconds_total),
            dtype=dtype,
        )
        global_embed = values

    local_add_cond = None
    if local_add_cond_dim > 0 and requirements.diffusion.local_add_cond_ids:
        mask = mx.zeros((batch_size, 1, latent_length), dtype=dtype)
        masked_input = mx.zeros((batch_size, io_channels, latent_length), dtype=dtype)
        local_add_cond = mx.concatenate([mask, masked_input], axis=1)
        if int(local_add_cond.shape[1]) != local_add_cond_dim:
            raise ValueError(
                "Dummy local_add_cond shape does not match local_add_cond_dim: "
                f"{local_add_cond.shape[1]} vs {local_add_cond_dim}"
            )

    padding_mask = mx.ones((batch_size, latent_length), dtype=mx.bool_)

    return MLXDiTSmokeInputs(
        x=x,
        t=t,
        cross_attn_cond=cross_attn_cond,
        cross_attn_cond_mask=cross_attn_cond_mask,
        global_embed=global_embed,
        local_add_cond=local_add_cond,
        padding_mask=padding_mask,
    )


def run_mlx_dit_forward_smoke(
    model_config: dict[str, tp.Any],
    *,
    batch_size: int = 1,
    latent_length: int = 64,
    cfg_scale: float = 1.0,
    dtype_name: str = "float32",
) -> dict[str, tp.Any]:
    dtype = getattr(mx, dtype_name)
    model = StableAudioMLXDiT.from_sao_model_config(model_config, param_dtype=dtype)
    inputs = build_dummy_dit_smoke_inputs(
        model_config,
        batch_size=batch_size,
        latent_length=latent_length,
        dtype_name=dtype_name,
    )

    kwargs = {
        "cross_attn_cond": inputs.cross_attn_cond,
        "cross_attn_cond_mask": inputs.cross_attn_cond_mask,
        "global_embed": inputs.global_embed,
        "local_add_cond": inputs.local_add_cond,
    }

    objective = model.diffusion_objective
    if objective == "rectified_flow":
        kwargs["padding_mask"] = inputs.padding_mask

    output = model(
        inputs.x,
        inputs.t,
        cfg_scale=cfg_scale,
        **kwargs,
    )
    mx.eval(output)

    return {
        "diffusion_objective": objective,
        "input_shape": tuple(int(x) for x in inputs.x.shape),
        "output_shape": tuple(int(x) for x in output.shape),
        "dtype": str(output.dtype),
        "cfg_scale": float(cfg_scale),
        "used_padding_mask": objective == "rectified_flow",
    }
