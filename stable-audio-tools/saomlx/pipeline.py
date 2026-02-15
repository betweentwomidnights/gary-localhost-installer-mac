"""Top-level SAO MLX generation pipeline."""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import math
import threading
import typing as tp
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import numpy as np
import soundfile as sf

from .conditioning_mlx import SAOMLXConditioner
from .dit import SAODiT
from .hf_download import download_file
from .logging import tensor_stats, write_json
from .quantization import quantize_dit
from .sampler_base import (
    sample_k_dpm_2_mlx,
    sample_k_dpmpp_2m_mlx,
    sample_k_dpmpp_2m_sde_mlx,
    sample_k_dpmpp_2s_ancestral_mlx,
    sample_k_dpmpp_3m_sde_mlx,
    sample_k_heun_mlx,
    sample_k_lms_mlx,
    sample_rf_dpmpp_mlx,
    sample_rf_euler_mlx,
    sample_v_ddim_mlx,
)
from .sampler_pingpong import PingPongSampler
from .sao_autoencoder_mlx import SAOAutoencoderMLX


@dataclass
class _TorchModelCacheEntry:
    model: tp.Any
    model_config: dict[str, tp.Any]
    model_source: dict[str, tp.Any]


@dataclass
class _ConditionerCacheEntry:
    conditioner: SAOMLXConditioner


@dataclass
class _ConvertedModelCacheEntry:
    dit: SAODiT
    vae: SAOAutoencoderMLX
    dit_report: tp.Any
    vae_report: tp.Any
    vae_config_repo: str
    quantization_report: dict[str, tp.Any]


_CACHE_LOCK = threading.Lock()
_TORCH_MODEL_CACHE: "OrderedDict[str, _TorchModelCacheEntry]" = OrderedDict()
_CONDITIONER_CACHE: "OrderedDict[str, _ConditionerCacheEntry]" = OrderedDict()
_CONVERTED_MODEL_CACHE: "OrderedDict[str, _ConvertedModelCacheEntry]" = OrderedDict()


def _cache_touch(cache: OrderedDict, key: str) -> None:
    if key in cache:
        cache.move_to_end(key)


def _cache_put_lru(cache: OrderedDict, key: str, value: tp.Any, max_entries: int) -> None:
    cache[key] = value
    cache.move_to_end(key)
    max_entries = max(1, int(max_entries))
    while len(cache) > max_entries:
        _, evicted = cache.popitem(last=False)
        del evicted
    gc.collect()


def clear_runtime_caches() -> None:
    with _CACHE_LOCK:
        _TORCH_MODEL_CACHE.clear()
        _CONDITIONER_CACHE.clear()
        _CONVERTED_MODEL_CACHE.clear()
    gc.collect()


def get_runtime_cache_info() -> dict[str, tp.Any]:
    with _CACHE_LOCK:
        return {
            "torch_model_keys": list(_TORCH_MODEL_CACHE.keys()),
            "conditioner_keys": list(_CONDITIONER_CACHE.keys()),
            "converted_model_keys": list(_CONVERTED_MODEL_CACHE.keys()),
            "torch_model_count": len(_TORCH_MODEL_CACHE),
            "conditioner_count": len(_CONDITIONER_CACHE),
            "converted_model_count": len(_CONVERTED_MODEL_CACHE),
        }


def _coerce_scalar(value: str) -> tp.Any:
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def parse_extra_cond(items: list[str]) -> dict[str, tp.Any]:
    parsed: dict[str, tp.Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --cond value '{item}'. Expected KEY=VALUE.")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Invalid --cond value '{item}'. Key cannot be empty.")
        parsed[key] = _coerce_scalar(value)
    return parsed


def build_conditioning(
    model_config: dict[str, tp.Any],
    prompt: str,
    negative_prompt: str,
    seconds_total: int,
    batch_size: int,
    extra_cond: dict[str, tp.Any],
) -> tuple[list[dict[str, tp.Any]], list[dict[str, tp.Any]] | None]:
    cond_cfg = model_config.get("model", {}).get("conditioning", {}).get("configs", [])

    cond: dict[str, tp.Any] = {}
    for item in cond_cfg:
        cond_id = item["id"]
        if cond_id == "prompt":
            cond[cond_id] = prompt
        elif cond_id == "seconds_start":
            cond[cond_id] = extra_cond.get(cond_id, 0)
        elif cond_id == "seconds_total":
            cond[cond_id] = seconds_total
        elif cond_id in extra_cond:
            cond[cond_id] = extra_cond[cond_id]
        else:
            raise ValueError(
                f"Missing conditioning key '{cond_id}' for this model. "
                "Provide it via --cond KEY=VALUE."
            )

    if not cond:
        cond = {"prompt": prompt}

    conditioning = [copy.deepcopy(cond) for _ in range(batch_size)]
    negative_conditioning = None
    if negative_prompt:
        neg = copy.deepcopy(cond)
        if "prompt" in neg:
            neg["prompt"] = negative_prompt
        negative_conditioning = [copy.deepcopy(neg) for _ in range(batch_size)]
    return conditioning, negative_conditioning


def _tensor_to_numpy(value: tp.Any) -> np.ndarray:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        try:
            return np.asarray(value.numpy())
        except Exception:
            pass
    try:
        return np.asarray(value)
    except Exception:
        cast_attempts: list[tp.Any] = [np.float32, "float32"]
        try:
            cast_attempts = [mx.float32, *cast_attempts]
        except Exception:
            pass
        if hasattr(value, "astype"):
            for cast_dtype in cast_attempts:
                try:
                    casted = value.astype(cast_dtype)
                    if hasattr(casted, "numpy"):
                        try:
                            return np.asarray(casted.numpy())
                        except Exception:
                            pass
                    return np.asarray(casted)
                except Exception:
                    continue
        raise


def _scalar(value: tp.Any) -> float:
    arr = _tensor_to_numpy(value)
    return float(np.asarray(arr).reshape(-1)[0])


def _summarize_conditioning_inputs(conditioning_inputs: dict[str, tp.Any]) -> dict[str, tp.Any]:
    out: dict[str, tp.Any] = {}
    for key, value in conditioning_inputs.items():
        if value is None:
            continue
        out[key] = tensor_stats(value).to_dict()
    return out


def _default_sampler_for_objective(diffusion_objective: str, sampler_type: str | None) -> str:
    if sampler_type:
        return sampler_type
    if diffusion_objective == "rf_denoiser":
        return "pingpong"
    if diffusion_objective == "rectified_flow":
        return "euler"
    if diffusion_objective == "v":
        return "dpmpp-3m-sde"
    return "unsupported"


def _parse_mx_dtype(value: str):
    normalized = value.strip().lower()
    mapping = {
        "float32": mx.float32,
        "bfloat16": mx.bfloat16,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype '{value}'. Expected one of {sorted(mapping)}")
    return mapping[normalized], normalized


def _load_vae_config(repo_id: str) -> tuple[dict[str, tp.Any], str]:
    # Open-small does not publish standalone VAE config file; use open-1.0 config shape.
    for candidate in (repo_id, "stabilityai/stable-audio-open-1.0"):
        try:
            cfg_path = download_file(candidate, "vae_model_config.json")
            with cfg_path.open("r", encoding="utf-8") as handle:
                return json.load(handle), candidate
        except Exception:
            continue
    raise FileNotFoundError("Could not load `vae_model_config.json` from repo or fallback.")


def _build_vae_config_from_model_config(model_config: dict[str, tp.Any]) -> dict[str, tp.Any] | None:
    pre_cfg = model_config.get("model", {}).get("pretransform", {}).get("config")
    if not isinstance(pre_cfg, dict):
        return None
    required = {"encoder", "decoder", "bottleneck", "latent_dim", "downsampling_ratio", "io_channels"}
    if not required.issubset(pre_cfg.keys()):
        return None
    sample_rate = model_config.get("sample_rate")
    audio_channels = model_config.get("audio_channels")
    if sample_rate is None or audio_channels is None:
        return None
    return {
        "model_type": "autoencoder",
        "sample_rate": int(sample_rate),
        "audio_channels": int(audio_channels),
        "model": pre_cfg,
    }


def _load_torch_model_and_config(
    *,
    model_id: str | None,
    model_config_path: str | Path | None,
    model_ckpt_path: str | Path | None,
    pretransform_ckpt_path: str | Path | None,
):
    if model_id:
        from stable_audio_tools.models.pretrained import get_pretrained_model

        model, model_config = get_pretrained_model(model_id)
        source = {"type": "pretrained", "model_id": model_id}
        return model, model_config, source

    if model_config_path is None or model_ckpt_path is None:
        raise ValueError("Provide either `model_id` or both `model_config_path` and `model_ckpt_path`.")

    from stable_audio_tools.models.factory import create_model_from_config
    from stable_audio_tools.models.utils import copy_state_dict, load_ckpt_state_dict

    config_path = Path(model_config_path).expanduser().resolve()
    ckpt_path = Path(model_ckpt_path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"model_config_path does not exist: {config_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"model_ckpt_path does not exist: {ckpt_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        model_config = json.load(handle)

    model = create_model_from_config(model_config)
    copy_state_dict(model, load_ckpt_state_dict(str(ckpt_path)))

    if pretransform_ckpt_path is not None:
        pre_path = Path(pretransform_ckpt_path).expanduser().resolve()
        if not pre_path.exists():
            raise FileNotFoundError(f"pretransform_ckpt_path does not exist: {pre_path}")
        if model.pretransform is None:
            raise RuntimeError("pretransform_ckpt_path provided but model has no pretransform")
        model.pretransform.load_state_dict(load_ckpt_state_dict(str(pre_path)), strict=False)

    source = {
        "type": "local",
        "model_config_path": str(config_path),
        "model_ckpt_path": str(ckpt_path),
        "pretransform_ckpt_path": (
            str(Path(pretransform_ckpt_path).expanduser().resolve()) if pretransform_ckpt_path else None
        ),
    }
    return model, model_config, source


def _path_signature(path_like: str | Path | None) -> str:
    if path_like is None:
        return "none"
    path = Path(path_like).expanduser().resolve()
    try:
        st = path.stat()
        return f"{path}|size={st.st_size}|mtime_ns={st.st_mtime_ns}"
    except Exception:
        return f"{path}|missing"


def _torch_model_cache_key(
    *,
    model_id: str | None,
    model_config_path: str | Path | None,
    model_ckpt_path: str | Path | None,
    pretransform_ckpt_path: str | Path | None,
) -> str:
    if model_id:
        return f"pretrained::{model_id}"
    payload = "|".join(
        [
            "local",
            _path_signature(model_config_path),
            _path_signature(model_ckpt_path),
            _path_signature(pretransform_ckpt_path),
        ]
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"local::{digest}"


def _conditioner_cache_key(*, torch_cache_key: str, dtype_name: str) -> str:
    return f"{torch_cache_key}|cond_dtype={dtype_name}"


def _converted_model_cache_key(
    *,
    torch_cache_key: str,
    dtype_name: str,
    dit_quantize: bool,
    dit_q_bits: int,
    dit_q_group_size: int,
    dit_q_scope: str,
) -> str:
    if not dit_quantize:
        return f"{torch_cache_key}|dit_dtype={dtype_name}|quant=off"
    return (
        f"{torch_cache_key}|dit_dtype={dtype_name}|quant=on"
        f"|bits={int(dit_q_bits)}|group={int(dit_q_group_size)}|scope={dit_q_scope}"
    )


def _load_torch_model_and_config_cached(
    *,
    model_id: str | None,
    model_config_path: str | Path | None,
    model_ckpt_path: str | Path | None,
    pretransform_ckpt_path: str | Path | None,
    cache_enabled: bool,
    cache_max_entries: int,
) -> tuple[tp.Any, dict[str, tp.Any], dict[str, tp.Any], str, bool]:
    import torch

    cache_key = _torch_model_cache_key(
        model_id=model_id,
        model_config_path=model_config_path,
        model_ckpt_path=model_ckpt_path,
        pretransform_ckpt_path=pretransform_ckpt_path,
    )

    if cache_enabled:
        with _CACHE_LOCK:
            entry = _TORCH_MODEL_CACHE.get(cache_key)
            if entry is not None:
                _cache_touch(_TORCH_MODEL_CACHE, cache_key)
                return entry.model, entry.model_config, entry.model_source, cache_key, True

    model, model_config, model_source = _load_torch_model_and_config(
        model_id=model_id,
        model_config_path=model_config_path,
        model_ckpt_path=model_ckpt_path,
        pretransform_ckpt_path=pretransform_ckpt_path,
    )
    model = model.to("cpu").eval().to(torch.float32)

    if cache_enabled:
        with _CACHE_LOCK:
            _cache_put_lru(
                _TORCH_MODEL_CACHE,
                cache_key,
                _TorchModelCacheEntry(
                    model=model,
                    model_config=model_config,
                    model_source=model_source,
                ),
                cache_max_entries,
            )

    return model, model_config, model_source, cache_key, False


def _get_or_build_conditioner_cached(
    *,
    torch_cache_key: str,
    torch_model: tp.Any,
    model_config: dict[str, tp.Any],
    dtype: mx.Dtype,
    dtype_name: str,
    cache_enabled: bool,
    cache_max_entries: int,
) -> tuple[SAOMLXConditioner, str, bool]:
    cache_key = _conditioner_cache_key(torch_cache_key=torch_cache_key, dtype_name=dtype_name)

    if cache_enabled:
        with _CACHE_LOCK:
            entry = _CONDITIONER_CACHE.get(cache_key)
            if entry is not None:
                _cache_touch(_CONDITIONER_CACHE, cache_key)
                return entry.conditioner, cache_key, True

    conditioner = SAOMLXConditioner.from_torch_model(
        torch_model=torch_model,
        model_config=model_config,
        dtype=dtype,
    )

    if cache_enabled:
        with _CACHE_LOCK:
            _cache_put_lru(
                _CONDITIONER_CACHE,
                cache_key,
                _ConditionerCacheEntry(conditioner=conditioner),
                cache_max_entries,
            )

    return conditioner, cache_key, False


def _get_or_build_converted_models_cached(
    *,
    torch_cache_key: str,
    torch_model: tp.Any,
    model_config: dict[str, tp.Any],
    model_id: str | None,
    dit_mx_dtype: mx.Dtype,
    dit_dtype_name: str,
    dit_quantize: bool,
    dit_q_bits: int,
    dit_q_group_size: int,
    dit_q_scope: str,
    cache_enabled: bool,
    cache_max_entries: int,
) -> tuple[_ConvertedModelCacheEntry, str, bool]:
    cache_key = _converted_model_cache_key(
        torch_cache_key=torch_cache_key,
        dtype_name=dit_dtype_name,
        dit_quantize=dit_quantize,
        dit_q_bits=dit_q_bits,
        dit_q_group_size=dit_q_group_size,
        dit_q_scope=dit_q_scope,
    )

    if cache_enabled:
        with _CACHE_LOCK:
            entry = _CONVERTED_MODEL_CACHE.get(cache_key)
            if entry is not None:
                _cache_touch(_CONVERTED_MODEL_CACHE, cache_key)
                return entry, cache_key, True

    dit_mlx, dit_report = SAODiT.from_torch_dit(
        torch_model.model.model,
        model_config,
        mlx_dtype=dit_mx_dtype,
    )

    vae_config = _build_vae_config_from_model_config(model_config)
    if vae_config is not None:
        vae_config_repo = "from-model-config"
    else:
        vae_config, vae_config_repo = _load_vae_config(model_id or "stabilityai/stable-audio-open-1.0")

    vae_mlx, vae_report = SAOAutoencoderMLX.from_torch_model(torch_model.pretransform.model, vae_config)

    quantization_report: dict[str, tp.Any] = {
        "enabled": False,
        "scope": dit_q_scope,
        "bits": int(dit_q_bits),
        "group_size": int(dit_q_group_size),
        "eligible_linear_modules": 0,
        "skipped_linear_modules": 0,
        "quantized_linear_modules": 0,
        "skipped_examples": [],
    }
    if dit_quantize:
        q_report = quantize_dit(
            dit_mlx,
            bits=int(dit_q_bits),
            group_size=int(dit_q_group_size),
            scope=dit_q_scope,
        )
        quantization_report = q_report.to_dict()
        mx.eval(dit_mlx.parameters())

    entry = _ConvertedModelCacheEntry(
        dit=dit_mlx,
        vae=vae_mlx,
        dit_report=dit_report,
        vae_report=vae_report,
        vae_config_repo=vae_config_repo,
        quantization_report=quantization_report,
    )

    if cache_enabled:
        with _CACHE_LOCK:
            _cache_put_lru(
                _CONVERTED_MODEL_CACHE,
                cache_key,
                entry,
                cache_max_entries,
            )

    return entry, cache_key, False


def _as_mx(value: tp.Any, *, dtype=mx.float32):
    if value is None:
        return None
    arr = _tensor_to_numpy(value)
    if arr.dtype == np.bool_:
        return mx.array(arr)
    return mx.array(arr.astype(np.float32, copy=False)).astype(dtype)


def _align_audio_length(audio_tc: np.ndarray, target_length: int) -> np.ndarray:
    if audio_tc.shape[0] == target_length:
        return audio_tc
    if audio_tc.shape[0] > target_length:
        return audio_tc[:target_length]
    pad = np.zeros((target_length - audio_tc.shape[0], audio_tc.shape[1]), dtype=audio_tc.dtype)
    return np.concatenate([audio_tc, pad], axis=0)


def _make_initial_noise(
    *,
    batch_size: int,
    io_channels: int,
    latent_size: int,
    seed: int,
    noise_backend: str,
):
    shape = (batch_size, io_channels, latent_size)
    if noise_backend == "mlx":
        return mx.random.normal(shape, dtype=mx.float32)
    if noise_backend == "torch":
        import torch

        torch.manual_seed(int(seed))
        t_noise = torch.randn(shape, device="cpu", dtype=torch.float32)
        return mx.array(t_noise.numpy())
    raise ValueError(f"Unknown noise_backend '{noise_backend}'. Expected one of ['mlx', 'torch'].")


def _cfg_gate_sigma(diffusion_objective: str, sampler_choice: str, sigma_logged: float) -> float:
    """
    Map callback sigma values into the sigma space used by SAT DiT CFG interval gating.

    For k-diffusion samplers with v-objective models, callbacks expose k-sigma
    while DiT compares against `sin(t*pi/2)` where `t = 2/pi * atan(k_sigma)`.
    """
    sigma_val = float(sigma_logged)
    if diffusion_objective == "v" and sampler_choice in {
        "dpmpp-3m-sde",
        "dpmpp-2m-sde",
        "dpmpp-2m",
        "k-heun",
        "k-lms",
        "k-dpmpp-2s-ancestral",
        "k-dpm-2",
    }:
        return sigma_val / math.sqrt(1.0 + sigma_val * sigma_val)
    return sigma_val


def _annotate_cfg_usage(
    *,
    step_stats: list[dict[str, tp.Any]],
    diffusion_objective: str,
    sampler_choice: str,
    cfg_scale: float,
    cfg_interval: tuple[float, float],
) -> dict[str, tp.Any]:
    steps_total = int(len(step_stats))
    steps_with_cfg = 0

    for row in step_stats:
        sigma_logged = row.get("sigma")
        if sigma_logged is None:
            row["cfg_applied"] = False
            continue
        sigma_gate = _cfg_gate_sigma(diffusion_objective, sampler_choice, float(sigma_logged))
        row["sigma_cfg_gate"] = float(sigma_gate)
        should_cfg = (
            float(cfg_scale) != 1.0
            and cfg_interval[0] <= sigma_gate <= cfg_interval[1]
        )
        row["cfg_applied"] = bool(should_cfg)
        if should_cfg:
            steps_with_cfg += 1

    model_eval_equivalent = steps_total + steps_with_cfg
    compute_multiplier = (model_eval_equivalent / steps_total) if steps_total > 0 else 1.0
    return {
        "steps_total": steps_total,
        "steps_with_cfg": int(steps_with_cfg),
        "model_eval_equivalent_steps": int(model_eval_equivalent),
        "estimated_compute_multiplier": float(compute_multiplier),
    }


def generate_diffusion_cond_mlx(
    *,
    model_id: str | None = None,
    model_config_path: str | Path | None = None,
    model_ckpt_path: str | Path | None = None,
    pretransform_ckpt_path: str | Path | None = None,
    prompt: str,
    negative_prompt: str = "",
    seed: int = 123,
    steps: int = 8,
    seconds: float = 4.0,
    cfg_scale: float = 1.0,
    cfg_interval: tuple[float, float] = (0.0, 1.0),
    batch_size: int = 1,
    sampler_type: str | None = None,
    sigma_max: float | None = None,
    sigma_min: float | None = None,
    rho: float = 1.0,
    eta: float = 1.0,
    s_noise: float = 1.0,
    sde_noise_backend: str = "brownian_torch",
    noise_backend: str = "mlx",
    conditioning_backend: str = "mlx",
    dit_dtype: str = "float32",
    dit_quantize: bool = False,
    dit_q_bits: int = 8,
    dit_q_group_size: int = 64,
    dit_q_scope: str = "transformer",
    cache_enabled: bool = True,
    cache_max_entries: int = 1,
    out_dir: str | Path | None = "runs/mlx_smoke",
    extra_cond: dict[str, tp.Any] | None = None,
    step_callback: tp.Callable[[dict[str, tp.Any]], None] | None = None,
) -> dict[str, tp.Any]:
    extra_cond = {} if extra_cond is None else dict(extra_cond)
    run_dir: Path | None = None
    if out_dir is not None:
        run_dir = (Path(out_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")).resolve()
        run_dir.mkdir(parents=True, exist_ok=False)

    t0 = perf_counter()
    model, model_config, model_source, torch_model_cache_key, torch_model_cache_hit = _load_torch_model_and_config_cached(
        model_id=model_id,
        model_config_path=model_config_path,
        model_ckpt_path=model_ckpt_path,
        pretransform_ckpt_path=pretransform_ckpt_path,
        cache_enabled=bool(cache_enabled),
        cache_max_entries=int(cache_max_entries),
    )
    t_model_fetched = perf_counter()
    t_model_ready = t_model_fetched

    if model.pretransform is None:
        raise RuntimeError("This pipeline currently expects latent diffusion models with a pretransform.")

    diffusion_objective = model.diffusion_objective
    sampler_choice = _default_sampler_for_objective(diffusion_objective, sampler_type)
    if sampler_choice == "unsupported":
        raise NotImplementedError(
            f"Diffusion objective '{diffusion_objective}' is not supported yet in MLX pipeline. "
            "Implement this objective's sampler path next."
        )

    sample_rate = int(model_config["sample_rate"])
    sample_size = int(seconds * sample_rate)
    latent_size = sample_size // int(model.pretransform.downsampling_ratio)
    seconds_total = max(1, math.ceil(seconds))

    if sigma_max is None:
        if diffusion_objective == "v":
            sigma_max = 500.0
        else:
            sigma_max = 1.0
    if sigma_min is None:
        if diffusion_objective == "v":
            sigma_min = 0.3
        else:
            sigma_min = 0.01

    if seed == -1:
        seed = int(np.random.randint(0, 2**32 - 1, dtype=np.uint32))
    if noise_backend not in {"mlx", "torch"}:
        raise ValueError(f"Unknown noise_backend '{noise_backend}'. Expected one of ['mlx', 'torch'].")
    mx.random.seed(int(seed))
    dit_mx_dtype, dit_dtype_name = _parse_mx_dtype(dit_dtype)
    cfg_interval = (float(cfg_interval[0]), float(cfg_interval[1]))
    if cfg_interval[0] > cfg_interval[1]:
        raise ValueError(f"cfg_interval min must be <= max, got {cfg_interval}")

    conditioning, negative_conditioning = build_conditioning(
        model_config=model_config,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seconds_total=seconds_total,
        batch_size=batch_size,
        extra_cond=extra_cond,
    )

    if conditioning_backend not in {"mlx", "torch"}:
        raise ValueError(
            f"Unknown conditioning_backend '{conditioning_backend}'. Expected one of ['mlx', 'torch']."
        )

    conditioner_cache_key = None
    conditioner_cache_hit = False
    cond_inputs: dict[str, tp.Any]
    neg_inputs: dict[str, tp.Any]
    if conditioning_backend == "mlx":
        conditioner_mlx, conditioner_cache_key, conditioner_cache_hit = _get_or_build_conditioner_cached(
            torch_cache_key=torch_model_cache_key,
            torch_model=model,
            model_config=model_config,
            dtype=dit_mx_dtype,
            dtype_name=dit_dtype_name,
            cache_enabled=bool(cache_enabled),
            cache_max_entries=int(cache_max_entries),
        )
        cond_inputs = conditioner_mlx.get_conditioning_inputs(conditioning)
        neg_inputs = (
            conditioner_mlx.get_conditioning_inputs(negative_conditioning, negative=True)
            if negative_conditioning is not None
            else {}
        )
    else:
        cond_tensors = model.conditioner(conditioning, "cpu")
        cond_inputs = model.get_conditioning_inputs(cond_tensors)
        neg_inputs = {}
        if negative_conditioning is not None:
            neg_tensors = model.conditioner(negative_conditioning, "cpu")
            neg_inputs = model.get_conditioning_inputs(neg_tensors, negative=True)
    t_conditioning_ready = perf_counter()

    converted_models, converted_model_cache_key, converted_model_cache_hit = _get_or_build_converted_models_cached(
        torch_cache_key=torch_model_cache_key,
        torch_model=model,
        model_config=model_config,
        model_id=model_id,
        dit_mx_dtype=dit_mx_dtype,
        dit_dtype_name=dit_dtype_name,
        dit_quantize=bool(dit_quantize),
        dit_q_bits=int(dit_q_bits),
        dit_q_group_size=int(dit_q_group_size),
        dit_q_scope=dit_q_scope,
        cache_enabled=bool(cache_enabled),
        cache_max_entries=int(cache_max_entries),
    )
    t_models_converted = perf_counter()
    dit_mlx = converted_models.dit
    vae_mlx = converted_models.vae
    dit_report = converted_models.dit_report
    vae_report = converted_models.vae_report
    vae_config_repo = converted_models.vae_config_repo
    quantization_report = converted_models.quantization_report
    t_dit_quantized = t_models_converted

    noise = _make_initial_noise(
        batch_size=batch_size,
        io_channels=int(model.io_channels),
        latent_size=int(latent_size),
        seed=int(seed),
        noise_backend=noise_backend,
    )
    step_stats: list[dict[str, tp.Any]] = []

    def _sampling_callback(payload: dict[str, tp.Any]) -> None:
        row: dict[str, tp.Any] = {"step": int(payload.get("i", len(step_stats)))}
        if "t" in payload:
            row["t"] = _scalar(payload["t"])
        if "sigma" in payload:
            row["sigma"] = _scalar(payload["sigma"])
        if "sigma_hat" in payload:
            row["sigma_hat"] = _scalar(payload["sigma_hat"])
        if "x" in payload:
            row["latent"] = tensor_stats(np.asarray(payload["x"])).to_dict()
        if "denoised" in payload:
            row["denoised"] = tensor_stats(np.asarray(payload["denoised"])).to_dict()
        step_stats.append(row)
        if step_callback is not None:
            try:
                step_callback(payload)
            except Exception:
                pass

    def cond_value(key: str):
        value = cond_inputs.get(key)
        return value if conditioning_backend == "mlx" else _as_mx(value)

    def neg_value(key: str):
        value = neg_inputs.get(key)
        return value if conditioning_backend == "mlx" else _as_mx(value)

    cond_args = {
        "cross_attn_cond": cond_value("cross_attn_cond"),
        "cross_attn_cond_mask": cond_value("cross_attn_mask"),
        "input_concat_cond": cond_value("input_concat_cond"),
        "global_embed": cond_value("global_cond"),
        "prepend_cond": cond_value("prepend_cond"),
        "prepend_cond_mask": cond_value("prepend_cond_mask"),
        "negative_cross_attn_cond": neg_value("negative_cross_attn_cond"),
        "negative_cross_attn_mask": neg_value("negative_cross_attn_mask"),
        "negative_global_embed": neg_value("negative_global_cond"),
        "cfg_scale": float(cfg_scale),
        "cfg_interval": cfg_interval,
    }

    if sampler_choice == "pingpong":
        latents = PingPongSampler(steps=steps, sigma_max=sigma_max).sample(
            dit_mlx,
            noise,
            callback=_sampling_callback,
            **cond_args,
        )
    elif sampler_choice == "euler":
        latents = sample_rf_euler_mlx(
            dit_mlx,
            noise,
            steps=steps,
            sigma_max=sigma_max,
            callback=_sampling_callback,
            **cond_args,
        )
    elif sampler_choice == "dpmpp":
        latents = sample_rf_dpmpp_mlx(
            dit_mlx,
            noise,
            steps=steps,
            sigma_max=sigma_max,
            callback=_sampling_callback,
            **cond_args,
        )
    elif sampler_choice == "v-ddim":
        latents = sample_v_ddim_mlx(
            dit_mlx,
            noise,
            steps=steps,
            eta=0.0,
            sigma_max=sigma_max,
            callback=_sampling_callback,
            **cond_args,
        )
    elif sampler_choice == "v-ddim-cfgpp":
        latents = sample_v_ddim_mlx(
            dit_mlx,
            noise,
            steps=steps,
            eta=0.0,
            sigma_max=sigma_max,
            cfg_pp=True,
            callback=_sampling_callback,
            **cond_args,
        )
    elif sampler_choice == "dpmpp-2m":
        latents = sample_k_dpmpp_2m_mlx(
            dit_mlx,
            noise,
            steps=steps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho=rho,
            callback=_sampling_callback,
            **cond_args,
        )
    elif sampler_choice == "dpmpp-2m-sde":
        brownian_seed = int(seed)
        if sde_noise_backend == "brownian_torch" and noise_backend == "torch":
            brownian_seed = None
        latents = sample_k_dpmpp_2m_sde_mlx(
            dit_mlx,
            noise,
            steps=steps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho=rho,
            eta=eta,
            s_noise=s_noise,
            sde_noise_backend=sde_noise_backend,
            seed=brownian_seed,
            callback=_sampling_callback,
            **cond_args,
        )
    elif sampler_choice == "k-heun":
        latents = sample_k_heun_mlx(
            dit_mlx,
            noise,
            steps=steps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho=rho,
            s_noise=s_noise,
            callback=_sampling_callback,
            **cond_args,
        )
    elif sampler_choice == "k-lms":
        latents = sample_k_lms_mlx(
            dit_mlx,
            noise,
            steps=steps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho=rho,
            callback=_sampling_callback,
            **cond_args,
        )
    elif sampler_choice == "k-dpmpp-2s-ancestral":
        brownian_seed = int(seed)
        if sde_noise_backend == "brownian_torch" and noise_backend == "torch":
            brownian_seed = None
        latents = sample_k_dpmpp_2s_ancestral_mlx(
            dit_mlx,
            noise,
            steps=steps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho=rho,
            eta=eta,
            s_noise=s_noise,
            sde_noise_backend=sde_noise_backend,
            seed=brownian_seed,
            callback=_sampling_callback,
            **cond_args,
        )
    elif sampler_choice == "k-dpm-2":
        latents = sample_k_dpm_2_mlx(
            dit_mlx,
            noise,
            steps=steps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho=rho,
            s_noise=s_noise,
            callback=_sampling_callback,
            **cond_args,
        )
    elif sampler_choice == "dpmpp-3m-sde":
        brownian_seed = int(seed)
        # SAT parity mode: with torch-initialized noise, k-diffusion's default
        # BrownianTreeNoiseSampler consumes torch RNG state instead of a fixed seed.
        if sde_noise_backend == "brownian_torch" and noise_backend == "torch":
            brownian_seed = None
        latents = sample_k_dpmpp_3m_sde_mlx(
            dit_mlx,
            noise,
            steps=steps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho=rho,
            eta=eta,
            s_noise=s_noise,
            sde_noise_backend=sde_noise_backend,
            seed=brownian_seed,
            callback=_sampling_callback,
            **cond_args,
        )
    else:
        raise ValueError(f"Unknown sampler_type '{sampler_choice}' for objective '{diffusion_objective}'")
    t_sampling_done = perf_counter()

    mx.eval(latents)
    decoded = vae_mlx.decode(latents)
    mx.eval(decoded)
    t_decode_done = perf_counter()

    cfg_summary = _annotate_cfg_usage(
        step_stats=step_stats,
        diffusion_objective=diffusion_objective,
        sampler_choice=sampler_choice,
        cfg_scale=float(cfg_scale),
        cfg_interval=cfg_interval,
    )

    decoded_nct = np.asarray(decoded)
    final_latents = np.asarray(latents)

    audio_files: list[str] = []
    audio_file_paths: list[Path] = []
    decoded_batches = []
    for i in range(decoded_nct.shape[0]):
        audio_tc = decoded_nct[i].T.astype(np.float32, copy=False)
        audio_tc = _align_audio_length(audio_tc, sample_size)
        decoded_batches.append(audio_tc)
        if run_dir is not None:
            wav_path = run_dir / f"output_{i:02d}.wav"
            sf.write(str(wav_path), audio_tc, sample_rate, format="WAV")
            audio_files.append(str(wav_path.relative_to(Path.cwd())))
            audio_file_paths.append(wav_path)

    decoded_np = np.stack(decoded_batches, axis=0)  # [B, T, C]
    rms = float(np.sqrt(np.mean(decoded_np**2)))
    peak = float(np.max(np.abs(decoded_np)))
    cache_info = get_runtime_cache_info()

    stats = {
        "model": model_id if model_id is not None else "local-model",
        "model_source": model_source,
        "cache": {
            "enabled": bool(cache_enabled),
            "max_entries": int(cache_max_entries),
            "torch_model_cache_key": torch_model_cache_key,
            "torch_model_cache_hit": bool(torch_model_cache_hit),
            "conditioner_cache_key": conditioner_cache_key,
            "conditioner_cache_hit": bool(conditioner_cache_hit),
            "converted_model_cache_key": converted_model_cache_key,
            "converted_model_cache_hit": bool(converted_model_cache_hit),
            "runtime_cache_counts": {
                "torch_models": int(cache_info.get("torch_model_count", 0)),
                "conditioners": int(cache_info.get("conditioner_count", 0)),
                "converted_models": int(cache_info.get("converted_model_count", 0)),
            },
        },
        "seed": int(seed),
        "steps": int(steps),
        "cfg": float(cfg_scale),
        "cfg_interval": [float(cfg_interval[0]), float(cfg_interval[1])],
        "cfg_summary": cfg_summary,
        "seconds": float(seconds),
        "sample_rate": sample_rate,
        "sample_size": sample_size,
        "diffusion_objective": diffusion_objective,
        "dit_dtype": dit_dtype_name,
        "quantization": quantization_report,
        "sampler_type": sampler_choice,
        "sampler_params": {
            "sigma_min": float(sigma_min),
            "sigma_max": float(sigma_max),
            "rho": float(rho),
            "eta": float(eta),
            "s_noise": float(s_noise),
            "sde_noise_backend": sde_noise_backend,
        },
        "sampler_notes": (
            "Brownian-tree SDE noise uses torch/k-diffusion precomputed CPU increments when "
            "`sde_noise_backend=brownian_torch`; set `gaussian` for faster i.i.d. fallback."
            if sampler_choice in {"dpmpp-3m-sde", "dpmpp-2m-sde", "k-dpmpp-2s-ancestral"}
            else ""
        ),
        "conditioning": {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "conditioning_batch": conditioning,
        },
        "conditioning_backend": conditioning_backend,
        "noise_backend": noise_backend,
        "text_embedding_stats": _summarize_conditioning_inputs(cond_inputs),
        "initial_latent_stats": tensor_stats(np.asarray(noise)).to_dict(),
        "step_stats": step_stats,
        "final_latent_stats": tensor_stats(final_latents).to_dict(),
        "decoded_waveform_stats": {
            **tensor_stats(decoded_np).to_dict(),
            "rms": rms,
            "peak": peak,
        },
        "conversion": {
            "dit": {
                "missing_keys": dit_report.missing_keys,
                "unexpected_keys_count": len(dit_report.unexpected_keys),
                "transposed_keys_count": len(dit_report.transposed_keys),
            },
            "vae": {
                "missing_keys": vae_report.missing_keys,
                "unexpected_keys_count": len(vae_report.unexpected_keys),
                "transposed_keys_count": len(vae_report.transposed_keys),
                "config_repo": vae_config_repo,
            },
        },
        "audio_files": audio_files,
        "timings_sec": {
            "model_fetch_torch": t_model_fetched - t0,
            "model_move_eval_torch": t_model_ready - t_model_fetched,
            "conditioning": t_conditioning_ready - t_model_ready,
            "conditioning_torch": (t_conditioning_ready - t_model_ready) if conditioning_backend == "torch" else 0.0,
            "conditioning_mlx": (t_conditioning_ready - t_model_ready) if conditioning_backend == "mlx" else 0.0,
            "mlx_convert": t_models_converted - t_conditioning_ready,
            "dit_quantize": t_dit_quantized - t_models_converted,
            "sampling_mlx": t_sampling_done - t_dit_quantized,
            "decode_mlx": t_decode_done - t_sampling_done,
            "total": t_decode_done - t0,
        },
    }
    stats_path: Path | None = None
    if run_dir is not None:
        stats_path = run_dir / "stats.json"
        write_json(stats_path, stats)

    return {
        "run_dir": run_dir,
        "stats_path": stats_path,
        "audio_files": audio_file_paths,
        "stats": stats,
        "audio": decoded_np,
        "sample_rate": sample_rate,
    }
