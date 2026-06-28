# Weight conversion from PyTorch AceStep DiT decoder to native MLX format.

import logging
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

_VARIANT_DIR = {
    "turbo": "acestep-v15-turbo",
    "base": "acestep-v15-base",
    "sft": "acestep-v15-sft",
    "xl-base": "acestep-v15-xl-base",
    "xl-turbo": "acestep-v15-xl-turbo",
}

_ADAPTER_ONLY_KEY_MARKERS = (
    ".lora_A.",
    ".lora_B.",
    ".lora_embedding_A.",
    ".lora_embedding_B.",
    ".lora_magnitude_vector.",
)


@dataclass(frozen=True)
class MLXCheckpointLoadReport:
    """Summary of a direct safetensors-to-MLX decoder load."""

    model_dir: Path
    tensor_count: int
    parameter_count: int
    shard_count: int
    dtype: str


def resolve_model_dir(checkpoint_dir: str | Path, variant: str) -> Path:
    """Resolve an ACE model directory without importing or loading PyTorch."""
    root = Path(checkpoint_dir).expanduser().resolve()
    candidates = []
    mapped = _VARIANT_DIR.get(str(variant))
    if mapped:
        candidates.append(root / mapped)
    candidates.append(root / str(variant))
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"ACE model directory not found; tried: {tried}")


def load_model_config(model_dir: str | Path) -> SimpleNamespace:
    """Load the lightweight decoder configuration from ``config.json``."""
    config_path = Path(model_dir) / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"ACE config.json not found at {config_path}")
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"ACE config must be a JSON object: {config_path}")
    return SimpleNamespace(**data)


def _normalize_decoder_key(key: str) -> Optional[str]:
    """Map PEFT-wrapped decoder keys to their plain MLX parameter names."""
    if "rotary_emb" in key:
        return None

    # Older PEFT wrappers may surface the base decoder under ``base_model.model``.
    if key.startswith("base_model.model."):
        key = key.removeprefix("base_model.model.")

    # ``modules_to_save`` wraps a saved module under the adapter name; strip it.
    key = re.sub(r"\.modules_to_save\.[^.]+\.", ".", key)

    if any(marker in key for marker in _ADAPTER_ONLY_KEY_MARKERS):
        return None

    # LoRA-wrapped linear layers expose their merged tensor under ``base_layer``.
    return key.replace(".base_layer.", ".")


def _local_decoder_key(key: str) -> Optional[str]:
    for prefix in (
        "decoder.",
        "base_model.model.decoder.",
        "model.decoder.",
    ):
        if key.startswith(prefix):
            return _normalize_decoder_key(key.removeprefix(prefix))
    return None


def _checkpoint_weight_map(model_dir: Path) -> dict[str, str]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError(f"Invalid safetensors index: {index_path}")
        return {str(key): str(value) for key, value in weight_map.items()}

    model_path = model_dir / "model.safetensors"
    if not model_path.is_file():
        raise FileNotFoundError(
            f"No model.safetensors or model.safetensors.index.json under {model_dir}"
        )
    from safetensors import safe_open

    with safe_open(str(model_path), framework="pt") as handle:
        return {key: model_path.name for key in handle.keys()}


def _convert_checkpoint_array(name: str, value, *, dtype):
    import mlx.core as mx

    if hasattr(value, "detach"):
        value = value.detach().cpu().float().numpy()
    array = mx.array(value)
    new_name = name
    if new_name.startswith("proj_in.1."):
        new_name = new_name.replace("proj_in.1.", "proj_in.")
        if new_name.endswith(".weight"):
            array = array.swapaxes(1, 2)
    elif new_name.startswith("proj_out.1."):
        new_name = new_name.replace("proj_out.1.", "proj_out.")
        if new_name.endswith(".weight"):
            array = array.transpose(1, 2, 0)
    if array.dtype in (mx.float16, mx.float32, mx.bfloat16):
        array = array.astype(dtype)
    return new_name, array


def load_decoder_safetensors(
    model_dir: str | Path,
    mlx_decoder,
    *,
    dtype,
) -> tuple[object | None, MLXCheckpointLoadReport]:
    """Stream ACE decoder tensors directly from safetensors into MLX.

    Only one source tensor is materialized at a time. This avoids constructing
    a full PyTorch model and a second model-sized MLX conversion list.
    """
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from safetensors import safe_open

    model_path = Path(model_dir).expanduser().resolve()
    weight_map = _checkpoint_weight_map(model_path)
    by_shard: dict[str, list[str]] = defaultdict(list)
    for source_name, shard_name in weight_map.items():
        if _local_decoder_key(source_name) is not None or source_name.endswith(
            "null_condition_emb"
        ):
            by_shard[shard_name].append(source_name)

    expected = {name for name, _ in tree_flatten(mlx_decoder.parameters())}
    loaded: set[str] = set()
    parameter_count = 0
    null_condition_emb = None

    for shard_name in sorted(by_shard):
        shard_path = model_path / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(f"Missing checkpoint shard: {shard_path}")
        with safe_open(str(shard_path), framework="pt") as handle:
            for source_name in sorted(by_shard[shard_name]):
                value = handle.get_tensor(source_name)
                if source_name.endswith("null_condition_emb"):
                    null_condition_emb = mx.array(value).astype(dtype)
                    mx.eval(null_condition_emb)
                    continue

                local_name = _local_decoder_key(source_name)
                if local_name is None:
                    continue
                target_name, array = _convert_checkpoint_array(
                    local_name,
                    value,
                    dtype=dtype,
                )
                if target_name not in expected:
                    raise ValueError(
                        f"Checkpoint tensor {source_name!r} maps to unknown MLX parameter "
                        f"{target_name!r}"
                    )
                mlx_decoder.load_weights([(target_name, array)], strict=False)
                mx.eval(array)
                loaded.add(target_name)
                parameter_count += int(array.size)
                del array, value
        _clear_mlx_cache(mx)

    missing = expected - loaded
    if missing:
        preview = ", ".join(sorted(missing)[:8])
        raise ValueError(
            f"Direct MLX checkpoint load is missing {len(missing)} decoder tensors: {preview}"
        )
    mx.eval(mlx_decoder.parameters())
    _clear_mlx_cache(mx)
    report = MLXCheckpointLoadReport(
        model_dir=model_path,
        tensor_count=len(loaded),
        parameter_count=parameter_count,
        shard_count=len(by_shard),
        dtype=str(dtype),
    )
    logger.info(
        "[MLX-DiT] Streamed %d decoder tensors (%d parameters) from %d shard(s) as %s.",
        report.tensor_count,
        report.parameter_count,
        report.shard_count,
        report.dtype,
    )
    return null_condition_emb, report


def _clear_mlx_cache(mx_module) -> None:
    clear_cache = getattr(mx_module, "clear_cache", None)
    if clear_cache is not None:
        clear_cache()
    else:
        mx_module.metal.clear_cache()


def convert_decoder_weights(
    pytorch_model,
) -> List[Tuple[str, "mx.array"]]:
    """Convert PyTorch decoder weights to a list of (name, mx.array) pairs
    suitable for ``mlx_decoder.load_weights()``.

    The function extracts weights from
    ``pytorch_model.decoder`` (``AceStepDiTModel``) and converts them to MLX
    format, handling:
        - Conv1d weight layout:  PT ``[out, in, K]`` -> MLX ``[out, K, in]``
        - ConvTranspose1d layout: PT ``[in, out, K]`` -> MLX ``[out, K, in]``
        - nn.Sequential index remapping (Lambda wrappers removed in MLX)
        - All other weights are transferred as-is

    Args:
        pytorch_model: The full ``AceStepConditionGenerationModel`` (PyTorch).

    Returns:
        List of (param_name, mx.array) pairs ready for ``model.load_weights()``.
    """
    import mlx.core as mx

    decoder = pytorch_model.decoder
    state_dict = decoder.state_dict()

    weights_by_name = {}
    skipped = 0

    for key, value in state_dict.items():
        normalized_key = _normalize_decoder_key(key)
        if normalized_key is None:
            skipped += 1
            continue

        np_val = value.detach().cpu().float().numpy()
        new_key = normalized_key

        # PyTorch proj_in is Sequential(Lambda, Conv1d, Lambda)
        # The Conv1d is at index 1.  In MLX we use a bare Conv1d.
        if new_key.startswith("proj_in.1."):
            new_key = new_key.replace("proj_in.1.", "proj_in.")
            if new_key.endswith(".weight"):
                # PT Conv1d weight: [out, in, K] -> MLX: [out, K, in]
                np_val = np_val.swapaxes(1, 2)

        # PyTorch proj_out is Sequential(Lambda, ConvTranspose1d, Lambda)
        elif new_key.startswith("proj_out.1."):
            new_key = new_key.replace("proj_out.1.", "proj_out.")
            if new_key.endswith(".weight"):
                # PT ConvTranspose1d weight: [in, out, K] -> MLX: [out, K, in]
                np_val = np_val.transpose(1, 2, 0)

        weights_by_name[new_key] = mx.array(np_val)

    weights = list(weights_by_name.items())

    logger.info(
        "[MLX-DiT] Converted %d decoder parameters to MLX format (%d adapter-only tensors skipped).",
        len(weights),
        skipped,
    )
    return weights


def convert_and_load(
    pytorch_model,
    mlx_decoder: "MLXDiTDecoder",
) -> None:
    """Convert PyTorch decoder weights and load them into an MLX decoder.

    Args:
        pytorch_model: The full AceStepConditionGenerationModel (PyTorch).
        mlx_decoder: An instance of ``MLXDiTDecoder`` (already constructed).
    """
    import mlx.core as mx

    weights = convert_decoder_weights(pytorch_model)
    mlx_decoder.load_weights(weights)
    mx.eval(mlx_decoder.parameters())
    logger.info("[MLX-DiT] Weights loaded and evaluated successfully.")
