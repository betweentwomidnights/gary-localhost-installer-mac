"""Construct a decoder-free PyTorch shell for ACE conditioning features."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import torch
from accelerate import init_empty_weights
from safetensors import safe_open

from acestep.models.common.configuration_acestep_v15 import AceStepConfig

_MODEL_CLASSES = {
    "acestep-v15-base": ("acestep.models.base.modeling_acestep_v15_base", "AceStepConditionGenerationModel"),
    "acestep-v15-sft": ("acestep.models.sft.modeling_acestep_v15_base", "AceStepConditionGenerationModel"),
    "acestep-v15-turbo": ("acestep.models.turbo.modeling_acestep_v15_turbo", "AceStepConditionGenerationModel"),
    "acestep-v15-xl-base": ("acestep.models.xl_base.modeling_acestep_v15_xl_base", "AceStepConditionGenerationModel"),
    "acestep-v15-xl-sft": ("acestep.models.xl_sft.modeling_acestep_v15_xl_base", "AceStepConditionGenerationModel"),
    "acestep-v15-xl-turbo": ("acestep.models.xl_turbo.modeling_acestep_v15_xl_turbo", "AceStepConditionGenerationModel"),
}


def _model_class(variant: str):
    """Resolve the bundled upstream model class for a supported variant."""
    try:
        module_name, class_name = _MODEL_CLASSES[variant]
    except KeyError as exc:
        raise ValueError(f"No conditioning shell model class for {variant!r}") from exc
    return getattr(importlib.import_module(module_name), class_name)


def _weight_map(root: Path) -> dict[str, str]:
    """Return tensor-to-file mappings for a single or sharded safetensors directory."""
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        return json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    model_path = root / "model.safetensors"
    if not model_path.is_file():
        raise FileNotFoundError(f"No conditioning safetensors found under {root}")
    with safe_open(str(model_path), framework="pt", device="cpu") as handle:
        return {name: model_path.name for name in handle.keys()}


def load_conditioning_shell(
    *,
    variant: str,
    config_root: str | Path,
    weights_root: str | Path,
    dtype: torch.dtype,
):
    """Build the upstream model on meta, discard its decoder, and assign auxiliary weights."""
    config = AceStepConfig.from_pretrained(Path(config_root), local_files_only=True)
    config._attn_implementation = "sdpa"
    # ResidualFSQ validates a small levels buffer with ``Tensor.item()`` during
    # construction, so buffers must remain on CPU while parameters stay meta.
    with init_empty_weights():
        model = _model_class(variant)(config)
    model.decoder = None
    expected = set(model.state_dict())
    mapping = _weight_map(Path(weights_root))
    absent = expected - set(mapping)
    if absent:
        raise ValueError(f"Conditioning checkpoint is missing {len(absent)} tensor(s)")

    loaded: set[str] = set()
    for shard_name in sorted({mapping[name] for name in expected}):
        names = sorted(name for name in expected if mapping[name] == shard_name)
        weights = {}
        with safe_open(str(Path(weights_root) / shard_name), framework="pt", device="cpu") as handle:
            for name in names:
                value = handle.get_tensor(name)
                if value.is_floating_point() and value.dtype != dtype:
                    value = value.to(dtype=dtype)
                weights[name] = value
        model.load_state_dict(weights, strict=False, assign=True)
        loaded.update(weights)
    if loaded != expected:
        raise ValueError(f"Conditioning shell loaded {len(loaded)} of {len(expected)} tensors")
    meta = [name for name, value in model.state_dict().items() if value.is_meta]
    if meta:
        raise ValueError(f"Conditioning shell retains meta tensors: {meta[:5]}")
    model.eval()
    return model
