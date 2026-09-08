"""Handler state bridge for native MLX LoRA and DoRA adapters."""

from __future__ import annotations

import time
from typing import Any

from loguru import logger

from acestep.models.mlx.artifact_runtime import load_decoder_artifact
from acestep.models.mlx.dit_quantization import (
    dequantize_dit_linears,
    quantize_dit,
)
from acestep.models.mlx.native_adapter import NativeMLXAdapter


def native_mlx_active(host: Any) -> bool:
    """Return whether the handler is using the decoder-free conditioning shell."""
    return bool(getattr(host, "_conditioning_shell_active", False))


def add_native_mlx_lora(host: Any, path: str, name: str) -> str:
    """Load one PEFT LoRA/DoRA adapter directly into the resident MLX decoder."""
    if getattr(host, "_mlx_native_adapter", None) is not None:
        return "❌ Native MLX inference currently supports one loaded adapter at a time."
    decoder = getattr(host, "mlx_decoder", None)
    if decoder is None:
        return "❌ MLX decoder is unavailable. Re-initialize Carey before loading the adapter."
    base_quantization = None
    try:
        artifact_dir = getattr(host, "_mlx_artifact_dir", None)
        if artifact_dir is None:
            return "❌ Pinned MLX base artifact is unavailable for safe adapter restoration."

        def restore_base() -> None:
            host._mlx_null_condition_emb = load_decoder_artifact(artifact_dir, decoder)

        base_quantization = dequantize_dit_linears(decoder)
        if base_quantization is not None:
            restore_base()
            logger.info(
                "Restored exact FP16 MLX weights before native adapter merge."
            )
        variant = (getattr(host, "last_init_params", None) or {}).get("config_path")
        resolve_started = time.monotonic()
        adapter = NativeMLXAdapter(
            decoder,
            path,
            restore_base,
            expected_variant=variant,
        )
        resolve_seconds = time.monotonic() - resolve_started
        merge_started = time.monotonic()
        adapter.apply(1.0)
        merge_seconds = time.monotonic() - merge_started
    except Exception as exc:
        if base_quantization is not None:
            restore_base()
            quantize_dit(decoder, base_quantization)
        logger.exception("Failed to load native MLX LoRA/DoRA adapter")
        return f"❌ Failed to load native MLX LoRA/DoRA: {exc}"
    host._mlx_adapter_base_quantization = base_quantization
    host._mlx_native_adapter = adapter
    host.lora_loaded = True
    host.use_lora = True
    host.lora_scale = 1.0
    host._adapter_type = "dora" if adapter.use_dora else "lora"
    host._active_loras = {name: 1.0}
    host._lora_active_adapter = name
    logger.info(
        "Native MLX {} '{}' loaded with {} targets "
        "(resolve={:.3f}s, merge={:.3f}s, eval_batch={})",
        host._adapter_type,
        name,
        adapter.target_count,
        resolve_seconds,
        merge_seconds,
        getattr(adapter, "eval_batch_size", 1),
    )
    return (
        f"✅ LoRA '{name}' loaded natively into MLX ({adapter.target_count} targets; "
        f"resolve {resolve_seconds:.2f}s, merge {merge_seconds:.2f}s)"
    )


def scale_native_mlx_lora(host: Any, name: str, scale: float) -> str:
    """Apply a scale to the active native adapter and update handler state."""
    adapter = getattr(host, "_mlx_native_adapter", None)
    if adapter is None or name != getattr(host, "_lora_active_adapter", None):
        return f"❌ Unknown native MLX adapter: {name}"
    effective = scale if getattr(host, "use_lora", False) else 0.0
    try:
        started = time.monotonic()
        adapter.apply(effective)
        elapsed = time.monotonic() - started
    except Exception as exc:
        logger.exception("Failed to scale native MLX adapter")
        return f"❌ Failed to scale native MLX adapter: {exc}"
    host._active_loras[name] = scale
    host.lora_scale = scale
    suffix = " (LoRA disabled)" if effective == 0.0 and not host.use_lora else ""
    return f"✅ LoRA scale ({name}): {scale:.2f} in {elapsed:.2f}s{suffix}"


def toggle_native_mlx_lora(host: Any, enabled: bool) -> str:
    """Enable or disable the native adapter without reconstructing the decoder."""
    adapter = getattr(host, "_mlx_native_adapter", None)
    if adapter is None:
        return "❌ No native MLX adapter loaded."
    name = host._lora_active_adapter
    host.use_lora = bool(enabled)
    scale = host._active_loras.get(name, 1.0) if enabled else 0.0
    try:
        started = time.monotonic()
        adapter.apply(scale)
        elapsed = time.monotonic() - started
    except Exception as exc:
        host.use_lora = not enabled
        logger.exception("Failed to toggle native MLX adapter")
        return f"❌ Failed to toggle native MLX adapter: {exc}"
    return f"✅ LoRA {'enabled' if enabled else 'disabled'} in {elapsed:.2f}s"


def unload_native_mlx_lora(host: Any) -> str:
    """Restore exact base arrays and clear native adapter bookkeeping."""
    adapter = getattr(host, "_mlx_native_adapter", None)
    if adapter is None:
        return "⚠️ No LoRA adapter loaded."
    started = time.monotonic()
    adapter.unload()
    base_quantization = getattr(host, "_mlx_adapter_base_quantization", None)
    if base_quantization is not None:
        quantize_dit(host.mlx_decoder, base_quantization)
    elapsed = time.monotonic() - started
    host._mlx_native_adapter = None
    host.lora_loaded = False
    host.use_lora = False
    host.lora_scale = 1.0
    host._adapter_type = None
    host._active_loras = {}
    host._lora_active_adapter = None
    host._mlx_adapter_base_quantization = None
    precision = f", restored {base_quantization.bits}-bit base" if base_quantization else ""
    return f"✅ LoRA unloaded in {elapsed:.2f}s, using base MLX model{precision}"
