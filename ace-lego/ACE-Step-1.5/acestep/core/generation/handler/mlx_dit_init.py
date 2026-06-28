"""MLX DiT initialization helpers for Apple Silicon acceleration."""

import copy
import os
from types import SimpleNamespace
from typing import Optional, Tuple
from loguru import logger
from time import perf_counter


class MlxDitInitMixin:
    """Initialize native MLX DiT decoder state used by generation runtime."""

    @staticmethod
    def _env_bool_flag(name: str, default: bool) -> bool:
        value = os.environ.get(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}

    def _maybe_materialize_mlx_dit_static_buffers(self, mlx_decoder) -> None:
        """Optionally materialize reusable MLX DiT static buffers for worker reuse."""
        if not self._env_bool_flag("ACESTEP_MLX_DIT_MATERIALIZE_STATIC_BUFFERS", False):
            return

        materialize = getattr(mlx_decoder, "materialize_static_buffers", None)
        if not callable(materialize):
            logger.info("[MLX-DiT] Static buffer materialization requested but unavailable on decoder.")
            return

        started = perf_counter()
        materialize()
        elapsed = perf_counter() - started
        logger.info(
            "[MLX-DiT] Materialized static buffers in {:.3f}s.",
            elapsed,
        )

    def _mlx_conversion_model(self):
        """Return a model-like object whose decoder exposes plain weights for MLX conversion."""
        model = getattr(self, "model", None)
        decoder = getattr(model, "decoder", None) if model is not None else None
        if model is None or decoder is None:
            raise RuntimeError("MLX DiT refresh failed: model decoder is not initialized.")

        try:
            from peft import PeftModel
        except ImportError:
            return model, None

        if not isinstance(decoder, PeftModel):
            return model, None

        get_base_model = getattr(decoder, "get_base_model", None)
        if not callable(get_base_model):
            raise RuntimeError("MLX DiT refresh failed: PEFT decoder does not expose get_base_model().")

        lora_enabled = bool(getattr(self, "lora_loaded", False) and getattr(self, "use_lora", False))
        if not lora_enabled:
            return SimpleNamespace(decoder=get_base_model()), None

        active_adapter = getattr(self, "_lora_active_adapter", None)
        merge_adapter = getattr(decoder, "merge_adapter", None)
        unmerge_adapter = getattr(decoder, "unmerge_adapter", None)
        if callable(merge_adapter) and callable(unmerge_adapter):
            merge_kwargs = {"safe_merge": False}
            if active_adapter:
                merge_kwargs["adapter_names"] = [active_adapter]
            try:
                merge_adapter(**merge_kwargs)
            except TypeError:
                # Older PEFT builds may not accept keyword arguments on merge_adapter().
                if active_adapter:
                    merge_adapter([active_adapter])
                else:
                    merge_adapter()
            return SimpleNamespace(decoder=get_base_model()), unmerge_adapter

        merge_and_unload = getattr(decoder, "merge_and_unload", None)
        if callable(merge_and_unload):
            decoder_copy = copy.deepcopy(decoder)
            merge_and_unload_copy = getattr(decoder_copy, "merge_and_unload", None)
            if not callable(merge_and_unload_copy):
                raise RuntimeError("MLX DiT refresh failed: copied PEFT decoder lacks merge_and_unload().")
            merge_kwargs = {"safe_merge": False}
            if active_adapter:
                merge_kwargs["adapter_names"] = [active_adapter]
            try:
                merged_decoder = merge_and_unload_copy(**merge_kwargs)
            except TypeError:
                if active_adapter:
                    merged_decoder = merge_and_unload_copy([active_adapter])
                else:
                    merged_decoder = merge_and_unload_copy()
            return SimpleNamespace(decoder=merged_decoder), None

        raise RuntimeError(
            "MLX DiT refresh failed: active PEFT LoRA cannot be converted because merge helpers are unavailable."
        )

    def _init_mlx_dit(self, compile_model: bool = False) -> bool:
        """Initialize the MLX DiT decoder when platform support is available.

        Args:
            compile_model: Whether MLX diffusion should use ``mx.compile``.

        Returns:
            bool: ``True`` when MLX DiT is initialized successfully, else ``False``.
        """
        try:
            from acestep.models.mlx import mlx_available

            if not mlx_available():
                logger.info("[MLX-DiT] MLX not available on this platform; skipping.")
                return False

            from acestep.models.mlx.dit_model import MLXDiTDecoder
            from acestep.models.mlx.dit_convert import convert_and_load

            mlx_decoder = MLXDiTDecoder.from_config(self.config)
            convert_and_load(self.model, mlx_decoder)
            self._maybe_materialize_mlx_dit_static_buffers(mlx_decoder)
            self.mlx_decoder = mlx_decoder
            self.use_mlx_dit = True
            self.mlx_dit_compiled = compile_model
            logger.info(
                "[MLX-DiT] Native MLX DiT decoder initialized successfully "
                f"(mx.compile={compile_model})."
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[MLX-DiT] Failed to initialize MLX decoder (non-fatal): {exc}")
            self.mlx_decoder = None
            self.use_mlx_dit = False
            self.mlx_dit_compiled = False
            return False

    def _sync_mlx_dit_from_model(self, reason: str = "model update") -> Tuple[bool, Optional[str]]:
        """Refresh the native MLX decoder weights from the current PyTorch model."""
        if not getattr(self, "use_mlx_dit", False):
            return True, None

        if getattr(self, "model", None) is None or getattr(self, "config", None) is None:
            return False, "MLX DiT refresh failed: model is not initialized."

        try:
            from acestep.models.mlx import mlx_available

            if not mlx_available():
                self.mlx_decoder = None
                return False, "MLX DiT refresh failed: MLX is not available on this platform."

            from acestep.models.mlx.dit_model import MLXDiTDecoder
            from acestep.models.mlx.dit_convert import convert_and_load

            if self.mlx_decoder is None:
                self.mlx_decoder = MLXDiTDecoder.from_config(self.config)

            started = perf_counter()
            logger.info(f"[MLX-DiT] Refreshing native MLX decoder after {reason}...")
            pytorch_model, cleanup = self._mlx_conversion_model()
            try:
                convert_and_load(pytorch_model, self.mlx_decoder)
            finally:
                if callable(cleanup):
                    cleanup()
            self._maybe_materialize_mlx_dit_static_buffers(self.mlx_decoder)
            elapsed = perf_counter() - started
            logger.info(f"[MLX-DiT] Refreshed native MLX decoder after {reason} in {elapsed:.1f}s.")
            return True, None
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[MLX-DiT] Failed to refresh MLX decoder after {reason}: {exc}")
            self.mlx_decoder = None
            return False, f"MLX DiT refresh failed after {reason}: {exc}"
