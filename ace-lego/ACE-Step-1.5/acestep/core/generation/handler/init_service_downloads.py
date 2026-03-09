"""Download and precheck helpers for service initialization."""

import os
from pathlib import Path
from typing import Optional, Tuple

from loguru import logger

from acestep.model_downloader import (
    check_model_exists,
    ensure_dit_model,
    ensure_main_model,
)


class InitServiceDownloadsMixin:
    """Helpers that validate and fetch required model checkpoints."""

    @staticmethod
    def _env_bool(name: str, default: bool = False) -> bool:
        value = os.getenv(name, "")
        if value == "":
            return default
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _missing_required_files(checkpoint_path: Path, required_relative_paths: list[str]) -> list[str]:
        missing = []
        for relative_path in required_relative_paths:
            absolute_path = checkpoint_path / relative_path
            if not absolute_path.is_file():
                missing.append(relative_path)
        return missing

    def _ensure_models_present(
        self,
        *,
        checkpoint_path: Path,
        config_path: str,
        prefer_source: Optional[str],
    ) -> Optional[Tuple[str, bool]]:
        """Ensure required checkpoint assets exist locally, downloading when missing."""
        required_shared_files = [
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
            "Qwen3-Embedding-0.6B/config.json",
            "Qwen3-Embedding-0.6B/tokenizer.json",
            "Qwen3-Embedding-0.6B/tokenizer_config.json",
            "Qwen3-Embedding-0.6B/merges.txt",
            "Qwen3-Embedding-0.6B/vocab.json",
            "Qwen3-Embedding-0.6B/special_tokens_map.json",
            "Qwen3-Embedding-0.6B/added_tokens.json",
            "Qwen3-Embedding-0.6B/chat_template.jinja",
            "Qwen3-Embedding-0.6B/model.safetensors",
        ]
        missing_shared_files = self._missing_required_files(checkpoint_path, required_shared_files)
        if missing_shared_files:
            if self._env_bool("ACESTEP_ALLOW_MAIN_AUTODOWNLOAD", False):
                logger.info(
                    "[initialize_service] Missing shared files {}; "
                    "starting main model auto-download because "
                    "ACESTEP_ALLOW_MAIN_AUTODOWNLOAD=true.",
                    missing_shared_files,
                )
                success, msg = ensure_main_model(checkpoint_path, prefer_source=prefer_source)
                if not success:
                    return f"ERROR: Failed to download main model: {msg}", False
                logger.info(f"[initialize_service] {msg}")
                missing_shared_files = self._missing_required_files(checkpoint_path, required_shared_files)
                if missing_shared_files:
                    missing_str = ", ".join(missing_shared_files)
                    return (
                        "ERROR: Shared model files are still missing after auto-download "
                        f"in {checkpoint_path}: [{missing_str}]"
                    ), False
            else:
                missing_str = ", ".join(missing_shared_files)
                return (
                    "ERROR: Missing required shared model files "
                    f"[{missing_str}] in {checkpoint_path}. "
                    "Run the focused Carey downloader first, or set "
                    "ACESTEP_ALLOW_MAIN_AUTODOWNLOAD=true to permit full "
                    "main-model auto-download."
                ), False

        if config_path == "":
            logger.warning(
                "[initialize_service] Empty config_path; pass None to use the default model."
            )

        if not check_model_exists(config_path, checkpoint_path):
            logger.info(f"[initialize_service] DiT model '{config_path}' not found, starting auto-download...")
            success, msg = ensure_dit_model(config_path, checkpoint_path, prefer_source=prefer_source)
            if not success:
                return f"ERROR: Failed to download DiT model '{config_path}': {msg}", False
            logger.info(f"[initialize_service] {msg}")

        required_dit_files = [
            f"{config_path}/config.json",
            f"{config_path}/model.safetensors",
            f"{config_path}/silence_latent.pt",
        ]
        missing_dit_files = self._missing_required_files(checkpoint_path, required_dit_files)
        if missing_dit_files:
            missing_str = ", ".join(missing_dit_files)
            return (
                "ERROR: Missing required DiT files "
                f"[{missing_str}] in {checkpoint_path}. "
                "Run the focused Carey downloader first."
            ), False

        return None

    @staticmethod
    def _sync_model_code_if_needed(config_path: str, checkpoint_path: Path) -> None:
        """Sync model-side python files when checkpoint code metadata diverges."""
        from acestep.model_downloader import _check_code_mismatch, _sync_model_code_files

        mismatched = _check_code_mismatch(config_path, checkpoint_path)
        if mismatched:
            logger.warning(
                f"[initialize_service] Model code mismatch detected for '{config_path}': "
                f"{mismatched}. Auto-syncing from acestep/models/..."
            )
            _sync_model_code_files(config_path, checkpoint_path)
            logger.info("[initialize_service] Model code files synced successfully.")
