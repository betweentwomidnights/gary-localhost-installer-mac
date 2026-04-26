# Weight conversion from PyTorch AceStep DiT decoder to native MLX format.

import logging
import re
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

_ADAPTER_ONLY_KEY_MARKERS = (
    ".lora_A.",
    ".lora_B.",
    ".lora_embedding_A.",
    ".lora_embedding_B.",
    ".lora_magnitude_vector.",
)


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
