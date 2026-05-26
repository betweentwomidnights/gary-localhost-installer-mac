from __future__ import annotations

import json
import typing as tp
from dataclasses import dataclass
from pathlib import Path


JSONDict = dict[str, tp.Any]


@dataclass(frozen=True)
class ConditionerSpec:
    id: str
    type: str
    config: JSONDict


@dataclass(frozen=True)
class DiffusionSpec:
    objective: str
    cross_attention_cond_ids: tuple[str, ...]
    global_cond_ids: tuple[str, ...]
    input_concat_ids: tuple[str, ...]
    local_add_cond_ids: tuple[str, ...]
    prepend_cond_ids: tuple[str, ...]
    mask_padding_attention: bool
    use_effective_length_for_schedule: bool
    global_cond_type: str
    local_add_cond_dim: int
    num_memory_tokens: int
    qk_norm: str
    uses_differential_attention: bool


@dataclass(frozen=True)
class PretransformSpec:
    pretransform_type: str
    patch_size: int
    channels: int
    latent_dim: int
    downsampling_ratio: int
    io_channels: int
    bottleneck_type: str
    encoder_transformer_depths: tuple[int, ...]
    decoder_transformer_depths: tuple[int, ...]
    encoder_sliding_window: tuple[int, ...]
    decoder_sliding_window: tuple[int, ...]
    encoder_variable_stride: bool
    decoder_variable_stride: bool
    encoder_differential: bool
    decoder_differential: bool


@dataclass(frozen=True)
class MLXPortRequirements:
    source_name: str
    model_type: str
    sample_rate: int
    audio_channels: int
    diffusion: DiffusionSpec
    conditioners: tuple[ConditionerSpec, ...]
    pretransform: PretransformSpec
    uses_t5gemma: bool
    uses_learned_text_padding: bool
    needs_adaln_global_cond: bool
    needs_input_concat_cond: bool
    needs_local_add_cond: bool
    needs_prepend_cond: bool
    needs_memory_tokens: bool
    needs_differential_attention: bool
    needs_variable_length_schedule: bool
    needs_padding_attention: bool
    needs_patched_pretransform: bool
    needs_softnorm_bottleneck: bool
    needs_transformer_resampling_same: bool
    notes: tuple[str, ...]
    legacy_saomlx_blockers: tuple[str, ...]


def load_model_config(path: str | Path) -> JSONDict:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_mlx_port_requirements(
    model_config: JSONDict,
    *,
    source_name: str = "<in-memory>",
) -> MLXPortRequirements:
    model = model_config["model"]
    diffusion_cfg = model["diffusion"]
    dit_cfg = diffusion_cfg["config"]
    conditioning_cfg = model.get("conditioning", {})
    pre_cfg = model["pretransform"]["config"]
    pre_patch_cfg = pre_cfg.get("pretransform", {}).get("config", {})
    enc_cfg = pre_cfg["encoder"]["config"]
    dec_cfg = pre_cfg["decoder"]["config"]

    conditioners = tuple(
        ConditionerSpec(
            id=str(item["id"]),
            type=str(item["type"]),
            config=dict(item.get("config", {})),
        )
        for item in conditioning_cfg.get("configs", [])
    )

    diffusion = DiffusionSpec(
        objective=str(diffusion_cfg.get("diffusion_objective", "v")),
        cross_attention_cond_ids=_as_tuple(diffusion_cfg.get("cross_attention_cond_ids")),
        global_cond_ids=_as_tuple(diffusion_cfg.get("global_cond_ids")),
        input_concat_ids=_as_tuple(diffusion_cfg.get("input_concat_ids")),
        local_add_cond_ids=_as_tuple(diffusion_cfg.get("local_add_cond_ids")),
        prepend_cond_ids=_as_tuple(diffusion_cfg.get("prepend_cond_ids")),
        mask_padding_attention=bool(diffusion_cfg.get("mask_padding_attention", False)),
        use_effective_length_for_schedule=bool(
            diffusion_cfg.get("use_effective_length_for_schedule", False)
        ),
        global_cond_type=str(dit_cfg.get("global_cond_type", "prepend")),
        local_add_cond_dim=int(dit_cfg.get("local_add_cond_dim", 0) or 0),
        num_memory_tokens=int(dit_cfg.get("num_memory_tokens", 0) or 0),
        qk_norm=str(dit_cfg.get("attn_kwargs", {}).get("qk_norm", "none")),
        uses_differential_attention=bool(
            dit_cfg.get("attn_kwargs", {}).get("differential", False)
        ),
    )

    pretransform = PretransformSpec(
        pretransform_type=str(pre_cfg.get("pretransform", {}).get("type", "")),
        patch_size=int(pre_patch_cfg.get("patch_size", 1) or 1),
        channels=int(pre_patch_cfg.get("channels", pre_cfg.get("io_channels", 0)) or 0),
        latent_dim=int(pre_cfg.get("latent_dim", 0) or 0),
        downsampling_ratio=int(pre_cfg.get("downsampling_ratio", 0) or 0),
        io_channels=int(pre_cfg.get("io_channels", 0) or 0),
        bottleneck_type=str(pre_cfg.get("bottleneck", {}).get("type", "")),
        encoder_transformer_depths=_int_tuple(enc_cfg.get("transformer_depths")),
        decoder_transformer_depths=_int_tuple(dec_cfg.get("transformer_depths")),
        encoder_sliding_window=_int_tuple(enc_cfg.get("sliding_window")),
        decoder_sliding_window=_int_tuple(dec_cfg.get("sliding_window")),
        encoder_variable_stride=bool(enc_cfg.get("variable_stride", False)),
        decoder_variable_stride=bool(dec_cfg.get("variable_stride", False)),
        encoder_differential=bool(enc_cfg.get("differential", False)),
        decoder_differential=bool(dec_cfg.get("differential", False)),
    )

    uses_t5gemma = any(cond.type == "t5gemma" for cond in conditioners)
    uses_learned_text_padding = any(
        cond.config.get("padding_mode") == "learned" for cond in conditioners
    )
    needs_transformer_resampling_same = bool(
        pretransform.encoder_transformer_depths or pretransform.decoder_transformer_depths
    )

    notes = _build_notes(diffusion, conditioners, pretransform)
    blockers = _legacy_saomlx_blockers(diffusion, conditioners, pretransform)

    return MLXPortRequirements(
        source_name=source_name,
        model_type=str(model_config.get("model_type", "")),
        sample_rate=int(model_config.get("sample_rate", 0) or 0),
        audio_channels=int(model_config.get("audio_channels", 0) or 0),
        diffusion=diffusion,
        conditioners=conditioners,
        pretransform=pretransform,
        uses_t5gemma=uses_t5gemma,
        uses_learned_text_padding=uses_learned_text_padding,
        needs_adaln_global_cond=diffusion.global_cond_type == "adaLN",
        needs_input_concat_cond=bool(diffusion.input_concat_ids),
        needs_local_add_cond=bool(diffusion.local_add_cond_ids),
        needs_prepend_cond=bool(diffusion.prepend_cond_ids),
        needs_memory_tokens=diffusion.num_memory_tokens > 0,
        needs_differential_attention=diffusion.uses_differential_attention
        or pretransform.encoder_differential
        or pretransform.decoder_differential,
        needs_variable_length_schedule=diffusion.use_effective_length_for_schedule,
        needs_padding_attention=diffusion.mask_padding_attention,
        needs_patched_pretransform=pretransform.pretransform_type == "patched",
        needs_softnorm_bottleneck=pretransform.bottleneck_type == "softnorm",
        needs_transformer_resampling_same=needs_transformer_resampling_same,
        notes=notes,
        legacy_saomlx_blockers=blockers,
    )


def summarize_mlx_port_requirements(requirements: MLXPortRequirements) -> str:
    conditioner_types = ", ".join(
        f"{cond.id}:{cond.type}" for cond in requirements.conditioners
    )
    lines = [
        f"Source: {requirements.source_name}",
        f"Model type: {requirements.model_type}",
        f"Sample rate: {requirements.sample_rate}",
        f"Audio channels: {requirements.audio_channels}",
        f"Diffusion objective: {requirements.diffusion.objective}",
        f"Conditioners: {conditioner_types}",
        (
            "Core MLX requirements: "
            f"t5gemma={requirements.uses_t5gemma}, "
            f"adaln={requirements.needs_adaln_global_cond}, "
            f"local_add={requirements.needs_local_add_cond}, "
            f"memory_tokens={requirements.needs_memory_tokens}, "
            f"differential_attn={requirements.needs_differential_attention}"
        ),
        (
            "SAME requirements: "
            f"patched_pretransform={requirements.needs_patched_pretransform}, "
            f"softnorm={requirements.needs_softnorm_bottleneck}, "
            f"transformer_resampling={requirements.needs_transformer_resampling_same}"
        ),
    ]

    if requirements.notes:
        lines.append("Notes:")
        lines.extend(f"- {note}" for note in requirements.notes)

    if requirements.legacy_saomlx_blockers:
        lines.append("Known blockers vs legacy saomlx reference:")
        lines.extend(f"- {item}" for item in requirements.legacy_saomlx_blockers)

    return "\n".join(lines)


def _build_notes(
    diffusion: DiffusionSpec,
    conditioners: tuple[ConditionerSpec, ...],
    pretransform: PretransformSpec,
) -> tuple[str, ...]:
    notes: list[str] = []

    if diffusion.objective == "rf_denoiser":
        notes.append("ARC Medium uses rf_denoiser, so pingpong is the natural default sampler.")
    elif diffusion.objective == "rectified_flow":
        notes.append("RF Medium uses rectified_flow and keeps the same DiT/SAME-L structure as ARC.")

    if diffusion.local_add_cond_ids:
        ids = ", ".join(diffusion.local_add_cond_ids)
        notes.append(f"Inpaint conditioning is injected through local_add_cond_ids: {ids}.")

    if diffusion.local_add_cond_dim:
        notes.append(
            f"local_add_cond_dim={diffusion.local_add_cond_dim}; for Medium this should line up with mask + latent inpaint channels."
        )

    if any(cond.type == "t5gemma" for cond in conditioners):
        prompt_cond = next(cond for cond in conditioners if cond.type == "t5gemma")
        notes.append(
            f"Prompt conditioning uses T5Gemma with max_length={prompt_cond.config.get('max_length')} and padding_mode={prompt_cond.config.get('padding_mode')}."
        )

    if diffusion.mask_padding_attention or diffusion.use_effective_length_for_schedule:
        notes.append("The RF config opts into padding-aware attention and effective-length schedule shifts.")

    if pretransform.pretransform_type == "patched":
        notes.append(
            f"SAME-L uses a patched pretransform with patch_size={pretransform.patch_size} and downsampling_ratio={pretransform.downsampling_ratio}."
        )

    if pretransform.encoder_sliding_window or pretransform.decoder_sliding_window:
        notes.append(
            "The SAME-L encoder/decoder use sliding-window transformer resampling blocks."
        )

    return tuple(notes)


def _legacy_saomlx_blockers(
    diffusion: DiffusionSpec,
    conditioners: tuple[ConditionerSpec, ...],
    pretransform: PretransformSpec,
) -> tuple[str, ...]:
    blockers: list[str] = []

    if any(cond.type == "t5gemma" for cond in conditioners):
        blockers.append("Replace the legacy T5 text path with a T5Gemma encoder path.")

    if any(cond.config.get("padding_mode") == "learned" for cond in conditioners):
        blockers.append("Implement learned text-padding embeddings instead of simple mask-zeroing.")

    if diffusion.global_cond_type == "adaLN":
        blockers.append("Add adaLN/global_cond support to the MLX TransformerBlock path.")

    if diffusion.local_add_cond_ids:
        blockers.append("Add local_add_cond support to the MLX DiT for inpaint mask features.")

    if diffusion.input_concat_ids:
        blockers.append("Add input_concat_cond support to the MLX DiT path.")

    if diffusion.num_memory_tokens > 0:
        blockers.append(
            f"Add memory token support to the MLX transformer (required count: {diffusion.num_memory_tokens})."
        )

    if diffusion.uses_differential_attention:
        blockers.append("Port the differential attention path used by the Medium DiT.")

    if diffusion.mask_padding_attention or diffusion.use_effective_length_for_schedule:
        blockers.append("Port padding-aware attention and effective-length schedule handling.")

    if pretransform.pretransform_type == "patched":
        blockers.append("Port the SAME patched pretransform encode/decode path.")

    if pretransform.bottleneck_type == "softnorm":
        blockers.append("Port the SAME softnorm bottleneck used by Stable Audio 3.")

    if pretransform.encoder_transformer_depths or pretransform.decoder_transformer_depths:
        blockers.append(
            "Port the SAME-L transformer-resampling encoder/decoder, including sliding-window and variable-stride behavior."
        )

    return tuple(dict.fromkeys(blockers))


def _as_tuple(values: tp.Any) -> tuple[str, ...]:
    if not values:
        return ()
    return tuple(str(value) for value in values)


def _int_tuple(values: tp.Any) -> tuple[int, ...]:
    if not values:
        return ()
    return tuple(int(value) for value in values)
