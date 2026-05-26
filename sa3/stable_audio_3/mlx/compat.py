from __future__ import annotations

from dataclasses import dataclass

from stable_audio_3.mlx.runtime import mlx_runtime_available
from stable_audio_3.mlx.spec import MLXPortRequirements


@dataclass(frozen=True)
class CompatibilityIssue:
    code: str
    message: str


@dataclass(frozen=True)
class MLXImplementationStatus:
    supports_t5gemma: bool = True
    supports_learned_text_padding: bool = True
    supports_adaln_global_cond: bool = True
    supports_input_concat_cond: bool = True
    supports_local_add_cond: bool = True
    supports_prepend_cond: bool = True
    supports_memory_tokens: bool = True
    supports_differential_attention: bool = True
    supports_variable_length_schedule: bool = True
    supports_padding_attention: bool = True
    supports_patched_pretransform: bool = True
    supports_softnorm_bottleneck: bool = True
    supports_transformer_resampling_same: bool = True
    supports_torch_conditioning_bridge: bool = True
    supports_mlx_number_conditioner: bool = True
    supports_rf_schedule_helpers: bool = True
    supports_pingpong_sampler_scaffold: bool = True
    supports_rf_latent_sampler: bool = True
    supports_same_autoencoder_smoke: bool = True
    supports_integrated_generation_api: bool = False


@dataclass(frozen=True)
class MLXCompatibilityReport:
    requirements: MLXPortRequirements
    implementation_status: MLXImplementationStatus
    runtime_available: bool
    text_to_music_blockers: tuple[CompatibilityIssue, ...]
    inpainting_blockers: tuple[CompatibilityIssue, ...]
    notes: tuple[str, ...]

    @property
    def implementation_ready_for_text_to_music(self) -> bool:
        return not self.text_to_music_blockers

    @property
    def implementation_ready_for_inpainting(self) -> bool:
        return not self.inpainting_blockers


def current_mlx_implementation_status() -> MLXImplementationStatus:
    return MLXImplementationStatus()


def analyze_mlx_compatibility(
    requirements: MLXPortRequirements,
    *,
    status: MLXImplementationStatus | None = None,
) -> MLXCompatibilityReport:
    status = status or current_mlx_implementation_status()

    text_to_music_blockers = _collect_blockers(
        requirements,
        status,
        include_inpainting_specific=False,
    )
    inpainting_blockers = _collect_blockers(
        requirements,
        status,
        include_inpainting_specific=True,
    )

    notes = []
    if not mlx_runtime_available():
        notes.append(
            "MLX runtime is not installed in the current environment, so only import/config smoke tests are possible here."
        )
    if (
        status.supports_adaln_global_cond
        and status.supports_local_add_cond
        and status.supports_memory_tokens
        and status.supports_differential_attention
    ):
        notes.append(
            "The Medium DiT core now has an MLX forward path for dummy or precomputed-conditioning smoke tests."
        )
    if status.supports_torch_conditioning_bridge:
        notes.append(
            "A torch-conditioning bridge remains available for parity checks against the original conditioning path."
        )
    if status.supports_t5gemma and status.supports_learned_text_padding:
        notes.append(
            "Native MLX T5Gemma prompt conditioning, including learned padding, is available for smoke tests."
        )
    if status.supports_mlx_number_conditioner:
        notes.append("Native MLX number conditioning is available for seconds_total smoke tests.")
    if status.supports_rf_schedule_helpers:
        notes.append("RF schedule helpers are scaffolded in pure Python.")
    if status.supports_variable_length_schedule:
        notes.append("Effective-length RF distribution-shift schedules are available for MLX smoke tests.")
    if status.supports_padding_attention:
        notes.append("Padding-aware MLX attention masks are available for RF smoke tests.")
    if status.supports_pingpong_sampler_scaffold:
        notes.append("Pingpong sampler wiring is available for MLX latent-space smoke tests.")
    if status.supports_rf_latent_sampler:
        notes.append("A narrow MLX RF latent sampler path is available for prompt-conditioned smoke tests.")
    if status.supports_same_autoencoder_smoke:
        notes.append(
            "A narrow MLX SAME-L encode/decode path is available for autoencoder parity and ear-test smokes."
        )

    return MLXCompatibilityReport(
        requirements=requirements,
        implementation_status=status,
        runtime_available=mlx_runtime_available(),
        text_to_music_blockers=tuple(text_to_music_blockers),
        inpainting_blockers=tuple(inpainting_blockers),
        notes=tuple(notes),
    )


def summarize_mlx_compatibility(report: MLXCompatibilityReport) -> str:
    lines = [
        f"Implementation-ready for text-to-music: {report.implementation_ready_for_text_to_music}",
        f"Implementation-ready for inpainting: {report.implementation_ready_for_inpainting}",
        f"MLX runtime available here: {report.runtime_available}",
    ]

    if report.notes:
        lines.append("Notes:")
        lines.extend(f"- {note}" for note in report.notes)

    if report.text_to_music_blockers:
        lines.append("Text-to-music blockers:")
        lines.extend(f"- {issue.code}: {issue.message}" for issue in report.text_to_music_blockers)

    if report.inpainting_blockers:
        lines.append("Inpainting blockers:")
        lines.extend(f"- {issue.code}: {issue.message}" for issue in report.inpainting_blockers)

    return "\n".join(lines)


def _collect_blockers(
    requirements: MLXPortRequirements,
    status: MLXImplementationStatus,
    *,
    include_inpainting_specific: bool,
) -> list[CompatibilityIssue]:
    blockers: list[CompatibilityIssue] = []

    if requirements.uses_t5gemma and not status.supports_t5gemma:
        blockers.append(
            CompatibilityIssue(
                "t5gemma",
                "Prompt conditioning still needs a T5Gemma MLX path.",
            )
        )

    if requirements.uses_learned_text_padding and not status.supports_learned_text_padding:
        blockers.append(
            CompatibilityIssue(
                "learned_text_padding",
                "Text conditioning still needs learned padding embeddings.",
            )
        )

    if requirements.needs_adaln_global_cond and not status.supports_adaln_global_cond:
        blockers.append(
            CompatibilityIssue(
                "adaln_global_cond",
                "The Medium DiT uses adaLN global conditioning.",
            )
        )

    if requirements.needs_input_concat_cond and not status.supports_input_concat_cond:
        blockers.append(
            CompatibilityIssue(
                "input_concat_cond",
                "This checkpoint requires input-concat conditioning.",
            )
        )

    if requirements.needs_local_add_cond and not status.supports_local_add_cond:
        blockers.append(
            CompatibilityIssue(
                "local_add_cond",
                "The Medium DiT expects local additive conditioning for inpaint features, even on the plain generation path.",
            )
        )

    if requirements.needs_prepend_cond and not status.supports_prepend_cond:
        blockers.append(
            CompatibilityIssue(
                "prepend_cond",
                "This checkpoint requires prepend conditioning support.",
            )
        )

    if requirements.needs_memory_tokens and not status.supports_memory_tokens:
        blockers.append(
            CompatibilityIssue(
                "memory_tokens",
                "The Medium DiT uses learned memory tokens.",
            )
        )

    if requirements.needs_differential_attention and not status.supports_differential_attention:
        blockers.append(
            CompatibilityIssue(
                "differential_attention",
                "The Medium DiT and SAME-L path use differential attention.",
            )
        )

    if requirements.needs_variable_length_schedule and not status.supports_variable_length_schedule:
        blockers.append(
            CompatibilityIssue(
                "variable_length_schedule",
                "The RF path uses effective-length schedule shifting.",
            )
        )

    if requirements.needs_padding_attention and not status.supports_padding_attention:
        blockers.append(
            CompatibilityIssue(
                "padding_attention",
                "The RF path expects padding-aware attention.",
            )
        )

    if requirements.needs_patched_pretransform and not status.supports_patched_pretransform:
        blockers.append(
            CompatibilityIssue(
                "patched_pretransform",
                "SAME-L uses the patched pretransform encode/decode path.",
            )
        )

    if requirements.needs_softnorm_bottleneck and not status.supports_softnorm_bottleneck:
        blockers.append(
            CompatibilityIssue(
                "softnorm_bottleneck",
                "SAME-L uses the softnorm bottleneck.",
            )
        )

    if (
        requirements.needs_transformer_resampling_same
        and not status.supports_transformer_resampling_same
    ):
        blockers.append(
            CompatibilityIssue(
                "transformer_resampling_same",
                "SAME-L depends on transformer-resampling encoder/decoder blocks.",
            )
        )

    if not status.supports_integrated_generation_api:
        blockers.append(
            CompatibilityIssue(
                "integrated_generation_api",
                "The MLX components are not yet wired into a single generation API.",
            )
        )

    if include_inpainting_specific and not requirements.needs_local_add_cond:
        blockers.append(
            CompatibilityIssue(
                "inpainting_path",
                "This checkpoint does not expose inpainting local-add conditioning in the extracted spec.",
            )
        )

    return blockers
