from stable_audio_3.mlx.api import (
    inspect_mlx_pipeline,
    inspect_pretrained_mlx_pipeline,
    summarize_pretrained_mlx_pipeline,
)
from stable_audio_3.mlx.compat import (
    CompatibilityIssue,
    MLXCompatibilityReport,
    MLXImplementationStatus,
    analyze_mlx_compatibility,
    current_mlx_implementation_status,
    summarize_mlx_compatibility,
)
from stable_audio_3.mlx.pipeline import (
    MLXGenerationResult,
    StableAudioMLXPipeline,
    resolve_pretrained_config_path,
)
from stable_audio_3.mlx.runtime import (
    MLXRuntimeUnavailableError,
    mlx_runtime_available,
    require_mlx_runtime,
)
from stable_audio_3.mlx.sampling import (
    RFSchedulePreview,
    create_padding_mask_for_effective_lengths,
    create_padding_mask_from_lengths,
    default_sampler_type_for_objective,
    padding_lengths_from_effective_lengths,
    make_linear_schedule_values,
    make_rf_schedule_values,
    preview_rf_schedule,
    sample_rf_latents_mlx,
)
from stable_audio_3.mlx.spec import (
    MLXPortRequirements,
    extract_mlx_port_requirements,
    load_model_config,
    summarize_mlx_port_requirements,
)

_RUNTIME_EXPORTS = {
    "AutoencoderConversionReport": (
        "stable_audio_3.mlx.autoencoder",
        "AutoencoderConversionReport",
    ),
    "MLXAudioAutoencoder": (
        "stable_audio_3.mlx.autoencoder",
        "MLXAudioAutoencoder",
    ),
    "MLXNumberConditioner": (
        "stable_audio_3.mlx.conditioning",
        "MLXNumberConditioner",
    ),
    "T5GemmaTextConditioner": (
        "stable_audio_3.mlx.t5gemma",
        "T5GemmaTextConditioner",
    ),
    "assemble_conditioning_inputs_from_tensors": (
        "stable_audio_3.mlx.conditioning",
        "assemble_conditioning_inputs_from_tensors",
    ),
    "build_mlx_conditioning_inputs": (
        "stable_audio_3.mlx.conditioning",
        "build_mlx_conditioning_inputs",
    ),
    "build_mlx_conditioning_inputs_from_torch_model": (
        "stable_audio_3.mlx.conditioning",
        "build_mlx_conditioning_inputs_from_torch_model",
    ),
    "ConversionReport": ("stable_audio_3.mlx.dit", "ConversionReport"),
    "MLXLoRAApplyReport": ("stable_audio_3.mlx.lora", "MLXLoRAApplyReport"),
    "MLXLoRASet": ("stable_audio_3.mlx.lora", "MLXLoRASet"),
    "MLXDiTSmokeInputs": ("stable_audio_3.mlx.smoke", "MLXDiTSmokeInputs"),
    "StableAudioMLXDiT": ("stable_audio_3.mlx.dit", "StableAudioMLXDiT"),
    "apply_mlx_loras": ("stable_audio_3.mlx.lora", "apply_mlx_loras"),
    "build_dummy_dit_smoke_inputs": (
        "stable_audio_3.mlx.smoke",
        "build_dummy_dit_smoke_inputs",
    ),
    "run_mlx_dit_forward_smoke": (
        "stable_audio_3.mlx.smoke",
        "run_mlx_dit_forward_smoke",
    ),
}

__all__ = [
    "CompatibilityIssue",
    "AutoencoderConversionReport",
    "MLXCompatibilityReport",
    "MLXAudioAutoencoder",
    "MLXDiTSmokeInputs",
    "MLXGenerationResult",
    "MLXImplementationStatus",
    "MLXNumberConditioner",
    "MLXPortRequirements",
    "MLXRuntimeUnavailableError",
    "MLXLoRAApplyReport",
    "MLXLoRASet",
    "RFSchedulePreview",
    "StableAudioMLXPipeline",
    "StableAudioMLXDiT",
    "T5GemmaTextConditioner",
    "assemble_conditioning_inputs_from_tensors",
    "build_mlx_conditioning_inputs",
    "build_mlx_conditioning_inputs_from_torch_model",
    "analyze_mlx_compatibility",
    "apply_mlx_loras",
    "build_dummy_dit_smoke_inputs",
    "create_padding_mask_for_effective_lengths",
    "create_padding_mask_from_lengths",
    "current_mlx_implementation_status",
    "default_sampler_type_for_objective",
    "extract_mlx_port_requirements",
    "inspect_mlx_pipeline",
    "inspect_pretrained_mlx_pipeline",
    "load_model_config",
    "padding_lengths_from_effective_lengths",
    "make_rf_schedule_values",
    "make_linear_schedule_values",
    "mlx_runtime_available",
    "preview_rf_schedule",
    "require_mlx_runtime",
    "resolve_pretrained_config_path",
    "run_mlx_dit_forward_smoke",
    "sample_rf_latents_mlx",
    "summarize_mlx_compatibility",
    "summarize_mlx_port_requirements",
    "summarize_pretrained_mlx_pipeline",
    "ConversionReport",
]


def __getattr__(name: str):
    if name not in _RUNTIME_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _RUNTIME_EXPORTS[name]
    module = __import__(module_name, fromlist=[attr_name])
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
