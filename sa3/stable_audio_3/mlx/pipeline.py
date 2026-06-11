from __future__ import annotations

import json
import math
import secrets
import typing as tp
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from stable_audio_3.model_configs import all_models
from stable_audio_3.mlx.compat import (
    MLXCompatibilityReport,
    MLXImplementationStatus,
    analyze_mlx_compatibility,
    summarize_mlx_compatibility,
)
from stable_audio_3.mlx.runtime import import_mlx_core, require_mlx_runtime
from stable_audio_3.mlx.sampling import preview_rf_schedule
from stable_audio_3.mlx.spec import (
    MLXPortRequirements,
    extract_mlx_port_requirements,
    load_model_config,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class MLXGenerationResult:
    audio: tp.Any | None
    latents: tp.Any
    report: dict[str, object]


@dataclass
class StableAudioMLXPipeline:
    source_name: str
    model_config: dict[str, object]
    requirements: MLXPortRequirements
    compatibility: MLXCompatibilityReport
    resolved_config_path: Path | None = None
    torch_pipeline: tp.Any | None = None
    mlx_dit: tp.Any | None = None
    text_conditioner: tp.Any | None = None
    number_conditioner: tp.Any | None = None
    autoencoder: tp.Any | None = None
    dtype_name: str = "float32"
    autoencoder_dtype_name: str = "float32"
    attention: str = "sliding"
    conversion_reports: dict[str, tp.Any] = field(default_factory=dict)
    lora_paths: tuple[Path, ...] = ()
    lora_labels: tuple[str, ...] = ()
    lora_strengths: tuple[float, ...] = ()
    _dit_lora_set: tp.Any | None = field(default=None, init=False, repr=False)
    _text_lora_set: tp.Any | None = field(default=None, init=False, repr=False)
    _number_lora_set: tp.Any | None = field(default=None, init=False, repr=False)
    _active_text_number_lora_signature: tuple[tp.Any, ...] | None = field(default=None, init=False, repr=False)
    _active_dit_lora_signature: tuple[tp.Any, ...] | None = field(default=None, init=False, repr=False)
    _cached_text_lora_report: tp.Any | None = field(default=None, init=False, repr=False)
    _cached_number_lora_report: tp.Any | None = field(default=None, init=False, repr=False)
    _cached_dit_lora_report: tp.Any | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_config(
        cls,
        config_or_path,
        *,
        source_name: str | None = None,
    ) -> "StableAudioMLXPipeline":
        if isinstance(config_or_path, (str, Path)):
            resolved_path = Path(config_or_path).expanduser().resolve()
            model_config = load_model_config(resolved_path)
            source_name = source_name or resolved_path.name
        else:
            resolved_path = None
            model_config = dict(config_or_path)
            source_name = source_name or "<in-memory>"

        requirements = extract_mlx_port_requirements(
            model_config,
            source_name=source_name,
        )
        compatibility = analyze_mlx_compatibility(requirements)

        return cls(
            source_name=source_name,
            model_config=model_config,
            requirements=requirements,
            compatibility=compatibility,
            resolved_config_path=resolved_path,
        )

    @classmethod
    def from_pretrained_config(
        cls,
        model_name_or_path: str,
        *,
        search_roots=None,
    ) -> "StableAudioMLXPipeline":
        resolved = resolve_pretrained_config_path(
            model_name_or_path,
            search_roots=search_roots,
        )
        return cls.from_config(resolved)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        search_roots=None,
    ) -> "StableAudioMLXPipeline":
        return cls.from_pretrained_config(
            model_name_or_path,
            search_roots=search_roots,
        )

    @classmethod
    def from_torch_pretrained(
        cls,
        model_name_or_path: str,
        *,
        torch_device: str | None = "auto",
        dtype: str = "float32",
        dit_dtype: str | None = None,
        text_dtype: str | None = None,
        number_dtype: str | None = None,
        autoencoder_dtype: str | None = None,
        attention: str = "sliding",
        model_half: bool = False,
        search_roots=None,
    ) -> "StableAudioMLXPipeline":
        """Load the upstream torch checkpoint and convert the inference modules to MLX.

        This is the practical generation path until direct MLX checkpoint loading
        exists. The torch pipeline is retained for config helpers and optional
        future A/B checks, but generation itself runs through converted MLX
        components.
        """
        require_mlx_runtime()
        if attention not in {"sliding", "full"}:
            raise ValueError(f"attention must be 'sliding' or 'full', got {attention!r}.")

        try:
            from stable_audio_3.pipeline import StableAudioPipeline
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "The torch Stable Audio pipeline dependencies are required for "
                "runtime MLX conversion."
            ) from exc

        torch_device_arg = None if torch_device == "auto" else torch_device
        torch_pipeline = StableAudioPipeline.from_pretrained(
            model_name_or_path,
            device=torch_device_arg,
            model_half=model_half,
        )

        resolved_config_path = None
        try:
            resolved_config_path = resolve_pretrained_config_path(
                model_name_or_path,
                search_roots=search_roots,
            )
        except (FileNotFoundError, TypeError):
            resolved_config_path = None

        return cls._from_loaded_torch_pipeline(
            torch_pipeline,
            source_name=str(model_name_or_path),
            resolved_config_path=resolved_config_path,
            dtype=dtype,
            dit_dtype=dit_dtype,
            text_dtype=text_dtype,
            number_dtype=number_dtype,
            autoencoder_dtype=autoencoder_dtype,
            attention=attention,
        )

    @classmethod
    def from_torch_checkpoint(
        cls,
        config_path: str | Path,
        checkpoint_path: str | Path,
        *,
        torch_device: str | None = "cpu",
        dtype: str = "float32",
        dit_dtype: str | None = None,
        text_dtype: str | None = None,
        number_dtype: str | None = None,
        autoencoder_dtype: str | None = None,
        attention: str = "sliding",
        model_half: bool = False,
    ) -> "StableAudioMLXPipeline":
        """Load a local torch checkpoint and convert its runtime modules to MLX."""
        require_mlx_runtime()
        if attention not in {"sliding", "full"}:
            raise ValueError(f"attention must be 'sliding' or 'full', got {attention!r}.")

        resolved_config_path = Path(config_path).expanduser().absolute()
        # Keep the Hugging Face snapshot symlink name intact. Resolving it to the
        # extensionless blob path prevents the checkpoint loader from recognizing
        # safetensors files.
        resolved_checkpoint_path = Path(checkpoint_path).expanduser().absolute()
        if not resolved_config_path.is_file():
            raise FileNotFoundError(f"Model config not found: {resolved_config_path}")
        if not resolved_checkpoint_path.is_file():
            raise FileNotFoundError(f"Model checkpoint not found: {resolved_checkpoint_path}")

        try:
            from stable_audio_3.loading_utils import load_diffusion_cond
            from stable_audio_3.pipeline import StableAudioPipeline
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "The torch Stable Audio pipeline dependencies are required for "
                "runtime MLX conversion."
            ) from exc

        with resolved_config_path.open() as handle:
            model_config = json.load(handle)

        torch_device_arg = (
            "cpu" if torch_device is None or torch_device == "auto" else torch_device
        )
        torch_model = load_diffusion_cond(
            model_config,
            str(resolved_checkpoint_path),
            device=torch_device_arg,
            model_half=model_half,
        )
        torch_model.use_lora = False
        torch_model.lora_names = []
        torch_pipeline = StableAudioPipeline(
            torch_model,
            model_config,
            torch_device_arg,
            model_half,
        )

        return cls._from_loaded_torch_pipeline(
            torch_pipeline,
            source_name=resolved_checkpoint_path.stem,
            resolved_config_path=resolved_config_path,
            dtype=dtype,
            dit_dtype=dit_dtype,
            text_dtype=text_dtype,
            number_dtype=number_dtype,
            autoencoder_dtype=autoencoder_dtype,
            attention=attention,
        )

    @classmethod
    def _from_loaded_torch_pipeline(
        cls,
        torch_pipeline,
        *,
        source_name: str,
        resolved_config_path: Path | None,
        dtype: str,
        dit_dtype: str | None,
        text_dtype: str | None,
        number_dtype: str | None,
        autoencoder_dtype: str | None,
        attention: str,
    ) -> "StableAudioMLXPipeline":
        from stable_audio_3.mlx.autoencoder import MLXAudioAutoencoder
        from stable_audio_3.mlx.conditioning import MLXNumberConditioner
        from stable_audio_3.mlx.dit import StableAudioMLXDiT
        from stable_audio_3.mlx.t5gemma import T5GemmaTextConditioner

        mx = import_mlx_core(required=True)
        dit_dtype = dit_dtype or dtype
        text_dtype = text_dtype or dit_dtype
        number_dtype = number_dtype or dit_dtype
        autoencoder_dtype = autoencoder_dtype or dtype
        mlx_dtype = getattr(mx, dit_dtype)
        text_mlx_dtype = getattr(mx, text_dtype)
        number_mlx_dtype = getattr(mx, number_dtype)
        autoencoder_mlx_dtype = getattr(mx, autoencoder_dtype)
        model_config = torch_pipeline.model_config
        requirements = extract_mlx_port_requirements(model_config, source_name=source_name)
        compatibility = analyze_mlx_compatibility(
            requirements,
            status=MLXImplementationStatus(supports_integrated_generation_api=True),
        )

        mlx_dit, dit_conversion = StableAudioMLXDiT.from_torch_dit(
            torch_pipeline.dit,
            model_config,
            mlx_dtype=mlx_dtype,
        )
        text_conditioner, text_conversion = T5GemmaTextConditioner.from_torch_conditioner(
            torch_pipeline.model.conditioner.conditioners["prompt"],
            mlx_dtype=text_mlx_dtype,
        )
        number_conditioner, number_conversion = MLXNumberConditioner.from_torch_conditioner(
            torch_pipeline.model.conditioner.conditioners["seconds_total"],
            mlx_dtype=number_mlx_dtype,
        )
        autoencoder, ae_conversion = MLXAudioAutoencoder.from_torch_autoencoder(
            torch_pipeline.model.pretransform,
            model_config,
            mlx_dtype=autoencoder_mlx_dtype,
            use_sliding_window=attention == "sliding",
        )
        # Match torch eval behavior for prompt-generation decode: no stochastic
        # softnorm noise unless a caller deliberately changes the module later.
        autoencoder.bottleneck.noise_regularize = False

        return cls(
            source_name=source_name,
            model_config=model_config,
            requirements=requirements,
            compatibility=compatibility,
            resolved_config_path=resolved_config_path,
            torch_pipeline=torch_pipeline,
            mlx_dit=mlx_dit,
            text_conditioner=text_conditioner,
            number_conditioner=number_conditioner,
            autoencoder=autoencoder,
            dtype_name=dit_dtype,
            autoencoder_dtype_name=autoencoder_dtype,
            attention=attention,
            conversion_reports={
                "dit": dit_conversion,
                "text": text_conversion,
                "number": number_conversion,
                "autoencoder": ae_conversion,
            },
        )

    def smoke_test_report(self, *, steps: int = 8, sigma_max: float = 1.0) -> dict[str, object]:
        report = {
            "source_name": self.source_name,
            "resolved_config_path": (
                str(self.resolved_config_path) if self.resolved_config_path else None
            ),
            "model_type": self.requirements.model_type,
            "diffusion_objective": self.requirements.diffusion.objective,
            "compatibility": {
                "implementation_ready_for_text_to_music": self.compatibility.implementation_ready_for_text_to_music,
                "implementation_ready_for_inpainting": self.compatibility.implementation_ready_for_inpainting,
                "runtime_available": self.compatibility.runtime_available,
                "text_to_music_blockers": [
                    issue.code for issue in self.compatibility.text_to_music_blockers
                ],
                "inpainting_blockers": [
                    issue.code for issue in self.compatibility.inpainting_blockers
                ],
            },
        }

        if self.requirements.diffusion.objective in {"rf_denoiser", "rectified_flow"}:
            schedule = preview_rf_schedule(
                self.requirements.diffusion.objective,
                steps=steps,
                sigma_max=sigma_max,
            )
            report["rf_schedule_preview"] = {
                "sampler_type": schedule.sampler_type,
                "steps": schedule.steps,
                "sigma_max": schedule.sigma_max,
                "values": schedule.values,
            }

        return report

    def validate_for_text_to_music(self, *, require_runtime: bool = False) -> None:
        if self.compatibility.text_to_music_blockers:
            raise NotImplementedError(summarize_mlx_compatibility(self.compatibility))
        if require_runtime:
            require_mlx_runtime()

    def validate_for_inpainting(self, *, require_runtime: bool = False) -> None:
        if self.compatibility.inpainting_blockers:
            raise NotImplementedError(summarize_mlx_compatibility(self.compatibility))
        if require_runtime:
            require_mlx_runtime()

    @property
    def generation_ready(self) -> bool:
        return all(
            component is not None
            for component in (
                self.torch_pipeline,
                self.mlx_dit,
                self.text_conditioner,
                self.number_conditioner,
                self.autoencoder,
            )
        )

    def load_lora(
        self,
        lora_ckpt_paths: tp.Sequence[str | Path],
        *,
        names: tp.Sequence[str] | None = None,
        strength: float | tp.Sequence[float] = 1.0,
    ) -> "StableAudioMLXPipeline":
        self._require_generation_ready()
        from stable_audio_3.mlx.lora import MLXLoRASet

        paths = tuple(Path(path).expanduser().resolve() for path in lora_ckpt_paths)
        labels = _normalize_lora_labels(names, paths)
        strengths = _normalize_lora_strengths(strength, len(paths))
        self.lora_paths = paths
        self.lora_labels = labels
        self.lora_strengths = strengths
        self._dit_lora_set = MLXLoRASet.from_checkpoints(
            paths,
            self.mlx_dit,
            target_label="dit",
            names=labels,
        )
        self._text_lora_set = MLXLoRASet.from_checkpoints(
            paths,
            self.text_conditioner.encoder,
            target_label="text_conditioner",
            names=labels,
        )
        self._number_lora_set = MLXLoRASet.from_checkpoints(
            paths,
            self.number_conditioner,
            target_label="number_conditioner",
            names=labels,
        )
        return self

    def clear_lora(self) -> None:
        if self._dit_lora_set is not None:
            self._dit_lora_set.apply_to(self.mlx_dit, strength=0.0)
        if self._text_lora_set is not None:
            self._text_lora_set.apply_to(self.text_conditioner.encoder, strength=0.0)
        if self._number_lora_set is not None:
            self._number_lora_set.apply_to(self.number_conditioner, strength=0.0)
        self.lora_paths = ()
        self.lora_labels = ()
        self.lora_strengths = ()
        self._dit_lora_set = None
        self._text_lora_set = None
        self._number_lora_set = None
        self._active_text_number_lora_signature = None
        self._active_dit_lora_signature = None
        self._cached_text_lora_report = None
        self._cached_number_lora_report = None
        self._cached_dit_lora_report = None

    def set_lora_strength(self, strength: float, lora_index: int | None = None) -> None:
        if not self.lora_paths:
            return
        strengths = list(self.lora_strengths or (1.0 for _ in self.lora_paths))
        if lora_index is None:
            strengths = [float(strength) for _ in strengths]
        else:
            strengths[int(lora_index)] = float(strength)
        self.lora_strengths = tuple(strengths)

    def _active_lora_reports(self) -> list[dict[str, object]]:
        reports = []
        for report in (
            self._cached_text_lora_report,
            self._cached_number_lora_report,
        ):
            if report is not None and _include_lora_report(report):
                reports.append(report.to_dict())
        return reports

    def _apply_text_number_loras(
        self,
        strengths: tuple[float, ...],
        configs: tuple[dict[str, tp.Any], ...],
    ) -> list[dict[str, object]]:
        if self._text_lora_set is None or self._number_lora_set is None:
            return []

        has_active = _lora_has_active_strengths(strengths)
        signature = _lora_runtime_signature(strengths, configs) if has_active else None
        if signature == self._active_text_number_lora_signature:
            return self._active_lora_reports()

        if not has_active and self._active_text_number_lora_signature is None:
            return []

        self._cached_text_lora_report = self._text_lora_set.apply_to(
            self.text_conditioner.encoder,
            strength=strengths,
        )
        self._cached_number_lora_report = self._number_lora_set.apply_to(
            self.number_conditioner,
            strength=strengths,
        )
        self._active_text_number_lora_signature = signature
        return self._active_lora_reports()

    def _clear_applied_dit_lora(self) -> None:
        if self._dit_lora_set is None:
            return
        if self._active_dit_lora_signature is not None:
            self._dit_lora_set.apply_to(self.mlx_dit, strength=0.0)
        self._active_dit_lora_signature = None
        self._cached_dit_lora_report = None

    def _prepare_dit_lora(
        self,
        strengths: tuple[float, ...],
        configs: tuple[dict[str, tp.Any], ...],
    ) -> tuple[tp.Any, tp.Any | None, dict[str, object] | None]:
        from stable_audio_3.mlx.lora import MLXLoRAScheduledModule

        if self._dit_lora_set is None:
            return self.mlx_dit, None, None

        has_active = _lora_has_active_strengths(strengths)
        if not has_active:
            self._clear_applied_dit_lora()
            return self.mlx_dit, None, None

        if _lora_configs_are_static(configs):
            signature = _lora_runtime_signature(strengths, configs)
            if signature != self._active_dit_lora_signature:
                self._cached_dit_lora_report = self._dit_lora_set.apply_to(
                    self.mlx_dit,
                    strength=strengths,
                    lora_configs=configs,
                )
                self._active_dit_lora_signature = signature
            return (
                self.mlx_dit,
                None,
                _dit_lora_report_payload(
                    self,
                    strengths,
                    configs,
                    report=self._cached_dit_lora_report,
                    schedule_updates=(
                        [self._cached_dit_lora_report.to_dict()]
                        if self._cached_dit_lora_report is not None
                        else []
                    ),
                ),
            )

        self._clear_applied_dit_lora()
        runtime = MLXLoRAScheduledModule(
            self.mlx_dit,
            self._dit_lora_set,
            strength=strengths,
            lora_configs=configs,
        )
        return runtime, runtime, None

    def generate(
        self,
        *,
        prompt: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = None,
        duration: float | list[float] = 10.0,
        steps: int = 8,
        cfg_scale: float = 1.0,
        batch_size: int = 1,
        sample_size: int = 5_292_032,
        truncate_output_to_duration: bool = True,
        conditioning: list[dict[str, tp.Any]] | None = None,
        negative_conditioning: list[dict[str, tp.Any]] | None = None,
        seed: int | None = 0,
        sampler_type: str | None = None,
        init_audio: tuple[int, tp.Any] | None = None,
        init_noise_level: float = 1.0,
        inpaint_audio: tuple[int, tp.Any] | None = None,
        inpaint_mask=None,
        inpaint_mask_start_seconds: float | None = None,
        inpaint_mask_end_seconds: float | None = None,
        fixed_prefix_data=None,
        fixed_prefix_mask=None,
        fixed_prefix_noise=None,
        duration_padding_sec: float = 6.0,
        dist_shift: tp.Any = "default",
        cfg_interval: tuple[float, float] = (0.0, 1.0),
        scale_phi: float = 0.0,
        cfg_norm_threshold: float = 0.0,
        apg_scale: float = 0.0,
        lora_configs: tp.Sequence[dict[str, tp.Any]] | None = None,
        lora_strengths: float | tp.Sequence[float] | None = None,
        chunked_decode: bool = False,
        decode_chunk_size: int = 128,
        decode_overlap: int = 32,
        decode_chunk_batch_size: int = 1,
        callback: tp.Callable[[dict[str, tp.Any]], None] | None = None,
        return_latents: bool = False,
        return_dict: bool = False,
    ):
        self._require_generation_ready()
        if init_audio is not None and inpaint_audio is not None:
            raise ValueError("init_audio and inpaint_audio are mutually exclusive.")
        if fixed_prefix_data is not None and (init_audio is not None or inpaint_audio is not None):
            raise ValueError("fixed_prefix_data is mutually exclusive with init_audio and inpaint_audio.")
        if cfg_interval[0] > cfg_interval[1]:
            raise ValueError("cfg_interval min must be <= max.")

        import torch

        from stable_audio_3.mlx.conditioning import (
            assemble_conditioning_inputs_from_tensors,
            build_mlx_conditioning_tensors,
        )
        from stable_audio_3.mlx.dit import extract_diffusion_objective
        from stable_audio_3.mlx.sampling import (
            create_padding_mask_for_effective_lengths,
            default_sampler_type_for_objective,
            distribution_shift_spec_to_jsonable,
            effective_latent_lengths_from_durations,
            make_linear_schedule_values,
            make_shifted_linear_schedule_values,
            padding_lengths_from_effective_lengths,
            sample_rf_latents_mlx,
        )

        mx = import_mlx_core(required=True)
        mlx_dtype = getattr(mx, self.dtype_name)
        autoencoder_dtype = getattr(mx, self.autoencoder_dtype_name)
        run_seed = secrets.randbelow(2**31) if seed is None or int(seed) < 0 else int(seed)
        mx.random.seed(run_seed)
        torch.manual_seed(run_seed)
        use_negative_conditioning = float(cfg_scale) != 1.0

        if conditioning is None:
            conditioning, negative_conditioning_from_prompt = _build_conditioning_dicts(
                prompt,
                negative_prompt,
                duration,
                batch_size,
            )
            if use_negative_conditioning and negative_conditioning is None:
                negative_conditioning = negative_conditioning_from_prompt
        batch_size = len(conditioning)
        duration_values = _conditioning_durations(conditioning, duration, batch_size)
        audio_sample_size, latent_length = _infer_lengths(
            self.torch_pipeline,
            conditioning,
            sample_size=sample_size,
            duration_padding_sec=duration_padding_sec,
        )
        sample_rate = int(self.torch_pipeline.model.sample_rate)
        downsampling_ratio = int(self.torch_pipeline.model.pretransform.downsampling_ratio)
        effective_seq_len = effective_latent_lengths_from_durations(
            duration_values,
            sample_rate=sample_rate,
            downsampling_ratio=downsampling_ratio,
        )
        padding_headroom_tokens = max(
            int(float(duration_padding_sec) * sample_rate / downsampling_ratio),
            0,
        )
        padding_valid_lengths = padding_lengths_from_effective_lengths(
            effective_seq_len,
            latent_length,
            headroom_tokens=padding_headroom_tokens,
        )
        padding_mask = create_padding_mask_for_effective_lengths(
            effective_seq_len,
            latent_length,
            headroom_tokens=padding_headroom_tokens,
        )
        mx.eval(padding_mask)
        dist_shift_spec, dist_shift_label = _resolve_mlx_dist_shift(dist_shift, self)

        lora_reports: list[dict[str, object]] = []
        sampling_model = self.mlx_dit
        dit_lora_runtime = None
        static_dit_report = None
        active_lora_strengths = ()
        active_lora_configs: tuple[dict[str, tp.Any], ...] = ()
        if self.lora_paths:
            active_lora_strengths = _normalize_lora_strengths(
                lora_strengths if lora_strengths is not None else self.lora_strengths,
                len(self.lora_paths),
            )
            active_lora_configs = _normalize_lora_configs(lora_configs, len(self.lora_paths))
            sampling_model, dit_lora_runtime, static_dit_report = self._prepare_dit_lora(
                active_lora_strengths,
                active_lora_configs,
            )
            lora_reports.extend(self._apply_text_number_loras(active_lora_strengths, active_lora_configs))

        init_latents = None
        inpaint_latents = None
        inpaint_mask_latent = None
        inpaint_masked_input = None
        if init_audio is not None:
            init_latents = self._encode_audio_input(
                init_audio,
                audio_sample_size=audio_sample_size,
                latent_length=latent_length,
                batch_size=batch_size,
                mlx_dtype=mlx_dtype,
                autoencoder_dtype=autoencoder_dtype,
            )
        elif inpaint_audio is not None:
            inpaint_latents = self._encode_audio_input(
                inpaint_audio,
                audio_sample_size=audio_sample_size,
                latent_length=latent_length,
                batch_size=batch_size,
                mlx_dtype=mlx_dtype,
                autoencoder_dtype=autoencoder_dtype,
            )
            mask_audio = _build_inpaint_mask(
                inpaint_mask,
                audio_sample_size=audio_sample_size,
                sample_rate=sample_rate,
                duration=max(duration_values),
                mask_start_seconds=inpaint_mask_start_seconds,
                mask_end_seconds=inpaint_mask_end_seconds,
            )
            mask_latent_torch = _resize_mask_to_latents(mask_audio, latent_length)
            if int(mask_latent_torch.shape[0]) != batch_size:
                mask_latent_torch = mask_latent_torch.repeat(batch_size, 1)
            inpaint_mask_latent = mx.array(mask_latent_torch.numpy()).astype(mlx_dtype)
            inpaint_masked_input = inpaint_latents * inpaint_mask_latent[:, None, :]
            mx.eval(inpaint_latents, inpaint_mask_latent, inpaint_masked_input)

        conditioning_tensors = build_mlx_conditioning_tensors(
            self.model_config,
            conditioning,
            text_conditioners={"prompt": self.text_conditioner},
            number_conditioners={"seconds_total": self.number_conditioner},
        )
        if inpaint_mask_latent is not None and inpaint_masked_input is not None:
            conditioning_tensors["inpaint_mask"] = (inpaint_mask_latent[:, None, :], None)
            conditioning_tensors["inpaint_masked_input"] = (inpaint_masked_input, None)

        cond_inputs = assemble_conditioning_inputs_from_tensors(
            self.model_config,
            conditioning_tensors,
            negative=False,
            latent_length=latent_length,
            dtype_name=self.dtype_name,
        )
        if use_negative_conditioning and negative_conditioning is not None:
            negative_tensors = build_mlx_conditioning_tensors(
                self.model_config,
                negative_conditioning,
                text_conditioners={"prompt": self.text_conditioner},
                number_conditioners={"seconds_total": self.number_conditioner},
            )
            if inpaint_mask_latent is not None and inpaint_masked_input is not None:
                negative_tensors["inpaint_mask"] = (inpaint_mask_latent[:, None, :], None)
                negative_tensors["inpaint_masked_input"] = (inpaint_masked_input, None)
            cond_inputs.update(
                assemble_conditioning_inputs_from_tensors(
                    self.model_config,
                    negative_tensors,
                    negative=True,
                    latent_length=latent_length,
                    dtype_name=self.dtype_name,
                )
            )

        io_channels = int(self.torch_pipeline.model.io_channels)
        noise = mx.random.normal((batch_size, io_channels, latent_length), dtype=mlx_dtype)
        diffusion_objective = extract_diffusion_objective(self.model_config)
        sampler_type = sampler_type or default_sampler_type_for_objective(diffusion_objective)
        sigma_max = float(init_noise_level) if init_latents is not None else 1.0
        shifted_schedule = make_shifted_linear_schedule_values(
            steps,
            sigma_max=sigma_max,
            dist_shift=dist_shift_spec,
            effective_seq_len=effective_seq_len,
            fallback_seq_len=latent_length,
        )
        latents = sample_rf_latents_mlx(
            sampling_model,
            noise,
            cond_inputs=cond_inputs,
            diffusion_objective=diffusion_objective,
            steps=steps,
            cfg_scale=cfg_scale,
            sampler_type=sampler_type,
            init_data=init_latents,
            init_noise_level=init_noise_level,
            dist_shift=dist_shift_spec,
            effective_seq_len=effective_seq_len,
            padding_mask=padding_mask,
            cfg_interval=cfg_interval,
            scale_phi=scale_phi,
            cfg_norm_threshold=cfg_norm_threshold,
            apg_scale=apg_scale,
            callback=callback,
            fixed_prefix_data=fixed_prefix_data,
            fixed_prefix_mask=fixed_prefix_mask,
            fixed_prefix_noise=fixed_prefix_noise,
        )
        audio = None
        if not return_latents:
            decode_latents = (
                latents if latents.dtype == autoencoder_dtype else latents.astype(autoencoder_dtype)
            )
            if chunked_decode:
                audio = self.autoencoder.decode_audio(
                    decode_latents,
                    chunked=True,
                    chunk_size=int(decode_chunk_size),
                    overlap=int(decode_overlap),
                    chunk_batch_size=int(decode_chunk_batch_size),
                    add_bottleneck_noise=False,
                )
            else:
                audio = self.autoencoder.decode(decode_latents, add_bottleneck_noise=False)
            if truncate_output_to_duration:
                max_samples = _truncate_samples(duration, sample_rate)
                if max_samples is not None:
                    audio = audio[:, :, :max_samples]
            mx.eval(audio)
        mx.eval(latents)

        if static_dit_report is not None:
            lora_reports.insert(0, static_dit_report)
        elif dit_lora_runtime is not None:
            lora_reports.insert(
                0,
                _dit_lora_report_payload(
                    self,
                    active_lora_strengths,
                    active_lora_configs,
                    report=dit_lora_runtime.reports[-1] if dit_lora_runtime.reports else None,
                    schedule_updates=[report.to_dict() for report in dit_lora_runtime.reports],
                ),
            )
            self._clear_applied_dit_lora()

        latents_np = np.asarray(latents, dtype=np.float32)
        audio_np = np.asarray(audio, dtype=np.float32) if audio is not None else None
        report = {
            "model": self.source_name,
            "prompt": prompt,
            "duration": duration,
            "duration_padding_sec": float(duration_padding_sec),
            "sample_rate": sample_rate,
            "audio_sample_size": audio_sample_size,
            "latent_length": latent_length,
            "steps": int(steps),
            "cfg_scale": float(cfg_scale),
            "cfg_interval": tuple(float(x) for x in cfg_interval),
            "cfg_rescale": float(scale_phi),
            "cfg_norm_threshold": float(cfg_norm_threshold),
            "apg_scale": float(apg_scale),
            "sampler_type": sampler_type,
            "diffusion_objective": diffusion_objective,
            "dist_shift": {
                "requested": "default" if dist_shift is None else str(dist_shift),
                "resolved": dist_shift_label,
                "spec": distribution_shift_spec_to_jsonable(dist_shift_spec),
                "effective_seq_len": tuple(int(x) for x in effective_seq_len),
            },
            "padding_mask": {
                "shape": tuple(int(x) for x in padding_mask.shape),
                "headroom_tokens": int(padding_headroom_tokens),
                "valid_lengths": tuple(int(x) for x in padding_valid_lengths),
                "mean": float(np.asarray(padding_mask).mean()),
            },
            "unshifted_schedule": make_linear_schedule_values(steps, sigma_max=sigma_max),
            "schedule": shifted_schedule,
            "seed": run_seed,
            "batch_size": batch_size,
            "dtype": self.dtype_name,
            "autoencoder_dtype": self.autoencoder_dtype_name,
            "attention": self.attention,
            "chunked_decode": bool(chunked_decode),
            "decode_chunk_size": int(decode_chunk_size) if chunked_decode else None,
            "decode_overlap": int(decode_overlap) if chunked_decode else None,
            "decode_chunk_batch_size": int(decode_chunk_batch_size) if chunked_decode else None,
            "lora": {
                "paths": [str(path) for path in self.lora_paths],
                "labels": list(self.lora_labels),
                "strength": active_lora_strengths[0]
                if active_lora_strengths and len(set(active_lora_strengths)) == 1
                else None,
                "strengths": list(active_lora_strengths),
                "intervals": [
                    {"min": float(config["interval"][0]), "max": float(config["interval"][1])}
                    for config in active_lora_configs
                ],
                "layer_filters": [str(config["layer_filter"]) for config in active_lora_configs],
                "reports": lora_reports,
            },
            "mlx_latent_shape": tuple(int(x) for x in latents.shape),
            "mlx_audio_shape": tuple(int(x) for x in audio.shape) if audio is not None else None,
            "mlx_latent_stats": _array_stats(latents_np),
            "mlx_audio_stats": _array_stats(audio_np) if audio_np is not None else None,
            "conversion": _conversion_summary(self.conversion_reports),
        }
        if return_dict:
            return MLXGenerationResult(audio=audio, latents=latents, report=report)
        return latents if return_latents else audio

    def to_json(self) -> str:
        return json.dumps(self.smoke_test_report(), indent=2)

    def _require_generation_ready(self) -> None:
        if not self.generation_ready:
            raise NotImplementedError(
                "This StableAudioMLXPipeline was created for config inspection only. "
                "Use StableAudioMLXPipeline.from_torch_pretrained(...) for runtime "
                "MLX generation."
            )

    def _encode_audio_input(
        self,
        audio_input: tuple[int, tp.Any],
        *,
        audio_sample_size: int,
        latent_length: int,
        batch_size: int,
        mlx_dtype,
        autoencoder_dtype,
    ):
        from stable_audio_3.inference.audio_utils import prepare_audio

        mx = import_mlx_core(required=True)
        source_sr, source_audio = audio_input
        prepared_target_length = _input_audio_target_length(
            source_audio,
            in_sr=int(source_sr),
            target_sr=int(self.torch_pipeline.model.sample_rate),
            max_target_length=int(audio_sample_size),
        )
        prepared = prepare_audio(
            source_audio,
            in_sr=int(source_sr),
            target_sr=int(self.torch_pipeline.model.sample_rate),
            target_length=prepared_target_length,
            target_channels=int(self.torch_pipeline.model.pretransform.io_channels),
            device="cpu",
        ).float()
        latents = self.autoencoder.encode(mx.array(prepared.numpy()).astype(autoencoder_dtype))
        latents = _fit_latent_length(latents, latent_length)
        if latents.dtype != mlx_dtype:
            latents = latents.astype(mlx_dtype)
        if int(latents.shape[0]) != int(batch_size):
            if int(latents.shape[0]) != 1:
                raise ValueError("Only single-source audio can be broadcast to batch generation.")
            latents = mx.broadcast_to(
                latents,
                (int(batch_size), int(latents.shape[1]), int(latents.shape[2])),
            )
        mx.eval(latents)
        return latents


def _build_conditioning_dicts(
    prompt: str | list[str] | None,
    negative_prompt: str | list[str] | None,
    duration: float | list[float],
    batch_size: int,
) -> tuple[list[dict[str, tp.Any]], list[dict[str, tp.Any]] | None]:
    if prompt is None:
        raise ValueError("prompt is required when conditioning is not provided.")

    prompts = _broadcast_values(prompt, batch_size, label="prompt")
    durations = _broadcast_values(duration, batch_size, label="duration")
    conditioning = [
        {"prompt": prompt_value, "seconds_total": float(duration_value)}
        for prompt_value, duration_value in zip(prompts, durations, strict=True)
    ]

    negative_conditioning = None
    if negative_prompt is not None:
        negative_prompts = _broadcast_values(
            negative_prompt,
            batch_size,
            label="negative_prompt",
        )
        negative_conditioning = [
            {"prompt": prompt_value, "seconds_total": float(duration_value)}
            for prompt_value, duration_value in zip(
                negative_prompts,
                durations,
                strict=True,
            )
        ]
    return conditioning, negative_conditioning


def _broadcast_values(value, count: int, *, label: str) -> list:
    if isinstance(value, (str, bytes)) or not isinstance(value, list | tuple):
        return [value for _ in range(int(count))]
    if len(value) == int(count):
        return list(value)
    if len(value) == 1:
        return [value[0] for _ in range(int(count))]
    raise ValueError(f"{label} length must be 1 or batch_size ({count}), got {len(value)}.")


def _conditioning_durations(
    conditioning: list[dict[str, tp.Any]],
    fallback_duration: float | list[float],
    batch_size: int,
) -> list[float]:
    values = []
    for item in conditioning:
        if "seconds_total" in item:
            values.append(float(item["seconds_total"]))
    if values:
        return values
    return [float(value) for value in _broadcast_values(fallback_duration, batch_size, label="duration")]


def _infer_lengths(
    torch_pipeline,
    conditioning: list[dict[str, tp.Any]],
    *,
    sample_size: int,
    duration_padding_sec: float,
) -> tuple[int, int]:
    audio_sample_size = torch_pipeline._adapt_sample_size(
        conditioning,
        sample_size,
        duration_padding_sec,
    )
    downsampling_ratio = int(torch_pipeline.model.pretransform.downsampling_ratio)
    latent_sample_size = int(math.ceil(audio_sample_size / downsampling_ratio))
    return int(audio_sample_size), latent_sample_size


def _resolve_mlx_dist_shift(value, pipeline: StableAudioMLXPipeline):
    from stable_audio_3.mlx.sampling import (
        DistributionShiftSpec,
        distribution_shift_spec_from_object,
        make_distribution_shift_spec,
    )

    if isinstance(value, DistributionShiftSpec):
        return value, value.kind
    if value is None or str(value).lower() == "default":
        return distribution_shift_spec_from_object(
            pipeline.torch_pipeline.model.sampling_dist_shift
        ), "default"
    selection = str(value).lower()
    if selection == "none":
        return make_distribution_shift_spec("none"), "none"
    if selection == "full":
        return make_distribution_shift_spec("full"), "full"
    if selection == "flux":
        return make_distribution_shift_spec("flux", alpha_min=6.93, alpha_max=6.93), "flux"
    if selection == "logsnr":
        return (
            make_distribution_shift_spec(
                "logsnr",
                rate=0,
                anchor_logsnr=-6.2,
                logsnr_end=2.0,
            ),
            "logsnr",
        )
    raise ValueError(f"Unknown dist_shift selection: {value!r}")


def _normalize_lora_labels(
    names: tp.Sequence[str] | None,
    paths: tp.Sequence[Path],
) -> tuple[str, ...]:
    if names is None:
        return tuple(path.stem for path in paths)
    if len(names) != len(paths):
        raise ValueError(f"Expected {len(paths)} LoRA names, got {len(names)}.")
    return tuple(str(name) for name in names)


def _normalize_lora_strengths(
    strength: float | tp.Sequence[float],
    count: int,
) -> tuple[float, ...]:
    if count <= 0:
        return ()
    if isinstance(strength, int | float):
        return tuple(float(strength) for _ in range(count))
    values = tuple(float(value) for value in strength)
    if len(values) == 1:
        return tuple(values[0] for _ in range(count))
    if len(values) != count:
        raise ValueError(f"Expected 1 or {count} LoRA strengths, got {len(values)}.")
    return values


def _normalize_lora_configs(
    configs: tp.Sequence[dict[str, tp.Any]] | None,
    count: int,
) -> tuple[dict[str, tp.Any], ...]:
    if count <= 0:
        return ()
    if configs is None:
        return tuple(
            {"lora_index": index, "interval": (0.0, 1.0), "layer_filter": ""}
            for index in range(count)
        )
    if len(configs) != count:
        raise ValueError(f"Expected {count} LoRA configs, got {len(configs)}.")
    normalized = []
    for index, config in enumerate(configs):
        interval = config.get("interval", (0.0, 1.0))
        if len(interval) != 2:
            raise ValueError(f"LoRA config {index} interval must have two values.")
        interval_min = float(interval[0])
        interval_max = float(interval[1])
        if interval_min > interval_max:
            raise ValueError(f"LoRA config {index} interval min must be <= max.")
        normalized.append(
            {
                "lora_index": int(config.get("lora_index", index)),
                "interval": (interval_min, interval_max),
                "layer_filter": str(config.get("layer_filter", "") or ""),
            }
        )
    return tuple(normalized)


def _lora_has_active_strengths(strengths: tuple[float, ...]) -> bool:
    return any(abs(float(strength)) > 1e-8 for strength in strengths)


def _lora_configs_are_static(configs: tuple[dict[str, tp.Any], ...]) -> bool:
    for config in configs:
        interval = config.get("interval", (0.0, 1.0))
        if float(interval[0]) > 0.0 or float(interval[1]) < 1.0:
            return False
        if str(config.get("layer_filter", "") or ""):
            return False
    return True


def _lora_runtime_signature(
    strengths: tuple[float, ...],
    configs: tuple[dict[str, tp.Any], ...],
) -> tuple[tp.Any, ...]:
    return (
        tuple(float(strength) for strength in strengths),
        tuple(
            (
                float(config["interval"][0]),
                float(config["interval"][1]),
                str(config.get("layer_filter", "") or ""),
            )
            for config in configs
        ),
    )


def _include_lora_report(report) -> bool:
    return bool(
        report.applied_layers
        or report.unsupported_adapters
        or report.skipped_layers
        or report.missing_targets
    )


def _dit_lora_report_payload(
    pipeline: StableAudioMLXPipeline,
    strengths: tuple[float, ...],
    configs: tuple[dict[str, tp.Any], ...],
    *,
    report,
    schedule_updates: list[dict[str, object]],
) -> dict[str, object]:
    lora_set = pipeline._dit_lora_set
    return {
        "target_label": "dit",
        "paths": [str(path) for path in pipeline.lora_paths],
        "names": list(lora_set.names) if lora_set is not None else [],
        "labels": list(pipeline.lora_labels),
        "strength": strengths[0] if strengths and len(set(strengths)) == 1 else None,
        "strengths": list(strengths),
        "intervals": [
            {"min": float(config["interval"][0]), "max": float(config["interval"][1])}
            for config in configs
        ],
        "layer_filters": [str(config["layer_filter"]) for config in configs],
        "loaded_layers": len(lora_set.layers) if lora_set is not None else 0,
        "schedule_updates": schedule_updates,
        "missing_targets": list(report.missing_targets) if report is not None else [],
        "unsupported_adapters": list(report.unsupported_adapters) if report is not None else [],
        "skipped_layers": list(report.skipped_layers) if report is not None else [],
    }


def _fit_latent_length(latents, latent_length: int):
    mx = import_mlx_core(required=True)
    current_length = int(latents.shape[-1])
    latent_length = int(latent_length)
    if current_length > latent_length:
        return latents[..., :latent_length]
    if current_length < latent_length:
        pad = mx.zeros(
            (*latents.shape[:-1], latent_length - current_length),
            dtype=latents.dtype,
        )
        return mx.concatenate([latents, pad], axis=-1)
    return latents


def _input_audio_target_length(
    audio,
    *,
    in_sr: int,
    target_sr: int,
    max_target_length: int,
) -> int:
    max_target_length = int(max_target_length)
    if max_target_length <= 1:
        return 1
    if in_sr <= 0 or target_sr <= 0 or not hasattr(audio, "shape"):
        return max_target_length

    try:
        source_samples = int(audio.shape[-1])
    except (TypeError, ValueError, IndexError):
        return max_target_length
    if source_samples <= 0:
        return max_target_length

    if in_sr == target_sr:
        target_length = source_samples
    else:
        target_length = int(math.ceil(source_samples * float(target_sr) / float(in_sr)))
    return max(1, min(max_target_length, target_length))


def _build_inpaint_mask(
    inpaint_mask,
    *,
    audio_sample_size: int,
    sample_rate: int,
    duration: float,
    mask_start_seconds: float | None,
    mask_end_seconds: float | None,
):
    import torch

    if inpaint_mask is not None:
        mask = torch.as_tensor(inpaint_mask, dtype=torch.float32)
        if mask.ndim == 1:
            mask = mask[None, :]
        if mask.shape[-1] > audio_sample_size:
            mask = mask[:, :audio_sample_size]
        elif mask.shape[-1] < audio_sample_size:
            mask = torch.nn.functional.pad(mask, (0, audio_sample_size - mask.shape[-1]))
    else:
        if mask_start_seconds is None or mask_end_seconds is None:
            raise ValueError(
                "inpaint_audio requires either inpaint_mask or "
                "inpaint_mask_start_seconds/inpaint_mask_end_seconds."
            )
        mask_start_samples = min(int(float(mask_start_seconds) * sample_rate), audio_sample_size)
        mask_end_samples = min(int(float(mask_end_seconds) * sample_rate), audio_sample_size)
        if mask_end_samples < mask_start_samples:
            raise ValueError("inpaint mask end must be greater than or equal to mask start.")
        mask = torch.ones(1, audio_sample_size, dtype=torch.float32)
        mask[:, mask_start_samples:mask_end_samples] = 0.0

    effective_audio_len = int(float(duration) * sample_rate)
    if effective_audio_len < audio_sample_size:
        mask = mask.clone()
        mask[:, effective_audio_len:] = 0.0
    return mask


def _resize_mask_to_latents(mask, latent_length: int):
    import torch

    return torch.nn.functional.interpolate(
        mask.unsqueeze(1),
        size=int(latent_length),
        mode="nearest",
    ).squeeze(1)


def _truncate_samples(duration: float | list[float], sample_rate: int) -> int | None:
    if isinstance(duration, int | float):
        return int(float(duration) * sample_rate)
    if not duration:
        return None
    values = [float(value) for value in duration]
    if all(value == values[0] for value in values):
        return int(values[0] * sample_rate)
    return None


def _array_stats(array: np.ndarray) -> dict[str, float | bool]:
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
        "finite": bool(np.isfinite(array).all()),
    }


def _conversion_summary(reports: dict[str, tp.Any]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for label, report in reports.items():
        summary[f"{label}_unexpected_keys"] = len(getattr(report, "unexpected_keys", ()))
        summary[f"{label}_transposed_keys"] = len(getattr(report, "transposed_keys", ()))
        if hasattr(report, "synthesized_keys"):
            summary[f"{label}_synthesized_keys"] = len(getattr(report, "synthesized_keys", ()))
    return summary


def resolve_pretrained_config_path(
    model_name_or_path: str,
    *,
    search_roots=None,
) -> Path:
    candidate_path = Path(model_name_or_path).expanduser()
    if candidate_path.exists():
        return candidate_path.resolve()

    model_cfg = all_models.get(model_name_or_path)
    if model_cfg is not None:
        if hasattr(model_cfg, "resolve_config"):
            local_config = model_cfg.resolve_config()
        else:
            local_config, _ = model_cfg.resolve()
        return Path(local_config).expanduser().resolve()

    search_root_list = list(search_roots or ())
    search_root_list.extend(
        [
            _REPO_ROOT / ".hf_configs",
            _REPO_ROOT,
        ]
    )

    for root in search_root_list:
        root_path = Path(root).expanduser()
        candidate = root_path / model_name_or_path
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Could not resolve config for '{model_name_or_path}'. "
        f"Known model aliases: {sorted(all_models)}. "
        "If this is a gated Hugging Face model, set HF_TOKEN and make sure the "
        "required SA3 assets have been downloaded first."
    )
