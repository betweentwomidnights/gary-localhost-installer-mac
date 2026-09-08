"""Execution helpers for service generation diffusion and output assembly."""

import random
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from loguru import logger


class ServiceGenerateExecuteMixin:
    """Run diffusion execution for normalized service-generation requests."""

    def _prepare_service_condition(
        self,
        *,
        text_hidden_states: torch.Tensor,
        text_attention_mask: torch.Tensor,
        lyric_hidden_states: torch.Tensor,
        lyric_attention_mask: torch.Tensor,
        refer_audio_acoustic_hidden_states_packed: torch.Tensor,
        refer_audio_order_mask: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        silence_latent: torch.Tensor,
        src_latents: torch.Tensor,
        chunk_masks: torch.Tensor,
        is_covers: torch.Tensor,
        precomputed_lm_hints_25Hz: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare conditioning while skipping unused FSQ work for non-cover batches."""
        if isinstance(is_covers, torch.Tensor) and not bool(is_covers.any().item()):
            encoder_hidden_states, encoder_attention_mask = self.model.encoder(
                text_hidden_states=text_hidden_states,
                text_attention_mask=text_attention_mask,
                lyric_hidden_states=lyric_hidden_states,
                lyric_attention_mask=lyric_attention_mask,
                refer_audio_acoustic_hidden_states_packed=refer_audio_acoustic_hidden_states_packed,
                refer_audio_order_mask=refer_audio_order_mask,
            )
            context_latents = torch.cat([src_latents, chunk_masks.to(hidden_states.dtype)], dim=-1)
            return encoder_hidden_states, encoder_attention_mask, context_latents

        return self.model.prepare_condition(
            text_hidden_states=text_hidden_states,
            text_attention_mask=text_attention_mask,
            lyric_hidden_states=lyric_hidden_states,
            lyric_attention_mask=lyric_attention_mask,
            refer_audio_acoustic_hidden_states_packed=refer_audio_acoustic_hidden_states_packed,
            refer_audio_order_mask=refer_audio_order_mask,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            silence_latent=silence_latent,
            src_latents=src_latents,
            chunk_masks=chunk_masks,
            is_covers=is_covers,
            precomputed_lm_hints_25Hz=precomputed_lm_hints_25Hz,
        )

    def _unpack_service_processed_data(self, processed_data: Tuple[Any, ...]) -> Dict[str, Any]:
        """Convert batch preprocessing tuple into a keyed payload."""
        (
            keys,
            text_inputs,
            src_latents,
            target_latents,
            text_hidden_states,
            text_attention_mask,
            lyric_hidden_states,
            lyric_attention_mask,
            _audio_attention_mask,
            refer_audio_acoustic_hidden_states_packed,
            refer_audio_order_mask,
            chunk_mask,
            spans,
            is_covers,
            _audio_codes,
            lyric_token_idss,
            precomputed_lm_hints_25Hz,
            non_cover_text_hidden_states,
            non_cover_text_attention_masks,
        ) = processed_data
        return {
            "keys": keys,
            "text_inputs": text_inputs,
            "src_latents": src_latents,
            "target_latents": target_latents,
            "text_hidden_states": text_hidden_states,
            "text_attention_mask": text_attention_mask,
            "lyric_hidden_states": lyric_hidden_states,
            "lyric_attention_mask": lyric_attention_mask,
            "refer_audio_acoustic_hidden_states_packed": refer_audio_acoustic_hidden_states_packed,
            "refer_audio_order_mask": refer_audio_order_mask,
            "chunk_mask": chunk_mask,
            "spans": spans,
            "is_covers": is_covers,
            "lyric_token_idss": lyric_token_idss,
            "precomputed_lm_hints_25Hz": precomputed_lm_hints_25Hz,
            "non_cover_text_hidden_states": non_cover_text_hidden_states,
            "non_cover_text_attention_masks": non_cover_text_attention_masks,
        }

    def _resolve_service_seed_param(self, seed_list: Optional[List[int]]) -> Any:
        """Return model seed parameter: per-item seed list or random single seed."""
        if seed_list is not None:
            return seed_list
        return random.randint(0, 2**32 - 1)

    def _build_service_generate_kwargs(
        self,
        payload: Dict[str, Any],
        seed_param: Any,
        infer_steps: int,
        guidance_scale: float,
        audio_cover_strength: float,
        cover_noise_strength: float,
        repaint_injection_ratio: float,
        infer_method: str,
        use_adg: bool,
        cfg_interval_start: float,
        cfg_interval_end: float,
        shift: float,
        timesteps: Optional[List[float]],
        task_type: str,
    ) -> Dict[str, Any]:
        """Build kwargs passed to model generation backends."""
        if task_type in ("cover", "cover-nofsq"):
            is_cover_task = True
        else:
            is_covers = payload.get("is_covers")
            if isinstance(is_covers, torch.Tensor):
                # Legacy direct service_generate callers may imply cover mode by
                # supplying audio-code hints without going through generate_music.
                is_cover_task = bool(is_covers.any().item())
            else:
                is_cover_task = any(is_covers or [])

        kwargs = {
            "text_hidden_states": payload["text_hidden_states"],
            "text_attention_mask": payload["text_attention_mask"],
            "lyric_hidden_states": payload["lyric_hidden_states"],
            "lyric_attention_mask": payload["lyric_attention_mask"],
            "refer_audio_acoustic_hidden_states_packed": payload["refer_audio_acoustic_hidden_states_packed"],
            "refer_audio_order_mask": payload["refer_audio_order_mask"],
            "src_latents": payload["src_latents"],
            "chunk_masks": payload["chunk_mask"],
            "is_covers": payload["is_covers"],
            "silence_latent": self.silence_latent,
            "seed": seed_param,
            "non_cover_text_hidden_states": payload["non_cover_text_hidden_states"],
            "non_cover_text_attention_mask": payload["non_cover_text_attention_masks"],
            "precomputed_lm_hints_25Hz": payload["precomputed_lm_hints_25Hz"],
            "audio_cover_strength": audio_cover_strength,
            "cover_noise_strength": cover_noise_strength,
            # Only cover requests get the source-latent remix anchor. This is
            # the critical gate that keeps lego/complete from regressing into
            # source-audio reconstruction instead of stem generation.
            "clean_src_latents": (
                payload.get("target_latents")
                if (is_cover_task and repaint_injection_ratio > 0.0)
                else None
            ),
            "repaint_injection_ratio": repaint_injection_ratio if is_cover_task else 0.0,
            "infer_method": infer_method,
            "infer_steps": infer_steps,
            "diffusion_guidance_sale": guidance_scale,
            "use_adg": use_adg,
            "cfg_interval_start": cfg_interval_start,
            "cfg_interval_end": cfg_interval_end,
            "shift": shift,
        }
        if timesteps is not None:
            kwargs["timesteps"] = torch.tensor(timesteps, dtype=torch.float32, device=self.device)
        return kwargs

    def _execute_service_generate_diffusion(
        self,
        payload: Dict[str, Any],
        generate_kwargs: Dict[str, Any],
        seed_param: Any,
        infer_method: str,
        shift: float,
        audio_cover_strength: float,
        cover_noise_strength: float,
    ) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
        """Execute condition preparation and diffusion using MLX or PyTorch backend."""
        dit_backend = (
            "MLX (native)" if (self.use_mlx_dit and self.mlx_decoder is not None) else f"PyTorch ({self.device})"
        )
        logger.info(f"[service_generate] Generating audio... (DiT backend: {dit_backend})")
        spans = payload.get("spans") or []
        if spans:
            full_count = sum(1 for span in spans if span and span[0] == "full")
            repaint_count = sum(1 for span in spans if span and span[0] == "repainting")
            logger.info(
                "[service_generate] Conditioning mask mode: full={}, repainting={}, first_span={}",
                full_count,
                repaint_count,
                spans[0],
            )

        src_latents = payload.get("src_latents")
        if isinstance(src_latents, torch.Tensor) and src_latents.numel() > 0:
            logger.info(
                "[service_generate] Source latent stats: mean_abs={:.6f}, max_abs={:.6f}",
                float(src_latents.abs().mean().item()),
                float(src_latents.abs().max().item()),
            )

        requested_steps = int(generate_kwargs.get("infer_steps", 8))
        logger.info(
            "[service_generate] Request params: infer_method={}, infer_steps={}, shift={}, audio_cover_strength={}, cover_noise_strength={}",
            infer_method,
            requested_steps,
            shift,
            audio_cover_strength,
            cover_noise_strength,
        )
        execute_started = time.perf_counter()
        condition_started = execute_started
        non_cover_condition_seconds = 0.0
        with torch.inference_mode():
            with self._load_model_context("model"):
                encoder_hidden_states, encoder_attention_mask, context_latents = self._prepare_service_condition(
                    text_hidden_states=payload["text_hidden_states"],
                    text_attention_mask=payload["text_attention_mask"],
                    lyric_hidden_states=payload["lyric_hidden_states"],
                    lyric_attention_mask=payload["lyric_attention_mask"],
                    refer_audio_acoustic_hidden_states_packed=payload["refer_audio_acoustic_hidden_states_packed"],
                    refer_audio_order_mask=payload["refer_audio_order_mask"],
                    hidden_states=payload["src_latents"],
                    attention_mask=torch.ones(
                        payload["src_latents"].shape[0],
                        payload["src_latents"].shape[1],
                        device=payload["src_latents"].device,
                        dtype=payload["src_latents"].dtype,
                    ),
                    silence_latent=self.silence_latent,
                    src_latents=payload["src_latents"],
                    chunk_masks=payload["chunk_mask"],
                    is_covers=payload["is_covers"],
                    precomputed_lm_hints_25Hz=payload["precomputed_lm_hints_25Hz"],
                )
                condition_seconds = time.perf_counter() - condition_started

                if self.use_mlx_dit and self.mlx_decoder is not None:
                    try:
                        enc_hs_nc, enc_am_nc, ctx_nc = None, None, None
                        if audio_cover_strength < 1.0 and payload["non_cover_text_hidden_states"] is not None:
                            non_cover_started = time.perf_counter()
                            non_is_covers = torch.zeros_like(payload["is_covers"])
                            sil_exp = self.silence_latent[:, :payload["src_latents"].shape[1], :].expand(
                                payload["src_latents"].shape[0], -1, -1
                            )
                            enc_hs_nc, enc_am_nc, ctx_nc = self._prepare_service_condition(
                                text_hidden_states=payload["non_cover_text_hidden_states"],
                                text_attention_mask=payload["non_cover_text_attention_masks"],
                                lyric_hidden_states=payload["lyric_hidden_states"],
                                lyric_attention_mask=payload["lyric_attention_mask"],
                                refer_audio_acoustic_hidden_states_packed=payload["refer_audio_acoustic_hidden_states_packed"],
                                refer_audio_order_mask=payload["refer_audio_order_mask"],
                                hidden_states=sil_exp,
                                attention_mask=torch.ones(
                                    sil_exp.shape[0], sil_exp.shape[1], device=sil_exp.device, dtype=sil_exp.dtype
                                ),
                                silence_latent=self.silence_latent,
                                src_latents=sil_exp,
                                chunk_masks=payload["chunk_mask"],
                                is_covers=non_is_covers,
                            )
                            non_cover_condition_seconds = time.perf_counter() - non_cover_started

                        diffusion_started = time.perf_counter()
                        outputs = self._mlx_run_diffusion(
                            encoder_hidden_states=encoder_hidden_states,
                            encoder_attention_mask=encoder_attention_mask,
                            context_latents=context_latents,
                            src_latents=payload["src_latents"],
                            seed=seed_param,
                            infer_method=infer_method,
                            infer_steps=requested_steps,
                            shift=shift,
                            timesteps=generate_kwargs.get("timesteps"),
                            guidance_scale=float(
                                generate_kwargs.get(
                                    "diffusion_guidance_scale",
                                    generate_kwargs.get("diffusion_guidance_sale", 1.0),
                                )
                            ),
                            cfg_interval_start=float(generate_kwargs.get("cfg_interval_start", 0.0)),
                            cfg_interval_end=float(generate_kwargs.get("cfg_interval_end", 1.0)),
                            audio_cover_strength=audio_cover_strength,
                            cover_noise_strength=cover_noise_strength,
                            use_adg=bool(generate_kwargs.get("use_adg", False)),
                            encoder_hidden_states_non_cover=enc_hs_nc,
                            encoder_attention_mask_non_cover=enc_am_nc,
                            context_latents_non_cover=ctx_nc,
                        )
                        dispatch_seconds = time.perf_counter() - diffusion_started
                        _tc = outputs.get("time_costs", {})
                        _tc.update(
                            {
                                "condition_prepare_time_cost": condition_seconds,
                                "non_cover_condition_time_cost": non_cover_condition_seconds,
                                "diffusion_dispatch_time_cost": dispatch_seconds,
                                "execute_total_time_cost": time.perf_counter() - execute_started,
                            }
                        )
                        logger.info(
                            "[service_generate] DiT diffusion complete via MLX ({:.2f}s total, {:.3f}s/step, {} steps).",
                            float(_tc.get("diffusion_time_cost", 0.0)),
                            float(_tc.get("diffusion_per_step_time_cost", 0.0)),
                            int(_tc.get("num_steps", requested_steps)),
                        )
                    except Exception as exc:
                        if getattr(self, "_conditioning_shell_active", False):
                            raise RuntimeError(
                                "MLX diffusion failed while the decoder-free conditioning shell was active"
                            ) from exc
                        logger.warning("[service_generate] MLX diffusion failed (%s); falling back to PyTorch.", exc)
                        outputs = self.model.generate_audio(**generate_kwargs)
                else:
                    logger.info("[service_generate] DiT diffusion via PyTorch ({})...", self.device)
                    outputs = self.model.generate_audio(**generate_kwargs)

        timing = outputs.setdefault("time_costs", {})
        timing.setdefault("condition_prepare_time_cost", condition_seconds)
        timing.setdefault("non_cover_condition_time_cost", non_cover_condition_seconds)
        timing.setdefault("execute_total_time_cost", time.perf_counter() - execute_started)
        logger.info(
            "[timing] execute total={:.3f}s condition={:.3f}s non_cover={:.3f}s "
            "diffusion_dispatch={:.3f}s",
            float(timing["execute_total_time_cost"]),
            float(timing["condition_prepare_time_cost"]),
            float(timing["non_cover_condition_time_cost"]),
            float(timing.get("diffusion_dispatch_time_cost", 0.0)),
        )

        return outputs, encoder_hidden_states, encoder_attention_mask, context_latents
