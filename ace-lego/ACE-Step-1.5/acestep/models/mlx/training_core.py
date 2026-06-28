"""ACE-Step MLX flow-matching training primitives."""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass

try:  # Keep the module importable on systems without MLX installed.
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
except ModuleNotFoundError:  # pragma: no cover - exercised by non-MLX dev shells.
    mx = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    optim = None  # type: ignore[assignment]


LOSS_WEIGHTING_NONE = "none"
LOSS_WEIGHTING_MIN_SNR = "min_snr"
LOSS_WEIGHTING_CHOICES = (LOSS_WEIGHTING_NONE, LOSS_WEIGHTING_MIN_SNR)


@dataclass(frozen=True)
class ACEFlowMatchingConfig:
    """Configuration for one ACE MLX flow-matching training step."""

    cfg_ratio: float = 0.15
    timestep_mu: float = -0.4
    timestep_sigma: float = 1.0
    data_proportion: float = 0.0
    use_meanflow: bool = False
    loss_weighting: str = LOSS_WEIGHTING_MIN_SNR
    snr_gamma: float = 5.0


@dataclass(frozen=True)
class ACEAdamWConfig:
    """AdamW settings used by Gary's ACE MLX LoRA training loop."""

    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8


def sample_timesteps(
    batch_size: int,
    *,
    dtype: tp.Any = None,
    data_proportion: float = 0.0,
    timestep_mu: float = -0.4,
    timestep_sigma: float = 1.0,
    use_meanflow: bool = False,
) -> tuple[tp.Any, tp.Any]:
    """Sample ACE-Step logit-normal ``(t, r)`` timesteps in MLX."""
    _require_mlx()
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    dtype = dtype or mx.float32
    t = mx.sigmoid(
        mx.random.normal((batch_size,), dtype=dtype) * timestep_sigma + timestep_mu
    )
    r = mx.sigmoid(
        mx.random.normal((batch_size,), dtype=dtype) * timestep_sigma + timestep_mu
    )
    upper = mx.maximum(t, r)
    lower = mx.minimum(t, r)

    if not use_meanflow:
        data_proportion = 1.0
    data_size = int(batch_size * float(data_proportion))
    zero_mask = mx.arange(batch_size) < data_size
    r = mx.where(zero_mask, upper, lower)
    return upper.astype(dtype), r.astype(dtype)


def apply_cfg_dropout(
    encoder_hidden_states,
    null_condition_emb,
    *,
    cfg_ratio: float = 0.15,
):
    """Apply per-sample classifier-free guidance dropout to ACE conditions."""
    _require_mlx()
    if cfg_ratio <= 0.0:
        return encoder_hidden_states
    if cfg_ratio >= 1.0:
        return mx.broadcast_to(null_condition_emb, encoder_hidden_states.shape).astype(
            encoder_hidden_states.dtype
        )

    bsz = int(encoder_hidden_states.shape[0])
    keep = (
        mx.random.uniform(shape=(bsz,), dtype=encoder_hidden_states.dtype)
        >= float(cfg_ratio)
    )
    keep = keep[:, None, None]
    null_states = mx.broadcast_to(null_condition_emb, encoder_hidden_states.shape)
    return mx.where(keep, encoder_hidden_states, null_states).astype(
        encoder_hidden_states.dtype
    )


def flow_min_snr_weights(timesteps, *, gamma: float = 5.0):
    """Return per-sample Min-SNR weights for ACE flow timesteps."""
    _require_mlx()
    if gamma <= 0:
        raise ValueError("Min-SNR gamma must be greater than zero")

    t = mx.clip(timesteps.astype(mx.float32), 1e-4, 1.0 - 1e-4)
    snr = mx.minimum(((1.0 - t) / t) ** 2, 1e6)
    gamma_values = mx.full(snr.shape, float(gamma), dtype=mx.float32)
    return mx.minimum(snr, gamma_values) / mx.maximum(snr, 1e-6)


def flow_matching_loss_from_prediction(
    prediction,
    flow_target,
    timesteps,
    *,
    loss_weighting: str = LOSS_WEIGHTING_MIN_SNR,
    snr_gamma: float = 5.0,
):
    """Compute ACE flow-matching MSE with optional Min-SNR weighting."""
    _require_mlx()
    loss_weighting = _canonical_loss_weighting(loss_weighting)
    element_loss = (prediction.astype(mx.float32) - flow_target.astype(mx.float32)) ** 2
    if loss_weighting == LOSS_WEIGHTING_NONE:
        return mx.mean(element_loss)

    reduce_axes = tuple(range(1, len(element_loss.shape)))
    per_sample_loss = mx.mean(element_loss, axis=reduce_axes)
    weights = flow_min_snr_weights(timesteps, gamma=snr_gamma)
    return mx.mean(weights * per_sample_loss)


def ace_flow_matching_loss(
    decoder,
    batch: dict[str, tp.Any],
    *,
    null_condition_emb=None,
    config: ACEFlowMatchingConfig | None = None,
    noise=None,
    timesteps=None,
):
    """Run one ACE MLX decoder loss pass against a preprocessed tensor batch."""
    _require_mlx()
    config = config or ACEFlowMatchingConfig()
    loss_weighting = _canonical_loss_weighting(config.loss_weighting)

    target_latents = batch["target_latents"]
    encoder_hidden_states = batch["encoder_hidden_states"]
    encoder_attention_mask = batch.get("encoder_attention_mask")
    context_latents = batch["context_latents"]
    bsz = int(target_latents.shape[0])
    dtype = target_latents.dtype

    if null_condition_emb is not None and config.cfg_ratio > 0.0:
        encoder_hidden_states = apply_cfg_dropout(
            encoder_hidden_states,
            null_condition_emb,
            cfg_ratio=config.cfg_ratio,
        )

    if noise is None:
        noise = mx.random.normal(target_latents.shape, dtype=dtype)
    else:
        noise = noise.astype(dtype)

    if timesteps is None:
        t, r = sample_timesteps(
            bsz,
            dtype=dtype,
            data_proportion=config.data_proportion,
            timestep_mu=config.timestep_mu,
            timestep_sigma=config.timestep_sigma,
            use_meanflow=config.use_meanflow,
        )
    else:
        t = timesteps.astype(dtype)
        r = t

    t_view = t[:, None, None].astype(dtype)
    noisy_latents = t_view * noise + (1.0 - t_view) * target_latents
    flow_target = noise.astype(mx.float32) - target_latents.astype(mx.float32)

    decoder_outputs = decoder(
        hidden_states=noisy_latents,
        timestep=t,
        timestep_r=r,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        context_latents=context_latents,
        cache=None,
        use_cache=False,
    )
    prediction = _first_decoder_output(decoder_outputs)
    return flow_matching_loss_from_prediction(
        prediction,
        flow_target,
        t,
        loss_weighting=loss_weighting,
        snr_gamma=config.snr_gamma,
    )


def create_adamw_optimizer(config: ACEAdamWConfig | None = None):
    """Create the AdamW optimizer used for ACE MLX LoRA training."""
    _require_mlx()
    config = config or ACEAdamWConfig()
    return optim.AdamW(
        learning_rate=float(config.learning_rate),
        betas=list(config.betas),
        eps=float(config.eps),
        weight_decay=float(config.weight_decay),
    )


def ace_adamw_update_step(
    model,
    optimizer,
    batch: dict[str, tp.Any],
    *,
    null_condition_emb=None,
    config: ACEFlowMatchingConfig | None = None,
    noise=None,
    timesteps=None,
):
    """Run one AdamW update step for an ACE MLX decoder/module."""
    _require_mlx()
    config = config or ACEFlowMatchingConfig()

    def loss_fn(local_model):
        return ace_flow_matching_loss(
            local_model,
            batch,
            null_condition_emb=null_condition_emb,
            config=config,
            noise=noise,
            timesteps=timesteps,
        )

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    loss, grads = loss_and_grad(model)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state, loss)
    return loss


def _first_decoder_output(decoder_outputs):
    if isinstance(decoder_outputs, (tuple, list)):
        return decoder_outputs[0]
    return decoder_outputs


def _canonical_loss_weighting(loss_weighting: str) -> str:
    value = str(loss_weighting or LOSS_WEIGHTING_NONE).strip().lower()
    if value in LOSS_WEIGHTING_CHOICES:
        return value
    raise ValueError(
        f"Unsupported ACE loss weighting {loss_weighting!r}; "
        f"expected one of {LOSS_WEIGHTING_CHOICES}."
    )


def _require_mlx() -> None:
    if mx is None or nn is None or optim is None:
        raise RuntimeError("MLX is required for ACE MLX training helpers.")
