#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from stable_audio_3.mlx.dit import StableAudioMLXDiT  # noqa: E402
from stable_audio_3.mlx.pipeline import StableAudioMLXPipeline  # noqa: E402
from stable_audio_3.mlx.smoke import build_dummy_dit_smoke_inputs  # noqa: E402
from stable_audio_3.mlx.training import (  # noqa: E402
    inject_trainable_lora,
    rectified_flow_loss,
    save_trainable_lora,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run one MLX LoRA gradient step through the real SA3 DiT structure "
            "using synthetic latents and conditioning."
        )
    )
    parser.add_argument("--model", default="medium")
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument(
        "--adapter-type",
        choices=(
            "lora",
            "dora",
            "dora-rows",
            "bora",
            "lora-xs",
            "dora-rows-xs",
            "dora-cols-xs",
            "bora-xs",
        ),
        default="dora",
        help="Train LoRA, DoRA, BoRA, or their extra-small SVD variants.",
    )
    parser.add_argument("--latent-length", type=int, default=64)
    parser.add_argument(
        "--include",
        action="append",
        default=["transformer.layers.23.self_attn.to_out"],
        help="Layer-name substring to adapt. Repeat for multiple filters.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("mlx_sa3_lora_gradient_smoke.safetensors"),
    )
    args = parser.parse_args()

    pipeline = StableAudioMLXPipeline.from_pretrained_config(args.model)
    model = StableAudioMLXDiT.from_sao_model_config(
        pipeline.model_config,
        param_dtype=mx.float16,
    )
    report = inject_trainable_lora(
        model,
        rank=args.rank,
        alpha=args.rank,
        include=args.include,
        adapter_type=args.adapter_type,
    )
    inputs = build_dummy_dit_smoke_inputs(
        pipeline.model_config,
        batch_size=1,
        latent_length=args.latent_length,
        dtype_name="float16",
    )
    model_kwargs = {
        "cross_attn_cond": inputs.cross_attn_cond,
        "global_embed": inputs.global_embed,
        "local_add_cond": inputs.local_add_cond,
        "cfg_scale": 1.0,
    }
    if model.diffusion_objective == "rectified_flow":
        model_kwargs["padding_mask"] = inputs.padding_mask

    noise = mx.random.normal(inputs.x.shape, dtype=inputs.x.dtype)

    def loss_fn(local_model, clean, timesteps):
        return rectified_flow_loss(
            local_model,
            clean,
            timesteps,
            noise=noise,
            model_kwargs=model_kwargs,
        )

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    optimizer = optim.AdamW(learning_rate=args.learning_rate)
    loss, grads = loss_and_grad(model, inputs.x, inputs.t)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state, loss)

    checkpoint = save_trainable_lora(
        model,
        args.output,
        rank=args.rank,
        alpha=args.rank,
        include=args.include,
        adapter_type=args.adapter_type,
        extra_metadata={
            "step": 1,
            "base_model": args.model,
            "smoke_test": True,
        },
    )
    print(f"model={args.model}")
    print(f"adapter_type={report.adapter_type}")
    print(f"objective={model.diffusion_objective}")
    print(f"layers={report.layer_count}")
    print(f"trainable_parameters={report.trainable_parameters}")
    print(f"loss={float(loss):.8f}")
    print(f"finite={bool(mx.isfinite(loss))}")
    print(f"checkpoint={checkpoint}")


if __name__ == "__main__":
    main()
