"""Runnable ACE-Step MLX LoRA/DoRA training job helpers."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import typing as tp
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from acestep.models.mlx.lora_training import (
    BALANCED_PROFILE,
    inject_trainable_lora,
    save_trainable_lora_adapter,
)
from acestep.models.mlx.training_core import (
    ACEAdamWConfig,
    ACEFlowMatchingConfig,
    ace_apply_gradients,
    ace_flow_matching_loss,
    ace_loss_and_grad,
    create_adamw_optimizer,
)

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_map
except ModuleNotFoundError:  # pragma: no cover - exercised by non-MLX shells.
    mx = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    tree_map = None  # type: ignore[assignment]


@dataclass(frozen=True)
class TensorExampleGroup:
    """One logical ACE training row, with an optional genre tensor variant."""

    path: Path
    genre_path: Path | None = None


def parse_args(argv: tp.Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ACE-Step LoRA/DoRA adapters with native MLX."
    )
    parser.add_argument("--tensor-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--model-variant", default="base")
    parser.add_argument("--fake-decoder", action="store_true")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--alpha", type=int)
    parser.add_argument("--adapter-type", choices=("lora", "dora"), default="dora")
    parser.add_argument(
        "--module-profile",
        choices=("attention", "balanced"),
        default=BALANCED_PROFILE,
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--lr-scheduler",
        choices=("cosine", "linear", "constant"),
        default="cosine",
    )
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--save-best", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-best-after", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--cfg-ratio", type=float, default=0.15)
    parser.add_argument("--loss-weighting", choices=("none", "min_snr"), default="none")
    parser.add_argument("--snr-gamma", type=float, default=5.0)
    parser.add_argument("--timestep-mu", type=float, default=-0.4)
    parser.add_argument("--timestep-sigma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument(
        "--artifact-storage-upcast",
        action="store_true",
        help="Load an FP16 artifact, then upcast its floating weights for BF16 training.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--memory-limit-gb", type=float, default=0.0)
    parser.add_argument("--allow-unsafe-xl", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--cancel-path", type=Path)
    return parser.parse_args(argv)


def load_tensor_example_groups(tensor_dir: Path) -> tuple[list[TensorExampleGroup], dict[str, tp.Any]]:
    """Load ACE preprocessed tensor paths from ``manifest.json`` or directory scan."""
    tensor_dir = tensor_dir.expanduser().resolve()
    manifest_path = tensor_dir / "manifest.json"
    metadata: dict[str, tp.Any] = {}
    groups: list[TensorExampleGroup] = []

    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metadata = manifest.get("metadata") if isinstance(manifest.get("metadata"), dict) else {}
        sample_groups = manifest.get("sample_groups")
        if isinstance(sample_groups, list) and sample_groups:
            for item in sample_groups:
                if not isinstance(item, dict) or not item.get("path"):
                    continue
                groups.append(
                    TensorExampleGroup(
                        path=_resolve_tensor_path(tensor_dir, item["path"]),
                        genre_path=(
                            _resolve_tensor_path(tensor_dir, item["genre_path"])
                            if item.get("genre_path")
                            else None
                        ),
                    )
                )
        else:
            samples = manifest.get("samples", [])
            for item in samples:
                if isinstance(item, str) and item:
                    groups.append(TensorExampleGroup(path=_resolve_tensor_path(tensor_dir, item)))
    else:
        groups = [
            TensorExampleGroup(path=path)
            for path in sorted(tensor_dir.glob("*.pt"))
            if not path.name.endswith(".tmp.pt")
        ]

    groups = [group for group in groups if group.path.is_file()]
    if not groups:
        raise FileNotFoundError(f"No preprocessed ACE tensor files found in {tensor_dir}.")
    return groups, metadata


def select_epoch_tensor_paths(
    groups: tp.Sequence[TensorExampleGroup],
    *,
    epoch: int,
    metadata: dict[str, tp.Any],
    seed: int,
) -> list[Path]:
    """Select caption or genre tensors for one epoch, then shuffle order."""
    rng = np.random.default_rng(int(seed) + int(epoch))
    paths = [group.path for group in groups]

    genre_ratio = int(metadata.get("target_genre_ratio", metadata.get("genre_ratio", 0)) or 0)
    genre_ratio = max(0, min(100, genre_ratio))
    eligible = [index for index, group in enumerate(groups) if group.genre_path is not None]
    if genre_ratio > 0 and eligible:
        count = int(len(eligible) * genre_ratio / 100)
        if count == 0:
            count = 1
        count = min(count, len(eligible))
        selected = set(rng.choice(eligible, size=count, replace=False).tolist())
        for index in selected:
            genre_path = groups[index].genre_path
            if genre_path is not None:
                paths[index] = genre_path

    order = rng.permutation(len(paths))
    return [paths[int(index)] for index in order]


def resolve_tensor_batch_cache_path(
    path: Path,
    *,
    tensor_dir: Path,
    cache_root: Path,
) -> Path:
    """Return the MLX cache path for a preprocessed ACE tensor file."""
    relative = path.expanduser().resolve().relative_to(tensor_dir.expanduser().resolve())
    cache_name = relative.name.removesuffix(relative.suffix) + ".mlx.npz"
    return cache_root / relative.parent / cache_name


def _load_cached_tensor_batch(path: Path) -> dict[str, tp.Any]:
    """Load one cached MLX batch created via ``mx.savez``."""
    _require_mlx()
    batch = mx.load(str(path))
    expected = {
        "target_latents",
        "attention_mask",
        "encoder_hidden_states",
        "encoder_attention_mask",
        "context_latents",
    }
    missing = expected.difference(batch)
    if missing:
        raise ValueError(f"Cached MLX batch {path} is missing keys: {sorted(missing)}")
    return {key: batch[key] for key in expected}


def _write_cached_tensor_batch(path: Path, batch: dict[str, tp.Any]) -> None:
    """Persist one MLX batch atomically for reuse across epochs."""
    _require_mlx()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.stem + ".tmp" + path.suffix)
    try:
        mx.savez(str(temp_path), **batch)
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def load_tensor_batch(
    path: Path,
    *,
    dtype: tp.Any,
    tensor_dir: Path | None = None,
    cache_root: Path | None = None,
) -> dict[str, tp.Any]:
    """Load one preprocessed ``.pt`` tensor file into MLX batch arrays."""
    _require_mlx()
    if tensor_dir is not None and cache_root is not None:
        cache_path = resolve_tensor_batch_cache_path(
            path,
            tensor_dir=tensor_dir,
            cache_root=cache_root,
        )
        try:
            if cache_path.is_file() and cache_path.stat().st_mtime_ns >= path.stat().st_mtime_ns:
                return _load_cached_tensor_batch(cache_path)
        except Exception:
            try:
                cache_path.unlink(missing_ok=True)
            except OSError:
                pass

    data = torch.load(str(path), map_location="cpu", weights_only=False)
    encoder_attention_mask = _torch_to_mx_batch(
        data["encoder_attention_mask"],
        dtype=mx.float32,
    )
    encoder_hidden_states = _torch_to_mx_batch(data["encoder_hidden_states"], dtype=dtype)
    encoder_hidden_states = _sanitize_encoder_hidden_states(
        encoder_hidden_states,
        encoder_attention_mask,
    )
    batch = {
        "target_latents": _torch_to_mx_batch(data["target_latents"], dtype=dtype),
        "attention_mask": _torch_to_mx_batch(data["attention_mask"], dtype=mx.float32),
        "encoder_hidden_states": encoder_hidden_states,
        "encoder_attention_mask": encoder_attention_mask,
        "context_latents": _torch_to_mx_batch(data["context_latents"], dtype=dtype),
    }
    if tensor_dir is not None and cache_root is not None:
        _write_cached_tensor_batch(cache_path, batch)
    return batch


def build_mlx_decoder(args: argparse.Namespace):
    """Build an MLX decoder directly from sharded safetensors."""
    _require_mlx()
    if args.fake_decoder:
        decoder = _TinyAceTrainingDecoder(hidden_size=2)
        null_condition_emb = mx.zeros((1, 1, 2), dtype=_mlx_dtype(args.dtype))
        return decoder, null_condition_emb, None

    if args.checkpoint_dir is None:
        raise ValueError("--checkpoint-dir is required unless --fake-decoder is set.")

    from acestep.models.mlx.dit_convert import (
        load_decoder_safetensors,
        load_model_config,
        resolve_model_dir,
    )
    from acestep.models.mlx.dit_model import MLXDiTDecoder

    model_dir = resolve_model_dir(args.checkpoint_dir, args.model_variant)
    config = load_model_config(model_dir)
    decoder = MLXDiTDecoder.from_config(config)
    decoder.gradient_checkpointing = _resolve_gradient_checkpointing(args)
    from acestep.models.mlx.training_artifact import (
        load_training_artifact_if_compatible,
    )

    artifact_result = load_training_artifact_if_compatible(
        variant=args.model_variant,
        training_dtype=args.dtype,
        decoder=decoder,
        allow_storage_upcast=getattr(args, "artifact_storage_upcast", False),
    )
    if artifact_result is not None:
        null_condition_emb, artifact_path = artifact_result
        if getattr(args, "artifact_storage_upcast", False):
            target_dtype = _mlx_dtype(args.dtype)
            decoder.set_dtype(target_dtype)
            null_condition_emb = null_condition_emb.astype(target_dtype)
            mx.eval(decoder.parameters(), null_condition_emb)
            print("model_storage_dtype=fp16", flush=True)
            print(f"model_compute_dtype={args.dtype}", flush=True)
        print("model_loader=pinned_mlx_artifact", flush=True)
        print(f"model_artifact={artifact_path}", flush=True)
        return decoder, null_condition_emb, str(model_dir)
    null_condition_emb, report = load_decoder_safetensors(
        model_dir,
        decoder,
        dtype=_mlx_dtype(args.dtype),
    )
    print("model_loader=direct_safetensors", flush=True)
    print(f"model_tensors={report.tensor_count}", flush=True)
    print(f"model_parameters={report.parameter_count}", flush=True)
    print(f"model_shards={report.shard_count}", flush=True)
    return decoder, null_condition_emb, str(model_dir)


def run_training(args: argparse.Namespace) -> Path:
    """Run ACE MLX adapter training and return the final adapter directory."""
    _require_mlx()
    if args.batch_size != 1:
        raise ValueError("ACE MLX training currently supports --batch-size 1.")
    if args.gradient_accumulation <= 0:
        raise ValueError("--gradient-accumulation must be greater than zero.")
    if args.epochs <= 0:
        raise ValueError("--epochs must be greater than zero.")
    if args.save_best_after <= 0:
        raise ValueError("--save-best-after must be greater than zero.")
    if args.memory_limit_gb < 0:
        raise ValueError("--memory-limit-gb must be non-negative.")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative.")
    if args.max_grad_norm <= 0:
        raise ValueError("--max-grad-norm must be greater than zero.")

    physical_memory = _physical_memory_bytes()
    is_xl_model = _is_xl_variant(args.model_variant)
    if is_xl_model and physical_memory <= 40 * 1024**3 and not args.allow_unsafe_xl:
        raise RuntimeError(
            "XL ACE training is temporarily blocked on systems with 40 GiB or less "
            "to prevent another watchdog reset. Use the regular base model."
        )

    tensor_dir = args.tensor_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    groups, metadata = load_tensor_example_groups(tensor_dir)
    dtype = _mlx_dtype(args.dtype)
    mx.random.seed(int(args.seed))
    memory_limit = _configure_mlx_memory(
        physical_memory=physical_memory,
        requested_gb=_resolve_memory_limit_gb(args, physical_memory=physical_memory),
    )
    _print_mlx_memory("configured")

    decoder, null_condition_emb, base_model_path = build_mlx_decoder(args)
    _print_mlx_memory("model_loaded")
    injection = inject_trainable_lora(
        decoder,
        rank=args.rank,
        alpha=args.alpha,
        module_profile=args.module_profile,
        adapter_type=args.adapter_type,
    )
    mx.eval(decoder.parameters())
    _clear_mlx_cache()
    _print_mlx_memory("adapters_injected")

    total_steps = len(groups) * args.epochs
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    optimizer_steps = _optimizer_step_count(
        example_count=len(groups),
        epochs=args.epochs,
        gradient_accumulation=args.gradient_accumulation,
        max_steps=args.max_steps,
    )
    optimizer = create_adamw_optimizer(
        ACEAdamWConfig(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            scheduler=args.lr_scheduler,
            warmup_steps=args.warmup_steps,
            total_steps=optimizer_steps,
        )
    )
    flow_config = ACEFlowMatchingConfig(
        cfg_ratio=args.cfg_ratio,
        timestep_mu=args.timestep_mu,
        timestep_sigma=args.timestep_sigma,
        loss_weighting=args.loss_weighting,
        snr_gamma=args.snr_gamma,
    )
    batch_cache_dir = tensor_dir / ".mlx-cache" / args.dtype

    run_config = {
        "tensor_dir": str(tensor_dir),
        "batch_cache_dir": str(batch_cache_dir),
        "rank": args.rank,
        "alpha": args.alpha if args.alpha is not None else args.rank * 2,
        "adapter_type": injection.adapter_type,
        "module_profile": injection.module_profile,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "lr_scheduler": args.lr_scheduler,
        "warmup_steps": args.warmup_steps,
        "max_grad_norm": args.max_grad_norm,
        "gradient_accumulation": args.gradient_accumulation,
        "optimizer_steps": optimizer_steps,
        "epochs": args.epochs,
        "save_every": args.save_every,
        "save_best": args.save_best,
        "save_best_after": args.save_best_after,
        "cfg_ratio": args.cfg_ratio,
        "loss_weighting": args.loss_weighting,
        "snr_gamma": args.snr_gamma,
        "timestep_mu": args.timestep_mu,
        "timestep_sigma": args.timestep_sigma,
        "seed": args.seed,
        "dtype": args.dtype,
        "gradient_checkpointing": _resolve_gradient_checkpointing(args),
        "memory_limit_bytes": memory_limit,
        "example_count": len(groups),
        "trainable_parameters": injection.trainable_parameters,
        "adapted_layers": list(injection.layer_names),
    }
    (output_dir / "run.json").write_text(json.dumps(run_config, indent=2) + "\n")

    print(f"example_count={len(groups)}", flush=True)
    print(f"adapter_type={injection.adapter_type}", flush=True)
    print(f"module_profile={injection.module_profile}", flush=True)
    print(f"adapted_layers={injection.layer_count}", flush=True)
    print(f"trainable_parameters={injection.trainable_parameters}", flush=True)
    print(f"batch_cache_dir={batch_cache_dir}", flush=True)

    global_step = 0
    optimizer_step = 0
    best_loss = math.inf
    best_dir = output_dir / "best"
    loss_log_path = output_dir / "loss.jsonl"
    started = time.perf_counter()
    with loss_log_path.open("a", encoding="utf-8") as loss_log:
        for epoch in range(args.epochs):
            epoch_paths = select_epoch_tensor_paths(
                groups,
                epoch=epoch,
                metadata=metadata,
                seed=args.seed,
            )
            accumulated_grads = None
            accumulated_loss = 0.0
            accumulation_count = 0
            for sample_index, tensor_path in enumerate(epoch_paths):
                if args.cancel_path is not None and args.cancel_path.exists():
                    raise KeyboardInterrupt("Training cancelled.")
                if args.max_steps > 0 and global_step >= args.max_steps:
                    break

                batch = load_tensor_batch(
                    tensor_path,
                    dtype=dtype,
                    tensor_dir=tensor_dir,
                    cache_root=batch_cache_dir,
                )
                _print_mlx_memory("step_start")
                step_started = time.perf_counter()
                loss, grads = ace_loss_and_grad(
                    decoder,
                    batch,
                    null_condition_emb=null_condition_emb,
                    config=flow_config,
                )
                global_step += 1
                mx.eval(loss, grads)
                loss_value = float(loss)
                if not math.isfinite(loss_value):
                    raise FloatingPointError(
                        f"Non-finite loss at step {global_step}: {loss_value}"
                    )
                step_seconds = time.perf_counter() - step_started
                accumulated_grads = (
                    grads
                    if accumulated_grads is None
                    else tree_map(lambda left, right: left + right, accumulated_grads, grads)
                )
                mx.eval(accumulated_grads)
                accumulation_count += 1
                accumulated_loss += loss_value
                reached_limit = args.max_steps > 0 and global_step >= args.max_steps
                end_of_epoch = sample_index + 1 == len(epoch_paths)
                should_update = (
                    accumulation_count >= args.gradient_accumulation
                    or end_of_epoch
                    or reached_limit
                )
                grad_norm_value = None
                update_loss = None
                if should_update:
                    averaged_grads = tree_map(
                        lambda value: value / float(args.gradient_accumulation),
                        accumulated_grads,
                    )
                    grad_norm = ace_apply_gradients(
                        decoder,
                        optimizer,
                        averaged_grads,
                        max_grad_norm=args.max_grad_norm,
                    )
                    mx.eval(grad_norm)
                    grad_norm_value = float(grad_norm)
                    update_loss = accumulated_loss / float(accumulation_count)
                    optimizer_step += 1
                    accumulated_grads = None
                    accumulated_loss = 0.0
                    accumulation_count = 0
                loss_log.write(
                    json.dumps(
                        {
                            "step": global_step,
                            "total_steps": total_steps,
                            "epoch": epoch + 1,
                            "loss": loss_value,
                            "optimizer_step": optimizer_step,
                            "optimizer_updated": should_update,
                            "update_loss": update_loss,
                            "grad_norm": grad_norm_value,
                            "seconds": step_seconds,
                            "tensor_path": str(tensor_path),
                        }
                    )
                    + "\n"
                )
                loss_log.flush()
                print(
                    f"step={global_step}/{total_steps} loss={loss_value:.6f} "
                    f"epoch={epoch + 1}/{args.epochs} sample={tensor_path.name}",
                    flush=True,
                )
                _print_mlx_memory("step_complete")

                if (
                    should_update
                    and args.save_best
                    and epoch + 1 >= args.save_best_after
                    and update_loss is not None
                    and update_loss < best_loss
                ):
                    best_loss = update_loss
                    save_trainable_lora_adapter(
                        decoder,
                        best_dir,
                        rank=args.rank,
                        alpha=args.alpha,
                        module_profile=args.module_profile,
                        adapter_type=args.adapter_type,
                        base_model_name_or_path=base_model_path,
                    )
                    print(
                        f"best_checkpoint={best_dir} loss={best_loss:.6f}",
                        flush=True,
                    )
                del batch, loss, grads
                _clear_mlx_cache()

            if (
                args.save_every > 0
                and (epoch + 1) % args.save_every == 0
                and global_step > 0
            ):
                checkpoint_dir = output_dir / f"checkpoint-epoch-{epoch + 1:06d}"
                save_trainable_lora_adapter(
                    decoder,
                    checkpoint_dir,
                    rank=args.rank,
                    alpha=args.alpha,
                    module_profile=args.module_profile,
                    adapter_type=args.adapter_type,
                    base_model_name_or_path=base_model_path,
                )
                print(f"checkpoint={checkpoint_dir}", flush=True)
            if args.max_steps > 0 and global_step >= args.max_steps:
                break
            _clear_mlx_cache()

    final_dir = output_dir / "final"
    save_trainable_lora_adapter(
        decoder,
        final_dir,
        rank=args.rank,
        alpha=args.alpha,
        module_profile=args.module_profile,
        adapter_type=args.adapter_type,
        base_model_name_or_path=base_model_path,
    )
    elapsed = time.perf_counter() - started
    print(f"training_seconds={elapsed:.3f}", flush=True)
    print(f"final_checkpoint={final_dir}", flush=True)
    return final_dir


def _optimizer_step_count(
    *,
    example_count: int,
    epochs: int,
    gradient_accumulation: int,
    max_steps: int,
) -> int:
    """Count optimizer updates, including each epoch's partial accumulation."""
    available_microsteps = max(0, int(example_count)) * max(0, int(epochs))
    microsteps = (
        min(available_microsteps, int(max_steps))
        if int(max_steps) > 0
        else available_microsteps
    )
    if microsteps <= 0:
        return 1
    full_epochs, partial_epoch = divmod(microsteps, int(example_count))
    updates_per_epoch = math.ceil(int(example_count) / int(gradient_accumulation))
    return full_epochs * updates_per_epoch + math.ceil(
        partial_epoch / int(gradient_accumulation)
    )


def main(argv: tp.Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.tensor_dir = args.tensor_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.checkpoint_dir = (
        args.checkpoint_dir.expanduser().resolve()
        if args.checkpoint_dir is not None
        else None
    )
    args.cancel_path = (
        args.cancel_path.expanduser().resolve()
        if args.cancel_path is not None
        else None
    )
    try:
        run_training(args)
    except KeyboardInterrupt as exc:
        print(f"cancelled={exc}", file=sys.stderr, flush=True)
        return 130
    return 0


class _TinyAttention(nn.Module if nn is not None else object):  # type: ignore[misc]
    def __init__(self, hidden_size: int):
        _require_mlx()
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)


class _TinyMLP(nn.Module if nn is not None else object):  # type: ignore[misc]
    def __init__(self, hidden_size: int):
        _require_mlx()
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.down_proj = nn.Linear(hidden_size, hidden_size, bias=False)


class _TinyBlock(nn.Module if nn is not None else object):  # type: ignore[misc]
    def __init__(self, hidden_size: int):
        _require_mlx()
        super().__init__()
        self.self_attn = _TinyAttention(hidden_size)
        self.cross_attn = _TinyAttention(hidden_size)
        self.mlp = _TinyMLP(hidden_size)


class _TinyAceTrainingDecoder(nn.Module if nn is not None else object):  # type: ignore[misc]
    def __init__(self, hidden_size: int):
        _require_mlx()
        super().__init__()
        self.layers = [_TinyBlock(hidden_size)]

    def __call__(
        self,
        *,
        hidden_states,
        timestep,
        timestep_r,
        encoder_hidden_states,
        context_latents,
        encoder_attention_mask=None,
        cache=None,
        use_cache=False,
    ):
        block = self.layers[0]
        outputs = [
            block.self_attn.q_proj(hidden_states),
            block.self_attn.k_proj(hidden_states),
            block.self_attn.v_proj(hidden_states),
            block.self_attn.o_proj(hidden_states),
            block.cross_attn.q_proj(hidden_states),
            block.cross_attn.k_proj(hidden_states),
            block.cross_attn.v_proj(hidden_states),
            block.cross_attn.o_proj(hidden_states),
            block.mlp.down_proj(block.mlp.gate_proj(hidden_states)),
            block.mlp.down_proj(block.mlp.up_proj(hidden_states)),
        ]
        prediction = sum(outputs) / float(len(outputs))
        return prediction, None


def _resolve_tensor_path(tensor_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = tensor_dir / path
    return path.expanduser().resolve()


def _torch_to_mx_batch(tensor, *, dtype):
    array = _torch_to_mx_array(tensor, dtype=dtype)
    if len(array.shape) == 2:
        array = array[None, :, :]
    elif len(array.shape) == 1:
        array = array[None, :]
    return array


def _torch_to_mx_array(tensor, *, dtype):
    _require_mlx()
    if hasattr(tensor, "detach"):
        tensor = tensor.detach().cpu().float().numpy()
    return mx.array(tensor).astype(dtype)


def _sanitize_encoder_hidden_states(encoder_hidden_states, encoder_attention_mask):
    _require_mlx()
    if len(encoder_attention_mask.shape) == 1:
        encoder_attention_mask = encoder_attention_mask[None, :]
    if encoder_attention_mask.shape[0] != encoder_hidden_states.shape[0]:
        raise ValueError(
            "encoder_attention_mask batch size does not match encoder_hidden_states"
        )
    if encoder_attention_mask.shape[1] != encoder_hidden_states.shape[1]:
        raise ValueError(
            "encoder_attention_mask length does not match encoder_hidden_states"
        )

    zeros = mx.zeros_like(encoder_hidden_states)
    encoder_hidden_states = mx.where(
        mx.isfinite(encoder_hidden_states),
        encoder_hidden_states,
        zeros,
    )
    valid = (encoder_attention_mask > 0)[:, :, None]
    return mx.where(valid, encoder_hidden_states, zeros)


def _mlx_dtype(name: str):
    _require_mlx()
    if name == "fp16":
        return mx.float16
    if name == "bf16":
        return mx.bfloat16
    return mx.float32


def _is_xl_variant(variant: str) -> bool:
    return "xl" in str(variant).lower()


def _resolve_gradient_checkpointing(args: argparse.Namespace) -> bool:
    explicit = getattr(args, "gradient_checkpointing", None)
    if explicit is not None:
        return bool(explicit)
    return True


def _resolve_memory_limit_gb(
    args: argparse.Namespace,
    *,
    physical_memory: int,
) -> float:
    requested = float(getattr(args, "memory_limit_gb", 0.0) or 0.0)
    if requested > 0:
        return requested
    if _is_xl_variant(args.model_variant) and physical_memory <= 40 * 1024**3:
        return 20.0
    return 0.0


def _physical_memory_bytes() -> int:
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (OSError, ValueError):
        return 32 * 1024**3


def _configure_mlx_memory(*, physical_memory: int, requested_gb: float) -> int:
    _require_mlx()
    if requested_gb > 0:
        limit = int(requested_gb * 1024**3)
    else:
        limit = int(min(physical_memory * 0.75, max(8 * 1024**3, physical_memory - 8 * 1024**3)))
    _set_mlx_memory_limit(limit)
    _set_mlx_cache_limit(min(512 * 1024**2, max(64 * 1024**2, limit // 16)))
    _reset_mlx_peak_memory()
    print(f"memory_limit_gb={limit / 1024**3:.2f}", flush=True)
    return limit


def _print_mlx_memory(stage: str) -> None:
    _require_mlx()
    active = _get_mlx_active_memory()
    cache = _get_mlx_cache_memory()
    peak = _get_mlx_peak_memory()
    print(
        f"memory stage={stage} active_gb={active / 1024**3:.3f} "
        f"cache_gb={cache / 1024**3:.3f} peak_gb={peak / 1024**3:.3f}",
        flush=True,
    )


def _clear_mlx_cache() -> None:
    clear_cache = getattr(mx, "clear_cache", None)
    if clear_cache is not None:
        clear_cache()
    else:
        mx.metal.clear_cache()


def _set_mlx_memory_limit(limit: int) -> None:
    setter = getattr(mx, "set_memory_limit", None)
    if setter is not None:
        setter(limit)
    else:
        mx.metal.set_memory_limit(limit)


def _set_mlx_cache_limit(limit: int) -> None:
    setter = getattr(mx, "set_cache_limit", None)
    if setter is not None:
        setter(limit)
    else:
        mx.metal.set_cache_limit(limit)


def _reset_mlx_peak_memory() -> None:
    resetter = getattr(mx, "reset_peak_memory", None)
    if resetter is not None:
        resetter()
    else:
        mx.metal.reset_peak_memory()


def _get_mlx_active_memory() -> int:
    getter = getattr(mx, "get_active_memory", None)
    return int(getter() if getter is not None else mx.metal.get_active_memory())


def _get_mlx_cache_memory() -> int:
    getter = getattr(mx, "get_cache_memory", None)
    return int(getter() if getter is not None else mx.metal.get_cache_memory())


def _get_mlx_peak_memory() -> int:
    getter = getattr(mx, "get_peak_memory", None)
    return int(getter() if getter is not None else mx.metal.get_peak_memory())


def _require_mlx() -> None:
    if mx is None or nn is None:
        raise RuntimeError("MLX is required for ACE MLX training jobs.")
