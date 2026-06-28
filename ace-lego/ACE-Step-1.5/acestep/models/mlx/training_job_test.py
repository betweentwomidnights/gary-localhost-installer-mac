"""Tests for the ACE MLX training job runner."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

pytest.importorskip("mlx.core")

import mlx.core as mx  # noqa: E402

from acestep.models.mlx import training_job  # noqa: E402


def test_model_family_defaults_are_resolved_from_variant() -> None:
    base_args = training_job.parse_args(
        [
            "--tensor-dir",
            "/tmp/tensors",
            "--output-dir",
            "/tmp/output",
            "--fake-decoder",
            "--model-variant",
            "base",
        ]
    )
    xl_args = training_job.parse_args(
        [
            "--tensor-dir",
            "/tmp/tensors",
            "--output-dir",
            "/tmp/output",
            "--fake-decoder",
            "--model-variant",
            "xl-base",
        ]
    )

    assert training_job._resolve_gradient_checkpointing(base_args) is True
    assert training_job._resolve_gradient_checkpointing(xl_args) is True
    assert (
        training_job._resolve_memory_limit_gb(
            xl_args,
            physical_memory=32 * 1024**3,
        )
        == 20.0
    )


def _write_tensor(path: Path, *, value: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "target_latents": torch.full((4, 2), value),
            "attention_mask": torch.ones(4),
            "encoder_hidden_states": torch.zeros(1, 2),
            "encoder_attention_mask": torch.ones(1),
            "context_latents": torch.zeros(4, 2),
            "metadata": {"caption": path.stem},
        },
        path,
    )


def test_load_tensor_example_groups_supports_pc_variant_manifest(tmp_path: Path) -> None:
    _write_tensor(tmp_path / "song.pt")
    _write_tensor(tmp_path / "song.genre.pt")
    _write_tensor(tmp_path / "plain.pt")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "metadata": {"target_genre_ratio": 20},
                "samples": ["song.pt", "plain.pt"],
                "sample_groups": [
                    {"path": "song.pt", "genre_path": "song.genre.pt"},
                    {"path": "plain.pt"},
                ],
            }
        )
    )

    groups, metadata = training_job.load_tensor_example_groups(tmp_path)
    selected = training_job.select_epoch_tensor_paths(
        groups,
        epoch=0,
        metadata=metadata,
        seed=11,
    )

    assert metadata["target_genre_ratio"] == 20
    assert len(groups) == 2
    assert groups[0].path == (tmp_path / "song.pt").resolve()
    assert groups[0].genre_path == (tmp_path / "song.genre.pt").resolve()
    assert (tmp_path / "song.genre.pt").resolve() in selected


def test_load_tensor_batch_sanitizes_encoder_hidden_states(tmp_path: Path) -> None:
    tensor_path = tmp_path / "sample.pt"
    torch.save(
        {
            "target_latents": torch.zeros(4, 2),
            "attention_mask": torch.ones(4),
            "encoder_hidden_states": torch.tensor(
                [
                    [1.0, float("nan")],
                    [2.0, 3.0],
                    [float("inf"), 5.0],
                ],
                dtype=torch.float32,
            ),
            "encoder_attention_mask": torch.tensor([1.0, 1.0, 0.0]),
            "context_latents": torch.zeros(4, 2),
        },
        tensor_path,
    )

    batch = training_job.load_tensor_batch(tensor_path, dtype=mx.float32)

    assert batch["target_latents"].shape == (1, 4, 2)
    assert batch["encoder_hidden_states"].shape == (1, 3, 2)
    assert batch["encoder_attention_mask"].shape == (1, 3)
    assert bool(mx.all(mx.isfinite(batch["encoder_hidden_states"])))
    assert mx.allclose(
        batch["encoder_hidden_states"],
        mx.array([[[1.0, 0.0], [2.0, 3.0], [0.0, 0.0]]], dtype=mx.float32),
    )


def test_load_tensor_batch_reuses_mlx_cache_when_source_is_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tensor_path = tmp_path / "nested" / "sample.pt"
    _write_tensor(tensor_path, value=0.5)
    cache_root = tmp_path / ".mlx-cache" / "bf16"

    first = training_job.load_tensor_batch(
        tensor_path,
        dtype=mx.bfloat16,
        tensor_dir=tmp_path,
        cache_root=cache_root,
    )
    cache_path = training_job.resolve_tensor_batch_cache_path(
        tensor_path,
        tensor_dir=tmp_path,
        cache_root=cache_root,
    )

    assert cache_path.is_file()
    assert first["target_latents"].dtype == mx.bfloat16

    def _unexpected_torch_load(*_args, **_kwargs):
        raise AssertionError("torch.load should not run when the MLX cache is fresh")

    monkeypatch.setattr(training_job.torch, "load", _unexpected_torch_load)
    second = training_job.load_tensor_batch(
        tensor_path,
        dtype=mx.bfloat16,
        tensor_dir=tmp_path,
        cache_root=cache_root,
    )

    assert mx.array_equal(first["target_latents"], second["target_latents"])
    assert mx.array_equal(first["encoder_hidden_states"], second["encoder_hidden_states"])
    assert second["target_latents"].dtype == mx.bfloat16


def test_fake_decoder_training_job_writes_peft_adapter(tmp_path: Path, capsys) -> None:
    tensor_dir = tmp_path / "tensors"
    _write_tensor(tensor_dir / "a.pt", value=0.0)
    _write_tensor(tensor_dir / "b.pt", value=0.25)
    (tensor_dir / "manifest.json").write_text(
        json.dumps({"samples": ["a.pt", "b.pt"], "num_samples": 2})
    )
    output_dir = tmp_path / "run"

    code = training_job.main(
        [
            "--tensor-dir",
            str(tensor_dir),
            "--output-dir",
            str(output_dir),
            "--fake-decoder",
            "--rank",
            "1",
            "--alpha",
            "2",
            "--adapter-type",
            "lora",
            "--module-profile",
            "balanced",
            "--learning-rate",
            "0.05",
            "--epochs",
            "2",
            "--save-every",
            "2",
            "--save-best-after",
            "1",
            "--cfg-ratio",
            "0",
            "--loss-weighting",
            "none",
            "--seed",
            "17",
        ]
    )
    captured = capsys.readouterr()

    assert code == 0
    assert "trainable_parameters=" in captured.out
    assert "step=4/4" in captured.out
    assert f"best_checkpoint={output_dir / 'best'}" in captured.out
    assert f"final_checkpoint={output_dir / 'final'}" in captured.out
    assert (output_dir / "run.json").is_file()
    assert (output_dir / "loss.jsonl").is_file()
    assert (output_dir / "checkpoint-epoch-000002" / "adapter_model.safetensors").is_file()
    assert (output_dir / "best" / "adapter_model.safetensors").is_file()
    assert (output_dir / "final" / "adapter_model.safetensors").is_file()
    run_config = json.loads((output_dir / "run.json").read_text())
    assert run_config["weight_decay"] == 0.01
    assert run_config["save_best"] is True
    assert run_config["save_best_after"] == 1
    config = json.loads((output_dir / "final" / "adapter_config.json").read_text())
    assert config["peft_type"] == "LORA"
    assert config["use_dora"] is False
    assert config["rank_pattern"]["self_attn.v_proj"] == 1
