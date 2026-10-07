"""Plugin request parity without model loading or GPU generation."""

import ast
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import api


@pytest.fixture
def client(monkeypatch):
    # Queue the real endpoint's parameters, but never start a worker.
    monkeypatch.setattr(api.threading, "Thread", Mock())
    monkeypatch.setattr(api, "MAX_DURATION", 380)
    monkeypatch.setattr(api, "TAIL_PAD_SECONDS", 19)
    monkeypatch.setattr(api.parse_tail_pad, "__defaults__", (19,))
    monkeypatch.setattr(api, "CONTINUE_TAIL_PAD", 19)
    monkeypatch.setattr(api, "CONTINUE_TAIL_MODE", "regen_past")
    with api.app.test_client() as client:
        yield client


def queued_params():
    return api.threading.Thread.call_args.kwargs["args"][1]


@pytest.mark.parametrize("pad", [0, 6])
def test_generate_client_choice_overrides_saved_default(client, pad):
    response = client.post("/generate", json={"prompt": "piano", "duration": 30, "tail_pad": pad, "seed": 42})
    assert response.status_code == 200
    result = response.get_json()
    params = queued_params()
    assert result["duration"] == 30
    assert result["tail_pad"] == pad
    assert result["gen_duration"] == params["duration"] == 30 + pad
    assert params["target_samples"] == 30 * api.OUTPUT_SAMPLE_RATE
    assert result["seed"] == 42
    assert api.get_session(result["session_id"])["meta"]["duration"] == 30


@pytest.mark.parametrize("pad", [0, 6])
@pytest.mark.parametrize("mode", ["inpaint", "latent_prefix"])
def test_continue_client_choice_overrides_saved_default(client, monkeypatch, pad, mode):
    monkeypatch.setattr(api, "decode_audio_data", lambda _: (10, SimpleNamespace(shape=(2, 80))))
    response = client.post("/continue", json={"prompt": "piano", "audio_data": "mock",
        "continuation_seconds": 22, "continuation_tail_pad": pad, "continuation_mode": mode,
        "seed": 42})
    assert response.status_code == 200
    result = response.get_json()
    params = queued_params()
    assert result["total_duration"] == 30
    assert result["tail_pad"] == pad
    assert result["gen_duration"] == params["duration"] == 30 + pad
    assert params["inpaint_mask_end_seconds"] == 30 + pad
    assert params["target_samples"] == 30 * api.OUTPUT_SAMPLE_RATE
    assert result["seed"] == 42


def test_omitted_pad_retains_server_default(client):
    response = client.post("/generate", json={"prompt": "piano", "duration": 30})
    assert response.get_json()["tail_pad"] == 19
    assert queued_params()["duration"] == 49


def test_tail_alias_precedence_and_clamps():
    assert api.parse_tail_pad({"tail_pad_seconds": 0, "tail_pad": 6, "continuation_tail_pad": 19}) == 0
    assert api.parse_tail_pad({"tail_pad_seconds": "", "tail_pad": "0"}) == 0
    assert api.parse_tail_pad({"continuation_tail_pad": 6}) == 6
    assert api.parse_tail_pad({"tail_pad": -4}) == 0
    assert api.parse_tail_pad({"tail_pad": 1000}) == api.TAIL_PAD_MAX


@pytest.mark.parametrize("duration", [300.01, 380])
def test_generate_accepts_long_local_output_with_internal_pad(client, duration):
    response = client.post("/generate", json={"prompt": "piano", "duration": duration, "tail_pad": 6})
    assert response.status_code == 200
    assert queued_params()["duration"] == duration + 6
    assert queued_params()["target_samples"] == round(duration * api.OUTPUT_SAMPLE_RATE)
    assert api.MAX_SAMPLE_SIZE > (duration + 6) * api.OUTPUT_SAMPLE_RATE


@pytest.mark.parametrize("duration", [380.01, 0, -1, "nan", "inf"])
def test_generate_rejects_invalid_output_before_queueing(client, duration):
    response = client.post("/generate", json={"prompt": "piano", "duration": duration})
    assert response.status_code == 400
    api.threading.Thread.assert_not_called()


@pytest.mark.parametrize("pad", ["not a number", "nan", "inf", True, []])
@pytest.mark.parametrize("route,key", [("/generate", "tail_pad"), ("/continue", "continuation_tail_pad")])
def test_invalid_pad_is_a_client_error(client, route, key, pad):
    response = client.post(route, json={"prompt": "piano", "duration": 30, key: pad})
    assert response.status_code == 400
    api.threading.Thread.assert_not_called()


@pytest.mark.parametrize("extra", [50, 50.01, "nan", "bad"])
def test_continue_total_limit_before_queueing(client, monkeypatch, extra):
    monkeypatch.setattr(api, "decode_audio_data", lambda _: (10, SimpleNamespace(shape=(2, 3300))))
    response = client.post("/continue", json={"prompt": "piano", "audio_data": "mock",
        "continuation_seconds": extra, "continuation_tail_pad": 6})
    if extra == 50:
        assert response.status_code == 200
        assert response.get_json()["total_duration"] == 380
        assert queued_params()["duration"] == 386
    else:
        assert response.status_code == 400
        api.threading.Thread.assert_not_called()


@pytest.mark.parametrize("duration", [380, 380.01])
def test_transform_input_limit(client, monkeypatch, duration):
    monkeypatch.setattr(api, "decode_audio_data", lambda _: (100, SimpleNamespace(shape=(2, round(duration * 100)))))
    response = client.post("/transform", json={"prompt": "piano", "audio_data": "mock"})
    assert response.status_code == (200 if duration == 380 else 400)
    if duration == 380:
        assert queued_params()["target_samples"] == 380 * api.OUTPUT_SAMPLE_RATE
    else:
        api.threading.Thread.assert_not_called()


def test_generated_tail_is_discarded_not_returned():
    audio = torch.arange(36, dtype=torch.float32).view(1, 1, 36)
    trimmed = api.apply_target_length(audio, 30)
    torch.testing.assert_close(trimmed, audio[..., :30])
    assert trimmed.shape[-1] == 30


def test_loop_keeps_its_separate_bar_padding(client):
    response = client.post("/generate/loop", json={"prompt": "piano", "bpm": 120, "bars": 8})
    assert response.status_code == 200
    assert queued_params()["mode"] == "loop"
    assert math.isclose(queued_params()["duration"], 16 + api.LOOP_PAD_SECONDS)


@pytest.mark.parametrize("override,expected", [(None, 380), ("275", 275)])
def test_default_duration_limit_and_environment_override(monkeypatch, override, expected):
    monkeypatch.delenv("SA3_MAX_DURATION", raising=False)
    if override is not None:
        monkeypatch.setenv("SA3_MAX_DURATION", override)
    tree = ast.parse(Path(api.__file__).read_text())
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "MAX_DURATION" for target in node.targets))
    assert eval(compile(ast.Expression(assignment.value), "api.py", "eval"), {"os": api.os}) == expected
