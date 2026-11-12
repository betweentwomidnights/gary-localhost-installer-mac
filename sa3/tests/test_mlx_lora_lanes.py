from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mlx.core")

import mlx.core as mx
from mlx.utils import tree_flatten, tree_map

from stable_audio_3.mlx.dit import StableAudioMLXDiT
from stable_audio_3.mlx.lora import MLXLoRASet, _MLXLoRALayer
from stable_audio_3.mlx.lora_lanes import (
    LaneError,
    LaneRunner,
    isolation_bias,
    parse_lane_plan,
    sample_envelope,
)

NAMES = {"kev": 0, "keygen": 1, "ratatat": 2}


def resolve(name: str) -> int:
    if name not in NAMES:
        raise ValueError(f"unknown LoRA '{name}'. available: {list(NAMES)}")
    return NAMES[name]


def shared(*names: str, strength: float = 1.0):
    return [
        {"lora_index": NAMES[n], "name": n, "strength": strength, "interval": (0.0, 1.0), "layer_filter": ""}
        for n in names
    ]


def plan(data, shared_loras=(), prompt="tone"):
    return parse_lane_plan(data, list(shared_loras), resolve, prompt)


# --------------------------------------------------------------------------- parsing


@pytest.mark.parametrize(
    "data",
    [
        {"lora_lanes": {"loras": []}},
        {"lora_lanes": [{"loras": [{"name": "kev"}]}]},
        {"lora_lanes": [{"loras": [{"name": "kev"}], "envelope": [[0]]}]},
        {"lora_lanes": [{"loras": [{"name": "nope"}], "envelope": [[0, 1]]}]},
        {"lora_lanes": [{"loras": [{"name": "kev"}, {"name": "kev"}], "envelope": [[0, 1]]}]},
        {"lora_lanes": [{"loras": [{"name": "kev", "interval_max": 0.5}], "envelope": [[0, 1]]}]},
        {"lora_lanes": [{"envelope": [[0, 1]], "prompt": "bells"}] * 9},
        {"lora_lanes": [{"envelope": [[0, 101]], "prompt": "bells"}]},
        {"lora_lanes": [{"envelope": [[0, float("nan")]], "prompt": "bells"}]},
        {"lora_envelope": [[t, 1] for t in range(257)]},
        {"lora_envelope": [[0, 1]], "lora_envelope_mode": "sideways"},
        {"lora_envelope": [[0, 1]], "lora_envelope_mode": "layer"},
        {"lora_lanes": [{"envelope": [[0, 1]], "prompt": "bells"}], "lane_isolation": -1},
    ],
)
def test_malformed_automation_is_refused(data):
    with pytest.raises(LaneError):
        plan(data)


def test_shared_adapter_cannot_also_ride_a_lane():
    with pytest.raises(LaneError, match="also a shared LoRA"):
        plan({"lora_lanes": [{"loras": [{"name": "kev"}], "envelope": [[0, 1]]}]}, shared("kev"))
    custom = shared("kev")
    custom[0]["interval"] = (0.0, 0.5)
    with pytest.raises(LaneError, match="interval"):
        plan({"lora_lanes": [{"loras": [{"name": "keygen"}], "envelope": [[0, 1]]}]}, custom)


def test_lanes_blend_and_prompts_are_parsed_like_sa3_server():
    result = plan(
        {
            "lora_lanes": [
                {
                    "loras": [{"name": "keygen", "strength": 1.2}],
                    "envelope": [[0, 0], {"t": 20, "amount": 2}],
                    "prompt": "  demoscene chiptune  ",
                },
                {"envelope": [[0, 1], [20, 0]]},  # no adapter, no prompt: dropped
                {"loras": [{"name": "ratatat"}], "envelope": [[0, 0], [20, 0]]},  # all zero: dropped
                {"envelope": [[0, 1]], "prompt": "tone"},  # the request prompt: dropped
            ],
            "lane_isolation": 1,
            "lane_isolation_until": 0,
            "lane_average_until": 0.9,
        },
        shared("kev", strength=0.5),
    )
    assert result.shared == [(0, 0.5)]
    assert len(result.lanes) == 1
    lane = result.lanes[0]
    assert lane.loras == [(1, 1.2)] and lane.peak == 2.0 and lane.prompt == "demoscene chiptune"
    assert (result.isolation, result.isolation_until, result.average_until) == (1.0, 0.0, 0.9)
    assert result.describe()["lanes"][0]["loras"] == [["keygen", 1.2]]


def test_lone_envelope_moves_every_request_lora_and_nothing_else_is_automation():
    result = plan({"lora_envelope": [[0, 0.2], [10, 1.3]]}, shared("kev", "keygen"))
    assert result.shared == []
    assert result.lanes[0].loras == [(0, 1.0), (1, 1.0)]
    assert (result.isolation, result.isolation_until, result.average_until) == (0.0, 0.9, 0.0)
    assert plan({"lora_lanes": None, "lane_isolation": 1}) is None
    assert plan({"lora_envelope": [[0, 1]]}) is None  # no request LoRA to move
    assert plan({"lora_lanes": [{"loras": [{"name": "kev"}], "envelope": [[0, 0]]}]}) is None


def test_envelope_is_linear_held_flat_and_peak_normalized():
    frames = 12
    envelope = sample_envelope([(0.5, 0.0), (0.25, 0.0), (0.75, 2.0)], 2.0, frames)
    centres = (np.arange(frames, dtype=np.float32) + 0.5) * 4096 / 44100
    expected = np.clip((centres - 0.5) / 0.25, 0.0, 1.0)
    assert envelope.dtype == np.float32
    np.testing.assert_allclose(envelope, expected, atol=1e-6)
    bias = isolation_bias([envelope], 3.0)
    assert bias[0, 0] == 0.0 and bias[0, -1] == pytest.approx(-3.0)


# --------------------------------------------------------------------------- runtime

FRAMES = 12


@pytest.fixture()
def dit():
    mx.random.seed(3)
    model = StableAudioMLXDiT(
        {
            "io_channels": 4,
            "embed_dim": 32,
            "depth": 2,
            "num_heads": 2,
            "cond_token_dim": 16,
            "project_cond_tokens": False,
            "transformer_type": "continuous_transformer",
            "num_memory_tokens": 3,
        },
        diffusion_objective="rf_denoiser",
    )
    # Branch outputs start at zero, which would make every pass return the same thing.
    model.update(tree_map(lambda p: mx.random.normal(p.shape).astype(p.dtype) * 0.2, model.parameters()))
    mx.eval(model.parameters())
    return model


def lora_set_for(dit, count=2) -> MLXLoRASet:
    params = dict(tree_flatten(dit.parameters()))
    keys = [k for k, v in params.items() if "self_attn" in k and v.ndim == 2][:2]
    assert keys
    rng = np.random.default_rng(5)
    layers = []
    for index in range(count):
        for key in keys:
            out_dim, in_dim = params[key].shape
            layers.append(
                _MLXLoRALayer(
                    lora_index=index,
                    lora_name=f"lora{index}",
                    source_name=key.removesuffix(".weight"),
                    target_key=key,
                    adapter_type="lora",
                    alpha=4.0,
                    rank=4,
                    params={
                        "lora_A": rng.normal(size=(4, in_dim)).astype(np.float32),
                        "lora_B": rng.normal(size=(out_dim, 4)).astype(np.float32),
                    },
                )
            )
    return MLXLoRASet("dit", tuple(f"/l{i}" for i in range(count)), tuple(f"lora{i}" for i in range(count)), layers)


def inputs(batch=1):
    mx.random.seed(11)
    x = mx.random.normal((batch, 4, FRAMES))
    t = mx.full((batch,), 0.7)
    cross = mx.random.normal((batch, 5, 16))
    return x, t, cross


def run(dit, **kwargs):
    x, t, cross = inputs()
    out = dit(x, t, cross_attn_cond=cross, **kwargs)
    mx.eval(out)
    return np.asarray(out, dtype=np.float32)


def static(dit, loras, strengths, **kwargs):
    loras.apply_to(dit, strength=strengths)
    try:
        return run(dit, **kwargs)
    finally:
        loras.apply_to(dit, strength=0.0)


def lanes(body, shared_loras=()):
    names = {"lora0": 0, "lora1": 1}
    shared_list = [
        {"lora_index": names[n], "name": n, "strength": s, "interval": (0.0, 1.0), "layer_filter": ""}
        for n, s in shared_loras
    ]
    return parse_lane_plan(body, shared_list, lambda n: names[n], "tone")


def test_each_frame_follows_its_lane_curve(dit):
    loras = lora_set_for(dit)
    before = {k: np.asarray(v) for k, v in tree_flatten(dit.parameters())}
    base = run(dit)
    v_shared = static(dit, loras, (0.5, 0.0))
    v_lane = static(dit, loras, (0.5, 0.8 * 1.5))
    assert np.abs(v_lane - v_shared).max() > 1e-3

    seconds = FRAMES * 4096 / 44100
    lane_plan = lanes(
        {"lora_lanes": [{"loras": [{"name": "lora1", "strength": 0.8}], "envelope": [[0, 0], [seconds, 1.5]]}]},
        [("lora0", 0.5)],
    )
    with LaneRunner(dit, loras, lane_plan, batch_size=1) as runner:
        mixed = run(dit)
    curve = sample_envelope(lane_plan.lanes[0].envelope, 1.5, FRAMES)
    np.testing.assert_allclose(mixed, v_shared + (v_lane - v_shared) * curve, atol=2e-5)
    assert runner.report()["passes"] == 2

    # The hooks and the merged weights are gone afterwards.
    np.testing.assert_allclose(run(dit), base, atol=1e-6)
    for key, value in tree_flatten(dit.parameters()):
        np.testing.assert_array_equal(np.asarray(value), before[key])


def test_averaging_plays_the_curve_mean_and_no_cache_gives_the_same_result(dit):
    loras = lora_set_for(dit)
    seconds = FRAMES * 4096 / 44100
    body = {
        "lora_lanes": [
            {"loras": [{"name": "lora0"}], "envelope": [[0, 0], [seconds, 1]]},
            {"loras": [{"name": "lora1"}], "envelope": [[0, 1], [seconds, 0]]},
        ],
        "lane_average_until": 0.5,  # sigma is 0.7, so the step is averaged
        "lane_isolation": 4.0,  # paused while averaging
        "lane_isolation_until": 0.0,
    }
    v0 = run(dit)
    v1 = static(dit, loras, (1.0, 0.0))
    v2 = static(dit, loras, (0.0, 1.0))
    means = [float(sample_envelope(l.envelope, 1.0, FRAMES).astype(np.float64).mean()) for l in lanes(body).lanes]
    expected = v0 + (v1 - v0) * means[0] + (v2 - v0) * means[1]
    with LaneRunner(dit, loras, lanes(body), batch_size=1) as runner:
        cached = run(dit)
    assert runner.report()["averaged_calls"] == 1 and runner.report()["isolated_calls"] == 0
    with LaneRunner(dit, loras, lanes(body), batch_size=1, cache_bytes=0) as runner:
        recomputed = run(dit)
    assert runner.report()["recomputed_passes"] == 2
    np.testing.assert_allclose(cached, expected, atol=2e-5)
    np.testing.assert_allclose(recomputed, cached, atol=1e-6)


def test_isolation_only_separates_frames_whose_lane_mix_differs(dit):
    loras = lora_set_for(dit)
    seconds = FRAMES * 4096 / 44100
    flat = {"lora_lanes": [{"loras": [{"name": "lora0"}], "envelope": [[0, 1]]}]}
    step = {"lora_lanes": [{"loras": [{"name": "lora0"}], "envelope": [[seconds / 2, 0], [seconds / 2 + 0.01, 1]]}]}
    blend = {"lane_isolation": 6.0, "lane_isolation_until": 0.0}

    def lane_run(body):
        with LaneRunner(dit, loras, lanes(body), batch_size=1) as runner:
            return run(dit), runner.report()

    flat_plain, _ = lane_run(flat)
    flat_isolated, report = lane_run({**flat, **blend})
    assert report["isolated_calls"] == 1
    np.testing.assert_allclose(flat_isolated, flat_plain, atol=1e-5)

    step_plain, _ = lane_run(step)
    step_isolated, _ = lane_run({**step, **blend})
    assert np.abs(step_isolated - step_plain).max() > 1e-3
    # Below the threshold the bias is off again.
    step_late, report = lane_run({**step, "lane_isolation": 6.0, "lane_isolation_until": 0.9})
    assert report["isolated_calls"] == 0
    np.testing.assert_allclose(step_late, step_plain, atol=1e-6)


def test_lane_prompt_replaces_only_the_conditioned_half_under_cfg(dit):
    body = {"lora_lanes": [{"envelope": [[0, 1]], "prompt": "bells"}]}
    x, t, cross = inputs()
    mx.random.seed(23)
    bells = mx.random.normal(cross.shape)
    negative = mx.random.normal(cross.shape)

    def guided(cond):
        out = dit(x, t, cross_attn_cond=cond, negative_cross_attn_cond=negative, cfg_scale=3.0)
        mx.eval(out)
        return np.asarray(out, dtype=np.float32)

    expected = guided(bells)
    assert np.abs(expected - guided(cross)).max() > 1e-3
    with LaneRunner(dit, None, lanes(body), batch_size=1, lane_cross={"bells": bells}):
        mixed = guided(cross)
    # A flat curve at 1 with a prompt of its own is just the lane prompt, guided as usual.
    np.testing.assert_allclose(mixed, expected, atol=5e-5)


@pytest.mark.parametrize("adapter_type", ["lora", "dora", "dora-rows", "dora-cols"])
def test_gpu_merge_matches_the_numpy_merge(adapter_type):
    rng = np.random.default_rng(9)
    out_dim, in_dim, rank = 24, 40, 4
    params = {
        "lora_A": rng.normal(size=(rank, in_dim)).astype(np.float32),
        "lora_B": rng.normal(size=(out_dim, rank)).astype(np.float32),
    }
    if adapter_type != "lora":
        size = in_dim if adapter_type == "dora-cols" else out_dim
        params["magnitude"] = rng.uniform(0.5, 2.0, size=(size,)).astype(np.float32)
    layer = _MLXLoRALayer(0, "l", "w", "w.weight", adapter_type, alpha=8.0, rank=rank, params=params)
    base = rng.normal(size=(out_dim, in_dim)).astype(np.float32)
    merged = layer.apply_mx(mx.array(base), strength=0.7)
    assert merged is not None
    np.testing.assert_allclose(np.asarray(merged), layer.apply(base, strength=0.7), rtol=2e-5, atol=2e-5)
    assert layer.apply_mx(mx.array(base), strength=0.0) is not None
    # A weight stored transposed, or an adapter family without a GPU path, falls back to numpy.
    assert layer.apply_mx(mx.array(base.T), strength=0.7) is None
    bora = _MLXLoRALayer(0, "l", "w", "w.weight", "bora", alpha=8.0, rank=rank, params=params)
    assert bora.apply_mx(mx.array(base), strength=0.7) is None
