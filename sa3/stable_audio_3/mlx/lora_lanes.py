"""LoRA automation lanes for the MLX Stable Audio 3 backend.

The request contract and the semantics are sa3.cpp's (docs/SERVER.md, "LoRA automation lanes"),
by way of the PyTorch port in sa3-api (lora_lanes.py), so gary4juce sends one request shape to
sa3.cpp, to the remote service and to this Mac service:

    "loras":      [{"name": "kev", "strength": 1.0}, ...]     # shared: in EVERY pass
    "lora_lanes": [{"loras": [{"name": "keygen", "strength": 1.0}],
                    "envelope": [[sec, amount], ...],
                    "prompt": "optional lane prompt"}, ...]   # <= 8 lanes, <= 256 points
    "lora_envelope": [[sec, amount], ...]   # one-lane shorthand: every request LoRA on one curve
    "lora_envelope_mode": "velocity"        # "layer" is sa3.cpp-only (rejected here)
    "lane_isolation": 0.0, "lane_isolation_until": 0.9, "lane_average_until": 0.0

  * Breakpoints are linear between points and held flat past either end, sampled at every latent
    frame's centre in float32 and divided by the lane's peak.
  * Pass 0 runs the shared adapters alone on the request prompt; pass k runs the shared adapters
    plus lane k's adapters at strength * peak, on the lane's prompt when it has its own. Per frame
    the velocity is v0 + sum_k e_k(t) * (v_k - v0). Under CFG the conditioned and unconditioned
    halves of the batch are mixed separately (the unconditioned half keeps the negative/null
    conditioning in every pass) and only then guided.
  * Lanes whose curve is all zero, or with no adapters and no prompt of their own, are dropped.
  * The conditioners (the seconds_total token) see the REQUEST-level loras only.
  * lane_average_until: while the step's sigma is above it, every lane plays at the mean of its
    curve (isolation is paused for those steps).
  * lane_isolation: while sigma > lane_isolation_until (and not averaged), self-attention between
    frames i and j is biased by -lane_isolation * sum_k |e_k(i) - e_k(j)|; memory and prepend
    tokens are unbiased. Applied to every pass.

What is different on MLX: adapters here are merged into the DiT weights rather than applied as
parametrizations. Each pass therefore owns a set of merged weights, built once per generation and
swapped into the DiT before the pass runs. Sets are kept up to SA3_LANE_WEIGHT_CACHE_GB; a pass
past the budget is merged again on every step (slower, same result).

Rejected with a clear error, as on the remote service, so a project behaves the same everywhere:
an adapter both shared and in a lane, or twice in one lane; "layer" envelope mode; shared LoRAs
with a non-default interval or a layer_filter while lanes are active.
"""
from __future__ import annotations

import math
import os
import typing as tp
from dataclasses import dataclass, field

import numpy as np

MAX_LANES = 8
MAX_ENVELOPE_POINTS = 256
MAX_LANE_AMOUNT = 100.0
LANE_FIELDS = (
    "lora_lanes",
    "lora_envelope",
    "lora_envelope_mode",
    "lane_isolation",
    "lane_isolation_until",
    "lane_average_until",
)


class LaneError(ValueError):
    """A request-level problem (HTTP 400)."""


@dataclass
class Lane:
    loras: list  # [(lora_index, strength)]
    envelope: list  # [(seconds, amount)]
    prompt: str = ""
    peak: float = 0.0
    names: list = field(default_factory=list)


@dataclass
class LanePlan:
    shared: list  # [(lora_index, strength)], in every pass
    lanes: list  # [Lane]
    isolation: float = 0.0
    isolation_until: float = 0.9
    average_until: float = 0.0

    def describe(self) -> dict[str, tp.Any]:
        return {
            "lanes": [
                {
                    "loras": [[name, strength] for name, (_, strength) in zip(lane.names, lane.loras)],
                    "points": len(lane.envelope),
                    "peak": lane.peak,
                    "prompt": lane.prompt or None,
                }
                for lane in self.lanes
            ],
            "lane_isolation": self.isolation,
            "lane_isolation_until": self.isolation_until,
            "lane_average_until": self.average_until,
        }


# --------------------------------------------------------------------------- request parsing


def wants_lanes(data: dict) -> bool:
    return bool(data.get("lora_lanes")) or bool(data.get("lora_envelope"))


def _finite(value, what: str) -> float:
    if isinstance(value, bool):
        raise LaneError(f"{what} must be a number")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise LaneError(f"{what} must be a number") from None
    if not math.isfinite(number):
        raise LaneError(f"{what} must be finite")
    return number


def _points(raw, what: str) -> list[tuple[float, float]]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise LaneError(f"{what}: expected [[seconds, amount], ...]")
    if len(raw) > MAX_ENVELOPE_POINTS:
        raise LaneError(f"{what}: at most {MAX_ENVELOPE_POINTS} envelope points")
    points = []
    for point in raw:
        # sa3.cpp also takes {"t": seconds, "amount": amount}.
        if isinstance(point, dict) and "t" in point and "amount" in point:
            point = [point["t"], point["amount"]]
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise LaneError(f"{what}: expected [[seconds, amount], ...]")
        seconds, amount = _finite(point[0], what), _finite(point[1], what)
        if amount < 0.0 or amount > MAX_LANE_AMOUNT:
            raise LaneError(f"{what}: amounts must be finite and in [0, {MAX_LANE_AMOUNT:g}]")
        points.append((seconds, amount))
    return points


def parse_lane_plan(
    data: dict,
    shared_loras: list[dict[str, tp.Any]],
    resolve_index: tp.Callable[[str], tp.Any],
    base_prompt: str,
) -> LanePlan | None:
    """Validate the lane fields of a request; None when it asks for no automation.

    shared_loras is the resolved request "loras" ([{lora_index, name, strength, interval,
    layer_filter}]). resolve_index(name) returns the adapter's identity in the registry or raises
    ValueError.
    """
    if not wants_lanes(data):
        return None
    mode = data.get("lora_envelope_mode")
    mode = "velocity" if mode is None else str(mode).strip().lower()
    if mode not in ("velocity", "layer"):
        raise LaneError("lora_envelope_mode must be velocity or layer")
    if mode == "layer":
        raise LaneError("lora envelope: 'layer' mode is not supported by this backend; use 'velocity'")
    isolation = _finite(data.get("lane_isolation") or 0.0, "lane_isolation")
    if isolation < 0.0:
        raise LaneError("lane_isolation must not be negative")
    isolation_until = data.get("lane_isolation_until")
    isolation_until = _finite(0.9 if isolation_until is None else isolation_until, "lane_isolation_until")
    average_until = _finite(data.get("lane_average_until") or 0.0, "lane_average_until")
    if isolation_until < 0.0 or average_until < 0.0:
        raise LaneError("lane_isolation_until and lane_average_until must not be negative")

    for config in shared_loras:
        if tuple(config.get("interval", (0.0, 1.0))) != (0.0, 1.0) or config.get("layer_filter"):
            raise LaneError(
                f"LoRA '{config['name']}': interval/layer_filter are not supported with lanes"
            )
    shared = [(config["lora_index"], float(config["strength"])) for config in shared_loras]
    shared_names = [config["name"] for config in shared_loras]

    envelope = _points(data.get("lora_envelope"), "lora_envelope")
    raw_lanes = data.get("lora_lanes") or []
    if not isinstance(raw_lanes, list):
        raise LaneError("lora_lanes must be a list")
    if len(raw_lanes) > MAX_LANES:
        raise LaneError(f"lora_lanes: at most {MAX_LANES} lanes")

    lanes = []
    for lane_index, raw in enumerate(raw_lanes):
        if not isinstance(raw, dict):
            raise LaneError("each lora_lanes entry must be an object")
        points = _points(raw.get("envelope"), "lora_lanes envelope")
        if not points:
            raise LaneError("lora_lanes: each lane needs an envelope")
        raw_loras = raw.get("loras") or []
        if not isinstance(raw_loras, list):
            raise LaneError("lora_lanes loras must be a list")
        lane_loras, names = [], []
        for entry in raw_loras:
            if not isinstance(entry, dict) or not str(entry.get("name") or "").strip():
                raise LaneError("each LoRA needs a catalog name")
            name = str(entry["name"]).strip()
            if (
                _finite(entry.get("interval_min", 0.0), "interval_min") != 0.0
                or _finite(entry.get("interval_max", 1.0), "interval_max") != 1.0
                or entry.get("layer_filter")
            ):
                raise LaneError(f"LoRA '{name}': interval/layer_filter are not supported with lanes")
            strength = _finite(entry.get("strength", 1.0), "lora lane: adapter strengths")
            try:
                index = resolve_index(name)
            except ValueError as exc:
                raise LaneError(str(exc)) from None
            if index in [shared_index for shared_index, _ in shared]:
                raise LaneError(
                    f"lora lane {lane_index}: '{name}' is also a shared LoRA; this backend "
                    "cannot stack an adapter with itself"
                )
            if index in [existing for existing, _ in lane_loras]:
                raise LaneError(f"lora lane {lane_index}: '{name}' listed twice")
            lane_loras.append((index, strength))
            names.append(name)
        prompt = raw.get("prompt")
        if prompt is not None and not isinstance(prompt, str):
            raise LaneError("lora_lanes prompt must be a string")
        lanes.append(Lane(lane_loras, points, (prompt or "").strip(), names=names))

    if not lanes and envelope and shared:
        # A lone lora_envelope is one lane carrying every DiT adapter, with nothing shared.
        lanes = [Lane(list(shared), envelope, "", names=list(shared_names))]
        shared = []

    for lane in lanes:
        lane.peak = max(0.0, max(amount for _, amount in lane.envelope))
    lanes = [
        lane
        for lane in lanes
        if lane.peak > 0.0 and not (not lane.loras and (not lane.prompt or lane.prompt == base_prompt))
    ]
    if not lanes:
        return None  # nothing can differ from the base pass: an ordinary generation
    return LanePlan(shared, lanes, isolation, isolation_until, average_until)


# --------------------------------------------------------------------------- envelope math


def sample_envelope(points, peak, frames, samples_per_frame=4096, sample_rate=44100.0) -> np.ndarray:
    """sa3.cpp's sample_envelope, in float32 with its operation order."""
    pts = sorted(points)
    f32 = np.float32
    spf, rate, pk = f32(samples_per_frame), f32(sample_rate), f32(peak)
    out = np.empty(frames, dtype=np.float32)
    for i in range(frames):
        t = (f32(i) + f32(0.5)) * spf / rate
        amount = f32(pts[-1][1])
        if t <= f32(pts[0][0]):
            amount = f32(pts[0][1])
        else:
            for k in range(1, len(pts)):
                if t <= f32(pts[k][0]):
                    t0, t1 = f32(pts[k - 1][0]), f32(pts[k][0])
                    u = (t - t0) / (t1 - t0) if t1 > t0 else f32(1.0)
                    amount = f32(pts[k - 1][1]) + u * (f32(pts[k][1]) - f32(pts[k - 1][1]))
                    break
        out[i] = amount / pk
    return out


def isolation_bias(envelopes: tp.Sequence[np.ndarray], isolation: float) -> np.ndarray:
    """[T, T] additive attention bias: -isolation * sum_k |e_k(i) - e_k(j)|."""
    frames = len(envelopes[0])
    distance = np.zeros((frames, frames), dtype=np.float32)
    for envelope in envelopes:
        distance += np.abs(envelope[:, None] - envelope[None, :])
    return np.float32(-isolation) * distance


def mix_passes(outputs: tp.Sequence[np.ndarray], curves: tp.Sequence[np.ndarray]) -> np.ndarray:
    """Reference mix in numpy (the runner does the same in MLX): v0 + sum e_k * (v_k - v0)."""
    v0 = outputs[0].astype(np.float32)
    mixed = v0.copy()
    for vk, curve in zip(outputs[1:], curves):
        mixed += (vk.astype(np.float32) - v0) * curve
    return mixed


# --------------------------------------------------------------------------- runtime


def default_weight_cache_bytes() -> float:
    raw = (os.environ.get("SA3_LANE_WEIGHT_CACHE_GB") or "").strip()
    if raw:
        return max(0.0, float(raw)) * 2**30
    try:
        physical = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        physical = 16 * 2**30
    # Unified memory: leave most of it to the model, the decoder and the DAW.
    return float(min(12 * 2**30, max(2 * 2**30, physical * 0.25)))


class LaneRunner:
    """Per-generation state: owns the hooks installed on the MLX DiT.

    Use as a context manager around the sampler. The DiT must hold its unadapted weights on
    entry (StableAudioMLXPipeline.generate clears any merged LoRA first), and holds them again on
    exit.
    """

    def __init__(
        self,
        dit,
        lora_set,
        plan: LanePlan,
        *,
        batch_size: int,
        lane_cross: dict[str, tp.Any] | None = None,
        samples_per_frame: int = 4096,
        sample_rate: float = 44100.0,
        cache_bytes: float | None = None,
    ) -> None:
        import mlx.core as mx
        from mlx.utils import tree_flatten, tree_unflatten

        self._mx = mx
        self._tree_unflatten = tree_unflatten
        self.dit = dit
        self.lora_set = lora_set
        self.plan = plan
        self.batch_size = int(batch_size)
        self.lane_cross = lane_cross or {}
        self.samples_per_frame = int(samples_per_frame)
        self.sample_rate = float(sample_rate)
        self.passes = [list(plan.shared)]
        for lane in plan.lanes:
            self.passes.append(list(plan.shared) + [(i, s * lane.peak) for i, s in lane.loras])
        if lora_set is None and any(strength != 0.0 for p in self.passes for _, strength in p):
            raise LaneError("lora lanes: no LoRA is loaded")
        if plan.isolation > 0.0 and int(getattr(dit, "patch_size", 1)) != 1:
            raise LaneError("lane isolation is not supported by this model")

        self.cache_budget = default_weight_cache_bytes() if cache_bytes is None else float(cache_bytes)
        self.cache_used = 0.0
        self._cache: dict[int, dict[str, tp.Any]] = {}
        self._uncached: set[int] = set()
        self._loaded_keys: set[str] = set()
        self._base = dict(tree_flatten(dit.parameters())) if lora_set is not None else {}
        self._frames: int | None = None
        self._iso_active = False
        self._iso_mask: dict[tuple, tp.Any] = {}
        self._saved: list[tuple[tp.Any, str, tp.Any]] = []
        self.steps_averaged = 0
        self.steps_isolated = 0
        self.steps = 0

    # -- envelopes, built once per canvas length
    def _envelopes(self, frames: int):
        if self._frames != frames:
            mx = self._mx
            sampled = [
                sample_envelope(lane.envelope, lane.peak, frames, self.samples_per_frame, self.sample_rate)
                for lane in self.plan.lanes
            ]
            self._curves = [mx.array(e) for e in sampled]
            self._curves_avg = [
                mx.array(np.full(frames, np.float32(e.astype(np.float64).mean()), dtype=np.float32))
                for e in sampled
            ]
            self._bias = (
                isolation_bias(sampled, self.plan.isolation) if self.plan.isolation > 0.0 else None
            )
            self._iso_mask = {}
            self._frames = frames
        return self._curves, self._curves_avg

    # -- per-pass merged weights
    def _pass_weights(self, index: int) -> dict[str, tp.Any]:
        cached = self._cache.get(index)
        if cached is not None:
            return cached
        pairs = [(i, s) for i, s in self.passes[index] if float(s) != 0.0]
        if not pairs or self.lora_set is None:
            self._cache[index] = {}
            return self._cache[index]
        weights = self.lora_set.adapted_weights(self.dit, pairs, base=self._base)
        if index in self._uncached:
            return weights
        size = float(sum(array.nbytes for array in weights.values()))
        if self.cache_used + size <= self.cache_budget:
            self._cache[index] = weights
            self.cache_used += size
        else:
            self._uncached.add(index)
        return weights

    def _load_pass(self, index: int) -> None:
        weights = self._pass_weights(index)
        updates = {key: self._base[key] for key in self._loaded_keys if key not in weights}
        updates.update(weights)
        if updates:
            self.dit.update(self._tree_unflatten(list(updates.items())))
        self._loaded_keys = set(weights)

    def _restore_base(self) -> None:
        if self._loaded_keys:
            self.dit.update(
                self._tree_unflatten([(key, self._base[key]) for key in self._loaded_keys])
            )
            self._loaded_keys = set()

    # -- StableAudioMLXDiT._forward wrapper: one call per pass, then the per-frame mix
    def _wrap_forward(self, real):
        mx = self._mx

        def forward(x, t, *, cross_attn_cond=None, **kwargs):
            if kwargs.get("return_info") or kwargs.get("exit_layer_ix") is not None:
                return real(x, t, cross_attn_cond=cross_attn_cond, **kwargs)
            plan = self.plan
            sigma = float(t.reshape(-1)[0].item())
            curves, curves_avg = self._envelopes(int(x.shape[-1]))
            averaged = plan.average_until > 0.0 and sigma > plan.average_until
            self._iso_active = plan.isolation > 0.0 and not averaged and sigma > plan.isolation_until
            self.steps += 1
            self.steps_averaged += int(averaged)
            self.steps_isolated += int(self._iso_active)
            n_cond = min(self.batch_size, int(x.shape[0]))
            outputs = []
            try:
                for index in range(len(self.passes)):
                    self._load_pass(index)
                    cross = cross_attn_cond
                    if index > 0 and cross is not None:
                        lane_cond = self.lane_cross.get(plan.lanes[index - 1].prompt)
                        if lane_cond is not None:
                            lane_cond = lane_cond.astype(cross.dtype)
                            if tuple(lane_cond.shape[1:]) != tuple(cross.shape[1:]):
                                raise RuntimeError("lora lanes: lane prompt conditioning has a different shape")
                            if int(lane_cond.shape[0]) != n_cond:
                                lane_cond = mx.broadcast_to(lane_cond[:1], (n_cond, *lane_cond.shape[1:]))
                            cross = (
                                lane_cond
                                if n_cond == int(cross.shape[0])
                                else mx.concatenate([lane_cond, cross[n_cond:]], axis=0)
                            )
                    out = real(x, t, cross_attn_cond=cross, **kwargs)
                    # The pass must be evaluated while its weights are the ones in the model.
                    mx.eval(out)
                    outputs.append(out)
            finally:
                self._iso_active = False
            v0 = outputs[0].astype(mx.float32)
            mixed = v0
            for vk, curve in zip(outputs[1:], curves_avg if averaged else curves):
                mixed = mixed + (vk.astype(mx.float32) - v0) * curve
            mixed = mixed.astype(outputs[0].dtype)
            mx.eval(mixed)
            return mixed

        return forward

    # -- self-attention with the isolation bias when armed
    def _wrap_attention(self, attention, real):
        mx = self._mx

        def apply_attn(q, k, v, *, causal=None, padding_mask=None, sliding_window=None):
            if not self._iso_active:
                return real(q, k, v, causal=causal, padding_mask=padding_mask, sliding_window=sliding_window)
            use_causal = attention.causal if causal is None else causal
            if sliding_window is not None or use_causal:
                raise RuntimeError("lane isolation: sliding-window/causal attention is not supported")
            if attention.num_heads != attention.kv_heads:
                repeat = attention.num_heads // attention.kv_heads
                k = mx.repeat(k, repeat, axis=1)
                v = mx.repeat(v, repeat, axis=1)
            if padding_mask is not None and k.shape[-2] == padding_mask.shape[-1]:
                v = v * padding_mask[:, None, :, None].astype(v.dtype)
            tokens = int(q.shape[-2])
            key = (tokens, str(q.dtype))
            mask = self._iso_mask.get(key)
            if mask is None:
                frames = self._bias.shape[0]
                prefix = tokens - frames  # memory + prepend tokens: unbiased
                if prefix < 0 or int(k.shape[-2]) != tokens:
                    raise RuntimeError("lane isolation: unexpected attention geometry")
                full = np.zeros((tokens, tokens), dtype=np.float32)
                full[prefix:, prefix:] = self._bias
                mask = mx.array(full).astype(q.dtype)
                self._iso_mask[key] = mask
            return mx.fast.scaled_dot_product_attention(
                q, k, v, scale=attention.dim_heads**-0.5, mask=mask
            )

        return apply_attn

    def _patch(self, obj, name: str, wrap) -> None:
        self._saved.append((obj, name, vars(obj).get(name)))
        vars(obj)[name] = wrap(getattr(obj, name))

    def __enter__(self) -> "LaneRunner":
        self._patch(self.dit, "_forward", self._wrap_forward)
        if self.plan.isolation > 0.0:
            for layer in self.dit.transformer.layers:
                attention = getattr(layer, "self_attn", None)
                if attention is not None:
                    self._patch(attention, "apply_attn", lambda real, a=attention: self._wrap_attention(a, real))
        return self

    def __exit__(self, *exc) -> None:
        for obj, name, previous in reversed(self._saved):
            if previous is None:
                vars(obj).pop(name, None)
            else:
                vars(obj)[name] = previous
        self._saved = []
        self._restore_base()
        self._cache.clear()
        self._iso_mask = {}

    def report(self) -> dict[str, tp.Any]:
        return {
            **self.plan.describe(),
            "passes": len(self.passes),
            "weight_cache_gb": round(self.cache_used / 2**30, 3),
            "recomputed_passes": len(self._uncached),
            "model_calls": self.steps,
            "averaged_calls": self.steps_averaged,
            "isolated_calls": self.steps_isolated,
        }
