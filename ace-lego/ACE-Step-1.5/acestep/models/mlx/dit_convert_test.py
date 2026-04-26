"""Unit tests for extracted MLX DiT weight conversion helpers."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def _load_mlx_module(filename: str, module_name: str):
    """Load MLX helper module directly from file path."""
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    package_paths = {
        "acestep": repo_root / "acestep",
        "acestep.models": repo_root / "acestep" / "models",
        "acestep.models.mlx": repo_root / "acestep" / "models" / "mlx",
    }
    for package_name, package_path in package_paths.items():
        if package_name in sys.modules:
            continue
        package_module = types.ModuleType(package_name)
        package_module.__path__ = [str(package_path)]
        sys.modules[package_name] = package_module
    module_path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


DIT_CONVERT_MODULE = _load_mlx_module(
    "dit_convert.py",
    "acestep.models.mlx.dit_convert",
)
convert_decoder_weights = DIT_CONVERT_MODULE.convert_decoder_weights


class _FakeTensor:
    """Tensor stub implementing the subset of the torch API used by conversion."""

    def __init__(self, value):
        self._value = _FakeArray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        self._value = self._value.astype("float32", copy=False)
        return self

    def numpy(self):
        return self._value


class _FakeDecoder:
    """Decoder stub exposing a deterministic state dict."""

    def __init__(self, state_dict):
        self._state_dict = state_dict

    def state_dict(self):
        return self._state_dict


class _FakeModel:
    """Model stub exposing a decoder for conversion."""

    def __init__(self, state_dict):
        self.decoder = _FakeDecoder(state_dict)


class DitConvertTests(unittest.TestCase):
    """Behavior tests for MLX weight conversion."""

    def setUp(self):
        """Install a minimal ``mlx.core`` shim for conversion tests."""
        self._fake_mx_core = types.ModuleType("mlx.core")
        self._fake_mx_core.array = lambda value: value
        self._fake_mlx = types.ModuleType("mlx")
        self._fake_mlx.core = self._fake_mx_core
        self._modules = {"mlx": self._fake_mlx, "mlx.core": self._fake_mx_core}
        self._patcher = patch.dict(sys.modules, self._modules)
        self._patcher.start()

    def tearDown(self):
        """Remove the temporary ``mlx`` shim."""
        self._patcher.stop()

    def test_convert_decoder_weights_strips_peft_wrapper_keys(self):
        """It maps PEFT ``base_layer`` weights to plain decoder names and skips adapter tensors."""
        model = _FakeModel(
            {
                "base_model.model.layers.0.self_attn.q_proj.base_layer.weight": _FakeTensor([[1, 2], [3, 4]]),
                "base_model.model.layers.0.self_attn.q_proj.lora_A.mayer.weight": _FakeTensor([[9, 9]]),
                "base_model.model.layers.0.self_attn.q_proj.lora_B.mayer.weight": _FakeTensor([[8], [8]]),
                "base_model.model.layers.0.self_attn.q_proj.lora_magnitude_vector.mayer.weight": _FakeTensor([7, 7]),
                "base_model.model.rotary_emb.inv_freq": _FakeTensor([1, 2, 3]),
            }
        )

        weights = convert_decoder_weights(model)

        self.assertEqual([name for name, _value in weights], ["layers.0.self_attn.q_proj.weight"])
        self.assertEqual(weights[0][1].tolist(), [[1, 2], [3, 4]])

    def test_convert_decoder_weights_keeps_conv_layout_remaps_after_unwrapping(self):
        """It still applies proj_in/proj_out transposes when the tensors come from ``base_layer``."""
        model = _FakeModel(
            {
                "proj_in.1.base_layer.weight": _FakeTensor([[[1, 2, 3], [4, 5, 6]]]),
                "proj_out.1.base_layer.weight": _FakeTensor(
                    [
                        [[1, 2, 3]],
                        [[4, 5, 6]],
                    ]
                ),
            }
        )

        weights = dict(convert_decoder_weights(model))

        self.assertEqual(sorted(weights), ["proj_in.weight", "proj_out.weight"])
        self.assertEqual(weights["proj_in.weight"].tolist(), [[[1, 4], [2, 5], [3, 6]]])
        self.assertEqual(weights["proj_out.weight"].tolist(), [[[1, 4], [2, 5], [3, 6]]])


def _shape(value):
    """Return the recursive shape of a nested list/tuple structure."""
    if not isinstance(value, (list, tuple)):
        return ()
    if not value:
        return (0,)
    return (len(value),) + _shape(value[0])


def _get(value, indices):
    """Index into a nested list/tuple using ``indices``."""
    current = value
    for index in indices:
        current = current[index]
    return current


def _build(shape, getter, prefix=()):
    """Build a nested list for ``shape`` using ``getter``."""
    if not shape:
        return getter(prefix)
    return [_build(shape[1:], getter, prefix + (index,)) for index in range(shape[0])]


def _permute_axes(value, permutation):
    """Permute axes of a nested list/tuple."""
    source_shape = _shape(value)
    target_shape = tuple(source_shape[axis] for axis in permutation)
    return _build(
        target_shape,
        lambda indices: _get(
            value,
            tuple(indices[permutation.index(axis)] for axis in range(len(permutation))),
        ),
    )


class _FakeArray:
    """Minimal ndarray-like wrapper used by conversion tests."""

    def __init__(self, value):
        self._value = value

    def astype(self, _dtype, copy=False):
        return _FakeArray(self.tolist() if copy else self._value)

    def swapaxes(self, axis1, axis2):
        rank = len(_shape(self._value))
        permutation = list(range(rank))
        permutation[axis1], permutation[axis2] = permutation[axis2], permutation[axis1]
        return _FakeArray(_permute_axes(self._value, permutation))

    def transpose(self, *permutation):
        return _FakeArray(_permute_axes(self._value, permutation))

    def tolist(self):
        return self._value


if __name__ == "__main__":
    unittest.main()
