"""Unit tests for extracted MLX DiT initialization mixin."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


def _load_handler_module(filename: str, module_name: str):
    """Load handler mixin module directly from file path."""
    repo_root = Path(__file__).resolve().parents[4]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    package_paths = {
        "acestep": repo_root / "acestep",
        "acestep.core": repo_root / "acestep" / "core",
        "acestep.core.generation": repo_root / "acestep" / "core" / "generation",
        "acestep.core.generation.handler": repo_root / "acestep" / "core" / "generation" / "handler",
    }
    for package_name, package_path in package_paths.items():
        if package_name in sys.modules:
            continue
        package_module = types.ModuleType(package_name)
        package_module.__path__ = [str(package_path)]
        sys.modules[package_name] = package_module
    if "loguru" not in sys.modules:
        fake_loguru = types.ModuleType("loguru")
        fake_loguru.logger = SimpleNamespace(info=lambda *_a, **_k: None, warning=lambda *_a, **_k: None)
        sys.modules["loguru"] = fake_loguru
    module_path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


MLX_DIT_INIT_MODULE = _load_handler_module(
    "mlx_dit_init.py",
    "acestep.core.generation.handler.mlx_dit_init",
)
MlxDitInitMixin = MLX_DIT_INIT_MODULE.MlxDitInitMixin


class _DitHost(MlxDitInitMixin):
    """Minimal host exposing DiT init state used by tests."""

    def __init__(self):
        """Initialize deterministic model/config placeholders."""
        self.config = {"size": "tiny"}
        self.model = SimpleNamespace(decoder=object())
        self.mlx_decoder = None
        self.use_mlx_dit = False
        self.mlx_dit_compiled = False
        self.lora_loaded = False
        self.use_lora = False
        self._lora_active_adapter = None


class _FakePeftModel:
    """PEFT wrapper stub exposing reversible merge helpers."""

    def __init__(self, base_decoder):
        self._base_decoder = base_decoder
        self.merge_calls = []
        self.unmerge_calls = 0

    def get_base_model(self):
        return self._base_decoder

    def merge_adapter(self, adapter_names=None, safe_merge=False):
        self.merge_calls.append((adapter_names, safe_merge))

    def unmerge_adapter(self):
        self.unmerge_calls += 1


def _fake_dit_model_module():
    """Return a stub MLX DiT module used by refresh tests."""
    module = types.ModuleType("acestep.models.mlx.dit_model")
    module.MLXDiTDecoder = type(
        "FakeDecoder",
        (),
        {"from_config": classmethod(lambda _cls, _cfg: object())},
    )
    return module


class MlxDitInitMixinTests(unittest.TestCase):
    """Behavior tests for extracted ``MlxDitInitMixin``."""

    def test_init_mlx_dit_unavailable_returns_false(self):
        """It returns False and leaves MLX DiT flags unset when unavailable."""
        host = _DitHost()
        fake_mlx = types.ModuleType("acestep.models.mlx")
        fake_mlx.mlx_available = lambda: False
        with patch.dict(sys.modules, {"acestep.models.mlx": fake_mlx}):
            self.assertFalse(host._init_mlx_dit(compile_model=True))
        self.assertIsNone(host.mlx_decoder)
        self.assertFalse(host.use_mlx_dit)

    def test_init_mlx_dit_success_sets_decoder(self):
        """It loads converted MLX DiT decoder and stores compile flag."""
        host = _DitHost()
        fake_mlx = types.ModuleType("acestep.models.mlx")
        fake_mlx.mlx_available = lambda: True
        fake_dit_model = types.ModuleType("acestep.models.mlx.dit_model")
        fake_dit_model.MLXDiTDecoder = type(
            "FakeDecoder",
            (),
            {"from_config": classmethod(lambda _cls, _cfg: object())},
        )
        fake_dit_convert = types.ModuleType("acestep.models.mlx.dit_convert")
        fake_dit_convert.convert_and_load = Mock()
        with patch.dict(
            sys.modules,
            {
                "acestep.models.mlx": fake_mlx,
                "acestep.models.mlx.dit_model": fake_dit_model,
                "acestep.models.mlx.dit_convert": fake_dit_convert,
            },
        ):
            self.assertTrue(host._init_mlx_dit(compile_model=True))
        self.assertTrue(host.use_mlx_dit)
        self.assertTrue(host.mlx_dit_compiled)
        fake_dit_convert.convert_and_load.assert_called_once()

    def test_sync_mlx_dit_from_model_reuses_existing_decoder(self):
        """It reloads the current MLX decoder weights from the PyTorch model."""
        host = _DitHost()
        host.use_mlx_dit = True
        host.mlx_decoder = object()
        fake_mlx = types.ModuleType("acestep.models.mlx")
        fake_mlx.mlx_available = lambda: True
        fake_dit_model = _fake_dit_model_module()
        fake_dit_convert = types.ModuleType("acestep.models.mlx.dit_convert")
        fake_dit_convert.convert_and_load = Mock()

        with patch.dict(
            sys.modules,
            {
                "acestep.models.mlx": fake_mlx,
                "acestep.models.mlx.dit_model": fake_dit_model,
                "acestep.models.mlx.dit_convert": fake_dit_convert,
            },
        ):
            ok, message = host._sync_mlx_dit_from_model("LoRA load")

        self.assertTrue(ok)
        self.assertIsNone(message)
        fake_dit_convert.convert_and_load.assert_called_once_with(host.model, host.mlx_decoder)

    def test_sync_mlx_dit_from_model_merges_active_peft_adapter(self):
        """It converts plain merged decoder weights when a PEFT LoRA is active."""
        host = _DitHost()
        host.use_mlx_dit = True
        host.mlx_decoder = object()
        host.lora_loaded = True
        host.use_lora = True
        host._lora_active_adapter = "mayer"
        base_decoder = object()
        host.model.decoder = _FakePeftModel(base_decoder)

        fake_mlx = types.ModuleType("acestep.models.mlx")
        fake_mlx.mlx_available = lambda: True
        fake_dit_model = _fake_dit_model_module()
        fake_dit_convert = types.ModuleType("acestep.models.mlx.dit_convert")
        fake_dit_convert.convert_and_load = Mock()
        fake_peft = types.ModuleType("peft")
        fake_peft.PeftModel = _FakePeftModel

        with patch.dict(
            sys.modules,
            {
                "acestep.models.mlx": fake_mlx,
                "acestep.models.mlx.dit_model": fake_dit_model,
                "acestep.models.mlx.dit_convert": fake_dit_convert,
                "peft": fake_peft,
            },
        ):
            ok, message = host._sync_mlx_dit_from_model("LoRA load")

        self.assertTrue(ok)
        self.assertIsNone(message)
        converted_model = fake_dit_convert.convert_and_load.call_args.args[0]
        self.assertIs(converted_model.decoder, base_decoder)
        self.assertEqual(host.model.decoder.merge_calls, [(["mayer"], False)])
        self.assertEqual(host.model.decoder.unmerge_calls, 1)

    def test_sync_mlx_dit_from_model_uses_base_decoder_when_lora_disabled(self):
        """It converts the plain base decoder when a loaded LoRA is toggled off."""
        host = _DitHost()
        host.use_mlx_dit = True
        host.mlx_decoder = object()
        host.lora_loaded = True
        host.use_lora = False
        base_decoder = object()
        host.model.decoder = _FakePeftModel(base_decoder)

        fake_mlx = types.ModuleType("acestep.models.mlx")
        fake_mlx.mlx_available = lambda: True
        fake_dit_model = _fake_dit_model_module()
        fake_dit_convert = types.ModuleType("acestep.models.mlx.dit_convert")
        fake_dit_convert.convert_and_load = Mock()
        fake_peft = types.ModuleType("peft")
        fake_peft.PeftModel = _FakePeftModel

        with patch.dict(
            sys.modules,
            {
                "acestep.models.mlx": fake_mlx,
                "acestep.models.mlx.dit_model": fake_dit_model,
                "acestep.models.mlx.dit_convert": fake_dit_convert,
                "peft": fake_peft,
            },
        ):
            ok, message = host._sync_mlx_dit_from_model("LoRA toggle (disabled)")

        self.assertTrue(ok)
        self.assertIsNone(message)
        converted_model = fake_dit_convert.convert_and_load.call_args.args[0]
        self.assertIs(converted_model.decoder, base_decoder)
        self.assertEqual(host.model.decoder.merge_calls, [])
        self.assertEqual(host.model.decoder.unmerge_calls, 0)

    def test_sync_mlx_dit_from_model_failure_clears_decoder(self):
        """It clears the active MLX decoder when refresh fails after a model mutation."""
        host = _DitHost()
        host.use_mlx_dit = True
        host.mlx_decoder = object()
        fake_mlx = types.ModuleType("acestep.models.mlx")
        fake_mlx.mlx_available = lambda: True
        fake_dit_model = _fake_dit_model_module()
        fake_dit_convert = types.ModuleType("acestep.models.mlx.dit_convert")
        fake_dit_convert.convert_and_load = Mock(side_effect=RuntimeError("bad refresh"))

        with patch.dict(
            sys.modules,
            {
                "acestep.models.mlx": fake_mlx,
                "acestep.models.mlx.dit_model": fake_dit_model,
                "acestep.models.mlx.dit_convert": fake_dit_convert,
            },
        ):
            ok, message = host._sync_mlx_dit_from_model("LoRA scale")

        self.assertFalse(ok)
        self.assertIn("bad refresh", message)
        self.assertIsNone(host.mlx_decoder)


if __name__ == "__main__":
    unittest.main()
