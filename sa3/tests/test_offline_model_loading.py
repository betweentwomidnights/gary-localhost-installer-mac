"""Cached SA3 weights must resolve without online authentication or downloads."""

from unittest.mock import Mock

import api
from stable_audio_3 import model_configs


def test_configuring_auth_does_not_validate_token_online(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "offline-test-token")
    login = Mock(side_effect=AssertionError("must not log in"))
    monkeypatch.setattr(api, "login", login)
    api.configure_hf_auth()
    assert api.os.environ["HF_TOKEN"] == "offline-test-token"
    assert api.os.environ["HUGGING_FACE_HUB_TOKEN"] == "offline-test-token"
    login.assert_not_called()


def test_no_credentials_does_not_log_in(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    login = Mock()
    monkeypatch.setattr(api, "login", login)
    api.configure_hf_auth()
    login.assert_not_called()


def test_cached_model_resolution_never_calls_download(monkeypatch, tmp_path):
    config = model_configs.ModelConfig("example/model", "model_config.json", "model.safetensors")
    files = [tmp_path / config.config_path, tmp_path / config.ckpt_path]
    for file in files:
        file.write_bytes(b"fixture")
    monkeypatch.setattr(model_configs, "try_to_load_from_cache", Mock(side_effect=list(map(str, files))))
    download = Mock(side_effect=AssertionError("cached weights must not download"))
    monkeypatch.setattr(model_configs, "hf_hub_download", download)
    assert config.resolve() == tuple(map(str, files))
    download.assert_not_called()


def test_partial_cache_still_downloads_missing_assets(monkeypatch, tmp_path):
    config = model_configs.ModelConfig("example/model", "model_config.json", "model.safetensors")
    cached_config = tmp_path / config.config_path
    cached_config.write_bytes(b"fixture")
    monkeypatch.setattr(model_configs, "try_to_load_from_cache", Mock(side_effect=[str(cached_config), None]))
    download = Mock(side_effect=["downloaded-config", "downloaded-weights"])
    monkeypatch.setattr(model_configs, "hf_hub_download", download)
    assert config.resolve() == ("downloaded-config", "downloaded-weights")
    assert download.call_count == 2
