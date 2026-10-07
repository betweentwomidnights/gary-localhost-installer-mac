"""No-network coverage of the optional decoder download's API and CLI paths."""
import pytest

import api
import sa3_predownload_cli as cli


@pytest.fixture
def public_download(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    calls = []
    monkeypatch.setattr(api, "perform_predownload_target", lambda target, *, token: calls.append((target, token)))
    monkeypatch.setattr(api, "configure_hf_auth", lambda: pytest.fail("public decoder must not log in"))
    monkeypatch.setattr(api, "sa3_required_download_targets", lambda: pytest.fail("decoder must not fetch base models"))
    return calls


def test_decoder_downloads_only_configured_file_without_auth(public_download):
    api.run_sa3_predownload_task("decoder-public-test", {"target_type": "decoder"})
    assert public_download == [(api.SA3_DECODER_LORA_TARGET, False)]
    session = api.read_predownload_session("decoder-public-test")
    assert session["status"] == "completed"
    assert session["progress"] == 100
    assert session["target"] == "SAME-L decoder squeak fix"
    assert session["queue_status"]["stage_total"] == 1


def test_decoder_ignores_stale_saved_credentials(public_download, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "invalid-test-token")
    api.run_sa3_predownload_task("decoder-stale-token", {"target_type": "decoder"})
    assert public_download[0][1] is False
    assert api.read_predownload_session("decoder-stale-token")["status"] == "completed"


def test_decoder_ignores_unrelated_repo_and_file_inputs(public_download):
    api.run_sa3_predownload_task("decoder-fixed-target", {
        "target_type": "decoder", "repo_id": "unrelated/repo", "filename": "other.bin",
    })
    assert public_download == [(api.SA3_DECODER_LORA_TARGET, False)]


@pytest.mark.parametrize("target_type", ["required", "repo"])
def test_gated_downloads_still_require_token(public_download, target_type):
    session_id = "missing-token-" + target_type
    api.run_sa3_predownload_task(session_id, {"target_type": target_type, "repo_id": "test/repo"})
    assert not public_download
    assert api.read_predownload_session(session_id)["status"] == "failed"
    assert "HF_TOKEN" in api.read_predownload_session(session_id)["error"]


def test_http_decoder_endpoint_accepts_no_token(public_download, monkeypatch):
    class ImmediateThread:
        def __init__(self, target, args, **kwargs):
            self.target, self.args = target, args

        def start(self):
            self.target(*self.args)

    monkeypatch.setattr(api.threading, "Thread", ImmediateThread)
    with api.app.test_client() as client:
        response = client.post("/models/predownload", json={"target_type": "decoder"})
        assert response.status_code == 200
        session = api.read_predownload_session(response.get_json()["session_id"])
        assert session["status"] == "completed"
        assert session["target"] == "SAME-L decoder squeak fix"
        assert client.post("/models/predownload", json={"target_type": "required"}).status_code == 400
    assert len(public_download) == 1


def test_decoder_download_failure_is_reported(public_download, monkeypatch):
    def fail(target, *, token):
        raise RuntimeError("test download unavailable")

    monkeypatch.setattr(api, "perform_predownload_target", fail)
    api.run_sa3_predownload_task("decoder-error-test", {"target_type": "decoder"})
    session = api.read_predownload_session("decoder-error-test")
    assert session["status"] == "failed"
    assert "test download unavailable" in session["error"]


def test_file_download_disables_implicit_saved_token(monkeypatch):
    calls = []
    monkeypatch.setattr(api, "hf_hub_download", lambda **kwargs: calls.append(kwargs))
    api.perform_predownload_target(api.SA3_DECODER_LORA_TARGET, token=False)
    assert calls == [{"repo_id": api.DECODER_LORA_REPO, "filename": api.DECODER_LORA_FILENAME, "token": False}]


def test_stopped_service_cli_routes_decoder_command(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "download_command", lambda target: calls.append(target) or 0)
    assert cli.main(["download-decoder"]) == 0
    assert cli.main(["download-required"]) == 0
    assert calls == ["decoder", "required"]
