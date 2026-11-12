"""Contract tests for the ace-server job API on the Mac Carey wrapper, with a mocked api_server."""

import io
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import quote

import httpx
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent))

import main as wrapper  # noqa: E402
import native_api  # noqa: E402

BASE = "acestep-v15-base"
TURBO = "acestep-v15-xl-turbo"
XL_BASE = "acestep-v15-xl-base"


def wav_bytes(seconds=2.0, rate=44100, channels=1, subtype="FLOAT") -> bytes:
    frames = int(seconds * rate)
    tone = 0.25 * np.sin(np.linspace(0, 440 * 2 * np.pi * seconds, frames, dtype=np.float32))
    buffer = io.BytesIO()
    sf.write(buffer, np.stack([tone] * channels, axis=1), rate, subtype=subtype, format="WAV")
    return buffer.getvalue()


def request_json(**overrides) -> dict:
    body = {
        "task_type": "text2music", "caption": "warm analog house", "lyrics": "[Instrumental]",
        "bpm": 120, "vocal_language": "en", "inference_steps": 8, "guidance_scale": 1.0,
        "seed": 1234, "output_format": "wav16", "synth_model": BASE, "duration": 30,
    }
    body.update(overrides)
    return body


class Backend:
    """Stands in for the ACE-Step api_server subprocess."""

    def __init__(self, folder: Path):
        self.calls: list[tuple[str, object]] = []
        self.submissions: list[httpx.Request] = []
        self.finish = True
        self.fail_lora = False
        self.final_status = 1
        self.progress_text = "MLX DiT diffusion:  25%|##5       | 2/8 [00:01<00:03]"
        self.result = folder / "backend-result.wav"
        self.result.write_bytes(wav_bytes(seconds=3.0, rate=48000, channels=2, subtype="FLOAT"))

    def paths(self) -> list[str]:
        return [path for path, _ in self.calls]

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.calls.append((path, dict(request.url.params)))
        if path == "/v1/load":
            return httpx.Response(200, json={"data": {"model": request.url.params["config_path"]}})
        if path in ("/v1/unload", "/v1/lora/unload", "/health"):
            return httpx.Response(200, json={"data": {}})
        if path == "/v1/lora/load":
            if self.fail_lora:
                return httpx.Response(400, json={"detail": "❌ incompatible adapter"})
            return httpx.Response(200, json={"data": {"message": "ok"}, "code": 200})
        if path == "/v1/lora/scale":
            self.calls[-1] = (path, json.loads(request.content))
            return httpx.Response(200, json={"data": {"scale": 1}, "code": 200})
        if path == "/release_task":
            self.submissions.append(request)
            return httpx.Response(200, json={"data": {"task_id": "ace-1"}})
        if path == "/query_result":
            if not self.finish:
                return httpx.Response(200, json={"data": [{"status": 0, "result": "[]",
                                                             "progress_text": self.progress_text}]})
            item = [{"file": f"/v1/audio?path={quote(str(self.result))}", "status": self.final_status}]
            return httpx.Response(200, json={"data": [{"status": self.final_status,
                                                         "result": json.dumps(item)}]})
        return httpx.Response(404, json={"detail": "not found"})


class NativeCareyApiTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.backend = Backend(Path(self.folder.name))
        real_client = httpx.AsyncClient
        transport = httpx.MockTransport(self.backend.handle)
        self.patches = [
            patch.object(native_api.httpx, "AsyncClient", lambda *a, **k: real_client(transport=transport)),
            patch.object(native_api, "installed_dits", lambda: [BASE, XL_BASE, TURBO]),
            patch.object(native_api, "POLL_SECONDS", 0.01),
            patch.object(wrapper, "MANAGE_MODEL_LIFECYCLE", True),
            patch.object(wrapper, "KEEP_MODEL_LOADED", False),
            patch.object(wrapper, "_current_model", None),
        ]
        for item in self.patches:
            item.start()
        self.context = TestClient(wrapper.app)
        self.client = self.context.__enter__()
        wrapper.app.state.native_jobs.clear()
        wrapper.LORA_REGISTRY.clear()
        wrapper.LORA_REGISTRY.update({
            "billie": {"path": "/loras/billie", "scale": 1.0, "backends": ["base", "turbo"], "model_family": "standard"},
            "mayer": {"path": "/loras/mayer", "scale": 1.0, "backends": ["base"], "model_family": "xl"},
        })
        wrapper._caption_pools.clear()
        wrapper._caption_pools.update({"billie": ["whispered bedroom pop"], "default": ["x"]})

    def tearDown(self):
        self.context.__exit__(None, None, None)
        for item in reversed(self.patches):
            item.stop()
        wrapper.LORA_REGISTRY.clear()
        wrapper._caption_pools.clear()
        self.folder.cleanup()

    def submit(self, body: dict, audio: bytes | None = None, ref: bytes | None = None):
        files = {"request": (None, json.dumps(body), "application/json")}
        if audio is not None:
            files["audio"] = ("audio.wav", audio, "audio/wav")
        if ref is not None:
            files["ref_audio"] = ("ref_audio.wav", ref, "audio/wav")
        return self.client.post("/synth", files=files)

    def wait(self, job_id: str, until=("done", "failed", "cancelled"), seconds=10.0) -> dict:
        deadline = time.time() + seconds
        while time.time() < deadline:
            status = self.client.get("/job", params={"id": job_id}).json()
            if status["status"] in until:
                return status
            time.sleep(0.02)
        self.fail(f"job stayed {status}")

    def wait_inactive(self, job_id: str) -> None:
        deadline = time.time() + 10
        while wrapper.app.state.native_jobs[job_id].active and time.time() < deadline:
            time.sleep(0.02)
        self.assertFalse(wrapper.app.state.native_jobs[job_id].active)

    def test_props_lists_installed_dits_and_every_adapter_with_its_family(self):
        props = self.client.get("/props").json()
        self.assertEqual(props["models"]["dit"], [BASE, XL_BASE, TURBO])
        self.assertEqual(props["adapters"], ["billie", "mayer"])
        self.assertEqual(props["adapter_info"]["billie"],
                         {"model_family": "standard", "captions": ["whispered bedroom pop"]})
        self.assertEqual(props["adapter_info"]["mayer"], {"model_family": "xl", "captions": []})
        self.assertEqual(self.backend.calls, [])  # never touches the backend

    def test_generate_runs_exactly_what_was_asked_and_returns_wav16(self):
        response = self.submit(request_json(keyscale="F# minor", timesignature="4"))
        self.assertEqual(response.status_code, 200, response.text)
        job_id = response.json()["id"]
        self.assertRegex(job_id, r"^[0-9a-zA-Z]{1,64}$")
        self.assertEqual(self.wait(job_id), {"status": "done"})
        self.wait_inactive(job_id)

        sent = json.loads(self.backend.submissions[0].content)
        self.assertEqual(sent["task_type"], "text2music")
        self.assertEqual(sent["prompt"], "warm analog house")
        self.assertEqual((sent["inference_steps"], sent["guidance_scale"]), (8, 1.0))
        self.assertEqual((sent["seed"], sent["use_random_seed"]), (1234, False))
        self.assertEqual((sent["key_scale"], sent["time_signature"], sent["audio_duration"]), ("F# minor", "4", 30))
        self.assertEqual(sent["shift"], 1.0)  # base, not api_server's turbo default
        self.assertEqual((sent["thinking"], sent["batch_size"], sent["audio_format"]), (False, 1, "wav"))
        self.assertNotIn("track_name", sent)
        self.assertNotIn("repainting_start", sent)
        # Loaded the DiT the request named, started from no adapter, released the model after.
        self.assertEqual(self.backend.calls[0], ("/v1/load", {"config_path": BASE}))
        self.assertEqual(self.backend.paths()[-1], "/v1/unload")
        self.assertNotIn("/v1/lora/load", self.backend.paths())

        result = self.client.get("/job", params={"id": job_id, "result": "1"})
        self.assertEqual(result.headers["content-type"], "multipart/mixed; boundary=ace-batch-boundary")
        head = b"--ace-batch-boundary\r\nContent-Type: audio/wav\r\n\r\n"
        tail = b"\r\n--ace-batch-boundary--\r\n"
        self.assertTrue(result.content.startswith(head) and result.content.endswith(tail))
        info = sf.info(io.BytesIO(result.content[len(head):-len(tail)]))
        self.assertEqual((info.samplerate, info.channels, info.subtype), (48000, 2, "PCM_16"))
        self.assertAlmostEqual(info.frames / 48000, 3.0, places=3)  # full length, untrimmed

    def test_complete_sends_the_source_as_48k_stereo_and_each_track_class(self):
        body = request_json(task_type="complete", caption="", track="drums | bass", duration=2,
                            inference_steps=50, guidance_scale=7.0, audio_cover_strength=0.6)
        job_id = self.submit(body, audio=wav_bytes(2.0, 44100, 1), ref=wav_bytes(2.0, 44100, 1)).json()["id"]
        self.assertEqual(self.wait(job_id)["status"], "done")
        upload = self.backend.submissions[0]
        self.assertTrue(upload.headers["content-type"].startswith("multipart/form-data"))
        text = upload.content.decode("latin-1")
        self.assertEqual(text.count('name="track_classes"'), 2)
        self.assertIn('name="track_classes"\r\n\r\ndrums\r\n', text)
        self.assertIn('name="track_classes"\r\n\r\nbass\r\n', text)
        self.assertIn('name="use_random_seed"\r\n\r\nfalse\r\n', text)
        self.assertIn('name="audio_cover_strength"\r\n\r\n0.6\r\n', text)
        self.assertIn('name="ctx_audio"; filename="audio.wav"', text)
        self.assertIn('name="ref_audio"; filename="ref_audio.wav"', text)
        start = upload.content.index(b"RIFF")
        info = sf.info(io.BytesIO(upload.content[start:]))
        self.assertEqual((info.samplerate, info.channels, info.subtype), (48000, 2, "PCM_16"))

    def test_translation_per_task(self):
        def payload(**fields):
            return native_api.backend_payload(native_api.AceRequest.model_validate(request_json(**fields)))

        lego = payload(task_type="lego", track="vocals", repainting_start=0, repainting_end=-1)
        self.assertEqual((lego["track_name"], lego["repainting_start"], lego["repainting_end"]), ("vocals", 0, -1))
        repaint = payload(task_type="repaint", synth_model=TURBO, repainting_start=20, repainting_end=60, duration=60)
        self.assertEqual((repaint["repainting_start"], repaint["repainting_end"], repaint["shift"]), (20, 60, 3.0))
        cover = payload(task_type="cover-nofsq", cover_noise_strength=0.3, audio_cover_strength=0.8)
        self.assertEqual((cover["task_type"], cover["cover_noise_strength"], cover["audio_cover_strength"]),
                         ("cover-nofsq", 0.3, 0.8))
        self.assertNotIn("bpm", payload(bpm=0))
        self.assertEqual(payload(synth_model="acestep-v15-xl-sft")["shift"], 1.0)
        self.assertEqual(payload(task_type="extract", track="bass")["track_name"], "bass")

    def test_adapter_is_loaded_strictly_at_the_requested_scale(self):
        job_id = self.submit(request_json(adapter="billie", adapter_scale=0.8)).json()["id"]
        self.assertEqual(self.wait(job_id)["status"], "done")
        self.wait_inactive(job_id)
        paths = self.backend.paths()
        self.assertLess(paths.index("/v1/lora/load"), paths.index("/release_task"))
        self.assertIn(("/v1/lora/scale", {"scale": 0.8}), self.backend.calls)
        self.assertEqual(paths[-2:], ["/v1/lora/unload", "/v1/unload"])

        self.backend.fail_lora = True
        self.backend.submissions.clear()
        job_id = self.submit(request_json(adapter="billie")).json()["id"]
        self.assertEqual(self.wait(job_id), {"status": "failed", "error": "lora billie failed to load"})
        self.assertEqual(self.backend.submissions, [])  # never generated without the LoRA

    def test_requests_the_plugin_should_never_send_get_a_readable_400(self):
        audio = wav_bytes(2.0)
        cases = [
            (request_json(caption="  "), None, "Please enter a caption"),
            (request_json(synth_model="acestep-v15-sft"), None, "Requested model is not available"),
            (request_json(task_type="lego", track="vocals", synth_model=TURBO), audio, "This task needs a base model"),
            (request_json(adapter="nope"), None, "Requested lora is not available for this model"),
            (request_json(adapter="mayer"), None, "Requested lora is not available for this model"),
            (request_json(adapter="mayer", synth_model=TURBO), None, "Requested lora is not available for this model"),
            (request_json(task_type="lego", track="vocals"), None, "Please upload source audio"),
            (request_json(task_type="lego", track="kazoo"), audio, "Please choose valid tracks"),
            (request_json(task_type="lego", track="drums | bass"), audio, "Please choose valid tracks"),
            (request_json(task_type="repaint", duration=2.5), audio, "Target must be longer than the source"),
            (request_json(task_type="remix"), None, "Unknown task type"),
            (request_json(output_format="mp3"), None, "Only wav16 output is supported"),
            (request_json(duration=381), None, "Invalid or missing duration"),
            ({k: v for k, v in request_json().items() if k != "seed"}, None, "Invalid or missing seed"),
            (request_json(task_type="cover-nofsq"), b"not audio", "Invalid request or audio"),
        ]
        for body, source, message in cases:
            with self.subTest(message=message, body=body.get("adapter") or body["task_type"]):
                response = self.submit(body, audio=source)
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.json(), {"error": message})
        self.assertEqual(self.client.post("/synth", files={"audio": ("a.wav", audio)}).json(),
                         {"error": "Missing request part"})
        self.assertEqual(self.backend.calls, [])
        self.assertEqual(wrapper.app.state.native_jobs, {})

    def test_unknown_jobs_and_unready_results_are_404_with_an_error(self):
        for response in (self.client.get("/job", params={"id": "missing"}),
                         self.client.post("/job", params={"id": "missing", "cancel": "1"})):
            self.assertEqual((response.status_code, response.json()), (404, {"error": "Job not found"}))
        self.backend.finish = False
        job_id = self.submit(request_json()).json()["id"]
        response = self.client.get("/job", params={"id": job_id, "result": "1"})
        self.assertEqual((response.status_code, response.json()), (404, {"error": "Result not ready"}))
        self.backend.finish = True
        self.wait(job_id)

    def test_progress_reports_real_steps_and_a_failed_backend_job_fails(self):
        self.backend.finish = False
        job_id = self.submit(request_json()).json()["id"]
        deadline = time.time() + 5
        while time.time() < deadline:
            status = self.client.get("/job", params={"id": job_id}).json()
            if status.get("progress", {}).get("stage") == "dit":
                break
            time.sleep(0.02)
        self.assertEqual(status, {"status": "running", "progress": {"stage": "dit", "step": 2, "total": 8}})
        self.backend.final_status = 2
        self.backend.finish = True
        self.assertEqual(self.wait(job_id),
                         {"status": "failed", "error": "Carey generation failed; please retry"})

    def test_cancel_reports_at_once_and_holds_the_next_job_until_the_backend_is_done(self):
        self.backend.finish = False
        first = self.submit(request_json()).json()["id"]
        self.wait(first, until=("running",))
        deadline = time.time() + 5
        while "/release_task" not in self.backend.paths() and time.time() < deadline:
            time.sleep(0.02)
        second = self.submit(request_json(seed=2)).json()["id"]
        queued = self.submit(request_json(seed=3)).json()["id"]
        self.assertEqual(self.client.get("/job", params={"id": second}).json()["progress"]["stage"], "queued")

        cancelled = self.client.post("/job", params={"id": first, "cancel": "1"})
        self.assertEqual(cancelled.json(), {"status": "cancelled"})
        self.assertEqual(self.client.get("/job", params={"id": first}).json(), {"status": "cancelled"})
        # Cancelled while queued: never starts.
        self.assertEqual(self.client.post("/job", params={"id": queued, "cancel": "1"}).json(), {"status": "cancelled"})
        time.sleep(0.2)
        self.assertEqual(len(self.backend.submissions), 1)  # the drained job still owns the backend
        self.assertEqual(self.client.get("/job", params={"id": second}).json()["progress"]["stage"], "queued")

        self.backend.finish = True
        self.assertEqual(self.wait(second), {"status": "done"})
        self.wait_inactive(queued)
        self.assertEqual(len(self.backend.submissions), 2)
        self.assertEqual(self.client.get("/job", params={"id": first, "result": "1"}).status_code, 404)
        # A finished job answers a cancel with its own status.
        self.assertEqual(self.client.post("/job", params={"id": second, "cancel": "1"}).json(), {"status": "done"})

    def test_legacy_routes_are_still_served(self):
        routes = {route.path for route in wrapper.app.routes}
        for path in ("/lego", "/complete", "/cover", "/lego/status/{task_id}", "/loras", "/captions",
                     "/health", "/synth", "/job", "/props"):
            self.assertIn(path, routes)


class InstalledDitsTest(unittest.TestCase):
    def test_only_models_whose_weights_are_on_disk_are_offered(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            optimized = root / "artifacts"
            for name in ("acestep-v15-base", "acestep-v15-xl-turbo"):
                artifact = optimized / "hub" / "models--thepatch--acestep-1.5-mlx" / "snapshots" / "rev1" / "MLX" / f"{name}-f16"
                (artifact / "conditioning").mkdir(parents=True)
                for part in ("provenance.json", "model.safetensors.index.json", "config.json",
                             "conditioning/provenance.json", "conditioning/silence_latent.pt"):
                    (artifact / part).write_text("{}")
            (optimized / "hub" / "models--thepatch--acestep-1.5-mlx" / "snapshots" / "rev1" / "MLX"
             / "acestep-v15-xl-turbo-f16" / "config.json").unlink()  # incomplete download
            checkpoints = root / "checkpoints"
            (checkpoints / "acestep-v15-xl-sft").mkdir(parents=True)
            (checkpoints / "acestep-v15-xl-sft" / "config.json").write_text("{}")
            (checkpoints / "acestep-v15-xl-sft" / "model.safetensors").write_bytes(b"weights")
            (checkpoints / "acestep-v15-turbo").mkdir()
            (checkpoints / "acestep-v15-turbo" / "config.json").write_text("{}")  # metadata only

            def found(policy: str, mlx: str = "1") -> list[str]:
                env = {"HUGGINGFACE_HUB_CACHE": str(optimized / "hub"), "ACESTEP_CHECKPOINT_DIR": str(checkpoints),
                       "ACESTEP_MLX_WEIGHT_SOURCE": policy, "ACESTEP_USE_MLX_DIT": mlx}
                with patch.dict(os.environ, env), patch.object(native_api, "_pinned_artifact_revision", lambda: "rev1"):
                    return native_api.installed_dits()

            self.assertEqual(found("auto"), ["acestep-v15-base", "acestep-v15-xl-sft"])
            self.assertEqual(found("optimized"), ["acestep-v15-base"])
            self.assertEqual(found("legacy"), ["acestep-v15-xl-sft"])
            self.assertEqual(found("auto", mlx="0"), ["acestep-v15-xl-sft"])

    def test_the_pinned_revision_comes_from_the_bundled_api_server(self):
        self.assertRegex(native_api._pinned_artifact_revision() or "", r"^[0-9a-f]{40}$")


if __name__ == "__main__":
    unittest.main()
