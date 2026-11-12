"""Smoke test for the ace-server job API (/props, /synth, /job) on a running Carey wrapper.

    python smoke_native.py                      # discovery only
    python smoke_native.py generate --out /tmp/gen.wav
    python smoke_native.py lego --audio src.wav --track vocals --out /tmp/lego.wav
    python smoke_native.py complete --audio src.wav --track "drums | bass"
    python smoke_native.py continue --audio src.wav --duration 60
    python smoke_native.py cover --audio src.wav --ref
    python smoke_native.py generate --cancel-after 10

Sends what gary4juce sends, one tab per mode. Prints each progress change and the result's format.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time

import httpx
import soundfile as sf

TASKS = {"generate": "text2music", "lego": "lego", "complete": "complete",
         "continue": "repaint", "cover": "cover-nofsq", "extract": "extract"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", nargs="?", choices=sorted(TASKS))
    parser.add_argument("--url", default="http://127.0.0.1:8003")
    parser.add_argument("--model", help="a name from /props models.dit (default: the first base model)")
    parser.add_argument("--audio")
    parser.add_argument("--ref", action="store_true", help="also send the source as ref_audio")
    parser.add_argument("--caption", default="warm analog house groove, punchy drums, deep bass")
    parser.add_argument("--lyrics", default="[Instrumental]")
    parser.add_argument("--track", default="vocals")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--cfg", type=float)
    parser.add_argument("--bpm", type=int, default=120)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--adapter", default="")
    parser.add_argument("--adapter-scale", type=float, default=0.8)
    parser.add_argument("--cover-strength", type=float, default=1.0)
    parser.add_argument("--noise-strength", type=float, default=0.0)
    parser.add_argument("--cancel-after", type=float, help="cancel this many seconds after submitting")
    parser.add_argument("--out")
    args = parser.parse_args()

    client = httpx.Client(base_url=args.url, timeout=120)
    props = client.get("/props").json()
    print("dit:", props["models"]["dit"])
    for name in props["adapters"]:
        info = props["adapter_info"].get(name, {})
        print(f"adapter: {name} family={info.get('model_family')} captions={len(info.get('captions', []))}")
    if not args.mode:
        return 0

    model = args.model or next((d for d in props["models"]["dit"] if d.endswith("-base")), None)
    if not model:
        print("no base model installed; pass --model")
        return 1
    turbo = model.endswith("-turbo")
    request = {
        "task_type": TASKS[args.mode], "caption": args.caption, "lyrics": args.lyrics, "bpm": args.bpm,
        "vocal_language": "en", "inference_steps": args.steps or (8 if turbo else 32),
        "guidance_scale": args.cfg if args.cfg is not None else (1.0 if turbo else 7.0),
        "seed": args.seed, "output_format": "wav16", "synth_model": model, "duration": args.duration,
    }
    if args.adapter:
        request.update(adapter=args.adapter, adapter_scale=args.adapter_scale)
    files = {"request": (None, json.dumps(request), "application/json")}
    if args.mode != "generate":
        if not args.audio:
            parser.error("--audio is required for this mode")
        source = open(args.audio, "rb").read()
        seconds = sf.info(args.audio).duration
        if args.mode in ("lego", "extract"):
            request.update(track=args.track, duration=seconds)
            if args.mode == "lego":
                request.update(repainting_start=0.0, repainting_end=-1.0)
        elif args.mode == "complete":
            request.update(track=args.track, duration=seconds, audio_cover_strength=args.cover_strength)
        elif args.mode == "continue":
            request.update(repainting_start=seconds, repainting_end=args.duration)
        elif args.mode == "cover":
            request.update(duration=seconds, audio_cover_strength=args.cover_strength,
                           cover_noise_strength=args.noise_strength)
        files["request"] = (None, json.dumps(request), "application/json")
        files["audio"] = ("audio.wav", source, "audio/wav")
        if args.ref:
            files["ref_audio"] = ("ref_audio.wav", source, "audio/wav")

    started = time.time()
    submitted = client.post("/synth", files=files)
    print("submit:", submitted.status_code, submitted.text[:200])
    if submitted.status_code != 200:
        return 1
    job = submitted.json()["id"]
    last = None
    while True:
        status = client.get("/job", params={"id": job}).json()
        if status != last:
            print(f"{time.time() - started:7.1f}s {json.dumps(status)}")
            last = status
        if status["status"] != "running":
            break
        if args.cancel_after is not None and time.time() - started > args.cancel_after:
            print("cancel:", client.post("/job", params={"id": job, "cancel": "1"}).json())
            args.cancel_after = None
        time.sleep(0.5)
    if status["status"] != "done":
        return 0 if status["status"] == "cancelled" else 1

    result = client.get("/job", params={"id": job, "result": "1"})
    boundary = result.headers["content-type"].split("boundary=")[1]
    part = result.content.split(b"--" + boundary.encode())[1]
    headers, _, wav = part.partition(b"\r\n\r\n")
    wav = wav[:-2]
    info = sf.info(io.BytesIO(wav))
    print(f"result: {headers.decode().strip()} | {info.subtype} {info.samplerate} Hz "
          f"{info.channels} ch {info.duration:.2f} s in {time.time() - started:.1f} s")
    if args.out:
        open(args.out, "wb").write(wav)
        print("saved:", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
