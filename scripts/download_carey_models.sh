#!/usr/bin/env bash
set -Eeuo pipefail

# Downloads required ACE-Step checkpoint files for Carey (lego mode) on macOS.
# Uses the shared HF downloader environment (uv-managed), with adaptive
# HF Xet -> HTTP fallback for reliability. No Homebrew or aria2 dependency.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ACE_ROOT="${PROJECT_ROOT}/ace-lego/ACE-Step-1.5"
CHECKPOINTS_DIR="${ACESTEP_CHECKPOINT_DIR:-${ACE_ROOT}/checkpoints}"
LEGACY_CHECKPOINTS_DIR="${ACE_ROOT}/checkpoints"
HF_DOWNLOADER_PYTHON_VERSION="${G4L_HF_DOWNLOADER_PYTHON_VERSION:-3.11}"
HF_DOWNLOADER_VENV_DIR="${G4L_HF_DOWNLOADER_VENV_DIR:-${HOME}/Library/Application Support/GaryLocalhost/venvs/hf-downloader}"
HF_DOWNLOADER_PACKAGES=(
  "huggingface_hub==1.4.1"
  "hf_xet==1.2.0"
  "tqdm==4.67.2"
)

DOWNLOADER_PYTHON=""
WORKER_PATH=""

if [[ ! -d "${ACE_ROOT}" ]]; then
  echo "ACE-Step repo not found at: ${ACE_ROOT}"
  echo "Expected: ${PROJECT_ROOT}/ace-lego/ACE-Step-1.5"
  exit 1
fi

resolve_uv_path() {
  local candidates=()
  if command -v uv >/dev/null 2>&1; then
    candidates+=("$(command -v uv)")
  fi
  candidates+=(
    "${HOME}/.local/bin/uv"
    "${HOME}/Library/Application Support/gary4local/tools/uv/uv"
    "${HOME}/Library/Application Support/gary4local/tools/uv/bin/uv"
    "${HOME}/Library/Application Support/GaryLocalhost/tools/uv/uv"
    "${HOME}/Library/Application Support/GaryLocalhost/tools/uv/bin/uv"
  )

  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -n "${candidate}" && -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  return 1
}

bootstrap_uv_if_needed() {
  local resolved
  if resolved="$(resolve_uv_path)"; then
    printf '%s\n' "${resolved}"
    return 0
  fi

  local install_dir="${HOME}/Library/Application Support/gary4local/tools/uv"
  mkdir -p "${install_dir}"

  echo "uv not found. bootstrapping uv into ${install_dir}..." >&2
  UV_UNMANAGED_INSTALL="${install_dir}" PATH="${PATH:-/usr/bin:/bin:/usr/sbin:/sbin}" \
    /bin/sh -lc 'curl -LsSf https://astral.sh/uv/install.sh | sh' >&2

  if resolved="$(resolve_uv_path)"; then
    printf '%s\n' "${resolved}"
    return 0
  fi

  echo "failed to locate uv after bootstrap." >&2
  return 1
}

resolve_worker_path() {
  local candidates=(
    "${PROJECT_ROOT}/stable-audio-tools/hf_predownload_worker.py"
    "${PROJECT_ROOT}/audiocraft-mlx/audiocraft/hf_predownload_worker.py"
    "${PROJECT_ROOT}/melodyflow/hf_predownload_worker.py"
  )

  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -f "${candidate}" ]]; then
      WORKER_PATH="${candidate}"
      return 0
    fi
  done

  return 1
}

ensure_shared_downloader_python() {
  local uv_path
  uv_path="$(bootstrap_uv_if_needed)"
  uv_path="$(printf '%s\n' "${uv_path}" | tail -n 1)"
  if [[ -z "${uv_path}" || ! -x "${uv_path}" ]]; then
    echo "failed to resolve uv executable path." >&2
    return 1
  fi

  mkdir -p "$(dirname "${HF_DOWNLOADER_VENV_DIR}")"
  local venv_python="${HF_DOWNLOADER_VENV_DIR}/bin/python"

  if [[ ! -x "${venv_python}" ]]; then
    echo "creating shared HF downloader venv at ${HF_DOWNLOADER_VENV_DIR}..."
    "${uv_path}" python install "${HF_DOWNLOADER_PYTHON_VERSION}"
    "${uv_path}" venv --python "${HF_DOWNLOADER_PYTHON_VERSION}" --seed "${HF_DOWNLOADER_VENV_DIR}"
  fi

  echo "ensuring shared HF downloader packages are installed..."
  "${uv_path}" pip install --python "${venv_python}" --upgrade pip setuptools wheel
  "${uv_path}" pip install --python "${venv_python}" --upgrade "${HF_DOWNLOADER_PACKAGES[@]}"

  DOWNLOADER_PYTHON="${venv_python}"
}

seed_from_legacy_if_available() {
  local out_path="$1"
  local label="$2"

  if [[ "${CHECKPOINTS_DIR}" == "${LEGACY_CHECKPOINTS_DIR}" ]]; then
    return 0
  fi

  local prefix="${CHECKPOINTS_DIR%/}/"
  if [[ "${out_path}" != "${prefix}"* ]]; then
    return 0
  fi

  local relative_path="${out_path#${prefix}}"
  local legacy_path="${LEGACY_CHECKPOINTS_DIR%/}/${relative_path}"

  if [[ -f "${legacy_path}" && ! -f "${out_path}" ]]; then
    mkdir -p "$(dirname "${out_path}")"
    cp -p "${legacy_path}" "${out_path}"
    echo "[${label}] seeded from legacy path: ${legacy_path}"
  fi
}

download_with_shared_downloader() {
  local label="$1"
  local repo="$2"
  local hf_path="$3"
  local out_dir="$4"
  local out_name="$5"

  mkdir -p "${out_dir}"
  local out_path="${out_dir}/${out_name}"
  seed_from_legacy_if_available "${out_path}" "${label}"

  echo "[${label}] ensuring ${repo}/${hf_path}"
  "${DOWNLOADER_PYTHON}" - "${WORKER_PATH}" "${repo}" "${hf_path}" "${out_path}" "${label}" <<'PY'
import json
import os
import select
import shutil
import subprocess
import sys
import time

worker_path, repo_id, filename, out_path, label = sys.argv[1:6]
mode = str(
    os.environ.get("G4L_HF_DOWNLOADER_XET_MODE", os.environ.get("G4L_HF_XET_MODE", "adaptive"))
).strip().lower()
if mode not in {"on", "off", "adaptive"}:
    mode = "adaptive"

try:
    first_byte_timeout = max(
        5.0, float(str(os.environ.get("G4L_HF_XET_FIRST_BYTE_TIMEOUT_SECONDS", "25")).strip())
    )
except Exception:
    first_byte_timeout = 25.0
try:
    slow_speed_bps = max(
        64 * 1024, int(str(os.environ.get("G4L_HF_XET_SLOW_SPEED_BPS", str(1 * 1024 * 1024))).strip())
    )
except Exception:
    slow_speed_bps = 1 * 1024 * 1024
try:
    slow_grace_seconds = max(
        5.0, float(str(os.environ.get("G4L_HF_XET_SLOW_SPEED_GRACE_SECONDS", "45")).strip())
    )
except Exception:
    slow_grace_seconds = 45.0


def terminate_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=2)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def run_worker(*, use_xet: bool, force_download: bool = False):
    backend = "xet" if use_xet else "http"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if use_xet:
        env["HF_HUB_DISABLE_XET"] = "0"
        env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
        env["HF_XET_HIGH_PERFORMANCE"] = str(
            os.environ.get("G4L_HF_DOWNLOADER_XET_HIGH_PERFORMANCE", "1")
        )
        range_gets = str(os.environ.get("G4L_HF_DOWNLOADER_XET_NUM_CONCURRENT_RANGE_GETS", "64")).strip()
        if range_gets:
            env["HF_XET_NUM_CONCURRENT_RANGE_GETS"] = range_gets
    else:
        env["HF_HUB_DISABLE_XET"] = "1"
        env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    cmd = [sys.executable, worker_path, "--repo-id", repo_id, "--filename", filename]
    if force_download:
        cmd.append("--force-download")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        bufsize=1,
    )
    if proc.stdout is None:
        terminate_process(proc)
        return False, "no_stdout", "", f"{backend} worker missing stdout"

    started_at = time.time()
    first_byte_at = None
    slow_since = None
    completed_path = ""
    worker_error = ""
    last_percent = -1
    last_emit_t = 0.0

    while True:
        if use_xet:
            now = time.time()
            if first_byte_at is None and (now - started_at) > first_byte_timeout:
                terminate_process(proc)
                return False, "xet_no_first_byte", completed_path, "xet no-first-byte timeout"
            if first_byte_at is not None and slow_since is not None and (now - slow_since) > slow_grace_seconds:
                terminate_process(proc)
                return False, "xet_slow_throughput", completed_path, "xet sustained slow throughput"

        ready, _, _ = select.select([proc.stdout], [], [], 0.4)
        if not ready:
            if proc.poll() is not None:
                break
            continue

        raw_line = proc.stdout.readline()
        if raw_line == "":
            if proc.poll() is not None:
                break
            continue

        line = raw_line.strip()
        if not line:
            continue

        try:
            payload = json.loads(line)
        except Exception:
            continue

        event = str(payload.get("event") or "").lower()
        if event == "progress":
            downloaded = int(payload.get("downloaded_bytes") or 0)
            total = int(payload.get("total_bytes") or 0)
            percent = int(payload.get("percent") or 0)
            speed_bps = float(payload.get("speed_bps") or 0.0)
            if downloaded > 0 and first_byte_at is None:
                first_byte_at = time.time()
            if use_xet and downloaded > 0:
                if speed_bps >= slow_speed_bps:
                    slow_since = None
                else:
                    if slow_since is None:
                        slow_since = time.time()

            now = time.time()
            if percent >= last_percent + 5 or (now - last_emit_t) >= 8.0:
                speed_mbps = speed_bps / (1024.0 * 1024.0)
                print(
                    f"[{label}] {backend}: {percent}% "
                    f"({downloaded}/{total} bytes, {speed_mbps:.2f} MiB/s)",
                    flush=True,
                )
                last_percent = percent
                last_emit_t = now
        elif event == "complete":
            completed_path = str(payload.get("path") or completed_path)
        elif event == "error":
            worker_error = str(payload.get("error") or "unknown downloader error")

    rc = proc.wait()
    if rc != 0:
        reason = f"{backend}_worker_exit_{rc}"
        error_msg = worker_error or f"{backend} worker exited with code {rc}"
        return False, reason, completed_path, error_msg
    if not completed_path:
        return False, f"{backend}_no_path", completed_path, f"{backend} worker completed without file path"

    return True, "", completed_path, ""


def resolve_cached_file() -> str:
    if mode == "off":
        ok, reason, path, err = run_worker(use_xet=False)
        if not ok:
            raise RuntimeError(f"http mode failed: {reason}: {err}")
        return path

    if mode == "on":
        ok, reason, path, err = run_worker(use_xet=True)
        if not ok:
            raise RuntimeError(f"xet mode failed: {reason}: {err}")
        return path

    ok, reason, path, err = run_worker(use_xet=True)
    if ok:
        return path

    print(f"[{label}] xet path failed ({reason}: {err}); retrying with HTTP fallback...", flush=True)
    ok_http, reason_http, path_http, err_http = run_worker(use_xet=False, force_download=True)
    if not ok_http:
        raise RuntimeError(
            f"http fallback failed after xet failure ({reason}: {err}) -> ({reason_http}: {err_http})"
        )
    return path_http


cached_path = resolve_cached_file()
os.makedirs(os.path.dirname(out_path), exist_ok=True)
cached_size = os.path.getsize(cached_path)
if os.path.exists(out_path):
    existing_size = os.path.getsize(out_path)
    if existing_size == cached_size and existing_size > 0:
        print(f"[{label}] already complete: {out_path} ({existing_size} bytes)", flush=True)
    else:
        shutil.copy2(cached_path, out_path)
        print(f"[{label}] refreshed: {out_path} ({cached_size} bytes)", flush=True)
else:
    shutil.copy2(cached_path, out_path)
    print(f"[{label}] complete: {out_path} ({cached_size} bytes)", flush=True)
PY
}

download_required_file() {
  local label="$1"
  local repo="$2"
  local hf_path="$3"
  local relative_output_path="$4"
  local out_path="${CHECKPOINTS_DIR%/}/${relative_output_path}"
  local out_dir
  out_dir="$(dirname "${out_path}")"
  local out_name
  out_name="$(basename "${out_path}")"

  download_with_shared_downloader \
    "${label}" \
    "${repo}" \
    "${hf_path}" \
    "${out_dir}" \
    "${out_name}"
}

resolve_worker_path || {
  echo "could not locate hf_predownload_worker.py under runtime/workspace repositories."
  exit 1
}

ensure_shared_downloader_python

echo "Using ACE root: ${ACE_ROOT}"
echo "Checkpoints dir: ${CHECKPOINTS_DIR}"
echo "Worker: ${WORKER_PATH}"
echo "Shared downloader python: ${DOWNLOADER_PYTHON}"
echo

download_required_file "DiT Base Weights" "ACE-Step/acestep-v15-base" "model.safetensors" "acestep-v15-base/model.safetensors"
download_required_file "DiT Base Config" "ACE-Step/acestep-v15-base" "config.json" "acestep-v15-base/config.json"
download_required_file "DiT Silence Latent" "ACE-Step/acestep-v15-base" "silence_latent.pt" "acestep-v15-base/silence_latent.pt"

download_required_file "Qwen Weights" "ACE-Step/Ace-Step1.5" "Qwen3-Embedding-0.6B/model.safetensors" "Qwen3-Embedding-0.6B/model.safetensors"
download_required_file "Qwen Config" "ACE-Step/Ace-Step1.5" "Qwen3-Embedding-0.6B/config.json" "Qwen3-Embedding-0.6B/config.json"
download_required_file "Qwen Tokenizer" "ACE-Step/Ace-Step1.5" "Qwen3-Embedding-0.6B/tokenizer.json" "Qwen3-Embedding-0.6B/tokenizer.json"
download_required_file "Qwen Tokenizer Config" "ACE-Step/Ace-Step1.5" "Qwen3-Embedding-0.6B/tokenizer_config.json" "Qwen3-Embedding-0.6B/tokenizer_config.json"
download_required_file "Qwen Merges" "ACE-Step/Ace-Step1.5" "Qwen3-Embedding-0.6B/merges.txt" "Qwen3-Embedding-0.6B/merges.txt"
download_required_file "Qwen Vocab" "ACE-Step/Ace-Step1.5" "Qwen3-Embedding-0.6B/vocab.json" "Qwen3-Embedding-0.6B/vocab.json"
download_required_file "Qwen Special Tokens" "ACE-Step/Ace-Step1.5" "Qwen3-Embedding-0.6B/special_tokens_map.json" "Qwen3-Embedding-0.6B/special_tokens_map.json"
download_required_file "Qwen Added Tokens" "ACE-Step/Ace-Step1.5" "Qwen3-Embedding-0.6B/added_tokens.json" "Qwen3-Embedding-0.6B/added_tokens.json"
download_required_file "Qwen Chat Template" "ACE-Step/Ace-Step1.5" "Qwen3-Embedding-0.6B/chat_template.jinja" "Qwen3-Embedding-0.6B/chat_template.jinja"

download_required_file "VAE Weights" "ACE-Step/Ace-Step1.5" "vae/diffusion_pytorch_model.safetensors" "vae/diffusion_pytorch_model.safetensors"
download_required_file "VAE Config" "ACE-Step/Ace-Step1.5" "vae/config.json" "vae/config.json"

echo
echo "All required Carey model files are present."
echo "Quick check:"
ls -lh \
  "${CHECKPOINTS_DIR}/acestep-v15-base/model.safetensors" \
  "${CHECKPOINTS_DIR}/acestep-v15-base/config.json" \
  "${CHECKPOINTS_DIR}/acestep-v15-base/silence_latent.pt" \
  "${CHECKPOINTS_DIR}/Qwen3-Embedding-0.6B/model.safetensors" \
  "${CHECKPOINTS_DIR}/Qwen3-Embedding-0.6B/config.json" \
  "${CHECKPOINTS_DIR}/Qwen3-Embedding-0.6B/tokenizer.json" \
  "${CHECKPOINTS_DIR}/vae/diffusion_pytorch_model.safetensors"
