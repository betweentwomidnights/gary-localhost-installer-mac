#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
REPO_ROOT="$(cd "${PROJECT_DIR}/.." && pwd)"
RESOURCES_DIR="${TARGET_BUILD_DIR}/${UNLOCALIZED_RESOURCES_FOLDER_PATH}"
RUNTIME_DST="${RESOURCES_DIR}/runtime"
MANIFEST_DST="${RESOURCES_DIR}/manifest"

if [[ ! -d "${RESOURCES_DIR}" ]]; then
  echo "[stage-runtime] resources directory is missing: ${RESOURCES_DIR}"
  exit 1
fi

copy_tree() {
  local source_dir="$1"
  local destination_dir="$2"

  if [[ ! -d "${source_dir}" ]]; then
    echo "[stage-runtime] missing source directory: ${source_dir}"
    exit 1
  fi

  mkdir -p "${destination_dir}"

  rsync -a --delete \
    --exclude ".git/" \
    --exclude ".venv/" \
    --exclude ".claude/" \
    --exclude ".cache/" \
    --exclude ".pytest_cache/" \
    --exclude ".mlx_cache/" \
    --exclude "checkpoints/" \
    --exclude "__pycache__/" \
    --exclude "*.pyc" \
    --exclude "smoke-tests/" \
    --exclude "smoke.wav" \
    --exclude "smoke.mp3" \
    --exclude ".DS_Store" \
    "${source_dir}/" "${destination_dir}/"
}

echo "[stage-runtime] staging runtime payload into ${RUNTIME_DST}"
mkdir -p "${RUNTIME_DST}" "${MANIFEST_DST}"

copy_tree "${REPO_ROOT}/audiocraft-mlx" "${RUNTIME_DST}/audiocraft-mlx"
copy_tree "${REPO_ROOT}/ace-lego" "${RUNTIME_DST}/ace-lego"
copy_tree "${REPO_ROOT}/melodyflow" "${RUNTIME_DST}/melodyflow"
copy_tree "${REPO_ROOT}/stable-audio-tools" "${RUNTIME_DST}/stable-audio-tools"
copy_tree "${REPO_ROOT}/foundation" "${RUNTIME_DST}/foundation"

mkdir -p "${RUNTIME_DST}/scripts"
cp "${REPO_ROOT}/scripts/download_carey_models.sh" "${RUNTIME_DST}/scripts/download_carey_models.sh"
chmod +x "${RUNTIME_DST}/scripts/download_carey_models.sh"

# Keep app bundle lean; checkpoints will be managed in cache/runtime.
rm -rf "${RUNTIME_DST}/ace-lego/ACE-Step-1.5/checkpoints" || true
rm -rf "${RUNTIME_DST}/ace-lego/smoke-tests" || true
rm -rf "${RUNTIME_DST}/ace-lego/ACE-Step-1.5/.cache" || true
rm -f "${RUNTIME_DST}/ace-lego/smoke.wav" || true
# Remove non-runtime ACE-Step content (docs/media/examples/repo metadata).
rm -rf "${RUNTIME_DST}/ace-lego/ACE-Step-1.5/docs" || true
rm -rf "${RUNTIME_DST}/ace-lego/ACE-Step-1.5/examples" || true
rm -rf "${RUNTIME_DST}/ace-lego/ACE-Step-1.5/assets" || true
rm -rf "${RUNTIME_DST}/ace-lego/ACE-Step-1.5/.github" || true
rm -rf "${RUNTIME_DST}/ace-lego/ACE-Step-1.5/.githooks" || true
rm -rf "${RUNTIME_DST}/ace-lego/ACE-Step-1.5/docker-patches" || true
rm -rf "${RUNTIME_DST}/ace-lego/ACE-Step-1.5/gradio_outputs" || true

cp "${REPO_ROOT}/control-center/manifest/services.production.json" "${MANIFEST_DST}/services.production.json"

echo "[stage-runtime] done"
