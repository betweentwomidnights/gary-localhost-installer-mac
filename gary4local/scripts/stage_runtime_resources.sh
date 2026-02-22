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
    --exclude ".mlx_cache/" \
    --exclude "__pycache__/" \
    --exclude "*.pyc" \
    --exclude ".DS_Store" \
    "${source_dir}/" "${destination_dir}/"
}

echo "[stage-runtime] staging runtime payload into ${RUNTIME_DST}"
mkdir -p "${RUNTIME_DST}" "${MANIFEST_DST}"

copy_tree "${REPO_ROOT}/audiocraft-mlx" "${RUNTIME_DST}/audiocraft-mlx"
copy_tree "${REPO_ROOT}/melodyflow" "${RUNTIME_DST}/melodyflow"
copy_tree "${REPO_ROOT}/stable-audio-tools" "${RUNTIME_DST}/stable-audio-tools"

cp "${REPO_ROOT}/control-center/manifest/services.production.json" "${MANIFEST_DST}/services.production.json"

echo "[stage-runtime] done"
