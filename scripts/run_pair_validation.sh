#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 || $# -gt 7 ]]; then
  echo "usage: $0 <manifest> <pair-id> <gpu> <port> <noise-seed> <output-dir> [clean-screen-jsonl]" >&2
  exit 2
fi

manifest="$(realpath "$1")"
pair_id="$2"
gpu="$3"
port="$4"
noise_seed="$5"
output_dir="$(realpath -m "$6")"
clean_screen="${7:-}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "$output_dir"
if docker info >/dev/null 2>&1; then
  docker_command=(docker)
else
  docker_command=(sudo -n docker)
fi

extra_mounts=()
expected_args=()
if [[ -n "$clean_screen" ]]; then
  clean_screen="$(realpath "$clean_screen")"
  extra_mounts+=(--volume "$(dirname "$clean_screen"):/screen:ro")
  expected_args=(--expected-clean-screen "/screen/$(basename "$clean_screen")")
fi

"${docker_command[@]}" run --rm \
  --network=host \
  --gpus "device=${gpu}" \
  --volume "${repo_root}:/app:ro" \
  --volume "$(dirname "$manifest"):/pair:ro" \
  --volume "${output_dir}:/data" \
  "${extra_mounts[@]}" \
  --env MUJOCO_GL=egl \
  --env "MUJOCO_EGL_DEVICE_ID=${gpu}" \
  --env PYOPENGL_PLATFORM=egl \
  --env PYTHONPATH=/app/src:/app/third_party/openpi:/app/third_party/openpi/packages/openpi-client/src:/app/third_party/openpi/third_party/libero \
  action-chunking-libero-client \
  /bin/bash -lc \
  "source /.venv/bin/activate && python /app/scripts/validate_libero_pair_rollouts.py \
    --manifest /pair/$(basename "$manifest") \
    --pair-id '$pair_id' \
    --output /data \
    --port '$port' \
    --noise-seed '$noise_seed' \
    ${expected_args[*]}"
