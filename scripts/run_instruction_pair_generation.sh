#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 <suite> <base-task> <donor-task> <count> <render-gpu> <output-dir>" >&2
  exit 2
fi

suite="$1"
base_task="$2"
donor_task="$3"
count="$4"
render_gpu="$5"
output_dir="$(realpath -m "$6")"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "$output_dir"
if docker info >/dev/null 2>&1; then
  docker_command=(docker)
else
  docker_command=(sudo -n docker)
fi

"${docker_command[@]}" run --rm \
  --gpus "device=${render_gpu}" \
  --volume "${repo_root}:/app:ro" \
  --volume "${output_dir}:/data" \
  --env MUJOCO_GL=egl \
  --env "MUJOCO_EGL_DEVICE_ID=${render_gpu}" \
  --env PYOPENGL_PLATFORM=egl \
  --env PYTHONPATH=/app/src:/app/third_party/openpi:/app/third_party/openpi/packages/openpi-client/src:/app/third_party/openpi/third_party/libero \
  action-chunking-libero-client \
  /bin/bash -lc \
  "source /.venv/bin/activate && python /app/scripts/generate_libero_instruction_pairs.py \
    --suite '$suite' \
    --base-task '$base_task' \
    --donor-task '$donor_task' \
    --count '$count' \
    --output /data"
