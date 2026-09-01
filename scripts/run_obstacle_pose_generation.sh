#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 6 ]]; then
  echo "usage: $0 <manifest> <pair-id> <gpu> <output-dir> [fractions] [lateral-offsets-m]" >&2
  exit 2
fi

manifest="$(realpath "$1")"
pair_id="$2"
gpu="$3"
output_dir="$(realpath -m "$4")"
fractions="${5:-0.35,0.50,0.65}"
lateral_offsets="${6:-0.00,-0.05,0.05}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! [[ "$gpu" =~ ^[0-9]+$ ]]; then
  echo "gpu must be a nonnegative integer" >&2
  exit 2
fi
mkdir -p "$output_dir"
if docker info >/dev/null 2>&1; then
  docker_command=(docker)
else
  docker_command=(sudo -n docker)
fi

"${docker_command[@]}" run --rm \
  --network=host \
  --gpus "device=${gpu}" \
  --volume "${repo_root}:/app:ro" \
  --volume "$(dirname "$manifest"):/pair:ro" \
  --volume "${output_dir}:/data" \
  --env MUJOCO_GL=egl \
  --env "MUJOCO_EGL_DEVICE_ID=${gpu}" \
  --env PYOPENGL_PLATFORM=egl \
  --env PYTHONPATH=/app/src:/app/scripts:/app/third_party/openpi:/app/third_party/openpi/packages/openpi-client/src:/app/third_party/openpi/third_party/libero \
  action-chunking-libero-client \
  /bin/bash -lc \
  "source /.venv/bin/activate && python /app/scripts/generate_obstacle_pose_pairs.py \
    --source-manifest /pair/$(basename "$manifest") \
    --pair-id '$pair_id' \
    --fractions '$fractions' \
    --lateral-offsets '$lateral_offsets' \
    --output /data"
