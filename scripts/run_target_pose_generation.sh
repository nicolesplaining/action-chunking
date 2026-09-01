#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 7 || $# -gt 8 ]]; then
  echo "usage: $0 <manifest> <pair-id> <gpu> <rollout-dir> <origin> <snapshot-step> <output-dir> [offsets-m]" >&2
  exit 2
fi

manifest="$(realpath "$1")"
pair_id="$2"
gpu="$3"
rollout_dir="$(realpath "$4")"
origin="$5"
snapshot_step="$6"
output_dir="$(realpath -m "$7")"
offsets="${8:-0.02,0.04,0.06}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

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
  --volume "${rollout_dir}:/rollout:ro" \
  --volume "${output_dir}:/data" \
  --env MUJOCO_GL=egl \
  --env "MUJOCO_EGL_DEVICE_ID=${gpu}" \
  --env PYOPENGL_PLATFORM=egl \
  --env PYTHONPATH=/app/src:/app/scripts:/app/third_party/openpi:/app/third_party/openpi/packages/openpi-client/src:/app/third_party/openpi/third_party/libero \
  action-chunking-libero-client \
  /bin/bash -lc \
  "source /.venv/bin/activate && python /app/scripts/generate_target_pose_pairs.py \
    --source-manifest /pair/$(basename "$manifest") \
    --pair-id '$pair_id' \
    --rollout /rollout \
    --origin '$origin' \
    --snapshot-step '$snapshot_step' \
    --target-side base \
    --offsets '$offsets' \
    --output /data"
