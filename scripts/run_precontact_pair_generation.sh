#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
  echo "usage: $0 <manifest> <pair-id> <gpu> <rollout-dir> <output-dir> [precontact-offset]" >&2
  exit 2
fi

manifest="$(realpath "$1")"
pair_id="$2"
gpu="$3"
rollout_dir="$(realpath "$4")"
output_dir="$(realpath -m "$5")"
precontact_offset="${6:-10}"
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
  "source /.venv/bin/activate && python /app/scripts/generate_precontact_instruction_pairs.py \
    --source-manifest /pair/$(basename "$manifest") \
    --pair-id '$pair_id' \
    --rollout /rollout \
    --output /data \
    --precontact-offset '$precontact_offset'"
