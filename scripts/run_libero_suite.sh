#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 <suite> <gpu> <port> <trials-per-task> <output-dir>" >&2
  exit 2
fi

suite="$1"
gpu="$2"
port="$3"
trials="$4"
output_dir="$5"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "$output_dir/videos"

if docker info >/dev/null 2>&1; then
  docker_command=(docker)
else
  docker_command=(sudo -n docker)
fi

client_args="--args.host 0.0.0.0 --args.port ${port} --args.task-suite-name ${suite} --args.num-trials-per-task ${trials} --args.video-out-path /data/videos --args.seed 7"

"${docker_command[@]}" run --rm \
  --network=host \
  --gpus "device=${gpu}" \
  --volume "${repo_root}:/app" \
  --volume "${output_dir}:/data" \
  --env MUJOCO_GL=egl \
  --env "MUJOCO_EGL_DEVICE_ID=${gpu}" \
  --env PYOPENGL_PLATFORM=egl \
  --env "CLIENT_ARGS=${client_args}" \
  action-chunking-libero-client
