#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 <event-sae-root> <awe-root> <trajectory-jsonl> <output-dir> <mode> <threshold>" >&2
  exit 2
fi

event_sae_root="$(realpath "$1")"
awe_root="$(realpath "$2")"
trajectory="$(realpath "$3")"
output_dir="$(realpath -m "$4")"
mode="$5"
threshold="$6"

event_sae_commit="f7a000024a32d8b9ee8e92aab5e79694a2f2bc1c"
awe_commit="7197bb86a20784666dabed90e6eabcf8bb1e9912"
if [[ "$(git -C "$event_sae_root" rev-parse HEAD)" != "$event_sae_commit" ]]; then
  echo "Event-SAE checkout does not match pinned commit $event_sae_commit" >&2
  exit 1
fi
if [[ "$(git -C "$awe_root" rev-parse HEAD)" != "$awe_commit" ]]; then
  echo "AWE checkout does not match pinned commit $awe_commit" >&2
  exit 1
fi

mkdir -p "$output_dir"
if docker info >/dev/null 2>&1; then
  docker_command=(docker)
else
  docker_command=(sudo -n docker)
fi

"${docker_command[@]}" run --rm \
  --volume "${event_sae_root}:/event-sae:ro" \
  --volume "${awe_root}:/awe:ro" \
  --volume "$(dirname "$trajectory"):/records:ro" \
  --volume "${output_dir}:/output" \
  --env PYTHONPATH=/event-sae:/awe:/app/third_party/openpi/third_party/libero \
  action-chunking-event-keyframes \
  /bin/bash -lc \
  "source /.venv/bin/activate && python /event-sae/scripts/extract_keyframes.py \
    --trajectory-records-path /records/$(basename "$trajectory") \
    --output-dir /output \
    --waypoint-mode '$mode' \
    --err-threshold '$threshold'"
