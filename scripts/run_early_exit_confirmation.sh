#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 8 ]]; then
  echo "usage: $0 <repo-root> <checkpoint> <pilot-summary> <output> <policy-gpu> <sim-gpu> <port> <server-session>" >&2
  exit 2
fi

repo_root="$(realpath "$1")"
checkpoint="$(realpath "$2")"
pilot_summary="$(realpath "$3")"
output="$(realpath -m "$4")"
policy_gpu="$5"
sim_gpu="$6"
port="$7"
server_session="$8"
openpi="$repo_root/third_party/openpi"

if ! [[ "$policy_gpu" =~ ^[0-9]+$ && "$sim_gpu" =~ ^[0-9]+$ && "$port" =~ ^[1-9][0-9]*$ ]]; then
  echo "policy GPU, simulator GPU, and port must be nonnegative and positive integers" >&2
  exit 2
fi
if [[ "$policy_gpu" == "$sim_gpu" ]]; then
  echo "confirmation requires separate policy and simulator GPUs" >&2
  exit 2
fi
env PYTHONPATH="$repo_root/src" "$repo_root/.venv/bin/python" -c \
  'import json,sys; d=json.load(open(sys.argv[1])); assert d.get("pilot_positive") is True; assert int(d["eligible_scene_clusters"]) == 15; assert int(d["composite_preserved_clusters"]) >= 14; assert d.get("all_compute_counts_exact") is True' \
  "$pilot_summary"
if ss -H -ltn "sport = :$port" | grep -q .; then
  echo "policy port is already in use: $port" >&2
  exit 1
fi
if screen -list | grep -q "[.]${server_session}[[:space:]]"; then
  echo "policy screen already exists: $server_session" >&2
  exit 1
fi
if [[ -n "$(nvidia-smi --id="$policy_gpu" --query-compute-apps=pid --format=csv,noheader,nounits)" ]]; then
  echo "policy GPU has a competing compute process" >&2
  exit 1
fi

mkdir -p "$output"
nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total --format=csv >"$output/gpu_preflight.csv"
git -C "$repo_root" rev-parse HEAD >"$output/code_commit.txt"
sha256sum "$pilot_summary" >"$output/pilot_summary.sha256"

printf -v server_command '%q ' \
  env \
  "CUDA_VISIBLE_DEVICES=$policy_gpu" \
  "PYTHONPATH=$repo_root/src:$openpi:$openpi/packages/openpi-client/src" \
  "$openpi/.venv/bin/python" \
  "$repo_root/scripts/serve_intervention_policy.py" \
  --checkpoint "$checkpoint" \
  --config pi05_libero \
  --device cuda:0 \
  --port "$port" \
  --num-steps 10
printf -v server_log '%q' "$output/server.log"
screen -dmS "$server_session" bash -lc "$server_command > $server_log 2>&1"

cleanup() {
  if screen -list | grep -q "[.]${server_session}[[:space:]]"; then
    screen -S "$server_session" -X quit
  fi
}
trap cleanup EXIT

for _attempt in {1..120}; do
  if ss -H -ltn "sport = :$port" | grep -q .; then
    break
  fi
  if ! screen -list | grep -q "[.]${server_session}[[:space:]]"; then
    echo "confirmation policy server exited before becoming ready" >&2
    exit 1
  fi
  sleep 5
done
if ! ss -H -ltn "sport = :$port" | grep -q .; then
  echo "confirmation policy server did not become ready" >&2
  exit 1
fi

if docker info >/dev/null 2>&1; then
  docker_command=(docker)
else
  docker_command=(sudo -n docker)
fi
"${docker_command[@]}" run --rm \
  --network=host \
  --gpus "device=$sim_gpu" \
  --volume "$repo_root:/app:ro" \
  --volume "$output:/data" \
  --env MUJOCO_GL=egl \
  --env "MUJOCO_EGL_DEVICE_ID=$sim_gpu" \
  --env PYOPENGL_PLATFORM=egl \
  action-chunking-libero-client \
  /bin/bash -lc \
  "source /.venv/bin/activate && PYTHONPATH=/app/src:/app/third_party/openpi:/app/third_party/openpi/packages/openpi-client/src:/app/third_party/openpi/third_party/libero python /app/scripts/run_early_exit_suite_confirmation.py --host 0.0.0.0 --port '$port' --output /data --seed 7 --noise-seed 0"

env PYTHONPATH="$repo_root/src" "$repo_root/.venv/bin/python" \
  "$repo_root/scripts/analyze_early_exit_confirmation.py" \
  --root "$output" \
  --output "$output/analysis"
