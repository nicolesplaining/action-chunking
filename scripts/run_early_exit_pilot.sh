#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 8 ]]; then
  echo "usage: $0 <repo-root> <checkpoint> <manifest> <clean-validation> <output> <gpu> <port> <server-session>" >&2
  exit 2
fi

repo_root="$(realpath "$1")"
checkpoint="$(realpath "$2")"
manifest="$(realpath "$3")"
clean="$(realpath "$4")"
output="$(realpath -m "$5")"
gpu="$6"
port="$7"
server_session="$8"
openpi="$repo_root/third_party/openpi"

if ! [[ "$gpu" =~ ^[0-9]+$ && "$port" =~ ^[1-9][0-9]*$ ]]; then
  echo "gpu and port must be nonnegative and positive integers" >&2
  exit 2
fi
if ss -H -ltn "sport = :$port" | grep -q .; then
  echo "policy port is already in use: $port" >&2
  exit 1
fi
if screen -list | grep -q "[.]${server_session}[[:space:]]"; then
  echo "policy screen already exists: $server_session" >&2
  exit 1
fi

mkdir -p "$output"
printf -v server_command '%q ' \
  env \
  "CUDA_VISIBLE_DEVICES=$gpu" \
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
    echo "early-exit policy server exited before becoming ready" >&2
    exit 1
  fi
  sleep 5
done
if ! ss -H -ltn "sport = :$port" | grep -q .; then
  echo "early-exit policy server did not become ready" >&2
  exit 1
fi

env PYTHONPATH="$repo_root/src" "$repo_root/.venv/bin/python" \
  "$repo_root/scripts/run_manifest_pair_validations.py" \
  --manifest "$manifest" \
  --output "$output/full_control" \
  --gpu "$gpu" \
  --port "$port" \
  --noise-seed 0 \
  --intervention "$repo_root/configs/interventions/early_exit_10.json" \
  --intervene-replans all

env PYTHONPATH="$repo_root/src" "$repo_root/.venv/bin/python" \
  "$repo_root/scripts/run_manifest_pair_validations.py" \
  --manifest "$manifest" \
  --output "$output/early_exit_7" \
  --gpu "$gpu" \
  --port "$port" \
  --noise-seed 0 \
  --intervention "$repo_root/configs/interventions/early_exit_7.json" \
  --intervene-replans all

env PYTHONPATH="$repo_root/src" "$repo_root/.venv/bin/python" \
  "$repo_root/scripts/analyze_early_exit_pilot.py" \
  --manifest "$manifest" \
  --clean "$clean" \
  --full-control "$output/full_control" \
  --early-exit "$output/early_exit_7" \
  --output "$output/analysis"
