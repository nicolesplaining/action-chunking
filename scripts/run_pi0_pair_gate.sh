#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 9 ]]; then
  echo "usage: $0 <repo-root> <checkpoint-label-29999> <manifest> <suite-summary> <output-dir> <gpu> <port> <server-session> <noise-seed>" >&2
  exit 2
fi

repo_root="$(realpath "$1")"
checkpoint="$(realpath "$2")"
manifest="$(realpath "$3")"
suite_summary="$(realpath "$4")"
output="$(realpath -m "$5")"
gpu="$6"
port="$7"
server_session="$8"
noise_seed="$9"
openpi="$repo_root/third_party/openpi"

if ! PYTHONPATH="$repo_root/src" "$repo_root/.venv/bin/python" \
  "$repo_root/scripts/validate_pi0_final_checkpoint.py" --checkpoint "$checkpoint" >/dev/null; then
  echo "pi0 pair gate requires the frozen finalized 30,000-update checkpoint" >&2
  exit 1
fi
if ! [[ "$gpu" =~ ^[0-9]+$ && "$port" =~ ^[1-9][0-9]*$ && "$noise_seed" =~ ^[0-9]+$ ]]; then
  echo "gpu, port, and noise seed must be nonnegative integers" >&2
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
if [[ -e "$output/pairs/validation_summary.json" || -e "$output/competence_gate.json" ]]; then
  echo "pi0 pair-gate output already exists: $output" >&2
  exit 1
fi

mkdir -p "$output"
printf -v server_command '%q ' \
  env \
  "CUDA_VISIBLE_DEVICES=$gpu" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONPATH="$repo_root/src:$openpi:$openpi/packages/openpi-client/src" \
  "$openpi/.venv/bin/python" \
  "$repo_root/scripts/serve_noise_policy.py" \
  --checkpoint "$checkpoint" \
  --config pi0_libero \
  --port "$port"
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
    echo "pi0 explicit-noise server exited before becoming ready" >&2
    exit 1
  fi
  sleep 5
done
if ! ss -H -ltn "sport = :$port" | grep -q .; then
  echo "pi0 explicit-noise server did not become ready" >&2
  exit 1
fi

PYTHONPATH="$repo_root/src" "$repo_root/.venv/bin/python" \
  "$repo_root/scripts/run_manifest_pair_validations.py" \
  --manifest "$manifest" \
  --output "$output/pairs" \
  --gpu "$gpu" \
  --port "$port" \
  --noise-seed "$noise_seed" \
  --save-sim-states
PYTHONPATH="$repo_root/src" "$repo_root/.venv/bin/python" \
  "$repo_root/scripts/evaluate_pi0_competence.py" \
  --suite-summary "$suite_summary" \
  --pair-summary "$output/pairs/validation_summary.json" \
  --output "$output"
