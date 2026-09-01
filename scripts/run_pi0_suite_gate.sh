#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 <repo-root> <checkpoint-30000> <output-dir> <gpu> <port> <server-session>" >&2
  exit 2
fi

repo_root="$(realpath "$1")"
checkpoint="$(realpath "$2")"
output="$(realpath -m "$3")"
gpu="$4"
port="$5"
server_session="$6"
openpi="$repo_root/third_party/openpi"

if [[ "$(basename "$checkpoint")" != "30000" ]]; then
  echo "pi0 suite gate accepts only the finalized step-30000 checkpoint" >&2
  exit 1
fi
for required in _CHECKPOINT_METADATA params/manifest.ocdbt train_state/manifest.ocdbt; do
  if [[ ! -f "$checkpoint/$required" ]]; then
    echo "finalized checkpoint artifact is missing: $checkpoint/$required" >&2
    exit 1
  fi
done
if [[ ! -x "$openpi/.venv/bin/python" || ! -x "$repo_root/.venv/bin/python" ]]; then
  echo "required OpenPI and analysis environments are missing" >&2
  exit 1
fi
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
if [[ -e "$output/client.log" || -e "$output/summary/summary.json" ]]; then
  echo "pi0 suite output already exists: $output" >&2
  exit 1
fi

mkdir -p "$output" "$output/rollout" "$output/summary"
printf -v server_command '%q ' \
  env \
  "CUDA_VISIBLE_DEVICES=$gpu" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  "$openpi/.venv/bin/python" \
  "$openpi/scripts/serve_policy.py" \
  --port "$port" \
  policy:checkpoint \
  --policy.config pi0_libero \
  --policy.dir "$checkpoint"
printf -v server_log '%q' "$output/server.log"
screen -dmS "$server_session" bash -lc "$server_command > $server_log 2>&1"

owns_server=true
cleanup() {
  if [[ "$owns_server" == true ]] && screen -list | grep -q "[.]${server_session}[[:space:]]"; then
    screen -S "$server_session" -X quit
  fi
}
trap cleanup EXIT

for _attempt in {1..120}; do
  if ss -H -ltn "sport = :$port" | grep -q .; then
    break
  fi
  if ! screen -list | grep -q "[.]${server_session}[[:space:]]"; then
    echo "pi0 policy server exited before becoming ready" >&2
    exit 1
  fi
  sleep 5
done
if ! ss -H -ltn "sport = :$port" | grep -q .; then
  echo "pi0 policy server did not become ready" >&2
  exit 1
fi

"$repo_root/scripts/run_libero_suite.sh" \
  libero_goal "$gpu" "$port" 50 "$output/rollout" \
  > "$output/client.log" 2>&1
PYTHONPATH="$repo_root/src" "$repo_root/.venv/bin/python" \
  "$repo_root/scripts/summarize_libero_baseline.py" \
  --log "$output/client.log" \
  --suite libero_goal \
  --output "$output/summary" \
  --expected-episodes 500 \
  --expected-tasks 10 \
  --expected-episodes-per-task 50

echo "pi0 official libero_goal suite evaluation complete"
