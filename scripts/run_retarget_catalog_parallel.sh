#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 9 ]]; then
  echo "usage: $0 <repo-root> <checkpoint> <plan> <output> <gpu-a> <port-a> <gpu-b> <port-b> <server-session-prefix>" >&2
  exit 2
fi

repo_root="$(realpath "$1")"
checkpoint="$(realpath "$2")"
plan="$(realpath "$3")"
output="$(realpath -m "$4")"
gpu_a="$5"
port_a="$6"
gpu_b="$7"
port_b="$8"
session_prefix="$9"
openpi="$repo_root/third_party/openpi"
session_a="${session_prefix}_a"
session_b="${session_prefix}_b"

for value in "$gpu_a" "$gpu_b"; do
  if ! [[ "$value" =~ ^[0-9]+$ ]]; then
    echo "catalog GPUs must be nonnegative integers" >&2
    exit 2
  fi
done
for value in "$port_a" "$port_b"; do
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "catalog ports must be positive integers" >&2
    exit 2
  fi
done
if [[ "$gpu_a" == "$gpu_b" || "$port_a" == "$port_b" ]]; then
  echo "parallel catalog workers require distinct GPUs and ports" >&2
  exit 2
fi
if [[ -n "$(git -C "$repo_root" status --porcelain=v1 --untracked-files=all)" ]]; then
  echo "catalog screening requires a completely clean worktree" >&2
  exit 1
fi
expected_openpi_commit="215abfb217dbac7d5f1273282331b9b1866c0479"
if [[ "$(git -C "$openpi" rev-parse HEAD)" != "$expected_openpi_commit" ]]; then
  echo "catalog screening requires the pinned OpenPI revision" >&2
  exit 1
fi
if [[ ! -f "$checkpoint/model.safetensors" || ! -f "$checkpoint/config.json" ]]; then
  echo "catalog policy checkpoint is incomplete" >&2
  exit 1
fi
for gpu in "$gpu_a" "$gpu_b"; do
  if [[ -n "$(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)" ]]; then
    echo "catalog GPU $gpu has a competing compute process" >&2
    exit 1
  fi
done
for port in "$port_a" "$port_b"; do
  if ss -H -ltn "sport = :$port" | grep -q .; then
    echo "catalog policy port is already in use: $port" >&2
    exit 1
  fi
done
for session in "$session_a" "$session_b"; do
  if screen -list | grep -q "[.]${session}[[:space:]]"; then
    echo "catalog policy screen already exists: $session" >&2
    exit 1
  fi
done

code_commit="$(git -C "$repo_root" rev-parse HEAD)"
plan_digest="$(sha256sum "$plan" | cut -d ' ' -f 1)"
if [[ -e "$output" && ! -f "$output/code_commit.txt" ]]; then
  echo "existing catalog output lacks a code-commit binding" >&2
  exit 1
fi
mkdir -p "$output"
if [[ -f "$output/code_commit.txt" ]] && ! grep -qx "$code_commit" "$output/code_commit.txt"; then
  echo "existing catalog output uses a different code commit" >&2
  exit 1
fi
if [[ -f "$output/plan.sha256" ]] && ! grep -qx "$plan_digest" "$output/plan.sha256"; then
  echo "existing catalog output uses a different frozen plan" >&2
  exit 1
fi
printf '%s\n' "$code_commit" >"$output/code_commit.txt"
printf '%s\n' "$plan_digest" >"$output/plan.sha256"
nvidia-smi --id="$gpu_a,$gpu_b" \
  --query-gpu=index,uuid,name,driver_version,memory.total --format=csv \
  >"$output/gpu_preflight.csv"

start_server() {
  local gpu="$1"
  local port="$2"
  local session="$3"
  local log="$4"
  local command
  local quoted_log
  printf -v command '%q ' \
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
  printf -v quoted_log '%q' "$log"
  screen -dmS "$session" bash -lc "$command > $quoted_log 2>&1"
}

cleanup() {
  for session in "$session_a" "$session_b"; do
    if screen -list | grep -q "[.]${session}[[:space:]]"; then
      screen -S "$session" -X quit
    fi
  done
}
trap cleanup EXIT

start_server "$gpu_a" "$port_a" "$session_a" "$output/server_a.log"
start_server "$gpu_b" "$port_b" "$session_b" "$output/server_b.log"

for _attempt in {1..120}; do
  ready=0
  for port in "$port_a" "$port_b"; do
    if ss -H -ltn "sport = :$port" | grep -q .; then
      ready=$((ready + 1))
    fi
  done
  if [[ "$ready" -eq 2 ]]; then
    break
  fi
  for session in "$session_a" "$session_b"; do
    if ! screen -list | grep -q "[.]${session}[[:space:]]"; then
      echo "catalog policy server exited before becoming ready: $session" >&2
      exit 1
    fi
  done
  sleep 5
done
for port in "$port_a" "$port_b"; do
  if ! ss -H -ltn "sport = :$port" | grep -q .; then
    echo "catalog policy server did not become ready: $port" >&2
    exit 1
  fi
done

env PYTHONPATH="$repo_root/src" \
  "$repo_root/.venv/bin/python" "$repo_root/scripts/run_retarget_catalog_screen.py" \
  --plan "$plan" \
  --output "$output" \
  --workers "$gpu_a:$port_a,$gpu_b:$port_b" \
  --noise-seed 0
