#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 8 ]]; then
  echo "usage: $0 <repo-root> <checkpoint> <catalog-summary> <orientation-calibration> <utility-output> <gpu> <port> <server-session>" >&2
  exit 2
fi

repo_root="$(realpath "$1")"
checkpoint="$(realpath "$2")"
catalog_summary="$(realpath "$3")"
orientation_calibration="$(realpath "$4")"
utility_output="$(realpath -m "$5")"
gpu="$6"
port="$7"
server_session="$8"
openpi="$repo_root/third_party/openpi"

if ! [[ "$gpu" =~ ^[0-9]+$ && "$port" =~ ^[1-9][0-9]*$ ]]; then
  echo "utility GPU must be nonnegative and port must be positive" >&2
  exit 2
fi
if [[ -n "$(git -C "$repo_root" status --porcelain=v1 --untracked-files=all)" ]]; then
  echo "retarget utility requires a completely clean worktree" >&2
  exit 1
fi
action_chunking_commit="$(git -C "$repo_root" rev-parse HEAD)"
expected_openpi_commit="215abfb217dbac7d5f1273282331b9b1866c0479"
if [[ "$(git -C "$openpi" rev-parse HEAD)" != "$expected_openpi_commit" ]]; then
  echo "retarget utility requires the pinned OpenPI revision" >&2
  exit 1
fi
if [[ ! -f "$checkpoint/model.safetensors" || ! -f "$checkpoint/config.json" ]]; then
  echo "retarget utility policy checkpoint is incomplete" >&2
  exit 1
fi
if [[ -n "$(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)" ]]; then
  echo "retarget utility GPU has a competing compute process" >&2
  exit 1
fi
if ss -H -ltn "sport = :$port" | grep -q .; then
  echo "retarget utility policy port is already in use" >&2
  exit 1
fi
if screen -list | grep -q "[.]${server_session}[[:space:]]"; then
  echo "retarget utility policy screen already exists" >&2
  exit 1
fi

catalog_commit="$(
  "$repo_root/.venv/bin/python" -c \
    'import json,sys; value=json.load(open(sys.argv[1])); assert value.get("selection_uses_continuation_outcomes") is False; assert value.get("stop_threshold_reached") is True or value.get("catalog_exhausted") is True; print(value["code_commit"])' \
    "$catalog_summary"
)"
if [[ "$catalog_commit" != "$action_chunking_commit" ]]; then
  echo "catalog and utility must use the same action-chunking commit" >&2
  exit 1
fi
if [[ -e "$utility_output" && ! -f "$utility_output/code_commit.txt" ]]; then
  echo "existing retarget utility output lacks a code-commit binding" >&2
  exit 1
fi
if [[ -f "$utility_output/code_commit.txt" ]] && ! grep -qx "$action_chunking_commit" "$utility_output/code_commit.txt"; then
  echo "existing retarget utility output uses a different code commit" >&2
  exit 1
fi

handoff="$(dirname "$catalog_summary")/handoff"
env PYTHONPATH="$repo_root/src" "$repo_root/.venv/bin/python" \
  "$repo_root/scripts/prepare_catalog_retarget_study.py" \
  --catalog-summary "$catalog_summary" \
  --output "$handoff"

command=""
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
server_log="$utility_output.policy_server.log"
quoted_log=""
printf -v quoted_log '%q' "$server_log"
screen -dmS "$server_session" bash -lc "$command >> $quoted_log 2>&1"

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
    echo "retarget utility policy server exited before becoming ready" >&2
    exit 1
  fi
  sleep 5
done
if ! ss -H -ltn "sport = :$port" | grep -q .; then
  echo "retarget utility policy server did not become ready" >&2
  exit 1
fi

env PYTHONPATH="$repo_root/src" "$repo_root/.venv/bin/python" \
  "$repo_root/scripts/run_eligible_retarget_study.py" \
  --gate-summary "$handoff/gate_summary.json" \
  --candidate-index "$handoff/candidate_index.json" \
  --output "$utility_output" \
  --orientation-calibration "$orientation_calibration" \
  --gpu "$gpu" \
  --port "$port" \
  --noise-seed 0 \
  --action-chunking-commit "$action_chunking_commit" \
  --require-precomputed-predictions
