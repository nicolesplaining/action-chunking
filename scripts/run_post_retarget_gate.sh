#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 8 ]]; then
  echo "usage: $0 <gate-summary> <candidate-root> <utility-output> <catalog-plan> <catalog-output> <gpu> <port> <noise-seed>" >&2
  exit 2
fi

gate_summary="$(realpath "$1")"
candidate_root="$(realpath "$2")"
utility_output="$(realpath -m "$3")"
catalog_plan="$(realpath "$4")"
catalog_output="$(realpath -m "$5")"
gpu="$6"
port="$7"
noise_seed="$8"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! [[ "$gpu" =~ ^[0-9]+$ && "$port" =~ ^[1-9][0-9]*$ && "$noise_seed" =~ ^[0-9]+$ ]]; then
  echo "gpu, port, and noise seed must be nonnegative integers" >&2
  exit 2
fi
eligible_directions="$(
  PYTHONPATH="$repo_root/src" "$repo_root/.venv/bin/python" -c \
    'import json,sys; value=json.load(open(sys.argv[1])); assert value.get("selection_uses_continuation_outcomes") is False; print(int(value["eligible_directions"]))' \
    "$gate_summary"
)"

if (( eligible_directions > 0 )); then
  exec env PYTHONPATH="$repo_root/src" "$repo_root/.venv/bin/python" \
    "$repo_root/scripts/run_eligible_retarget_study.py" \
    --gate-summary "$gate_summary" \
    --candidate-root "$candidate_root" \
    --output "$utility_output" \
    --gpu "$gpu" \
    --port "$port" \
    --noise-seed "$noise_seed"
fi

exec env PYTHONPATH="$repo_root/src" "$repo_root/.venv/bin/python" \
  "$repo_root/scripts/run_retarget_catalog_screen.py" \
  --plan "$catalog_plan" \
  --output "$catalog_output" \
  --gpu "$gpu" \
  --port "$port" \
  --noise-seed "$noise_seed"
