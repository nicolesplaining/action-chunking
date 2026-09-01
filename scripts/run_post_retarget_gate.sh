#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 9 ]]; then
  echo "usage: $0 <gate-summary> <candidate-root> <utility-output> <catalog-plan> <catalog-output> <orientation-calibration> <gpu> <port> <noise-seed>" >&2
  exit 2
fi

gate_summary="$(realpath "$1")"
candidate_root="$(realpath "$2")"
utility_output="$(realpath -m "$3")"
catalog_plan="$(realpath "$4")"
catalog_output="$(realpath -m "$5")"
orientation_calibration="$(realpath "$6")"
gpu="$7"
port="$8"
noise_seed="$9"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -n "$(git -C "$repo_root" status --porcelain=v1 --untracked-files=all)" ]]; then
  echo "retarget utility requires a completely clean worktree" >&2
  exit 1
fi
action_chunking_commit="$(git -C "$repo_root" rev-parse HEAD)"

if ! [[ "$gpu" =~ ^[0-9]+$ && "$port" =~ ^[1-9][0-9]*$ && "$noise_seed" =~ ^[0-9]+$ ]]; then
  echo "gpu, port, and noise seed must be nonnegative integers" >&2
  exit 2
fi
eligible_directions="$(
  PYTHONPATH="$repo_root/src" "$repo_root/.venv/bin/python" -c \
    'import json,sys; value=json.load(open(sys.argv[1])); candidates=int(value["candidate_manifests"]); completed=int(value["completed_directions"]); rows=value["rows"]; assert value.get("selection_uses_continuation_outcomes") is False; assert candidates > 0; assert completed == 2 * candidates == len(rows), "endpoint gate is incomplete"; print(int(value["eligible_directions"]))' \
    "$gate_summary"
)"

if (( eligible_directions > 0 )); then
  env PYTHONPATH="$repo_root/src" "$repo_root/.venv/bin/python" \
    "$repo_root/scripts/run_eligible_retarget_study.py" \
    --gate-summary "$gate_summary" \
    --candidate-root "$candidate_root" \
    --output "$utility_output" \
    --orientation-calibration "$orientation_calibration" \
    --gpu "$gpu" \
    --port "$port" \
    --noise-seed "$noise_seed" \
    --action-chunking-commit "$action_chunking_commit"
fi

plan_digest="$(sha256sum "$catalog_plan" | cut -d ' ' -f 1)"
mkdir -p "$catalog_output"
if [[ -f "$catalog_output/code_commit.txt" ]] && ! grep -qx "$action_chunking_commit" "$catalog_output/code_commit.txt"; then
  echo "existing catalog output uses a different code commit" >&2
  exit 1
fi
if [[ -f "$catalog_output/plan.sha256" ]] && ! grep -qx "$plan_digest" "$catalog_output/plan.sha256"; then
  echo "existing catalog output uses a different frozen plan" >&2
  exit 1
fi
printf '%s\n' "$action_chunking_commit" >"$catalog_output/code_commit.txt"
printf '%s\n' "$plan_digest" >"$catalog_output/plan.sha256"

env PYTHONPATH="$repo_root/src" "$repo_root/.venv/bin/python" \
  "$repo_root/scripts/run_retarget_catalog_screen.py" \
  --plan "$catalog_plan" \
  --output "$catalog_output" \
  --gpu "$gpu" \
  --port "$port" \
  --noise-seed "$noise_seed"

handoff="$catalog_output/handoff"
env PYTHONPATH="$repo_root/src" "$repo_root/.venv/bin/python" \
  "$repo_root/scripts/prepare_catalog_retarget_study.py" \
  --catalog-summary "$catalog_output/summary.json" \
  --output "$handoff"

catalog_eligible_directions="$(
  PYTHONPATH="$repo_root/src" "$repo_root/.venv/bin/python" -c \
    'import json,sys; value=json.load(open(sys.argv[1])); assert value.get("selection_uses_continuation_outcomes") is False; print(int(value["eligible_directions"]))' \
    "$handoff/gate_summary.json"
)"
if (( catalog_eligible_directions > 0 )); then
  exec env PYTHONPATH="$repo_root/src" "$repo_root/.venv/bin/python" \
    "$repo_root/scripts/run_eligible_retarget_study.py" \
    --gate-summary "$handoff/gate_summary.json" \
    --candidate-index "$handoff/candidate_index.json" \
    --output "$catalog_output/utility" \
    --orientation-calibration "$orientation_calibration" \
    --gpu "$gpu" \
    --port "$port" \
    --noise-seed "$noise_seed" \
    --action-chunking-commit "$action_chunking_commit"
fi
