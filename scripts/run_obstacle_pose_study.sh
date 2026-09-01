#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
  echo "usage: $0 <repo-root> <source-manifest> <source-pair-id> <checkpoint> <output-root> <gpu> <policy-port>" >&2
  exit 2
fi

repo_root="$(realpath "$1")"
source_manifest="$(realpath "$2")"
source_pair_id="$3"
checkpoint="$(realpath "$4")"
output="$(realpath -m "$5")"
gpu="$6"
port="$7"
openpi="$repo_root/third_party/openpi"

if ! [[ "$gpu" =~ ^[0-9]+$ && "$port" =~ ^[1-9][0-9]*$ ]]; then
  echo "gpu must be nonnegative and policy port must be positive" >&2
  exit 2
fi

fixtures="$output/fixtures"
clean="$output/clean"
screening="$output/screen"
"$repo_root/scripts/run_obstacle_pose_generation.sh" \
  "$source_manifest" "$source_pair_id" "$gpu" "$fixtures"
PYTHONPATH="$repo_root/src" "$repo_root/.venv/bin/python" \
  "$repo_root/scripts/run_manifest_pair_validations.py" \
  --manifest "$fixtures/manifest.json" \
  --output "$clean" \
  --gpu "$gpu" \
  --port "$port" \
  --noise-seed 0 \
  --save-sim-states
PYTHONPATH="$repo_root/src" "$repo_root/.venv/bin/python" \
  "$repo_root/scripts/screen_obstacle_pose_pairs.py" \
  --manifest "$fixtures/manifest.json" \
  --clean-validation "$clean" \
  --output "$screening"

selected_pair_id="$(
  PYTHONPATH="$repo_root/src" "$repo_root/.venv/bin/python" -c \
    'import json,sys; value=json.load(open(sys.argv[1])); print(value.get("selected_pair_id") or "")' \
    "$screening/summary.json"
)"
if [[ -z "$selected_pair_id" ]]; then
  echo "obstacle placement screen produced no clean-eligible pair"
  exit 0
fi

run_grid() {
  local name="$1"
  local steps="$2"
  local layers="$3"
  local position_mode="$4"
  local dimension_mode="$5"
  local intervention_output="$output/interventions/$name"
  local analysis_output="$output/analysis/$name"
  env CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONPATH="$repo_root/src:$openpi:$openpi/packages/openpi-client/src" \
    "$openpi/.venv/bin/python" "$repo_root/scripts/run_pair_interventions.py" \
    --checkpoint "$checkpoint" \
    --manifest "$screening/selected_manifest.json" \
    --pair-id "$selected_pair_id" \
    --output "$intervention_output" \
    --config pi05_libero \
    --device cuda:0 \
    --noise-seed 0 \
    --num-steps 10 \
    --steps "$steps" \
    --layers "$layers" \
    --position-mode "$position_mode" \
    --dimension-mode "$dimension_mode" \
    --identity-sites anchors
  env PYTHONPATH="$repo_root/src" MPLBACKEND=Agg \
    "$repo_root/.venv/bin/python" "$repo_root/scripts/analyze_pair.py" \
    --input "$intervention_output" \
    --output "$analysis_output"
}

run_grid coarse all all all groups
run_grid positions 0,7,8,9 0,8,14,17 single none
