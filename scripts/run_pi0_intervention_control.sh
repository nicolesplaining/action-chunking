#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 12 ]]; then
  echo "usage: $0 <repo-root> <competence-gate> <parity-summary> <pytorch-checkpoint> <manifest> <pi0-clean-validation> <pi05-clean-validation> <pi05-coarse-interventions> <pi05-position-analysis> <output-root> <coarse-gpu> <position-gpu>" >&2
  exit 2
fi

repo_root="$(realpath "$1")"
competence_gate="$(realpath "$2")"
parity_summary="$(realpath "$3")"
checkpoint="$(realpath "$4")"
manifest="$(realpath "$5")"
pi0_clean="$(realpath "$6")"
pi05_clean="$(realpath "$7")"
pi05_coarse_interventions="$(realpath "$8")"
pi05_position_analysis="$(realpath "$9")"
output="$(realpath -m "${10}")"
coarse_gpu="${11}"
position_gpu="${12}"
openpi="$repo_root/third_party/openpi"

for selected_gpu in "$coarse_gpu" "$position_gpu"; do
  if ! [[ "$selected_gpu" =~ ^[0-9]+$ ]]; then
    echo "GPU indices must be nonnegative integers" >&2
    exit 2
  fi
  if [[ -n "$(nvidia-smi --id="$selected_gpu" --query-compute-apps=pid --format=csv,noheader,nounits)" ]]; then
    echo "pi0 intervention GPU $selected_gpu has a competing compute process" >&2
    exit 1
  fi
done
if [[ "$coarse_gpu" == "$position_gpu" ]]; then
  echo "registered pi0 intervention grids require two distinct GPUs" >&2
  exit 2
fi
if [[ -n "$(git -C "$repo_root" status --porcelain=v1 --untracked-files=all)" ]]; then
  echo "pi0 interventions require a completely clean worktree" >&2
  exit 1
fi
expected_openpi_commit="215abfb217dbac7d5f1273282331b9b1866c0479"
if [[ "$(git -C "$openpi" rev-parse HEAD)" != "$expected_openpi_commit" ]]; then
  echo "pi0 interventions require the pinned OpenPI revision" >&2
  exit 1
fi
if ! PYTHONPATH="$repo_root/src" "$repo_root/.venv/bin/python" -c \
  'import json,sys; value=json.load(open(sys.argv[1])); assert value.get("passed") is True and value.get("architecture_timing_claim_allowed") is True' \
  "$competence_gate"; then
  echo "pi0 competence gate did not authorize intervention timing" >&2
  exit 1
fi
if ! PYTHONPATH="$repo_root/src" "$repo_root/.venv/bin/python" \
  "$repo_root/scripts/validate_pi0_intervention_inputs.py" \
  --parity-summary "$parity_summary" \
  --pytorch-checkpoint "$checkpoint" \
  --manifest "$manifest" >/dev/null; then
  echo "pi0 conversion parity did not authorize interventions" >&2
  exit 1
fi
if [[ ! -f "$checkpoint/model.safetensors" || ! -f "$checkpoint/config.json" ]]; then
  echo "hookable pi0 checkpoint is incomplete" >&2
  exit 1
fi

code_commit="$(git -C "$repo_root" rev-parse HEAD)"
if [[ -e "$output" && ! -f "$output/code_commit.txt" ]]; then
  echo "existing pi0 intervention output lacks a resumable code-commit binding" >&2
  exit 1
fi
mkdir -p "$output"
if [[ -f "$output/code_commit.txt" ]] && ! grep -qx "$code_commit" "$output/code_commit.txt"; then
  echo "existing pi0 intervention output uses a different code commit" >&2
  exit 1
fi
printf '%s\n' "$code_commit" >"$output/code_commit.txt"
gpu_ids="$coarse_gpu,$position_gpu"
nvidia-smi --id="$gpu_ids" --query-gpu=index,uuid,name,driver_version,memory.total --format=csv >"$output/gpu_preflight.csv"
sha256sum "$competence_gate" "$parity_summary" "$manifest" >"$output/input_sha256.txt"
PYTHONPATH="$repo_root/src" "$repo_root/.venv/bin/python" \
  "$repo_root/scripts/validate_pi0_intervention_inputs.py" \
  --parity-summary "$parity_summary" \
  --pytorch-checkpoint "$checkpoint" \
  --manifest "$manifest" >"$output/intervention_input_binding.json"

run_grid() {
  local mode="$1"
  local selected_gpu="$2"
  local intervention_output="$output/interventions/$mode"
  local analysis_output="$output/analysis/$mode"
  env CUDA_VISIBLE_DEVICES="$selected_gpu" \
    PYTHONPATH="$repo_root/src:$openpi:$openpi/packages/openpi-client/src" \
    "$openpi/.venv/bin/python" "$repo_root/scripts/run_selected_pair_interventions.py" \
    --checkpoint "$checkpoint" \
    --manifest "$manifest" \
    --clean-validation "$pi0_clean" \
    --reference-clean-validation "$pi05_clean" \
    --output "$intervention_output" \
    --mode "$mode" \
    --eligibility dual_success \
    --config pi0_libero \
    --device cuda:0 \
    --noise-seeds 0 \
    --num-steps 10 \
    --minimum-selected-pairs 12
  env PYTHONPATH="$repo_root/src" MPLBACKEND=Agg \
    "$repo_root/.venv/bin/python" "$repo_root/scripts/analyze_selected_pair_interventions.py" \
    --input "$intervention_output" \
    --manifest "$manifest" \
    --output "$analysis_output"
}

pi05_coarse_analysis="$output/reference/pi05_coarse"
env PYTHONPATH="$repo_root/src" MPLBACKEND=Agg \
  "$repo_root/.venv/bin/python" "$repo_root/scripts/analyze_selected_pair_interventions.py" \
  --input "$pi05_coarse_interventions" \
  --manifest "$manifest" \
  --output "$pi05_coarse_analysis"

run_grid coarse "$coarse_gpu" &
coarse_pid="$!"
run_grid population_positions "$position_gpu" &
position_pid="$!"
set +e
wait "$coarse_pid"
coarse_status="$?"
wait "$position_pid"
position_status="$?"
set -e
if [[ "$coarse_status" -ne 0 || "$position_status" -ne 0 ]]; then
  echo "one or more parallel pi0 intervention grids failed" >&2
  exit 1
fi

env PYTHONPATH="$repo_root/src" \
  "$repo_root/.venv/bin/python" "$repo_root/scripts/compare_pi0_models.py" \
  --pi05-coarse-analysis "$pi05_coarse_analysis" \
  --pi0-coarse-analysis "$output/analysis/coarse" \
  --pi05-position-analysis "$pi05_position_analysis" \
  --pi0-position-analysis "$output/analysis/population_positions" \
  --output "$output/comparison"

env PYTHONPATH="$repo_root/src" \
  "$repo_root/.venv/bin/python" "$repo_root/scripts/audit_pi0_intervention_control.py" \
  --output-root "$output" \
  --parity-summary "$parity_summary" \
  --pytorch-checkpoint "$checkpoint" \
  --manifest "$manifest" \
  --output "$output/final_audit.json"
