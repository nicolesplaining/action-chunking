#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 7 || $# -gt 8 ]]; then
  echo "usage: $0 <manifest> <pair-id> <gpu> <port> <checkpoint> <intervention-output> <rollout-output> [seeds]" >&2
  exit 2
fi

manifest="$(realpath "$1")"
pair_id="$2"
gpu="$3"
port="$4"
checkpoint="$(realpath "$5")"
intervention_output="$(realpath -m "$6")"
rollout_output="$(realpath -m "$7")"
seed_list="${8:-0,1,2,3}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

IFS="," read -r -a seeds <<<"$seed_list"
if [[ ${#seeds[@]} -eq 0 ]]; then
  echo "at least one noise seed is required" >&2
  exit 2
fi

mkdir -p "$intervention_output/$pair_id" "$rollout_output"
for seed in "${seeds[@]}"; do
  if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
    echo "noise seeds must be nonnegative integers" >&2
    exit 2
  fi
  "$repo_root/scripts/run_pair_validation.sh" \
    "$manifest" \
    "$pair_id" \
    "$gpu" \
    "$port" \
    "$seed" \
    "$rollout_output/noise_$seed" \
    none \
    strict \
    false
  PYTHONPATH="$repo_root/src:$repo_root/third_party/openpi/src" \
    "$repo_root/third_party/openpi/.venv/bin/python" \
    "$repo_root/scripts/run_pair_interventions.py" \
      --checkpoint "$checkpoint" \
      --manifest "$manifest" \
      --pair-id "$pair_id" \
      --output "$intervention_output/$pair_id/noise_$seed" \
      --config pi05_libero \
      --device "cuda:$gpu" \
      --noise-seed "$seed" \
      --num-steps 10 \
      --steps all \
      --layers all \
      --skip-residual-patches \
      --position-mode all \
      --dimension-mode none \
      --identity-sites none
done
