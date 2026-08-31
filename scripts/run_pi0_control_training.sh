#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 <openpi-root> <checkpoint-base-dir> <experiment-name> [resume|fresh]" >&2
  exit 2
fi

openpi_root="$(realpath "$1")"
checkpoint_base="$(realpath -m "$2")"
experiment_name="$3"
run_mode="${4:-fresh}"
expected_revision="215abfb217dbac7d5f1273282331b9b1866c0479"
observed_revision="$(git -C "$openpi_root" rev-parse HEAD)"

if [[ "$observed_revision" != "$expected_revision" ]]; then
  echo "OpenPI revision mismatch: expected $expected_revision, found $observed_revision" >&2
  exit 1
fi
if [[ ! -f "$openpi_root/assets/pi0_libero/physical-intelligence/libero/norm_stats.json" ]]; then
  echo "pi0_libero normalization statistics are missing; run the pinned compute_norm_stats.py first" >&2
  exit 1
fi
if [[ "$run_mode" == "resume" ]]; then
  resume_args=(--resume)
elif [[ "$run_mode" == "fresh" ]]; then
  resume_args=(--no-resume)
else
  echo "run mode must be resume or fresh" >&2
  exit 2
fi
if [[ "$(nvidia-smi --list-gpus | wc -l | tr -d ' ')" -lt 2 ]]; then
  echo "the matched control launcher requires two visible GPUs" >&2
  exit 1
fi

mkdir -p "$checkpoint_base"
cd "$openpi_root"
export CUDA_VISIBLE_DEVICES=0,1
export WANDB_MODE=disabled
exec ./.venv/bin/python scripts/train.py pi0_libero \
  --exp-name "$experiment_name" \
  --assets-base-dir "$openpi_root/assets" \
  --checkpoint-base-dir "$checkpoint_base" \
  --batch-size 32 \
  --num-train-steps 30000 \
  --save-interval 1000 \
  --keep-period 5000 \
  --no-wandb-enabled \
  --fsdp-devices 2 \
  "${resume_args[@]}"
