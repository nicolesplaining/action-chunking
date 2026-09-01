#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 <openpi-root> <checkpoint-base-dir> <experiment-name> <lerobot-cache-root>" >&2
  exit 2
fi

openpi_root="$(realpath "$1")"
checkpoint_base="$(realpath -m "$2")"
experiment_name="$3"
lerobot_home="$(realpath "$4")"
expected_revision="215abfb217dbac7d5f1273282331b9b1866c0479"
observed_revision="$(git -C "$openpi_root" rev-parse HEAD)"
norm_stats="$openpi_root/assets/pi0_libero/physical-intelligence/libero/norm_stats.json"
expected_norm_sha256="f68a5fafe15e1577b7bb2c6fc4837a7d1669e2e9be3752f2589c3d327c6f8ccf"
dataset_root="$lerobot_home/physical-intelligence/libero"
expected_data_file_count=1693

if [[ "$observed_revision" != "$expected_revision" ]]; then
  echo "OpenPI revision mismatch: expected $expected_revision, found $observed_revision" >&2
  exit 1
fi
if [[ ! -f "$norm_stats" ]]; then
  echo "pi0_libero normalization statistics are missing" >&2
  exit 1
fi
observed_norm_sha256="$(sha256sum "$norm_stats" | cut -d" " -f1)"
if [[ "$observed_norm_sha256" != "$expected_norm_sha256" ]]; then
  echo "normalization hash mismatch: expected $expected_norm_sha256, found $observed_norm_sha256" >&2
  exit 1
fi
if [[ ! -f "$dataset_root/meta/info.json" ]]; then
  echo "LIBERO dataset metadata is missing from $dataset_root" >&2
  exit 1
fi
observed_data_file_count="$(find "$dataset_root/data" -type f -name '*.parquet' | wc -l | tr -d ' ')"
if [[ "$observed_data_file_count" -ne "$expected_data_file_count" ]]; then
  echo "LIBERO dataset cache is incomplete: expected $expected_data_file_count parquet files, found $observed_data_file_count" >&2
  exit 1
fi
if [[ "$(nvidia-smi --list-gpus | wc -l | tr -d " ")" -lt 2 ]]; then
  echo "the matched control smoke test requires two visible GPUs" >&2
  exit 1
fi

mkdir -p "$checkpoint_base"
cd "$openpi_root"
export CUDA_VISIBLE_DEVICES=0,1
export WANDB_MODE=disabled
export HF_LEROBOT_HOME="$lerobot_home"
exec ./.venv/bin/python scripts/train.py pi0_libero \
  --exp-name "$experiment_name" \
  --assets-base-dir "$openpi_root/assets" \
  --checkpoint-base-dir "$checkpoint_base" \
  --batch-size 32 \
  --num-train-steps 2 \
  --save-interval 1 \
  --keep-period 1 \
  --no-wandb-enabled \
  --fsdp-devices 2 \
  --no-resume
