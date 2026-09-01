#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
  echo "usage: $0 <repo-root> <jax-checkpoint-30000> <competence-gate> <manifest> <pytorch-output> <parity-output> <gpu>" >&2
  exit 2
fi

repo_root="$(realpath "$1")"
jax_checkpoint="$(realpath "$2")"
competence_gate="$(realpath "$3")"
manifest="$(realpath "$4")"
pytorch_output="$(realpath -m "$5")"
parity_output="$(realpath -m "$6")"
gpu="$7"
openpi="$repo_root/third_party/openpi"

if [[ "$(basename "$jax_checkpoint")" != "30000" || ! -f "$jax_checkpoint/_CHECKPOINT_METADATA" ]]; then
  echo "conversion accepts only the finalized step-30000 checkpoint" >&2
  exit 1
fi
if ! [[ "$gpu" =~ ^[0-9]+$ ]]; then
  echo "gpu must be a nonnegative integer" >&2
  exit 2
fi
if ! PYTHONPATH="$repo_root/src" "$repo_root/.venv/bin/python" -c \
  'import json,sys; value=json.load(open(sys.argv[1])); assert value.get("passed") is True and value.get("architecture_timing_claim_allowed") is True' \
  "$competence_gate"; then
  echo "pi0 competence gate did not authorize architecture timing" >&2
  exit 1
fi
if [[ -e "$pytorch_output" || -e "$parity_output/summary.json" ]]; then
  echo "conversion or parity output already exists" >&2
  exit 1
fi

env CUDA_VISIBLE_DEVICES="$gpu" XLA_PYTHON_CLIENT_PREALLOCATE=false \
  "$openpi/.venv/bin/python" "$openpi/examples/convert_jax_model_to_pytorch.py" \
  --checkpoint-dir "$jax_checkpoint" \
  --config-name pi0_libero \
  --output-path "$pytorch_output" \
  --precision bfloat16

if [[ ! -f "$pytorch_output/model.safetensors" || ! -f "$pytorch_output/config.json" ]]; then
  echo "official conversion did not produce the required PyTorch artifacts" >&2
  exit 1
fi
if [[ ! -d "$pytorch_output/assets" ]]; then
  cp -a "$jax_checkpoint/assets" "$pytorch_output/assets"
fi

PYTHONPATH="$repo_root/src:$openpi:$openpi/packages/openpi-client/src" \
  "$openpi/.venv/bin/python" "$repo_root/scripts/validate_conversion_manifest.py" \
  --jax-checkpoint "$jax_checkpoint" \
  --pytorch-checkpoint "$pytorch_output" \
  --manifest "$manifest" \
  --output "$parity_output" \
  --config pi0_libero \
  --gpu "$gpu" \
  --noise-seed 0 \
  --max-abs-tolerance 0.02 \
  --minimum-cosine-similarity 0.999
