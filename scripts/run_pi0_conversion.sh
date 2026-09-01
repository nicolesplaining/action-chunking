#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 8 ]]; then
  echo "usage: $0 <repo-root> <jax-checkpoint-label-29999> <competence-gate> <manifest> <prior-failed-parity> <pytorch-output> <parity-output> <gpu>" >&2
  exit 2
fi

repo_root="$(realpath "$1")"
jax_checkpoint="$(realpath "$2")"
competence_gate="$(realpath "$3")"
manifest="$(realpath "$4")"
prior_failed_parity="$(realpath "$5")"
pytorch_output="$(realpath -m "$6")"
parity_output="$(realpath -m "$7")"
gpu="$8"
openpi="$repo_root/third_party/openpi"
pinned_openpi="215abfb217dbac7d5f1273282331b9b1866c0479"
actual_openpi="$(git -C "$openpi" rev-parse HEAD)"
if [[ "$actual_openpi" != "$pinned_openpi" ]]; then
  echo "conversion requires pinned OpenPI commit $pinned_openpi, found $actual_openpi" >&2
  exit 1
fi
if [[ -n "$(git -C "$repo_root" status --porcelain=v1 --untracked-files=all)" ]]; then
  echo "conversion requires a completely clean worktree" >&2
  exit 1
fi
action_chunking_commit="$(git -C "$repo_root" rev-parse HEAD)"
checkpoint_assets="$jax_checkpoint/assets"
sibling_assets="$jax_checkpoint/../assets"
if [[ -d "$checkpoint_assets" ]]; then
  jax_assets="$(realpath "$checkpoint_assets")"
elif [[ -d "$sibling_assets" ]]; then
  jax_assets="$(realpath "$sibling_assets")"
else
  echo "finalized JAX checkpoint is missing normalization assets" >&2
  exit 1
fi

if ! PYTHONPATH="$repo_root/src" "$repo_root/.venv/bin/python" \
  "$repo_root/scripts/validate_pi0_final_checkpoint.py" --checkpoint "$jax_checkpoint" >/dev/null; then
  echo "conversion requires the frozen finalized 30,000-update checkpoint" >&2
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
if [[ -e "$pytorch_output" || -e "$parity_output" ]]; then
  echo "conversion or parity output already exists" >&2
  exit 1
fi
if ! PYTHONPATH="$repo_root/src" "$repo_root/.venv/bin/python" \
  "$repo_root/scripts/validate_prior_conversion_failure.py" \
  --summary "$prior_failed_parity" \
  --jax-checkpoint "$jax_checkpoint" \
  --manifest "$manifest" >/dev/null; then
  echo "lossless conversion is not bound to the preserved bfloat16 failure" >&2
  exit 1
fi

env CUDA_VISIBLE_DEVICES="$gpu" XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONPATH="$repo_root/src:$openpi" \
  "$openpi/.venv/bin/python" "$repo_root/scripts/convert_pi0_checkpoint_lossless.py" \
  --checkpoint-dir "$jax_checkpoint" \
  --config-name pi0_libero \
  --output-path "$pytorch_output" \
  --upstream-converter "$openpi/examples/convert_jax_model_to_pytorch.py" \
  --prior-failed-summary "$prior_failed_parity" \
  --upstream-revision "$actual_openpi" \
  --action-chunking-commit "$action_chunking_commit"

if [[ ! -f "$pytorch_output/model.safetensors" || ! -f "$pytorch_output/config.json" || ! -f "$pytorch_output/conversion_provenance.json" ]]; then
  echo "official conversion did not produce the required PyTorch artifacts" >&2
  exit 1
fi
if [[ -d "$pytorch_output/assets" ]]; then
  if ! diff -qr "$jax_assets" "$pytorch_output/assets" >/dev/null; then
    echo "converted normalization assets differ from the frozen JAX source" >&2
    exit 1
  fi
else
  cp -a "$jax_assets" "$pytorch_output/assets"
fi

PYTHONPATH="$repo_root/src:$openpi:$openpi/packages/openpi-client/src" \
  "$openpi/.venv/bin/python" "$repo_root/scripts/validate_conversion_manifest.py" \
  --jax-checkpoint "$jax_checkpoint" \
  --pytorch-checkpoint "$pytorch_output" \
  --upstream-converter "$openpi/examples/convert_jax_model_to_pytorch.py" \
  --prior-failed-summary "$prior_failed_parity" \
  --action-chunking-commit "$action_chunking_commit" \
  --manifest "$manifest" \
  --output "$parity_output" \
  --config pi0_libero \
  --gpu "$gpu" \
  --noise-seed 0 \
  --max-abs-tolerance 0.02 \
  --minimum-cosine-similarity 0.999
