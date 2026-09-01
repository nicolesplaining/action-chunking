#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 || $# -gt 18 ]]; then
  echo "usage: $0 <manifest> <pair-id> <gpu> <port> <noise-seed> <output-dir> [clean-screen-jsonl] [initial-input-mode] [save-sim-states] [intervention-json] [intervene-replans] [stop-after-first-task-contact] [stop-after-registered-destination] [dynamic-retarget-strategy] [dynamic-retarget-boundary] [max-steps] [sides] [noise-start-index]" >&2
  exit 2
fi

manifest="$(realpath "$1")"
pair_id="$2"
gpu="$3"
port="$4"
noise_seed="$5"
output_dir="$(realpath -m "$6")"
clean_screen="${7:-}"
if [[ "$clean_screen" == "none" ]]; then
  clean_screen=""
fi
initial_input_mode="${8:-strict}"
save_sim_states="${9:-false}"
intervention="${10:-}"
intervene_replans="${11:-0}"
stop_after_first_task_contact="${12:-false}"
stop_after_registered_destination="${13:-false}"
dynamic_retarget_strategy="${14:-}"
dynamic_retarget_boundary="${15:-}"
max_steps="${16:-400}"
sides="${17:-base,donor}"
noise_start_index="${18:-0}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "$output_dir"
if docker info >/dev/null 2>&1; then
  docker_command=(docker)
else
  docker_command=(sudo -n docker)
fi

extra_mounts=()
expected_args=()
sim_state_args=()
intervention_args=()
contact_args=()
destination_args=()
retarget_args=()
rollout_args=()
if ! [[ "$max_steps" =~ ^[1-9][0-9]*$ ]]; then
  echo "max-steps must be a positive integer" >&2
  exit 2
fi
if ! [[ "$noise_start_index" =~ ^[0-9]+$ ]]; then
  echo "noise-start-index must be a nonnegative integer" >&2
  exit 2
fi
rollout_args=(
  --max-steps "$max_steps"
  --sides "$sides"
  --noise-start-index "$noise_start_index"
)
if [[ -n "$clean_screen" ]]; then
  clean_screen="$(realpath "$clean_screen")"
  extra_mounts+=(--volume "$(dirname "$clean_screen"):/screen:ro")
  expected_args=(--expected-clean-screen "/screen/$(basename "$clean_screen")")
fi
if [[ -n "$intervention" ]]; then
  intervention="$(realpath "$intervention")"
  extra_mounts+=(--volume "$(dirname "$intervention"):/intervention:ro")
  intervention_args=(--intervention "/intervention/$(basename "$intervention")" --intervene-replans "$intervene_replans")
fi
if [[ -n "$dynamic_retarget_strategy" || -n "$dynamic_retarget_boundary" ]]; then
  if [[ -z "$dynamic_retarget_strategy" || -z "$dynamic_retarget_boundary" ]]; then
    echo "dynamic retargeting requires both strategy and boundary" >&2
    exit 2
  fi
  if [[ -n "$intervention" ]]; then
    echo "dynamic retargeting cannot be combined with an intervention file" >&2
    exit 2
  fi
  retarget_args=(
    --dynamic-retarget-strategy "$dynamic_retarget_strategy"
    --dynamic-retarget-boundary "$dynamic_retarget_boundary"
  )
fi
if [[ "$stop_after_first_task_contact" == "true" ]]; then
  contact_args=(--stop-after-first-task-contact)
elif [[ "$stop_after_first_task_contact" != "false" ]]; then
  echo "stop-after-first-task-contact must be true or false" >&2
  exit 2
fi
if [[ "$stop_after_registered_destination" == "true" ]]; then
  destination_args=(--stop-after-registered-destination)
elif [[ "$stop_after_registered_destination" != "false" ]]; then
  echo "stop-after-registered-destination must be true or false" >&2
  exit 2
fi
if [[ "$save_sim_states" == "true" ]]; then
  sim_state_args=(--save-sim-states)
elif [[ "$save_sim_states" != "false" ]]; then
  echo "save-sim-states must be true or false" >&2
  exit 2
fi

"${docker_command[@]}" run --rm \
  --network=host \
  --gpus "device=${gpu}" \
  --volume "${repo_root}:/app:ro" \
  --volume "$(dirname "$manifest"):/pair:ro" \
  --volume "${output_dir}:/data" \
  "${extra_mounts[@]}" \
  --env MUJOCO_GL=egl \
  --env "MUJOCO_EGL_DEVICE_ID=${gpu}" \
  --env PYOPENGL_PLATFORM=egl \
  --env PYTHONPATH=/app/src:/app/third_party/openpi:/app/third_party/openpi/packages/openpi-client/src:/app/third_party/openpi/third_party/libero \
  action-chunking-libero-client \
  /bin/bash -lc \
  "source /.venv/bin/activate && python /app/scripts/validate_libero_pair_rollouts.py \
    --manifest /pair/$(basename "$manifest") \
    --pair-id '$pair_id' \
    --output /data \
    --port '$port' \
    --noise-seed '$noise_seed' \
    --initial-input-mode '$initial_input_mode' \
    ${rollout_args[*]} \
    ${expected_args[*]} \
    ${sim_state_args[*]} \
    ${intervention_args[*]} \
    ${contact_args[*]} \
    ${destination_args[*]} \
    ${retarget_args[*]}"
