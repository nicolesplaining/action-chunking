#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <output-root> <count> <render-gpu>" >&2
  exit 2
fi

output_root="$1"
count="$2"
render_gpu="$3"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

labels=(alphabet_soup cream_cheese ketchup tomato_sauce)
tasks=(
  LIVING_ROOM_SCENE1_pick_up_the_alphabet_soup_and_put_it_in_the_basket.bddl
  LIVING_ROOM_SCENE1_pick_up_the_cream_cheese_box_and_put_it_in_the_basket.bddl
  LIVING_ROOM_SCENE1_pick_up_the_ketchup_and_put_it_in_the_basket.bddl
  LIVING_ROOM_SCENE1_pick_up_the_tomato_sauce_and_put_it_in_the_basket.bddl
)

mkdir -p "$output_root"
if docker info >/dev/null 2>&1; then
  docker_command=(docker)
else
  docker_command=(sudo -n docker)
fi

for ((base_index = 0; base_index < ${#tasks[@]}; base_index++)); do
  for ((donor_index = base_index + 1; donor_index < ${#tasks[@]}; donor_index++)); do
    pair_name="${labels[$base_index]}_vs_${labels[$donor_index]}"
    mkdir -p "$output_root/$pair_name"
    "${docker_command[@]}" run --rm \
      --gpus "device=${render_gpu}" \
      --volume "${repo_root}:/app" \
      --volume "${output_root}:/data" \
      --env MUJOCO_GL=egl \
      --env "MUJOCO_EGL_DEVICE_ID=${render_gpu}" \
      --env PYOPENGL_PLATFORM=egl \
      --env PYTHONPATH=/app/src:/app/third_party/openpi:/app/third_party/openpi/packages/openpi-client/src:/app/third_party/openpi/third_party/libero \
      action-chunking-libero-client \
      /bin/bash -lc \
      "source /.venv/bin/activate && python /app/scripts/generate_libero_instruction_pairs.py --suite libero_90 --base-task ${tasks[$base_index]} --donor-task ${tasks[$donor_index]} --count ${count} --output /data/${pair_name}"
  done
done
