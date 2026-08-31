#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 <event-sae-root> <awe-root> <trajectory-jsonl> <output-root> <thresholds>" >&2
  exit 2
fi

event_sae_root="$1"
awe_root="$2"
trajectory="$3"
output_root="$4"
IFS=',' read -r -a thresholds <<< "$5"
if [[ ${#thresholds[@]} -eq 0 ]]; then
  echo "threshold list must be nonempty" >&2
  exit 2
fi

launcher="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_event_sae_keyframes.sh"
for threshold in "${thresholds[@]}"; do
  tag="${threshold//./p}"
  "$launcher" \
    "$event_sae_root" \
    "$awe_root" \
    "$trajectory" \
    "$output_root/err_$tag" \
    pos_only \
    "$threshold"
done
