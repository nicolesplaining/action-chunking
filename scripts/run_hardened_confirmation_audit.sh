#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <repo-root> <confirmation-root> <audit-output>" >&2
  exit 2
fi

repo_root="$(realpath "$1")"
confirmation_root="$(realpath "$2")"
audit_output="$(realpath -m "$3")"
original="$confirmation_root/analysis/summary.json"
hardened="$audit_output/hardened/summary.json"

if [[ ! -f "$original" ]]; then
  echo "registered original confirmation analysis is missing" >&2
  exit 1
fi
if [[ -e "$audit_output" ]]; then
  echo "hardened audit output already exists" >&2
  exit 1
fi
if [[ -n "$(git -C "$repo_root" status --porcelain=v1 --untracked-files=all)" ]]; then
  echo "hardened confirmation audit requires a completely clean worktree" >&2
  exit 1
fi

mkdir -p "$audit_output/original" "$audit_output/hardened"
cp "$original" "$audit_output/original/summary.json"
git -C "$repo_root" rev-parse HEAD >"$audit_output/analysis_code_commit.txt"

set +e
env PYTHONPATH="$repo_root/src" "$repo_root/.venv/bin/python" \
  "$repo_root/scripts/analyze_early_exit_confirmation.py" \
  --root "$confirmation_root" \
  --output "$audit_output/hardened"
analysis_status="$?"
set -e
if [[ "$analysis_status" -ne 0 && "$analysis_status" -ne 1 ]]; then
  echo "hardened analyzer returned an unexpected status: $analysis_status" >&2
  exit 1
fi
if [[ ! -f "$hardened" ]]; then
  echo "hardened analyzer produced no summary" >&2
  exit 1
fi

env PYTHONPATH="$repo_root/src" "$repo_root/.venv/bin/python" \
  "$repo_root/scripts/compare_confirmation_analyses.py" \
  --original "$audit_output/original/summary.json" \
  --hardened "$hardened" \
  --output "$audit_output/comparison.json"
