#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 11 || $# -gt 12 ]]; then
  echo "usage: $0 <repo-root> <baseline-log> <analysis-root> <baseline-client-session> <policy-server-session> <policy-port> <checkpoint-base> <experiment-name> <lerobot-cache-root> <training-session> <training-log> [poll-seconds]" >&2
  exit 2
fi

repo_root="$(realpath "$1")"
baseline_log="$(realpath "$2")"
analysis_root="$(realpath -m "$3")"
baseline_client_session="$4"
policy_server_session="$5"
policy_port="$6"
checkpoint_base="$(realpath -m "$7")"
experiment_name="$8"
lerobot_home="$(realpath "$9")"
training_session="${10}"
training_log="$(realpath -m "${11}")"
poll_seconds="${12:-30}"
python="$repo_root/third_party/openpi/.venv/bin/python"

if [[ ! "$policy_port" =~ ^[0-9]+$ || ! "$poll_seconds" =~ ^[1-9][0-9]*$ ]]; then
  echo "policy port and poll interval must be positive integers" >&2
  exit 2
fi
if [[ ! -x "$python" ]]; then
  echo "OpenPI Python environment is missing: $python" >&2
  exit 1
fi
if ! command -v ss >/dev/null 2>&1; then
  echo "ss is required to verify that the policy port is free" >&2
  exit 1
fi
if tmux has-session -t "=$training_session" 2>/dev/null; then
  echo "training session already exists: $training_session" >&2
  exit 1
fi

echo "waiting for baseline session $baseline_client_session"
while tmux has-session -t "=$baseline_client_session" 2>/dev/null; do
  sleep "$poll_seconds"
done

echo "validating completed LIBERO-10 log"
cd "$repo_root"
PYTHONPATH=src "$python" scripts/summarize_libero_baseline.py \
  --log "$baseline_log" \
  --suite libero_10 \
  --output "$analysis_root/libero_10" \
  --expected-episodes 500 \
  --expected-tasks 10 \
  --expected-episodes-per-task 50

echo "combining four independently validated suites"
PYTHONPATH=src "$python" scripts/summarize_libero_benchmark.py \
  --summary "$analysis_root/libero_spatial/summary.json" \
  --summary "$analysis_root/libero_object/summary.json" \
  --summary "$analysis_root/libero_goal/summary.json" \
  --summary "$analysis_root/libero_10/summary.json" \
  --output "$analysis_root/combined" \
  --expected-episodes-per-suite 500

if ! tmux has-session -t "=$policy_server_session" 2>/dev/null; then
  echo "policy server session disappeared before the guarded handoff" >&2
  exit 1
fi
echo "stopping completed pi0.5 policy server $policy_server_session"
tmux kill-session -t "=$policy_server_session"

policy_port_listening() {
  [[ -n "$(ss -H -ltn "sport = :$policy_port")" ]]
}

for _attempt in {1..12}; do
  if ! policy_port_listening; then
    break
  fi
  sleep 5
done
if policy_port_listening; then
  echo "policy server on port $policy_port did not exit; pi0 training was not launched" >&2
  exit 1
fi

mkdir -p "$(dirname "$training_log")" "$checkpoint_base"
printf -v training_command '%q ' \
  "$repo_root/scripts/run_pi0_control_training.sh" \
  "$repo_root/third_party/openpi" \
  "$checkpoint_base" \
  "$experiment_name" \
  "$lerobot_home" \
  fresh
printf -v quoted_training_log '%q' "$training_log"
training_command+="> $quoted_training_log 2>&1"

echo "launching matched pi0 control in tmux session $training_session"
tmux new-session -d -s "$training_session" "$training_command"
echo "handoff complete"
