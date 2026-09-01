#!/usr/bin/env python3
"""Analyze the frozen 500-pair, 1,000-rollout early-exit confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from action_chunking.libero_logs import wilson_interval
from action_chunking.noninferiority import binomial_upper_bound
from action_chunking.pairs import file_digest

SUITE = "libero_goal"
TASKS = 10
TRIALS = 50
EXPECTED = TASKS * TRIALS
CONDITIONS = {"early_exit_7": 7, "full_control_10": 10}
NOISE_SHAPE = (10, 32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    warmups = _load_warmups(args.root)
    pairs = _load_pairs(args.root)
    rows, summary = analyze_confirmation(pairs)
    progress = json.loads((args.root / "progress.json").read_text())
    if int(progress.get("warmup_sessions", -1)) != len(warmups):
        raise ValueError("confirmation progress has the wrong warm-up session count")
    warmup_commits = {str(session["code_commit"]) for session in warmups}
    if warmup_commits != {summary["code_commit"]}:
        raise ValueError("warm-up sessions and scored pairs use different code commits")
    args.output.mkdir(parents=True, exist_ok=True)
    payload = {
        **summary,
        "confirmation_root": str(args.root),
        "progress_sha256": file_digest(args.root / "progress.json"),
        "warmup_sessions": len(warmups),
        "warmup_sessions_sha256": file_digest(args.root / "warmup_sessions.jsonl"),
        "rows": rows,
    }
    (args.output / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if summary["confirmation_positive"] else 1


def _load_pairs(root: Path) -> list[dict[str, Any]]:
    progress_path = root / "progress.json"
    progress = json.loads(progress_path.read_text())
    if (
        progress.get("schema_version") != 1
        or progress.get("suite") != SUITE
        or int(progress.get("expected_pairs", -1)) != EXPECTED
        or int(progress.get("completed_pairs", -1)) != EXPECTED
        or int(progress.get("expected_condition_rollouts", -1)) != 2 * EXPECTED
        or int(progress.get("completed_condition_rollouts", -1)) != 2 * EXPECTED
        or progress.get("conditions") != CONDITIONS
        or re.fullmatch(r"[0-9a-f]{40}", str(progress.get("code_commit"))) is None
        or len(progress.get("jobs", [])) != EXPECTED
    ):
        raise ValueError("confirmation progress is incomplete or incompatible")
    pairs = []
    for task_id in range(TASKS):
        for trial_index in range(TRIALS):
            path = root / "pairs" / _pair_key(task_id, trial_index) / "pair_summary.json"
            pairs.append(json.loads(path.read_text()))
    if {str(pair.get("code_commit")) for pair in pairs} != {str(progress["code_commit"])}:
        raise ValueError("confirmation progress and pairs use different code commits")
    return pairs


def analyze_confirmation(
    pairs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(pairs) != EXPECTED:
        raise ValueError("confirmation requires exactly 500 episode pairs")
    expected_keys = {(task_id, trial_index) for task_id in range(TASKS) for trial_index in range(TRIALS)}
    observed_keys = {(int(pair.get("task_id", -1)), int(pair.get("trial_index", -1))) for pair in pairs}
    if observed_keys != expected_keys:
        raise ValueError("confirmation task/trial grid is incomplete or duplicated")
    code_commits = {str(pair.get("code_commit")) for pair in pairs}
    if len(code_commits) != 1 or re.fullmatch(r"[0-9a-f]{40}", next(iter(code_commits))) is None:
        raise ValueError("confirmation pairs do not share one valid code commit")

    rows = []
    for pair in sorted(pairs, key=lambda value: (value["task_id"], value["trial_index"])):
        task_id = int(pair["task_id"])
        trial_index = int(pair["trial_index"])
        _validate_pair(pair, task_id, trial_index)
        early = pair["early_exit_7"]
        full = pair["full_control_10"]
        full_latency = _first_latency(full)
        early_latency = _first_latency(early)
        rows.append(
            {
                "pair_key": _pair_key(task_id, trial_index),
                "task_id": task_id,
                "trial_index": trial_index,
                "condition_order": pair["condition_order"],
                "early_exit_success": bool(early["success"]),
                "full_control_success": bool(full["success"]),
                "paired_loss": bool(full["success"] and not early["success"]),
                "paired_gain": bool(early["success"] and not full["success"]),
                "early_exit_replans": int(early["replans"]),
                "full_control_replans": int(full["replans"]),
                "early_exit_first_replan_integration_ms": early_latency,
                "full_control_first_replan_integration_ms": full_latency,
                "first_replan_latency_savings_fraction": ((full_latency - early_latency) / full_latency),
            }
        )

    losses = sum(row["paired_loss"] for row in rows)
    gains = sum(row["paired_gain"] for row in rows)
    early_successes = sum(row["early_exit_success"] for row in rows)
    full_successes = sum(row["full_control_success"] for row in rows)
    loss_upper = float(binomial_upper_bound(losses, EXPECTED, alpha=0.05))
    latency = np.asarray(
        [row["first_replan_latency_savings_fraction"] for row in rows],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(latency)):
        raise ValueError("confirmation latency differences must be finite")
    median_latency = float(np.median(latency))
    latency_interval = _bootstrap_median_interval(latency)
    order_counts = {
        "early_exit_first": sum(row["condition_order"][0] == "early_exit_7" for row in rows),
        "full_control_first": sum(row["condition_order"][0] == "full_control_10" for row in rows),
    }
    if order_counts != {"early_exit_first": 250, "full_control_first": 250}:
        raise ValueError("confirmation condition order is not exactly balanced")
    early_interval = wilson_interval(early_successes, EXPECTED)
    full_interval = wilson_interval(full_successes, EXPECTED)
    positive = bool(loss_upper < 0.02 and latency_interval[0] > 0.0 and losses <= 4)
    return rows, {
        "schema_version": 1,
        "analysis_unit": "paired_libero_episode_pair",
        "suite": SUITE,
        "code_commit": next(iter(code_commits)),
        "episode_pairs": EXPECTED,
        "episodes_per_condition": EXPECTED,
        "condition_rollouts": 2 * EXPECTED,
        "condition_order_counts": order_counts,
        "early_exit_successes": early_successes,
        "early_exit_success_rate": early_successes / EXPECTED,
        "early_exit_success_wilson_ci95": list(early_interval),
        "full_control_successes": full_successes,
        "full_control_success_rate": full_successes / EXPECTED,
        "full_control_success_wilson_ci95": list(full_interval),
        "paired_losses": losses,
        "paired_gains": gains,
        "paired_loss_rate": losses / EXPECTED,
        "paired_loss_clopper_pearson_upper95": loss_upper,
        "paired_loss_margin": 0.02,
        "maximum_passing_losses": 4,
        "all_compute_counts_exact": True,
        "velocity_evaluation_savings_fraction": 0.3,
        "median_first_replan_latency_savings_fraction": median_latency,
        "median_first_replan_latency_savings_fraction_bootstrap_ci95": latency_interval,
        "confirmation_positive": positive,
    }


def _validate_pair(pair: dict[str, Any], task_id: int, trial_index: int) -> None:
    if (
        pair.get("schema_version") != 1
        or pair.get("suite") != SUITE
        or re.fullmatch(r"[0-9a-f]{40}", str(pair.get("code_commit"))) is None
        or pair.get("pair_key") != _pair_key(task_id, trial_index)
        or pair.get("condition_order") != _condition_order(task_id, trial_index)
        or pair.get("order_digest_sha256") != _order_digest(task_id, trial_index)
        or pair.get("initial_inputs_exact") is not True
        or pair.get("initial_sim_state_exact") is not True
        or pair.get("shared_noise_exact") is not True
    ):
        raise ValueError(f"confirmation pair controls fail: {_pair_key(task_id, trial_index)}")
    early = pair.get("early_exit_7")
    full = pair.get("full_control_10")
    if not isinstance(early, dict) or not isinstance(full, dict):
        raise ValueError("confirmation pair lacks a condition")
    if early.get("initial_input_sha256") != full.get("initial_input_sha256"):
        raise ValueError("confirmation pair initial inputs differ")
    if set(early.get("initial_input_sha256", {})) != {
        "observation/image",
        "observation/state",
        "observation/wrist_image",
        "prompt",
    }:
        raise ValueError("confirmation pair initial input hashes are incomplete")
    if early.get("initial_sim_state_sha256") != full.get("initial_sim_state_sha256"):
        raise ValueError("confirmation pair simulator states differ")
    if early.get("initial_state_fixture_sha256") != full.get("initial_state_fixture_sha256"):
        raise ValueError("confirmation pair initial-state fixtures differ")
    common = min(int(early["replans"]), int(full["replans"]))
    if common <= 0 or int(pair.get("shared_noise_common_replans", -1)) != common:
        raise ValueError("confirmation pair has invalid shared-noise length")
    if early["noise_sha256_by_replan"][:common] != full["noise_sha256_by_replan"][:common]:
        raise ValueError("confirmation pair noise hashes differ")
    for condition, result in (("early_exit_7", early), ("full_control_10", full)):
        after_steps = CONDITIONS[condition]
        if (
            result.get("condition") != condition
            or result.get("code_commit") != pair.get("code_commit")
            or int(result.get("environment_seed", -1)) != 7
            or int(result.get("noise_seed", -1)) != 0
            or int(result.get("after_steps", -1)) != after_steps
            or int(result.get("total_flow_steps", -1)) != 10
            or int(result.get("replans", -1)) <= 0
            or len(result.get("early_exit_diagnostics", [])) != int(result["replans"])
        ):
            raise ValueError("confirmation condition metadata is invalid")
        for index, diagnostic in enumerate(result["early_exit_diagnostics"]):
            if int(diagnostic.get("replan_index", -1)) != index:
                raise ValueError("confirmation diagnostics are not replan indexed")
            _validate_diagnostic(diagnostic, after_steps)
        noise_hashes = result.get("noise_sha256_by_replan", [])
        action_hashes = result.get("action_sha256_by_replan", [])
        if len(noise_hashes) != int(result["replans"]) or len(action_hashes) != int(result["replans"]):
            raise ValueError("confirmation condition has incomplete action/noise hashes")
        for replan_index, observed in enumerate(noise_hashes):
            expected = _array_digest(_noise_for_replan(0, task_id, trial_index, replan_index))
            if observed != expected:
                raise ValueError("confirmation condition has unexpected action noise")
    expected_loss = bool(full["success"] and not early["success"])
    if bool(pair.get("paired_loss")) != expected_loss:
        raise ValueError("confirmation pair has inconsistent loss labeling")


def _validate_diagnostic(diagnostic: dict[str, Any], after_steps: int) -> None:
    savings = 10 - after_steps
    if (
        int(diagnostic.get("after_steps", -1)) != after_steps
        or int(diagnostic.get("total_flow_steps", -1)) != 10
        or int(diagnostic.get("velocity_field_evaluations", -1)) != after_steps
        or int(diagnostic.get("velocity_field_evaluation_savings", -1)) != savings
        or float(diagnostic.get("velocity_field_evaluation_savings_fraction", -1.0)) != savings / 10
        or float(diagnostic.get("integration_ms", 0.0)) <= 0.0
    ):
        raise ValueError("confirmation compute or latency diagnostic is invalid")


def _first_latency(result: dict[str, Any]) -> float:
    first = [item for item in result["early_exit_diagnostics"] if int(item["replan_index"]) == 0]
    if len(first) != 1:
        raise ValueError("confirmation condition lacks one first-replan latency")
    return float(first[0]["integration_ms"])


def _bootstrap_median_interval(values: np.ndarray, *, samples: int = 10_000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(samples, values.size))
    medians = np.median(values[indices], axis=1)
    return [float(value) for value in np.quantile(medians, [0.025, 0.975])]


def _load_warmups(root: Path) -> list[dict[str, Any]]:
    path = root / "warmup_sessions.jsonl"
    sessions = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not sessions:
        raise ValueError("confirmation has no warm-up session")
    for index, session in enumerate(sessions):
        if (
            int(session.get("session_index", -1)) != index
            or session.get("scored") is not False
            or re.fullmatch(r"[0-9a-f]{40}", str(session.get("code_commit"))) is None
        ):
            raise ValueError("confirmation warm-up session metadata is invalid")
        records = session.get("records", [])
        if [record.get("condition") for record in records] != [
            "full_control_10",
            "early_exit_7",
        ]:
            raise ValueError("confirmation warm-up session order is invalid")
        for record in records:
            _validate_diagnostic(record.get("diagnostic", {}), CONDITIONS[record["condition"]])
    return sessions


def _condition_order(task_id: int, trial_index: int) -> list[str]:
    ranked = sorted(range(TRIALS), key=lambda value: _order_digest(task_id, value))
    if trial_index in set(ranked[: TRIALS // 2]):
        return ["early_exit_7", "full_control_10"]
    return ["full_control_10", "early_exit_7"]


def _order_digest(task_id: int, trial_index: int) -> str:
    return hashlib.sha256(f"{SUITE}:{task_id}:{trial_index}".encode()).hexdigest()


def _pair_key(task_id: int, trial_index: int) -> str:
    return f"task_{task_id:02d}_trial_{trial_index:02d}"


def _noise_for_replan(noise_seed: int, task_id: int, trial_index: int, replan_index: int) -> np.ndarray:
    material = f"{noise_seed}:{SUITE}:{task_id}:{trial_index}:{replan_index}".encode()
    seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    return rng.standard_normal(NOISE_SHAPE, dtype=np.float32)


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
