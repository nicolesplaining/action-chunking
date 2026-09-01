#!/usr/bin/env python3
"""Run the frozen continue-without-restart utility sweep for one paired state."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--clean-screen", type=Path)
    parser.add_argument("--boundaries", default="0,7,8,9,10")
    parser.add_argument("--sides", default="base,donor")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    boundaries = _boundaries(args.boundaries)
    sides = _sides(args.sides)
    if 0 not in boundaries:
        raise ValueError("dynamic-retarget sweep requires boundary-zero exact controls")
    manifest = json.loads(args.manifest.read_text())
    entry = _manifest_entry(manifest, args.pair_id)
    if entry.get("semantic_role", "manipulated_object") != "manipulated_object":
        raise ValueError("dynamic target retargeting requires a manipulated-object pair")
    if entry.get("origin_side") is not None:
        expected_side = "donor" if entry["origin_side"] == "base" else "base"
        if sides != [expected_side]:
            raise ValueError(
                f"pre-contact retargeting must evaluate only new-task side {expected_side!r}"
            )
    args.output.mkdir(parents=True, exist_ok=True)
    launcher = Path(__file__).with_name("run_pair_validation.sh")
    jobs = [("restart", 0), *(('continue', boundary) for boundary in boundaries)]
    for strategy, boundary in jobs:
        run_output = args.output / f"{strategy}_after_{boundary}"
        summary_path = run_output / "summary.json"
        if summary_path.exists():
            _validate_run(json.loads(summary_path.read_text()), strategy, boundary, args)
            continue
        clean_screen = str(args.clean_screen) if boundary == 0 and args.clean_screen else ""
        command = [
            str(launcher),
            str(args.manifest),
            args.pair_id,
            str(args.gpu),
            str(args.port),
            str(args.noise_seed),
            str(run_output),
            clean_screen,
            "strict",
            "false",
            "",
            "0",
            "false",
            "false",
            strategy,
            str(boundary),
            "400",
            ",".join(sides),
        ]
        completed = subprocess.run(command, check=False)
        if completed.returncode not in {0, 1} or not summary_path.is_file():
            raise RuntimeError(f"dynamic retarget {strategy} boundary {boundary} produced no summary")
        _validate_run(json.loads(summary_path.read_text()), strategy, boundary, args)
        _write_tables(args.output, args.pair_id, entry, args.noise_seed, sides)
    _write_tables(args.output, args.pair_id, entry, args.noise_seed, sides)
    return 0


def _write_tables(
    output: Path,
    pair_id: str,
    entry: dict[str, Any],
    noise_seed: int,
    sides: list[str] | None = None,
) -> None:
    sides = sides or ["base", "donor"]
    rows = []
    summaries = []
    for path in sorted(output.glob("*_after_*/summary.json")):
        summary = json.loads(path.read_text())
        summaries.append(summary)
        spec = summary["dynamic_retarget"]
        strategy = spec["strategy"]
        boundary = int(spec["switch_after_steps"])
        for result in summary["results"]:
            side = result["side"]
            old_side = "donor" if side == "base" else "base"
            contacts = {str(key): int(value) for key, value in result["first_contact_step_by_object"].items()}
            first_object = min(contacts, key=contacts.get) if contacts else None
            diagnostics = result["retarget_diagnostics"]
            rows.append(
                {
                    "pair_id": pair_id,
                    "noise_seed": noise_seed,
                    "strategy": strategy,
                    "switch_after_steps": boundary,
                    "side": side,
                    "new_target": entry[f"{side}_target"],
                    "old_target": entry[f"{old_side}_target"],
                    "first_contact_object": first_object,
                    "new_target_first": first_object == entry[f"{side}_target"],
                    "old_target_first": first_object == entry[f"{old_side}_target"],
                    "eventual_new_task_success": bool(result["success"]),
                    "completion_steps": int(result["steps"]),
                    "post_event_velocity_evaluations": int(
                        diagnostics["post_event_velocity_evaluations"]
                    ),
                    "discarded_velocity_evaluations": int(
                        diagnostics["discarded_velocity_evaluations"]
                    ),
                    "post_event_evaluation_savings_fraction": float(
                        diagnostics["post_event_evaluation_savings_fraction"]
                    ),
                    "donor_condition_ms": float(diagnostics["donor_condition_ms"]),
                    "post_event_integration_ms": float(diagnostics["post_event_integration_ms"]),
                    "post_event_total_ms": float(diagnostics["post_event_total_ms"]),
                    "initial_input_exact": all(
                        field["array_equal"] for field in result["live_initial_input_diagnostics"].values()
                    ),
                    "simulator_state_exact": result["restored_sim_state_max_abs_error"] == 0.0,
                    "applied_only_at_first_replan": result["intervention_replans_applied"] == [0],
                }
            )
    rows.sort(key=lambda row: (row["strategy"], row["switch_after_steps"], row["side"]))
    if not rows:
        return
    with (output / "rollouts.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    restart = [row for row in rows if row["strategy"] == "restart" and row["switch_after_steps"] == 0]
    continuation = [row for row in rows if row["strategy"] == "continue"]
    boundary_zero_paths = [
        output / strategy / f"{side}_actions.json"
        for strategy in ("restart_after_0", "continue_after_0")
        for side in sides
    ]
    boundary_zero_actions_exact = (
        all(
            np.array_equal(
                _first_chunk(output / "restart_after_0" / f"{side}_actions.json"),
                _first_chunk(output / "continue_after_0" / f"{side}_actions.json"),
            )
            for side in sides
        )
        if all(path.is_file() for path in boundary_zero_paths)
        else None
    )
    summary = {
        "schema_version": 1,
        "pair_id": pair_id,
        "noise_seed": noise_seed,
        "registered_boundaries": sorted({row["switch_after_steps"] for row in continuation}),
        "directions": len(restart),
        "all_initial_inputs_exact": all(row["initial_input_exact"] for row in rows),
        "all_simulator_states_exact": all(row["simulator_state_exact"] for row in rows),
        "all_retargets_only_at_first_replan": all(
            row["applied_only_at_first_replan"] for row in rows
        ),
        "boundary_zero_continue_restart_actions_exact": boundary_zero_actions_exact,
        "restart_new_target_first_rate": _rate(restart, "new_target_first"),
        "restart_new_task_success_rate": _rate(restart, "eventual_new_task_success"),
        "continuation": [
            {
                "switch_after_steps": boundary,
                "new_target_first_rate": _rate(
                    [row for row in continuation if row["switch_after_steps"] == boundary],
                    "new_target_first",
                ),
                "new_task_success_rate": _rate(
                    [row for row in continuation if row["switch_after_steps"] == boundary],
                    "eventual_new_task_success",
                ),
                "mean_post_event_velocity_evaluations": _mean(
                    [row for row in continuation if row["switch_after_steps"] == boundary],
                    "post_event_velocity_evaluations",
                ),
                "mean_post_event_total_ms": _mean(
                    [row for row in continuation if row["switch_after_steps"] == boundary],
                    "post_event_total_ms",
                ),
            }
            for boundary in sorted({row["switch_after_steps"] for row in continuation})
        ],
        "source_summaries": len(summaries),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def _validate_run(
    summary: dict[str, Any], strategy: str, boundary: int, args: argparse.Namespace
) -> None:
    expected = {
        "family": "dynamic_retarget",
        "strategy": strategy,
        "switch_after_steps": boundary,
    }
    if summary.get("pair_id") != args.pair_id or int(summary.get("noise_seed", -1)) != args.noise_seed:
        raise ValueError("existing dynamic-retarget output has different pair or noise seed")
    if summary.get("dynamic_retarget") != expected:
        raise ValueError("existing dynamic-retarget output has different strategy or boundary")
    if {result["side"] for result in summary["results"]} != set(_sides(args.sides)):
        raise ValueError("existing dynamic-retarget output has different requested sides")
    for result in summary["results"]:
        if result.get("intervention_replans_applied") != [0]:
            raise ValueError("dynamic retargeting must be applied only at the first replan")
        if result.get("retarget_diagnostics") is None:
            raise ValueError("dynamic-retarget output omitted compute diagnostics")
        if not all(field["array_equal"] for field in result["live_initial_input_diagnostics"].values()):
            raise ValueError("dynamic-retarget rollout did not restore the exact model input")
        if result["restored_sim_state_max_abs_error"] != 0.0:
            raise ValueError("dynamic-retarget rollout did not restore the exact simulator state")


def _boundaries(value: str) -> list[int]:
    try:
        result = sorted({int(item) for item in value.split(",")})
    except ValueError as error:
        raise ValueError("boundaries must be comma-separated integers") from error
    if not result or result[0] < 0 or result[-1] > 10:
        raise ValueError("boundaries must lie within [0, 10]")
    return result


def _sides(value: str) -> list[str]:
    result = [side.strip() for side in value.split(",") if side.strip()]
    if not result or len(result) != len(set(result)) or not set(result) <= {"base", "donor"}:
        raise ValueError("sides must be a nonempty unique subset of base,donor")
    return result


def _manifest_entry(manifest: dict[str, Any], pair_id: str) -> dict[str, Any]:
    matches = [entry for entry in manifest["pairs"] if entry["pair_id"] == pair_id]
    if len(matches) != 1:
        raise ValueError(f"expected one manifest entry for {pair_id!r}, found {len(matches)}")
    return matches[0]


def _rate(rows: list[dict[str, Any]], key: str) -> float | None:
    return sum(bool(row[key]) for row in rows) / len(rows) if rows else None


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    return sum(float(row[key]) for row in rows) / len(rows) if rows else None


def _first_chunk(path: Path) -> np.ndarray:
    chunks = json.loads(path.read_text())
    if not chunks:
        raise ValueError(f"dynamic-retarget action log is empty: {path}")
    chunk = np.asarray(chunks[0])
    if chunk.ndim != 2:
        raise ValueError(f"dynamic-retarget first chunk has invalid shape: {path}")
    return chunk


if __name__ == "__main__":
    raise SystemExit(main())
