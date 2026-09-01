#!/usr/bin/env python3
"""Run the registered continue-versus-restart late visual-safety pilot."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

FIRST_CHUNK_EXECUTION_HORIZON = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--boundaries", default=",".join(str(value) for value in range(11)))
    parser.add_argument("--timing-isolated", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    boundaries = _boundaries(args.boundaries)
    if boundaries != list(range(11)):
        raise ValueError("registered obstacle sweep requires every boundary 0..10")
    manifest = json.loads(args.manifest.read_text())
    entry = _manifest_entry(manifest, args.pair_id)
    if manifest.get("pair_family") != "obstacle_pose" or entry.get("semantic_role") != "obstacle_pose":
        raise ValueError("dynamic obstacle sweep requires an obstacle-pose pair")
    args.output.mkdir(parents=True, exist_ok=True)
    launcher = Path(__file__).with_name("run_pair_validation.sh")
    for strategy, boundary in [("restart", 0), ("continue", 10)]:
        _run_one(args, launcher, strategy, boundary)
        _write_tables(args.output, entry, args.noise_seed, args.timing_isolated)
    endpoint = json.loads((args.output / "summary.json").read_text())
    if not endpoint["endpoint_gate_pass"]:
        return 0
    for strategy, boundary in [("continue", value) for value in boundaries if value != 10]:
        _run_one(args, launcher, strategy, boundary)
        _write_tables(args.output, entry, args.noise_seed, args.timing_isolated)
    return 0


def _run_one(
    args: argparse.Namespace,
    launcher: Path,
    strategy: str,
    boundary: int,
) -> None:
    run_output = args.output / f"{strategy}_after_{boundary}"
    summary_path = run_output / "summary.json"
    if not summary_path.is_file():
        command = [
            str(launcher),
            str(args.manifest),
            args.pair_id,
            str(args.gpu),
            str(args.port),
            str(args.noise_seed),
            str(run_output),
            "",
            "strict",
            "false",
            "",
            "0",
            "false",
            "false",
            strategy,
            str(boundary),
            "400",
            "donor",
            "0",
        ]
        completed = subprocess.run(command, check=False)
        if completed.returncode not in {0, 1} or not summary_path.is_file():
            raise RuntimeError(
                f"dynamic obstacle {strategy} boundary {boundary} produced no summary"
            )
    _validate_run(
        json.loads(summary_path.read_text()),
        args.pair_id,
        args.noise_seed,
        strategy,
        boundary,
    )


def _write_tables(
    output: Path,
    entry: dict[str, Any],
    noise_seed: int,
    timing_isolated: bool,
) -> None:
    rows = []
    obstacle = str(entry["obstacle"])
    obstacle_xy = np.asarray(entry["donor_obstacle_position"], dtype=np.float64)[:2]
    radius = float(entry["obstacle_bounding_radius_m"])
    for path in sorted(output.glob("*_after_*/summary.json")):
        summary = json.loads(path.read_text())
        spec = summary["dynamic_retarget"]
        result = summary["results"][0]
        boundary = int(spec["switch_after_steps"])
        contacts = {
            str(name): int(step)
            for name, step in result["first_contact_step_by_object"].items()
        }
        obstacle_contact_step = contacts.get(obstacle)
        trajectory_path = path.parent / "donor_trajectory_records.jsonl"
        clearance = _minimum_planar_clearance(trajectory_path, obstacle_xy, radius)
        diagnostics = result["retarget_diagnostics"]
        first_chunk_collision = bool(
            obstacle_contact_step is not None
            and obstacle_contact_step <= FIRST_CHUNK_EXECUTION_HORIZON
        )
        rows.append(
            {
                "pair_id": entry["pair_id"],
                "noise_seed": noise_seed,
                "strategy": spec["strategy"],
                "switch_after_steps": boundary,
                "obstacle": obstacle,
                "obstacle_contact_step": obstacle_contact_step,
                "first_chunk_obstacle_collision": first_chunk_collision,
                "first_chunk_collision_free": not first_chunk_collision,
                "eventual_task_success": bool(result["success"]),
                "collision_free_task_success": bool(
                    not first_chunk_collision and result["success"]
                ),
                "minimum_first_chunk_planar_clearance_m": clearance,
                "post_event_velocity_evaluations": int(
                    diagnostics["post_event_velocity_evaluations"]
                ),
                "post_event_total_ms": float(diagnostics["post_event_total_ms"]),
                "initial_input_exact": all(
                    field["array_equal"]
                    for field in result["live_initial_input_diagnostics"].values()
                ),
                "simulator_state_exact": result["restored_sim_state_max_abs_error"] == 0.0,
                "source_condition_is_frozen_fixture": bool(
                    result.get("source_condition_is_frozen_fixture")
                ),
                "donor_live_input_is_frozen_fixture": bool(
                    result.get("donor_live_input_is_frozen_fixture")
                ),
                "visual_condition_switch": bool(result.get("visual_condition_switch")),
                "applied_only_at_first_replan": result["intervention_replans_applied"] == [0],
            }
        )
    rows.sort(key=lambda row: (row["strategy"], row["switch_after_steps"]))
    if not rows:
        return
    with (output / "rollouts.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    restart = next(
        (row for row in rows if row["strategy"] == "restart" and row["switch_after_steps"] == 0),
        None,
    )
    continuation = {
        int(row["switch_after_steps"]): row
        for row in rows
        if row["strategy"] == "continue"
    }
    boundary_zero_actions_exact = _boundary_zero_actions_exact(output)
    exact_controls = all(
        row["initial_input_exact"]
        and row["simulator_state_exact"]
        and row["source_condition_is_frozen_fixture"]
        and row["donor_live_input_is_frozen_fixture"]
        and row["visual_condition_switch"]
        and row["applied_only_at_first_replan"]
        for row in rows
    )
    endpoint_gate_complete = restart is not None and 10 in continuation
    endpoint_exact_controls = endpoint_gate_complete and all(
        row["initial_input_exact"]
        and row["simulator_state_exact"]
        and row["source_condition_is_frozen_fixture"]
        and row["donor_live_input_is_frozen_fixture"]
        and row["visual_condition_switch"]
        and row["applied_only_at_first_replan"]
        for row in (restart, continuation[10])
    )
    endpoint_gate_pass = bool(
        endpoint_exact_controls
        and restart["collision_free_task_success"]
        and continuation[10]["first_chunk_obstacle_collision"]
    )
    complete = endpoint_gate_complete and set(continuation) == set(range(11))
    nfe_exact = complete and all(
        int(row["post_event_velocity_evaluations"]) == 10 - boundary
        for boundary, row in continuation.items()
    ) and int(restart["post_event_velocity_evaluations"]) == 10
    eligible = endpoint_gate_pass
    successful_boundaries = [
        boundary
        for boundary, row in sorted(continuation.items())
        if row["collision_free_task_success"]
    ]
    efficient_boundaries = [
        boundary
        for boundary in successful_boundaries
        if boundary > 0
        and int(continuation[boundary]["post_event_velocity_evaluations"]) < 10
        and float(continuation[boundary]["post_event_total_ms"])
        < float(restart["post_event_total_ms"])
    ] if restart is not None else []
    practical_positive = bool(
        eligible
        and boundary_zero_actions_exact is True
        and exact_controls
        and nfe_exact
        and timing_isolated
        and efficient_boundaries
    )
    payload = {
        "schema_version": 1,
        "pair_id": entry["pair_id"],
        "noise_seed": noise_seed,
        "registered_boundaries_complete": complete,
        "endpoint_gate_complete": endpoint_gate_complete,
        "endpoint_gate_pass": endpoint_gate_pass,
        "timing_isolated": timing_isolated,
        "all_exact_controls_pass": exact_controls,
        "boundary_zero_continue_restart_actions_exact": boundary_zero_actions_exact,
        "velocity_evaluation_counts_exact": nfe_exact,
        "behaviorally_eligible": eligible,
        "restart_collision_free_task_success": (
            bool(restart["collision_free_task_success"]) if restart is not None else None
        ),
        "fully_old_conditioned_chunk_collides": (
            bool(continuation[10]["first_chunk_obstacle_collision"])
            if 10 in continuation
            else None
        ),
        "successful_continued_boundaries": successful_boundaries,
        "efficient_continued_boundaries": efficient_boundaries,
        "last_successful_continued_boundary": (
            max(successful_boundaries) if successful_boundaries else None
        ),
        "practical_positive": practical_positive,
        "interpretation_scope": "exploratory_single_pair_gripper_obstacle_avoidance",
        "contact_definition": "gripper_geom_x_obstacle_contact_geom",
        "clearance_definition": "eef_center_planar_distance_minus_obstacle_bounding_radius",
        "whole_robot_collision_measured": False,
        "population_timing_claim_allowed": False,
        "rows": rows,
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _validate_run(
    summary: dict[str, Any],
    pair_id: str,
    noise_seed: int,
    strategy: str,
    boundary: int,
) -> None:
    expected = {
        "family": "dynamic_retarget",
        "strategy": strategy,
        "switch_after_steps": boundary,
    }
    if summary.get("pair_id") != pair_id or int(summary.get("noise_seed", -1)) != noise_seed:
        raise ValueError("existing dynamic-obstacle output has a different pair or seed")
    if summary.get("dynamic_retarget") != expected:
        raise ValueError("existing dynamic-obstacle output has a different strategy or boundary")
    if summary.get("requested_sides") != ["donor"] or len(summary.get("results", [])) != 1:
        raise ValueError("dynamic obstacle update must execute only in the donor state")
    result = summary["results"][0]
    if result.get("intervention_replans_applied") != [0]:
        raise ValueError("dynamic obstacle update must apply only at the first replan")
    if result.get("retarget_diagnostics") is None:
        raise ValueError("dynamic obstacle output omitted compute diagnostics")
    if result.get("restored_sim_state_max_abs_error") != 0.0:
        raise ValueError("dynamic obstacle output did not restore simulator state exactly")
    if not all(
        field["array_equal"] for field in result["live_initial_input_diagnostics"].values()
    ):
        raise ValueError("live moved-obstacle input differs from the frozen fixture")
    if not (
        result.get("visual_condition_switch")
        and result.get("source_condition_is_frozen_fixture")
        and result.get("donor_live_input_is_frozen_fixture")
    ):
        raise ValueError("dynamic obstacle output omitted exact paired visual conditions")


def _minimum_planar_clearance(path: Path, obstacle_xy: np.ndarray, radius: float) -> float:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(records) < FIRST_CHUNK_EXECUTION_HORIZON:
        raise ValueError("dynamic obstacle trajectory is shorter than the first chunk")
    eef_xy = np.asarray(
        [record["eef_pos"][:2] for record in records[:FIRST_CHUNK_EXECUTION_HORIZON]],
        dtype=np.float64,
    )
    return float(np.min(np.linalg.norm(eef_xy - obstacle_xy[None, :], axis=1)) - radius)


def _boundary_zero_actions_exact(output: Path) -> bool | None:
    paths = [
        output / strategy / "donor_actions.json"
        for strategy in ("restart_after_0", "continue_after_0")
    ]
    if not all(path.is_file() for path in paths):
        return None
    return bool(np.array_equal(_first_chunk(paths[0]), _first_chunk(paths[1])))


def _first_chunk(path: Path) -> np.ndarray:
    chunks = json.loads(path.read_text())
    if not chunks:
        raise ValueError(f"dynamic obstacle action log is empty: {path}")
    return np.asarray(chunks[0])


def _boundaries(value: str) -> list[int]:
    try:
        result = sorted({int(item) for item in value.split(",")})
    except ValueError as error:
        raise ValueError("boundaries must be comma-separated integers") from error
    if not result or result[0] < 0 or result[-1] > 10:
        raise ValueError("boundaries must lie within [0, 10]")
    return result


def _manifest_entry(manifest: dict[str, Any], pair_id: str) -> dict[str, Any]:
    matches = [entry for entry in manifest["pairs"] if entry["pair_id"] == pair_id]
    if len(matches) != 1:
        raise ValueError(f"expected one manifest entry for {pair_id!r}, found {len(matches)}")
    return matches[0]


if __name__ == "__main__":
    raise SystemExit(main())
