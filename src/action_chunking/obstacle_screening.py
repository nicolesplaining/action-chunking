"""Clean-only eligibility metrics for paired obstacle-pose episodes."""

from __future__ import annotations

from typing import Any

import numpy as np


def obstacle_screen_row(
    entry: dict[str, Any],
    summary: dict[str, Any],
    trajectories: dict[str, list[dict[str, Any]]],
    *,
    execution_horizon: int = 5,
    corridor_margin_m: float = 0.02,
    minimum_clearance_gain_m: float = 0.015,
    minimum_trajectory_contrast_m: float = 0.01,
) -> dict[str, Any]:
    """Classify one obstacle placement without inspecting causal interventions."""
    if execution_horizon <= 0:
        raise ValueError("execution horizon must be positive")
    if summary.get("pair_id") != entry.get("pair_id"):
        raise ValueError("obstacle entry and rollout summary have different pair ids")
    if set(trajectories) != {"base", "donor"}:
        raise ValueError("obstacle screening requires base and donor trajectories")
    by_side = {result["side"]: result for result in summary["results"]}
    if set(by_side) != {"base", "donor"}:
        raise ValueError("obstacle rollout summary requires base and donor results")
    exact = all(_result_exact(result) for result in by_side.values())
    target = str(entry["base_target"])
    if target != str(entry["donor_target"]):
        raise ValueError("obstacle-pose pair must retain one target")
    obstacle = str(entry["obstacle"])
    target_first = all(_first_contact(by_side[side]) == target for side in ("base", "donor"))
    both_successful = all(bool(by_side[side]["success"]) for side in ("base", "donor"))

    initial_eef = np.asarray(entry["end_effector_position"], dtype=np.float64)
    base_path = _eef_path(initial_eef, trajectories["base"], execution_horizon)
    donor_path = _eef_path(initial_eef, trajectories["donor"], execution_horizon)
    donor_obstacle = np.asarray(entry["donor_obstacle_position"], dtype=np.float64)
    radius = float(entry["obstacle_bounding_radius_m"])
    base_counterfactual_clearance = _planar_clearance(base_path, donor_obstacle)
    donor_clearance = _planar_clearance(donor_path, donor_obstacle)
    clearance_gain = donor_clearance - base_counterfactual_clearance
    trajectory_contrast = float(np.linalg.norm(donor_path[-1] - base_path[-1]))
    donor_obstacle_step = _contact_step(by_side["donor"], obstacle)
    donor_first_chunk_avoids_obstacle = (
        donor_obstacle_step is None or donor_obstacle_step > execution_horizon
    )
    nominal_path_intersects_corridor = base_counterfactual_clearance <= radius + corridor_margin_m
    donor_path_clears_obstacle_radius = donor_clearance > radius
    eligible = bool(
        exact
        and target_first
        and both_successful
        and donor_first_chunk_avoids_obstacle
        and nominal_path_intersects_corridor
        and donor_path_clears_obstacle_radius
        and clearance_gain >= minimum_clearance_gain_m
        and trajectory_contrast >= minimum_trajectory_contrast_m
    )
    return {
        "pair_id": entry["pair_id"],
        "source_pair_id": entry["source_pair_id"],
        "init_index": int(entry["init_index"]),
        "target": target,
        "obstacle": obstacle,
        "path_fraction": float(entry["path_fraction"]),
        "lateral_offset_m": float(entry["lateral_offset_m"]),
        "execution_horizon": execution_horizon,
        "exact_initial_state": exact,
        "target_first_both": target_first,
        "both_successful": both_successful,
        "donor_obstacle_contact_step": donor_obstacle_step,
        "donor_first_chunk_avoids_obstacle": donor_first_chunk_avoids_obstacle,
        "obstacle_bounding_radius_m": radius,
        "base_counterfactual_clearance_m": base_counterfactual_clearance,
        "donor_clearance_m": donor_clearance,
        "clearance_gain_m": clearance_gain,
        "first_horizon_endpoint_contrast_m": trajectory_contrast,
        "corridor_margin_m": corridor_margin_m,
        "minimum_clearance_gain_m": minimum_clearance_gain_m,
        "minimum_trajectory_contrast_m": minimum_trajectory_contrast_m,
        "nominal_path_intersects_corridor": nominal_path_intersects_corridor,
        "donor_path_clears_obstacle_radius": donor_path_clears_obstacle_radius,
        "eligible": eligible,
    }


def _eef_path(
    initial: np.ndarray, records: list[dict[str, Any]], horizon: int
) -> np.ndarray:
    if len(records) < horizon:
        raise ValueError("trajectory is shorter than the obstacle screening horizon")
    positions = [initial, *(np.asarray(record["eef_pos"], dtype=np.float64) for record in records[:horizon])]
    path = np.stack(positions)
    if path.shape != (horizon + 1, 3) or np.any(~np.isfinite(path)):
        raise ValueError("trajectory contains invalid end-effector positions")
    return path


def _planar_clearance(path: np.ndarray, obstacle: np.ndarray) -> float:
    return float(np.min(np.linalg.norm(path[:, :2] - obstacle[None, :2], axis=1)))


def _first_contact(result: dict[str, Any]) -> str | None:
    contacts = result.get("first_contact_step_by_object", {})
    return min(contacts, key=contacts.get) if contacts else None


def _contact_step(result: dict[str, Any], name: str) -> int | None:
    value = result.get("first_contact_step_by_object", {}).get(name)
    return int(value) if value is not None else None


def _result_exact(result: dict[str, Any]) -> bool:
    diagnostics = result.get("live_initial_input_diagnostics", {})
    return bool(
        diagnostics
        and all(field.get("array_equal") for field in diagnostics.values())
        and result.get("restored_sim_state_max_abs_error") == 0.0
    )
