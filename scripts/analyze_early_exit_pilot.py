#!/usr/bin/env python3
"""Analyze the frozen 15-cluster pi0.5 executed-action early-exit pilot."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from action_chunking.noninferiority import binomial_upper_bound
from action_chunking.pairs import file_digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--full-control", type=Path, required=True)
    parser.add_argument("--early-exit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    rows, summary = analyze_early_exit_pilot(
        manifest,
        args.clean,
        args.full_control,
        args.early_exit,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    payload = {
        **summary,
        "manifest": str(args.manifest),
        "manifest_sha256": file_digest(args.manifest),
        "clean_validation": str(args.clean),
        "full_control_validation": str(args.full_control),
        "early_exit_validation": str(args.early_exit),
        "rows": rows,
    }
    (args.output / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if summary["pilot_positive"] else 1


def analyze_early_exit_pilot(
    manifest: dict[str, Any],
    clean_root: Path,
    full_root: Path,
    early_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    entries = manifest.get("pairs", [])
    if len(entries) != 16:
        raise ValueError("early-exit pilot requires the frozen 16-state manifest")
    scene_hashes = [_scene_state_hash(entry) for entry in entries]
    if len(set(scene_hashes)) != len(scene_hashes):
        raise ValueError("early-exit pilot scene clusters must have unique state hashes")
    clean_catalog = _clean_catalog(clean_root, entries)
    full_catalog = _intervention_catalog(full_root, 16, 10)
    early_catalog = _intervention_catalog(early_root, 16, 7)
    rows = []
    for entry in entries:
        pair_id = entry["pair_id"]
        clean_job = clean_catalog[pair_id]
        full_job = full_catalog[pair_id]
        early_job = early_catalog[pair_id]
        eligible = bool(clean_job["exact_dual_success_target_first"])
        full_summary = _pair_summary(full_root, pair_id)
        early_summary = _pair_summary(early_root, pair_id)
        full_actions_exact = all(
            _actions_equal(clean_root, full_root, pair_id, side)
            for side in ("base", "donor")
        )
        if not full_actions_exact:
            raise ValueError(f"ten-step early-exit control differs from clean actions: {pair_id}")
        full_outcomes = _pair_outcomes(entry, full_summary)
        early_outcomes = _pair_outcomes(entry, early_summary)
        full_composite = full_outcomes["composite"]
        early_composite = early_outcomes["composite"]
        full_compute_exact = bool(full_job["early_exit_compute_exact"])
        early_compute_exact = bool(early_job["early_exit_compute_exact"])
        if not full_compute_exact or not early_compute_exact:
            raise ValueError(f"early-exit compute accounting failed: {pair_id}")
        full_latency = _first_replan_cluster_latency(full_summary)
        early_latency = _first_replan_cluster_latency(early_summary)
        rows.append(
            {
                "pair_id": pair_id,
                "scene_state_sha256": _scene_state_hash(entry),
                "eligible": eligible,
                "full_actions_exact": full_actions_exact,
                "full_target_first": full_outcomes["target_first"],
                "early_exit_target_first": early_outcomes["target_first"],
                "target_first_preserved": bool(
                    full_outcomes["target_first"] and early_outcomes["target_first"]
                ),
                "full_eventual_success": full_outcomes["eventual_success"],
                "early_exit_eventual_success": early_outcomes["eventual_success"],
                "eventual_success_preserved": bool(
                    full_outcomes["eventual_success"]
                    and early_outcomes["eventual_success"]
                ),
                "full_composite": full_composite,
                "early_exit_composite": early_composite,
                "composite_preserved": bool(full_composite and early_composite),
                "full_compute_exact": full_compute_exact,
                "early_exit_compute_exact": early_compute_exact,
                "full_first_replan_integration_ms": full_latency,
                "early_exit_first_replan_integration_ms": early_latency,
                "first_replan_latency_savings_fraction": (
                    (full_latency - early_latency) / full_latency
                ),
            }
        )
    eligible_rows = [row for row in rows if row["eligible"]]
    if len(eligible_rows) != 15:
        raise ValueError("early-exit pilot requires exactly 15 frozen clean-eligible clusters")
    preserved = sum(row["composite_preserved"] for row in eligible_rows)
    target_first_preserved = sum(
        row["target_first_preserved"] for row in eligible_rows
    )
    eventual_success_preserved = sum(
        row["eventual_success_preserved"] for row in eligible_rows
    )
    latency = np.asarray(
        [row["first_replan_latency_savings_fraction"] for row in eligible_rows],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(latency)):
        raise ValueError("early-exit latency savings must be finite")
    median_latency = float(np.median(latency))
    latency_interval = _bootstrap_median_interval(latency)
    positive_latency_clusters = int(np.count_nonzero(latency > 0.0))
    negative_latency_clusters = int(np.count_nonzero(latency < 0.0))
    latency_sign_p = _two_sided_sign_test_p(
        positive_latency_clusters, negative_latency_clusters
    )
    preservation_interval = _clopper_pearson_interval(
        preserved, len(eligible_rows)
    )
    compute_exact = all(
        row["full_compute_exact"] and row["early_exit_compute_exact"]
        for row in eligible_rows
    )
    return rows, {
        "schema_version": 1,
        "analysis_unit": "physical_scene_state",
        "frozen_scene_states": 16,
        "eligible_scene_clusters": len(eligible_rows),
        "full_control_exact_clusters": sum(row["full_actions_exact"] for row in eligible_rows),
        "target_first_preserved_clusters": target_first_preserved,
        "eventual_success_preserved_clusters": eventual_success_preserved,
        "composite_preserved_clusters": preserved,
        "composite_preservation_rate": preserved / len(eligible_rows),
        "composite_preservation_clopper_pearson_ci95": preservation_interval,
        "minimum_preserved_clusters": 14,
        "all_compute_counts_exact": compute_exact,
        "velocity_evaluation_savings_fraction": 0.3 if compute_exact else None,
        "median_first_replan_latency_savings_fraction": median_latency,
        "median_first_replan_latency_savings_fraction_bootstrap_ci95": latency_interval,
        "positive_latency_savings_clusters": positive_latency_clusters,
        "negative_latency_savings_clusters": negative_latency_clusters,
        "latency_savings_two_sided_sign_test_p": latency_sign_p,
        "pilot_positive": bool(
            preserved >= 14 and compute_exact and median_latency > 0.0
        ),
        "interpretation_scope": "exploratory_pilot_not_noninferiority",
    }


def _clean_catalog(
    root: Path, entries: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    summary_path = root / "validation_summary.json"
    summary = json.loads(summary_path.read_text())
    expected = len(entries)
    if int(summary.get("expected_jobs", -1)) != expected or int(
        summary.get("completed_jobs", -1)
    ) != expected:
        raise ValueError(f"legacy clean catalog is incomplete: {summary_path}")
    if not bool(summary.get("all_simulator_states_exact")) or not bool(
        summary.get("all_first_chunks_exact")
    ):
        raise ValueError(f"legacy clean catalog has inexact controls: {summary_path}")
    jobs = summary.get("jobs", [])
    catalog = {str(job["pair_id"]): job for job in jobs}
    if len(catalog) != expected:
        raise ValueError(f"legacy clean catalog has duplicate or missing pairs: {summary_path}")
    normalized = {}
    for entry in entries:
        pair_id = str(entry["pair_id"])
        if pair_id not in catalog:
            raise ValueError(f"legacy clean catalog omits manifest pair: {pair_id}")
        job = catalog[pair_id]
        if (
            int(job.get("noise_seed", -1)) != 0
            or job.get("status") != "completed"
            or not bool(job.get("simulator_state_exact"))
            or not bool(job.get("first_chunk_exact"))
            or job.get("initial_input_modes") != ["strict"]
            or job.get("scene_state_sha256") != _scene_state_hash(entry)
        ):
            raise ValueError(f"legacy clean catalog controls are invalid: {pair_id}")
        pair_summary = _pair_summary(root, pair_id)
        composite = _pair_composite(entry, pair_summary)
        if bool(job.get("both_successful")) != bool(pair_summary.get("both_successful")):
            raise ValueError(f"legacy clean catalog success mismatch: {pair_id}")
        normalized[pair_id] = {
            **job,
            "exact_dual_success_target_first": bool(composite),
        }
    return normalized


def _intervention_catalog(
    root: Path, expected: int, after_steps: int
) -> dict[str, dict[str, Any]]:
    summary_path = root / "validation_summary.json"
    summary = json.loads(summary_path.read_text())
    if int(summary.get("noise_seed", -1)) != 0:
        raise ValueError(f"validation catalog has the wrong noise seed: {summary_path}")
    if int(summary.get("expected_pairs", -1)) != expected or int(
        summary.get("completed_pairs", -1)
    ) != expected:
        raise ValueError(f"validation catalog is incomplete: {summary_path}")
    if not bool(summary.get("all_initial_states_exact")):
        raise ValueError(f"validation catalog has inexact initial states: {summary_path}")
    intervention = summary.get("intervention")
    if (
        not isinstance(intervention, dict)
        or intervention.get("family") != "early_exit"
        or int(intervention.get("after_steps", -1)) != after_steps
        or int(intervention.get("total_flow_steps", -1)) != 10
        or summary.get("intervene_replans") != "all"
    ):
        raise ValueError(f"validation catalog has the wrong early-exit condition: {summary_path}")
    jobs = summary.get("jobs", [])
    catalog = {str(job["pair_id"]): job for job in jobs}
    if len(catalog) != expected:
        raise ValueError(f"validation catalog has duplicate or missing pairs: {summary_path}")
    return catalog


def _pair_summary(root: Path, pair_id: str) -> dict[str, Any]:
    summary = json.loads((_artifact_pair_root(root, pair_id) / "summary.json").read_text())
    if summary.get("pair_id") != pair_id:
        raise ValueError("pair summary id mismatch")
    return summary


def _scene_state_hash(entry: dict[str, Any]) -> str:
    identity_hashes = entry.get("identity_hashes")
    if not isinstance(identity_hashes, dict):
        raise ValueError("manifest pair lacks identity hashes")
    value = identity_hashes.get("sim_state")
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("manifest pair lacks a valid simulator-state hash")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError("manifest simulator-state hash is not hexadecimal") from error
    return value


def _actions_equal(clean_root: Path, full_root: Path, pair_id: str, side: str) -> bool:
    clean_pair = _artifact_pair_root(clean_root, pair_id)
    full_pair = _artifact_pair_root(full_root, pair_id)
    clean = np.asarray(
        json.loads((clean_pair / f"{side}_actions.json").read_text()),
        dtype=np.float64,
    )
    full = np.asarray(
        json.loads((full_pair / f"{side}_actions.json").read_text()),
        dtype=np.float64,
    )
    return bool(clean.shape == full.shape and np.array_equal(clean, full))


def _artifact_pair_root(root: Path, pair_id: str) -> Path:
    candidates = [root / pair_id, root / pair_id / "noise_0"]
    selected = [path for path in candidates if (path / "summary.json").is_file()]
    if len(selected) != 1:
        raise ValueError(f"expected one pair artifact directory for {pair_id}, found {selected}")
    return selected[0]


def _pair_composite(entry: dict[str, Any], summary: dict[str, Any]) -> bool:
    return _pair_outcomes(entry, summary)["composite"]


def _pair_outcomes(entry: dict[str, Any], summary: dict[str, Any]) -> dict[str, bool]:
    by_side = {result["side"]: result for result in summary["results"]}
    if set(by_side) != {"base", "donor"}:
        raise ValueError("early-exit pair summary must contain both directions")
    target_first = all(
        _first_contact(by_side[side]) == entry[f"{side}_target"]
        for side in ("base", "donor")
    )
    eventual_success = all(
        bool(by_side[side]["success"]) for side in ("base", "donor")
    )
    return {
        "target_first": bool(target_first),
        "eventual_success": bool(eventual_success),
        "composite": bool(target_first and eventual_success),
    }


def _first_contact(result: dict[str, Any]) -> str | None:
    contacts = result.get("first_contact_step_by_object", {})
    if not contacts:
        return None
    first_step = min(contacts.values())
    first_objects = [name for name, step in contacts.items() if step == first_step]
    return first_objects[0] if len(first_objects) == 1 else None


def _first_replan_cluster_latency(summary: dict[str, Any]) -> float:
    values = []
    for result in summary["results"]:
        diagnostics = result.get("early_exit_diagnostics", [])
        first = [item for item in diagnostics if int(item["replan_index"]) == 0]
        if len(first) != 1 or float(first[0]["integration_ms"]) <= 0.0:
            raise ValueError("early-exit summary lacks one positive first-replan latency")
        values.append(float(first[0]["integration_ms"]))
    return float(np.mean(values))


def _bootstrap_median_interval(
    values: np.ndarray, *, samples: int = 10_000, seed: int = 0
) -> list[float]:
    if values.ndim != 1 or values.size == 0 or samples <= 0:
        raise ValueError("latency bootstrap requires nonempty one-dimensional values")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(samples, values.size))
    medians = np.median(values[indices], axis=1)
    return [float(value) for value in np.quantile(medians, [0.025, 0.975])]


def _clopper_pearson_interval(successes: int, trials: int) -> list[float]:
    if not 0 <= successes <= trials or trials <= 0:
        raise ValueError("invalid exact-binomial counts")
    alpha = 0.025
    lower = (
        0.0
        if successes == 0
        else 1.0 - binomial_upper_bound(trials - successes, trials, alpha)
    )
    upper = (
        1.0
        if successes == trials
        else binomial_upper_bound(successes, trials, alpha)
    )
    return [float(lower), float(upper)]


def _two_sided_sign_test_p(positive: int, negative: int) -> float:
    trials = positive + negative
    if trials == 0:
        return 1.0
    extreme = min(positive, negative)
    tail = sum(math.comb(trials, value) for value in range(extreme + 1))
    return min(1.0, 2.0 * tail / (2**trials))


if __name__ == "__main__":
    raise SystemExit(main())
