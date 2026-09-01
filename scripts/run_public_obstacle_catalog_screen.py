#!/usr/bin/env python3
"""Run the frozen public-catalog obstacle screen without patched outcomes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from action_chunking.pairs import file_digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--noise-seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.gpu < 0 or args.port <= 0 or args.noise_seed < 0:
        raise ValueError("gpu and noise seed must be nonnegative; port must be positive")
    plan = json.loads(args.plan.read_text())
    source_plan = json.loads(args.source_plan.read_text())
    _validate_plans(plan, source_plan, args.source_plan)
    source_rows = {row["screen_id"]: row for row in source_plan["rows"]}
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = []
    selected = None
    for plan_row in plan["rows"]:
        row = source_rows[plan_row["screen_id"]]
        index = int(plan_row["obstacle_plan_index"])
        row_root = args.output / "rows" / f"{index:05d}_{row['screen_id'][:12]}"
        result_path = row_root / "row_result.json"
        if result_path.is_file():
            result = json.loads(result_path.read_text())
        else:
            result = _run_row(index, row, row_root, args)
            result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        jobs.append(result)
        if result.get("selected_pair_id"):
            selected = result
        _write_summary(args, plan, jobs, selected)
        if selected is not None:
            break
    return 0


def _run_row(
    index: int,
    row: dict[str, Any],
    row_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[1]
    row_root.mkdir(parents=True, exist_ok=True)
    source = row_root / "source"
    source_manifest = source / "manifest.json"
    if not source_manifest.is_file():
        subprocess.run(
            [
                str(repo / "scripts" / "run_instruction_pair_generation.sh"),
                row["suite"],
                row["base_task"],
                row["donor_task"],
                "1",
                str(args.gpu),
                str(source),
                str(row["init_index"]),
            ],
            check=True,
        )
    source_data = json.loads(source_manifest.read_text())
    source_entries = source_data["pairs"]
    if len(source_entries) != 1:
        raise ValueError("public obstacle source generation must produce exactly one pair")
    source_entry = source_entries[0]
    if (
        int(source_entry["init_index"]) != int(row["init_index"])
        or source_entry["base_target"] != row["base_target"]
        or source_entry["donor_target"] != row["donor_target"]
    ):
        raise ValueError("generated obstacle source fixture differs from the frozen row")

    base_gate = row_root / "source_base_gate"
    base_gate_summary_path = base_gate / "summary.json"
    if not base_gate_summary_path.is_file():
        completed = subprocess.run(
            [
                str(repo / "scripts" / "run_pair_validation.sh"),
                str(source_manifest),
                source_entry["pair_id"],
                str(args.gpu),
                str(args.port),
                str(args.noise_seed),
                str(base_gate),
                "",
                "strict",
                "false",
                "",
                "0",
                "false",
                "false",
                "",
                "",
                "400",
                "base",
                "0",
            ],
            check=False,
        )
        if completed.returncode not in {0, 1} or not base_gate_summary_path.is_file():
            raise RuntimeError("source-base obstacle pre-gate produced no summary")
    base_gate_summary = json.loads(base_gate_summary_path.read_text())
    base_endpoint = _base_endpoint(base_gate_summary, row["base_target"])
    base_fields = {
        "source_base_gate": str(base_gate_summary_path),
        "source_base_gate_sha256": file_digest(base_gate_summary_path),
        **base_endpoint,
    }
    if not base_endpoint["source_base_endpoint_eligible"]:
        obstacle_manifest = row_root / "fixtures" / "manifest.json"
        existing_obstacle = (
            json.loads(obstacle_manifest.read_text()) if obstacle_manifest.is_file() else None
        )
        return {
            "obstacle_plan_index": index,
            "screen_id": row["screen_id"],
            "cluster_id": row["cluster_id"],
            "suite": row["suite"],
            "init_index": int(row["init_index"]),
            "source_pair_id": source_entry["pair_id"],
            "source_manifest": str(source_manifest),
            "source_manifest_sha256": file_digest(source_manifest),
            **base_fields,
            "obstacle_manifest": str(obstacle_manifest) if existing_obstacle else None,
            "obstacle_manifest_sha256": (
                file_digest(obstacle_manifest) if existing_obstacle else None
            ),
            "generated_candidates": (
                len(existing_obstacle["pairs"]) if existing_obstacle else 0
            ),
            "geometric_exclusions": (
                len(existing_obstacle["exclusions"]) if existing_obstacle else 0
            ),
            "geometry_exhausted": (
                bool(existing_obstacle.get("geometry_exhausted"))
                if existing_obstacle
                else None
            ),
            "geometry_not_evaluated": existing_obstacle is None,
            "selection_uses_interventions": False,
            "status": "source_base_endpoint_ineligible",
            "clean_screened_pairs": 0,
            "eligible_pairs": 0,
            "selected_pair_id": None,
        }

    fixtures = row_root / "fixtures"
    obstacle_manifest = fixtures / "manifest.json"
    if not obstacle_manifest.is_file():
        subprocess.run(
            [
                str(repo / "scripts" / "run_obstacle_pose_generation.sh"),
                str(source_manifest),
                source_entry["pair_id"],
                str(args.gpu),
                str(fixtures),
            ],
            check=True,
        )
    obstacle_data = json.loads(obstacle_manifest.read_text())
    generated = len(obstacle_data["pairs"])
    base = {
        "obstacle_plan_index": index,
        "screen_id": row["screen_id"],
        "cluster_id": row["cluster_id"],
        "suite": row["suite"],
        "init_index": int(row["init_index"]),
        "source_pair_id": source_entry["pair_id"],
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": file_digest(source_manifest),
        "obstacle_manifest": str(obstacle_manifest),
        "obstacle_manifest_sha256": file_digest(obstacle_manifest),
        "generated_candidates": generated,
        "geometric_exclusions": len(obstacle_data["exclusions"]),
        "geometry_exhausted": bool(obstacle_data.get("geometry_exhausted")),
        "geometry_not_evaluated": False,
        "selection_uses_interventions": False,
        **base_fields,
    }
    if generated == 0:
        return {
            **base,
            "status": "geometry_exhausted",
            "clean_screened_pairs": 0,
            "eligible_pairs": 0,
            "selected_pair_id": None,
        }

    clean = row_root / "clean"
    validation_path = clean / "validation_summary.json"
    if not validation_path.is_file():
        subprocess.run(
            [
                sys.executable,
                str(repo / "scripts" / "run_manifest_pair_validations.py"),
                "--manifest",
                str(obstacle_manifest),
                "--output",
                str(clean),
                "--gpu",
                str(args.gpu),
                "--port",
                str(args.port),
                "--noise-seed",
                str(args.noise_seed),
            ],
            check=True,
        )
    screen = row_root / "screen"
    screen_summary_path = screen / "summary.json"
    if not screen_summary_path.is_file():
        subprocess.run(
            [
                sys.executable,
                str(repo / "scripts" / "screen_obstacle_pose_pairs.py"),
                "--manifest",
                str(obstacle_manifest),
                "--clean-validation",
                str(clean),
                "--output",
                str(screen),
            ],
            check=True,
        )
    screen_summary = json.loads(screen_summary_path.read_text())
    if screen_summary.get("selection_uses_interventions") is not False:
        raise ValueError("public obstacle screen must exclude intervention outcomes")
    return {
        **base,
        "status": "clean_screened",
        "clean_validation": str(validation_path),
        "clean_validation_sha256": file_digest(validation_path),
        "screen_summary": str(screen_summary_path),
        "screen_summary_sha256": file_digest(screen_summary_path),
        "clean_screened_pairs": int(screen_summary["screened_pairs"]),
        "eligible_pairs": int(screen_summary["eligible_pairs"]),
        "selected_pair_id": screen_summary["selected_pair_id"],
        "selected_manifest": screen_summary["selected_manifest"],
        "selected_manifest_sha256": screen_summary["selected_manifest_sha256"],
    }


def _write_summary(
    args: argparse.Namespace,
    plan: dict[str, Any],
    jobs: list[dict[str, Any]],
    selected: dict[str, Any] | None,
) -> None:
    payload = {
        "schema_version": 1,
        "protocol_version": "0.15",
        "selection_uses_interventions": False,
        "plan": str(args.plan),
        "plan_sha256": file_digest(args.plan),
        "source_plan": str(args.source_plan),
        "source_plan_sha256": file_digest(args.source_plan),
        "planned_source_rows": int(plan["candidate_rows"]),
        "processed_source_rows": len(jobs),
        "catalog_exhausted": len(jobs) == int(plan["candidate_rows"]) and selected is None,
        "stop_threshold_reached": selected is not None,
        "selected_pair_id": selected.get("selected_pair_id") if selected else None,
        "selected_manifest": selected.get("selected_manifest") if selected else None,
        "selected_manifest_sha256": (
            selected.get("selected_manifest_sha256") if selected else None
        ),
        "total_geometric_exclusions": sum(job["geometric_exclusions"] for job in jobs),
        "total_clean_screened_pairs": sum(job["clean_screened_pairs"] for job in jobs),
        "jobs": jobs,
    }
    (args.output / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def _validate_plans(
    plan: dict[str, Any], source: dict[str, Any], source_path: Path
) -> None:
    if plan.get("selection_uses_obstacle_intervention_outcomes") is not False:
        raise ValueError("obstacle plan must exclude obstacle intervention outcomes")
    if source.get("selection_uses_intervention_outcomes") is not False:
        raise ValueError("source plan must exclude intervention outcomes")
    if plan["source_plan_sha256"] != file_digest(source_path):
        raise ValueError("source plan digest differs from frozen obstacle plan")
    source_ids = {row["screen_id"] for row in source["rows"]}
    ordered_ids = [row["screen_id"] for row in plan["rows"]]
    if len(ordered_ids) != len(set(ordered_ids)) or set(ordered_ids) != source_ids:
        raise ValueError("obstacle plan must contain every source row exactly once")


def _base_endpoint(summary: dict[str, Any], target: str) -> dict[str, Any]:
    if summary.get("requested_sides") != ["base"] or len(summary.get("results", [])) != 1:
        raise ValueError("source-base obstacle pre-gate must contain exactly the base side")
    result = summary["results"][0]
    contacts = {
        str(name): int(step)
        for name, step in result["first_contact_step_by_object"].items()
    }
    first_contact = min(contacts, key=contacts.get) if contacts else None
    input_exact = all(
        field["array_equal"] for field in result["live_initial_input_diagnostics"].values()
    )
    state_exact = result["restored_sim_state_max_abs_error"] == 0.0
    return {
        "source_base_input_exact": input_exact,
        "source_base_simulator_state_exact": state_exact,
        "source_base_task_success": bool(result["success"]),
        "source_base_first_contact_object": first_contact,
        "source_base_target_first": first_contact == target,
        "source_base_endpoint_eligible": bool(
            input_exact and state_exact and result["success"] and first_contact == target
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main())
