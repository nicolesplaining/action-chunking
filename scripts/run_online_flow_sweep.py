#!/usr/bin/env python3
"""Run and summarize closed-loop flow-switch interventions for one paired state."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--boundaries", default="all", help="all or comma-separated values in 0..10")
    parser.add_argument("--intervene-replans", default="all")
    parser.add_argument("--rollout-endpoint", choices=("first_contact", "full"), default="first_contact")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    boundaries = _boundaries(args.boundaries)
    manifest = json.loads(args.manifest.read_text())
    entry = _manifest_entry(manifest, args.pair_id)
    launcher = Path(__file__).with_name("run_pair_validation.sh")
    args.output.mkdir(parents=True, exist_ok=True)
    specs = args.output / "specs"
    specs.mkdir(exist_ok=True)

    for boundary in boundaries:
        run_output = args.output / f"switch_after_{boundary}"
        summary_path = run_output / "summary.json"
        if summary_path.exists():
            _validate_summary(json.loads(summary_path.read_text()), args, boundary)
            continue
        spec_path = specs / f"flow_switch_{boundary}.json"
        spec_path.write_text(
            json.dumps({"family": "flow_switch", "switch_after_steps": boundary}, indent=2) + "\n"
        )
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
            str(spec_path),
            args.intervene_replans,
            str(args.rollout_endpoint == "first_contact").lower(),
        ]
        completed = subprocess.run(command, check=False)
        if completed.returncode not in {0, 1} or not summary_path.exists():
            raise RuntimeError(f"flow boundary {boundary} failed without a behavioral summary")
        _validate_summary(json.loads(summary_path.read_text()), args, boundary)
        _write_tables(args.output, args.pair_id, entry, args.noise_seed)
    _write_tables(args.output, args.pair_id, entry, args.noise_seed)
    return 0


def _write_tables(output: Path, pair_id: str, entry: dict[str, Any], noise_seed: int) -> None:
    rows = []
    for path in sorted(output.glob("switch_after_*/summary.json")):
        summary = json.loads(path.read_text())
        boundary = int(summary["intervention"]["switch_after_steps"])
        for result in summary["results"]:
            side = result["side"]
            other_side = "donor" if side == "base" else "base"
            contacts = result["first_contact_step_by_object"]
            first_object = min(contacts, key=contacts.get) if contacts else None
            destination = result.get("destination_evaluation")
            endpoint_object = (
                destination["nearest_registered_destination"]
                if destination is not None
                else first_object
            )
            rows.append(
                {
                    "pair_id": pair_id,
                    "noise_seed": noise_seed,
                    "switch_after_steps": boundary,
                    "side": side,
                    "source_target": entry[f"{side}_target"],
                    "donor_target": entry[f"{other_side}_target"],
                    "first_contact_object": first_object,
                    "first_contact_step": contacts.get(first_object) if first_object is not None else None,
                    "first_contact_is_source": first_object == entry[f"{side}_target"],
                    "first_contact_is_donor": first_object == entry[f"{other_side}_target"],
                    "outcome_mode": "destination" if destination is not None else "first_contact",
                    "endpoint_choice": endpoint_object,
                    "endpoint_is_source": endpoint_object == entry[f"{side}_target"],
                    "endpoint_is_donor": endpoint_object == entry[f"{other_side}_target"],
                    "destination_margin_m": (
                        destination["nearest_destination_margin_m"] if destination is not None else None
                    ),
                    "success": bool(result["success"]),
                    "steps": int(result["steps"]),
                    "initial_input_exact": all(
                        value["array_equal"] for value in result["live_initial_input_diagnostics"].values()
                    ),
                    "simulator_state_exact": result["restored_sim_state_max_abs_error"] == 0.0,
                    "terminated_after_first_task_contact": result.get(
                        "terminated_after_first_task_contact", False
                    ),
                }
            )
    rows.sort(key=lambda row: (row["switch_after_steps"], row["side"]))
    if not rows:
        return
    with (output / "rollouts.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    by_boundary = []
    for boundary in sorted({row["switch_after_steps"] for row in rows}):
        selected = [row for row in rows if row["switch_after_steps"] == boundary]
        by_boundary.append(
            {
                "switch_after_steps": boundary,
                "sides": len(selected),
                "source_first_contacts": sum(row["first_contact_is_source"] for row in selected),
                "donor_first_contacts": sum(row["first_contact_is_donor"] for row in selected),
                "source_endpoint_choices": sum(row["endpoint_is_source"] for row in selected),
                "donor_endpoint_choices": sum(row["endpoint_is_donor"] for row in selected),
                "successes": sum(row["success"] for row in selected),
            }
        )
    summary = {
        "schema_version": 1,
        "pair_id": pair_id,
        "noise_seed": noise_seed,
        "intervention_applied_at_replans": json.loads(next(output.glob("switch_after_*/summary.json")).read_text())[
            "intervene_replans"
        ],
        "rollout_endpoint": (
            "first_contact"
            if all(row["terminated_after_first_task_contact"] for row in rows)
            else "full"
            if not any(row["terminated_after_first_task_contact"] for row in rows)
            else "mixed"
        ),
        "all_initial_inputs_exact": all(row["initial_input_exact"] for row in rows),
        "all_simulator_states_exact": all(row["simulator_state_exact"] for row in rows),
        "boundaries": by_boundary,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def _validate_summary(summary: dict[str, Any], args: argparse.Namespace, boundary: int) -> None:
    expected = {"family": "flow_switch", "switch_after_steps": boundary}
    if summary["pair_id"] != args.pair_id or int(summary["noise_seed"]) != args.noise_seed:
        raise ValueError("existing flow sweep output does not match pair or noise seed")
    if summary.get("intervention") != expected or summary.get("intervene_replans") != args.intervene_replans:
        raise ValueError("existing flow sweep output does not match the requested intervention")
    if summary.get("stop_after_first_task_contact", False) != (args.rollout_endpoint == "first_contact"):
        raise ValueError("existing flow sweep output does not match the requested rollout endpoint")
    for result in summary["results"]:
        if not all(value["array_equal"] for value in result["live_initial_input_diagnostics"].values()):
            raise ValueError("intervened rollout did not restore the exact initial model input")
        if result["restored_sim_state_max_abs_error"] != 0.0:
            raise ValueError("intervened rollout did not restore the exact simulator state")


def _boundaries(value: str) -> list[int]:
    if value == "all":
        return list(range(11))
    try:
        boundaries = sorted({int(item) for item in value.split(",")})
    except ValueError as error:
        raise ValueError("boundaries must be all or comma-separated integers") from error
    if not boundaries or boundaries[0] < 0 or boundaries[-1] > 10:
        raise ValueError("boundaries must lie in [0, 10]")
    return boundaries


def _manifest_entry(manifest: dict[str, Any], pair_id: str) -> dict[str, Any]:
    matches = [entry for entry in manifest["pairs"] if entry["pair_id"] == pair_id]
    if len(matches) != 1:
        raise ValueError(f"expected one manifest entry for {pair_id!r}, found {len(matches)}")
    return matches[0]


if __name__ == "__main__":
    raise SystemExit(main())
