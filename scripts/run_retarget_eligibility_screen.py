#!/usr/bin/env python3
"""Screen pre-contact states using only clean old-condition and restart endpoints."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any

from action_chunking.retarget_eligibility import eligibility_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--execution-horizon", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.execution_horizon <= 0:
        raise ValueError("execution horizon must be positive")
    manifests = sorted(args.candidate_root.glob("**/offset_*/manifest.json"))
    if not manifests:
        raise ValueError("candidate root contains no offset manifests")
    args.output.mkdir(parents=True, exist_ok=True)
    launcher = Path(__file__).with_name("run_pair_validation.sh")
    records: list[dict[str, Any]] = []
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("pair_family") != "instruction_target_precontact":
            raise ValueError(f"unexpected pair family in {manifest_path}")
        relative_parent = manifest_path.parent.relative_to(args.candidate_root)
        for entry in manifest["pairs"]:
            job_root = args.output / relative_parent / entry["pair_id"]
            event_summary = _run_endpoint(
                launcher,
                manifest_path,
                entry["pair_id"],
                job_root / "event_horizon",
                args,
                stop_after_contact=True,
                max_steps=args.execution_horizon,
            )
            row = eligibility_row(entry, event_summary, args.execution_horizon)
            if row["event_gate_pass"]:
                competence_summary = _run_endpoint(
                    launcher,
                    manifest_path,
                    entry["pair_id"],
                    job_root / "clean_competence",
                    args,
                    stop_after_contact=False,
                    max_steps=400,
                )
                row = eligibility_row(
                    entry,
                    event_summary,
                    args.execution_horizon,
                    competence_summary,
                )
            records.append(row)
            _write_outputs(args.output, records, len(manifests))
    return 0


def _run_endpoint(
    launcher: Path,
    manifest: Path,
    pair_id: str,
    output: Path,
    args: argparse.Namespace,
    *,
    stop_after_contact: bool,
    max_steps: int,
) -> dict[str, Any]:
    summary_path = output / "summary.json"
    if not summary_path.is_file():
        command = [
            str(launcher),
            str(manifest),
            pair_id,
            str(args.gpu),
            str(args.port),
            str(args.noise_seed),
            str(output),
            "",
            "strict",
            "false",
            "",
            "0",
            str(stop_after_contact).lower(),
            "false",
            "",
            "",
            str(max_steps),
        ]
        completed = subprocess.run(command, check=False)
        if completed.returncode not in {0, 1} or not summary_path.is_file():
            raise RuntimeError(f"clean eligibility endpoint produced no summary: {pair_id}")
    summary = json.loads(summary_path.read_text())
    if int(summary.get("noise_seed", -1)) != args.noise_seed:
        raise ValueError("existing eligibility result has a different noise seed")
    if bool(summary.get("stop_after_first_task_contact")) != stop_after_contact:
        raise ValueError("existing eligibility endpoint has a different stopping rule")
    return summary


def _write_outputs(output: Path, rows: list[dict[str, Any]], manifest_count: int) -> None:
    rows.sort(key=lambda row: (row["precontact_offset_steps"], row["origin_side"]))
    with (output / "eligibility.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema_version": 1,
        "selection_uses_continuation_outcomes": False,
        "candidate_manifests": manifest_count,
        "completed_directions": len(rows),
        "event_gate_directions": sum(bool(row["event_gate_pass"]) for row in rows),
        "eligible_directions": sum(bool(row["eligible"]) for row in rows),
        "eligible_pair_ids": [row["pair_id"] for row in rows if row["eligible"]],
        "rows": rows,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        f"screened {len(rows)} directions: {summary['eligible_directions']} eligible",
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
