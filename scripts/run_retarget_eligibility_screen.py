#!/usr/bin/env python3
"""Screen pre-contact states using only clean old-condition and restart endpoints."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from action_chunking.pairs import array_digest, load_instruction_pair
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
    manifests = sorted(
        {
            *args.candidate_root.glob("**/offset_*/manifest.json"),
            *args.candidate_root.glob("**/aligned/manifest.json"),
        }
    )
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
                noise_start_index=int(entry.get("source_replan_index") or 0),
            )
            source_chunk_exact = _source_chunk_exact(entry, job_root / "event_horizon")
            source_input_exact = _source_input_exact(
                entry, manifest_path.parent / entry["fixture"]
            )
            row = eligibility_row(
                entry,
                event_summary,
                args.execution_horizon,
                source_chunk_exact=source_chunk_exact,
                source_input_exact=source_input_exact,
            )
            if row["event_gate_pass"]:
                competence_summary = _run_endpoint(
                    launcher,
                    manifest_path,
                    entry["pair_id"],
                    job_root / "clean_competence",
                    args,
                    stop_after_contact=False,
                    max_steps=400,
                    noise_start_index=int(entry.get("source_replan_index") or 0),
                )
                row = eligibility_row(
                    entry,
                    event_summary,
                    args.execution_horizon,
                    competence_summary,
                    source_chunk_exact=source_chunk_exact,
                    source_input_exact=source_input_exact,
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
    noise_start_index: int,
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
            "fixture",
            "false",
            "",
            "0",
            str(stop_after_contact).lower(),
            "false",
            "",
            "",
            str(max_steps),
            "base,donor",
            str(noise_start_index),
        ]
        completed = subprocess.run(command, check=False)
        if completed.returncode not in {0, 1} or not summary_path.is_file():
            raise RuntimeError(f"clean eligibility endpoint produced no summary: {pair_id}")
    summary = json.loads(summary_path.read_text())
    if int(summary.get("noise_seed", -1)) != args.noise_seed:
        raise ValueError("existing eligibility result has a different noise seed")
    if bool(summary.get("stop_after_first_task_contact")) != stop_after_contact:
        raise ValueError("existing eligibility endpoint has a different stopping rule")
    if int(summary.get("noise_start_index", 0)) != noise_start_index:
        raise ValueError("existing eligibility endpoint has a different noise start index")
    return summary


def _source_chunk_exact(entry: dict[str, Any], event_output: Path) -> bool:
    expected = entry.get("source_action_chunk_sha256")
    if expected is None:
        return True
    origin = entry["origin_side"]
    chunks = json.loads((event_output / f"{origin}_actions.json").read_text())
    if len(chunks) != 1:
        raise ValueError("bounded event screen must sample exactly one source action chunk")
    return array_digest(np.asarray(chunks[0], dtype=np.float64)) == expected


def _source_input_exact(entry: dict[str, Any], fixture_path: Path) -> bool:
    expected = entry.get("source_replan_input_sha256")
    if expected is None:
        return True
    pair = load_instruction_pair(fixture_path)
    arrays = {
        "image": pair.base_image,
        "wrist_image": pair.base_wrist_image,
        "state": pair.base_state,
    }
    if any(
        not np.array_equal(value, getattr(pair, f"donor_{key}"))
        for key, value in arrays.items()
    ):
        return False
    return {key: array_digest(value) for key, value in arrays.items()} == expected


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
