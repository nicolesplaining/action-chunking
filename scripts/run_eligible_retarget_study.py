#!/usr/bin/env python3
"""Freeze all action-only predictions, then run held-out retargeting rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from action_chunking.utility_artifacts import (
    build_utility_job,
    build_utility_summary,
    cluster_id,
    select_primary_directions,
)
from action_chunking.utility_prediction import validate_eligible_retarget_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-summary", type=Path, required=True)
    candidates = parser.add_mutually_exclusive_group(required=True)
    candidates.add_argument("--candidate-root", type=Path)
    candidates.add_argument("--candidate-index", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--orientation-calibration", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--action-chunking-commit", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.noise_seed != 0:
        raise ValueError("registered retarget utility requires noise seed zero")
    _validate_execution_binding(args.output, args.action_chunking_commit)
    gate = json.loads(args.gate_summary.read_text())
    if gate.get("selection_uses_continuation_outcomes") is not False:
        raise ValueError("eligibility selection must explicitly exclude continuation outcomes")
    endpoint_eligible = [row for row in gate["rows"] if row["eligible"]]
    if not endpoint_eligible:
        raise ValueError("endpoint gate contains no eligible retargeting directions")
    for row in endpoint_eligible:
        validate_eligible_retarget_row(row)
    orientation_calibration = json.loads(args.orientation_calibration.read_text())
    if orientation_calibration.get("selection_uses_continuation_outcomes") is not False:
        raise ValueError("orientation calibration must exclude continuation outcomes")
    if not orientation_calibration.get("all_pairs_pass_contrast"):
        raise ValueError("orientation calibration did not pass its clean-control gate")
    orientation_calibration_digest = _digest(args.orientation_calibration)
    eligible, selection = select_primary_directions(endpoint_eligible)
    gate_digest = _digest(args.gate_summary)
    manifest_by_pair = _candidate_manifests(args.candidate_root, args.candidate_index)
    missing = sorted({row["pair_id"] for row in eligible} - set(manifest_by_pair))
    if missing:
        raise ValueError(f"eligible pairs are absent from candidate manifests: {missing}")
    args.output.mkdir(parents=True, exist_ok=True)

    prediction_entries = []
    prediction_script = Path(__file__).with_name("sample_retarget_prediction.py")
    for row in eligible:
        pair_id = row["pair_id"]
        prediction_output = args.output / "predictions" / pair_id / row["new_side"]
        prediction_path = prediction_output / "prediction.json"
        if not prediction_path.is_file():
            subprocess.run(
                [
                    sys.executable,
                    str(prediction_script),
                    "--manifest",
                    str(manifest_by_pair[pair_id]),
                    "--pair-id",
                    pair_id,
                    "--new-side",
                    row["new_side"],
                    "--output",
                    str(prediction_output),
                    "--port",
                    str(args.port),
                    "--noise-seed",
                    str(args.noise_seed),
                ],
                check=True,
            )
        prediction = json.loads(prediction_path.read_text())
        if prediction.get("pair_id") != pair_id or prediction.get("new_side") != row["new_side"]:
            raise ValueError("existing prediction does not match the eligible direction")
        prediction_entries.append(
            {
                "pair_id": pair_id,
                "new_side": row["new_side"],
                "cluster_id": cluster_id(row),
                "manifest": str(manifest_by_pair[pair_id]),
                "manifest_sha256": _digest(manifest_by_pair[pair_id]),
                "prediction": str(prediction_path),
                "prediction_sha256": _digest(prediction_path),
                "valid": bool(prediction["valid"]),
                "predicted_last_successful_boundary": prediction.get("predicted_last_successful_boundary"),
            }
        )
    frozen_manifest = {
        "schema_version": 1,
        "all_predictions_frozen_before_closed_loop": True,
        "selection_uses_continuation_outcomes": False,
        "primary_direction_selection_rule": "first_endpoint_eligible_in_frozen_gate_order",
        "endpoint_eligible_directions": len(endpoint_eligible),
        "selected_independent_clusters": len(eligible),
        "direction_selection": selection,
        "noise_seed": args.noise_seed,
        "action_chunking_commit": args.action_chunking_commit,
        "gate_summary": str(args.gate_summary),
        "gate_summary_sha256": gate_digest,
        "orientation_calibration": str(args.orientation_calibration),
        "orientation_calibration_sha256": orientation_calibration_digest,
        "entries": prediction_entries,
    }
    frozen_path = args.output / "frozen_predictions.json"
    serialized = json.dumps(frozen_manifest, indent=2, sort_keys=True) + "\n"
    if frozen_path.is_file():
        if frozen_path.read_text() != serialized:
            raise ValueError("existing frozen prediction manifest differs from current inputs")
    else:
        frozen_path.write_text(serialized)
    frozen_digest = _digest(frozen_path)

    sweep_script = Path(__file__).with_name("run_dynamic_retarget_sweep.py")
    orientation_script = Path(__file__).with_name("analyze_grasp_orientation_sweep.py")
    jobs = []
    for row in eligible:
        _validate_frozen_inputs(
            args.gate_summary,
            gate_digest,
            prediction_entries,
            frozen_path,
            frozen_digest,
            args.orientation_calibration,
            orientation_calibration_digest,
            args.output,
            args.action_chunking_commit,
        )
        pair_id = row["pair_id"]
        rollout_output = args.output / "rollouts" / pair_id / row["new_side"]
        subprocess.run(
            [
                sys.executable,
                str(sweep_script),
                "--manifest",
                str(manifest_by_pair[pair_id]),
                "--pair-id",
                pair_id,
                "--output",
                str(rollout_output),
                "--gpu",
                str(args.gpu),
                "--port",
                str(args.port),
                "--noise-seed",
                str(args.noise_seed),
                "--boundaries",
                ",".join(str(boundary) for boundary in range(11)),
                "--sides",
                row["new_side"],
            ],
            check=True,
        )
        orientation_path = rollout_output / "grasp_orientation.json"
        subprocess.run(
            [
                sys.executable,
                str(orientation_script),
                "--sweep",
                str(rollout_output),
                "--manifest",
                str(manifest_by_pair[pair_id]),
                "--pair-id",
                pair_id,
                "--calibration",
                str(args.orientation_calibration),
                "--output",
                str(orientation_path),
            ],
            check=True,
        )
        _validate_frozen_inputs(
            args.gate_summary,
            gate_digest,
            prediction_entries,
            frozen_path,
            frozen_digest,
            args.orientation_calibration,
            orientation_calibration_digest,
            args.output,
            args.action_chunking_commit,
        )
        jobs.append(
            build_utility_job(
                row,
                rollout_output,
                prediction_entries,
                orientation_path,
            )
        )
        _write_summary(args.output, jobs, len(eligible), frozen_path, frozen_digest)
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("audit_retarget_utility.py")),
            "--study-root",
            str(args.output),
            "--output",
            str(args.output / "final_audit.json"),
        ],
        check=True,
    )
    return 0


def _candidate_manifests(root: Path | None, index_path: Path | None) -> dict[str, Path]:
    if index_path is not None:
        index = json.loads(index_path.read_text())
        if index.get("selection_uses_continuation_outcomes") is not False:
            raise ValueError("candidate index must exclude continuation outcomes")
        result = {pair_id: Path(path) for pair_id, path in index["manifest_by_pair"].items()}
        expected_digests = index.get("manifest_sha256_by_pair")
        if not isinstance(expected_digests, dict) or set(expected_digests) != set(result):
            raise ValueError("candidate index must contain one frozen digest per manifest")
        missing = [str(path) for path in result.values() if not path.is_file()]
        if missing:
            raise ValueError(f"candidate index contains missing manifests: {missing}")
        changed = [pair_id for pair_id, path in result.items() if _digest(path) != expected_digests[pair_id]]
        if changed:
            raise ValueError(f"candidate manifests changed after catalog handoff: {changed}")
        return result
    if root is None:
        raise ValueError("candidate root or index is required")
    result = {}
    paths = {
        *root.glob("**/offset_*/manifest.json"),
        *root.glob("**/aligned/manifest.json"),
    }
    for path in sorted(paths):
        manifest = json.loads(path.read_text())
        for entry in manifest["pairs"]:
            pair_id = entry["pair_id"]
            if pair_id in result:
                raise ValueError(f"duplicate candidate pair id: {pair_id}")
            result[pair_id] = path
    return result


def _write_summary(
    output: Path,
    jobs: list[dict[str, Any]],
    expected: int,
    frozen_path: Path,
    frozen_digest: str,
) -> None:
    payload = build_utility_summary(jobs, expected, frozen_path, frozen_digest)
    (output / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"completed {len(jobs)}/{expected} primary scene clusters", flush=True)
def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_frozen_inputs(
    gate_path: Path,
    gate_digest: str,
    prediction_entries: list[dict[str, Any]],
    frozen_path: Path,
    frozen_digest: str,
    orientation_calibration_path: Path,
    orientation_calibration_digest: str,
    output: Path,
    action_chunking_commit: str,
) -> None:
    _validate_execution_binding(output, action_chunking_commit)
    if _digest(gate_path) != gate_digest:
        raise ValueError("endpoint gate changed after predictions were frozen")
    if _digest(frozen_path) != frozen_digest:
        raise ValueError("frozen prediction manifest changed after closed-loop rollout began")
    if _digest(orientation_calibration_path) != orientation_calibration_digest:
        raise ValueError("orientation calibration changed after closed-loop rollout began")
    for entry in prediction_entries:
        if _digest(Path(entry["manifest"])) != entry["manifest_sha256"]:
            raise ValueError("candidate manifest changed after predictions were frozen")
        if _digest(Path(entry["prediction"])) != entry["prediction_sha256"]:
            raise ValueError("action-only prediction changed after it was frozen")


def _validate_execution_binding(output: Path, action_chunking_commit: str) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", action_chunking_commit) is None:
        raise ValueError("retarget utility requires a full lowercase code commit")
    repo = Path(__file__).resolve().parents[1]
    actual = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual != action_chunking_commit:
        raise ValueError("retarget utility code commit differs from the current checkout")
    status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain=v1", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise ValueError("retarget utility requires a completely clean worktree")
    binding = output / "code_commit.txt"
    if output.exists() and not binding.is_file():
        raise ValueError("existing retarget utility output lacks a code-commit binding")
    output.mkdir(parents=True, exist_ok=True)
    if binding.is_file() and binding.read_text().strip() != action_chunking_commit:
        raise ValueError("existing retarget utility output uses a different code commit")
    binding.write_text(action_chunking_commit + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
