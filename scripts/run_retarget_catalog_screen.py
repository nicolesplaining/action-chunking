#!/usr/bin/env python3
"""Execute the frozen public-catalog endpoint screen in exact prefix order."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from action_chunking.catalog_progress import summarize_catalog_progress
from action_chunking.pairs import file_digest
from action_chunking.utility_prediction import audit_prediction_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument(
        "--workers",
        help="comma-separated gpu:port workers; overrides --gpu and --port",
    )
    parser.add_argument("--noise-seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.noise_seed != 0:
        raise ValueError("registered catalog screening requires noise seed zero")
    workers = _parse_workers(args.workers, args.gpu, args.port, args.noise_seed)
    plan = json.loads(args.plan.read_text())
    if plan.get("selection_uses_intervention_outcomes") is not True:
        raise ValueError("catalog plan must disclose action-only intervention selection")
    if (
        plan.get("selection_uses_continuation_outcomes") is not False
        or plan.get("selection_uses_action_only_prediction_validity") is not True
    ):
        raise ValueError("catalog plan must freeze outcome-blind prediction-validity selection")
    code_commit = _read_execution_binding(args.output, args.plan)
    run_catalog(plan, args.plan, args.output, workers, code_commit)
    return 0


@dataclass(frozen=True)
class WorkerConfig:
    gpu: int
    port: int
    noise_seed: int


def _parse_workers(
    value: str | None,
    gpu: int,
    port: int,
    noise_seed: int,
) -> list[WorkerConfig]:
    fields = value.split(",") if value else [f"{gpu}:{port}"]
    workers = []
    for field in fields:
        parts = field.split(":")
        if len(parts) != 2 or not all(part.isdigit() for part in parts):
            raise ValueError("catalog workers must use comma-separated gpu:port entries")
        workers.append(WorkerConfig(int(parts[0]), int(parts[1]), noise_seed))
    if not 1 <= len(workers) <= 2:
        raise ValueError("catalog screening supports one or two workers")
    if any(worker.port <= 0 for worker in workers):
        raise ValueError("catalog worker ports must be positive")
    if len({worker.gpu for worker in workers}) != len(workers):
        raise ValueError("parallel catalog workers require distinct GPUs")
    if len({worker.port for worker in workers}) != len(workers):
        raise ValueError("parallel catalog workers require distinct ports")
    return workers


def run_catalog(
    plan: dict[str, Any],
    plan_path: Path,
    output: Path,
    workers: list[WorkerConfig],
    code_commit: str,
) -> dict[str, Any]:
    if plan.get("selection_uses_intervention_outcomes") is not True:
        raise ValueError("catalog plan must disclose action-only intervention selection")
    if (
        plan.get("selection_uses_continuation_outcomes") is not False
        or plan.get("selection_uses_action_only_prediction_validity") is not True
    ):
        raise ValueError("catalog plan must freeze outcome-blind prediction-validity selection")
    if not plan.get("rows"):
        raise ValueError("catalog plan contains no screening rows")
    if not 1 <= len(workers) <= 2:
        raise ValueError("catalog execution requires one or two workers")
    if len(code_commit) != 40 or any(character not in "0123456789abcdef" for character in code_commit):
        raise ValueError("catalog execution requires a full lowercase code commit")
    output.mkdir(parents=True, exist_ok=True)
    jobs: list[dict[str, Any]] = []
    rows = plan["rows"]
    for batch_start in range(0, len(rows), len(workers)):
        batch = [
            (index, rows[index], workers[offset])
            for offset, index in enumerate(
                range(batch_start, min(batch_start + len(workers), len(rows)))
            )
        ]
        completed: dict[int, tuple[dict[str, Any], Path]] = {}
        pending = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch)) as pool:
            for index, row, worker in batch:
                row_root = output / "rows" / f"{index:05d}_{row['screen_id'][:12]}"
                result_path = row_root / "row_result.json"
                if result_path.is_file():
                    result = json.loads(result_path.read_text())
                    _validate_row_result(result, index, row)
                    completed[index] = (result, result_path)
                else:
                    pending[index] = (
                        pool.submit(_run_row, index, row, row_root, worker),
                        row,
                        result_path,
                    )
            for index, (future, row, result_path) in pending.items():
                result = future.result()
                _validate_row_result(result, index, row)
                _write_immutable_json(result_path, result)
                completed[index] = (result, result_path)

        ordered = [
            _catalog_job(*completed[index])
            for index, _row, _worker in batch
        ]
        for position, job in enumerate(ordered):
            jobs.append(job)
            progress = summarize_catalog_progress(plan, jobs)
            speculative = ordered[position + 1 :] if progress["stop_threshold_reached"] else []
            summary = _write_summary(
                output,
                plan_path,
                workers,
                code_commit,
                jobs,
                progress,
                speculative,
            )
            if progress["stop_threshold_reached"]:
                return summary
    return summary


def _read_execution_binding(output: Path, plan_path: Path) -> str:
    commit_path = output / "code_commit.txt"
    plan_digest_path = output / "plan.sha256"
    if not commit_path.is_file() or not plan_digest_path.is_file():
        raise ValueError("catalog execution requires launcher-written code and plan bindings")
    if plan_digest_path.read_text().strip() != file_digest(plan_path):
        raise ValueError("catalog launcher plan binding differs from the supplied plan")
    return commit_path.read_text().strip()


def _run_row(
    index: int, row: dict[str, Any], row_root: Path, args: WorkerConfig
) -> dict[str, Any]:
    row_root.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
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
    manifest = json.loads(source_manifest.read_text())
    entries = manifest.get("pairs", [])
    if len(entries) != 1 or int(entries[0]["init_index"]) != int(row["init_index"]):
        raise ValueError("generated source fixture does not match the frozen plan row")
    entry = entries[0]
    if (
        entry["base_target"] != row["base_target"]
        or entry["donor_target"] != row["donor_target"]
    ):
        raise ValueError("generated source targets do not match the frozen plan row")

    clean = row_root / "clean"
    validation_path = clean / "validation_summary.json"
    subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "run_manifest_pair_validations.py"),
            "--manifest",
            str(source_manifest),
            "--output",
            str(clean),
            "--gpu",
            str(args.gpu),
            "--port",
            str(args.port),
            "--noise-seed",
            str(args.noise_seed),
            "--save-sim-states",
        ],
        check=True,
    )
    validation = json.loads(validation_path.read_text())
    job = validation["jobs"][0]
    base_result = {
        "plan_index": index,
        "screen_id": row["screen_id"],
        "cluster_id": row["cluster_id"],
        "suite": row["suite"],
        "source_pair_id": entry["pair_id"],
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": file_digest(source_manifest),
        "clean_exact_dual_success_target_first": bool(
            job["exact_dual_success_target_first"]
        ),
    }
    if not job["exact_dual_success_target_first"]:
        return {
            **base_result,
            "status": "clean_endpoint_ineligible",
            "event_gate_directions": 0,
            "eligible_directions": 0,
            "eligible_pair_ids": [],
            "action_only_predictions": [],
        }

    candidate = row_root / "candidate"
    candidate_summary = candidate / "generation_summary.json"
    if not candidate_summary.is_file():
        subprocess.run(
            [
                sys.executable,
                str(repo / "scripts" / "generate_retarget_aligned_candidates.py"),
                "--source-manifest",
                str(source_manifest),
                "--rollout-root",
                str(clean),
                "--output",
                str(candidate),
                "--gpu",
                str(args.gpu),
                "--replan-steps",
                "5",
            ],
            check=True,
        )
    generation = json.loads(candidate_summary.read_text())
    if int(generation["generated_source_pairs"]) != 1:
        raise ValueError("clean-eligible catalog row produced no aligned candidate")

    gate = row_root / "gate"
    gate_summary_path = gate / "summary.json"
    if not gate_summary_path.is_file():
        subprocess.run(
            [
                sys.executable,
                str(repo / "scripts" / "run_retarget_eligibility_screen.py"),
                "--candidate-root",
                str(candidate),
                "--output",
                str(gate),
                "--gpu",
                str(args.gpu),
                "--port",
                str(args.port),
                "--noise-seed",
                str(args.noise_seed),
                "--execution-horizon",
                "5",
            ],
            check=True,
        )
    gate_summary = json.loads(gate_summary_path.read_text())
    eligible_rows = [row for row in gate_summary["rows"] if row["eligible"]]
    candidate_manifests = _candidate_manifests(candidate)
    action_only_predictions = []
    prediction_script = repo / "scripts" / "sample_retarget_prediction.py"
    for eligible_row in eligible_rows:
        pair_id = str(eligible_row["pair_id"])
        new_side = str(eligible_row["new_side"])
        if pair_id not in candidate_manifests:
            raise ValueError("eligible catalog direction lacks its candidate manifest")
        prediction_output = row_root / "predictions" / pair_id / new_side
        prediction_path = prediction_output / "prediction.json"
        actions_path = prediction_output / "actions.npz"
        if not prediction_path.is_file():
            subprocess.run(
                [
                    sys.executable,
                    str(prediction_script),
                    "--manifest",
                    str(candidate_manifests[pair_id]),
                    "--pair-id",
                    pair_id,
                    "--new-side",
                    new_side,
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
        if (
            prediction.get("pair_id") != pair_id
            or prediction.get("new_side") != new_side
            or not actions_path.is_file()
        ):
            raise ValueError("catalog action-only prediction has the wrong identity")
        audited = audit_prediction_artifacts(
            prediction_path,
            actions_path,
            candidate_manifests[pair_id],
            pair_id,
            new_side,
        )
        action_only_predictions.append(
            {
                "pair_id": pair_id,
                "new_side": new_side,
                "prediction": str(prediction_path),
                "prediction_sha256": file_digest(prediction_path),
                "actions": str(actions_path),
                "actions_sha256": file_digest(actions_path),
                "manifest": str(candidate_manifests[pair_id]),
                "manifest_sha256": file_digest(candidate_manifests[pair_id]),
                **audited,
            }
        )
    return {
        **base_result,
        "status": "endpoint_screened",
        "gate_summary": str(gate_summary_path),
        "gate_summary_sha256": file_digest(gate_summary_path),
        "event_gate_directions": int(gate_summary["event_gate_directions"]),
        "eligible_directions": int(gate_summary["eligible_directions"]),
        "eligible_pair_ids": list(gate_summary["eligible_pair_ids"]),
        "action_only_predictions": action_only_predictions,
    }


def _validate_row_result(result: dict[str, Any], index: int, row: dict[str, Any]) -> None:
    required = {
        "plan_index": index,
        "screen_id": row["screen_id"],
        "cluster_id": row["cluster_id"],
        "suite": row["suite"],
    }
    mismatched = {
        key: {"expected": value, "actual": result.get(key)}
        for key, value in required.items()
        if result.get(key) != value
    }
    if mismatched:
        raise ValueError(f"catalog row result differs from its frozen plan: {mismatched}")
    source = Path(str(result.get("source_manifest", "")))
    if not source.is_file() or result.get("source_manifest_sha256") != file_digest(source):
        raise ValueError("catalog row source manifest changed after screening")
    eligible = int(result.get("eligible_directions", -1))
    pair_ids = result.get("eligible_pair_ids")
    if eligible < 0 or not isinstance(pair_ids, list) or len(pair_ids) != eligible:
        raise ValueError("catalog row has inconsistent eligible-direction accounting")
    predictions = result.get("action_only_predictions")
    if not isinstance(predictions, list) or len(predictions) != eligible:
        raise ValueError("catalog row has inconsistent action-only predictions")
    if [str(item.get("pair_id")) for item in predictions] != [
        str(pair_id) for pair_id in pair_ids
    ]:
        raise ValueError("catalog prediction order differs from eligible gate order")
    for prediction in predictions:
        for path_key, digest_key in (
            ("prediction", "prediction_sha256"),
            ("actions", "actions_sha256"),
            ("manifest", "manifest_sha256"),
        ):
            path = Path(str(prediction.get(path_key, "")))
            if not path.is_file() or prediction.get(digest_key) != file_digest(path):
                raise ValueError("catalog action-only prediction changed after screening")
        raw_prediction = json.loads(Path(prediction["prediction"]).read_text())
        if (
            raw_prediction.get("pair_id") != prediction.get("pair_id")
            or raw_prediction.get("new_side") != prediction.get("new_side")
            or raw_prediction.get("valid") is not prediction.get("valid")
            or raw_prediction.get("predicted_last_successful_boundary")
            != prediction.get("predicted_last_successful_boundary")
        ):
            raise ValueError("catalog prediction metadata differs from its frozen artifact")
        audited = audit_prediction_artifacts(
            Path(prediction["prediction"]),
            Path(prediction["actions"]),
            Path(prediction["manifest"]),
            str(prediction["pair_id"]),
            str(prediction["new_side"]),
        )
        if audited != {
            "valid": prediction["valid"],
            "predicted_last_successful_boundary": prediction[
                "predicted_last_successful_boundary"
            ],
        }:
            raise ValueError("catalog prediction audit differs from frozen metadata")
    if eligible:
        gate = Path(str(result.get("gate_summary", "")))
        if not gate.is_file() or result.get("gate_summary_sha256") != file_digest(gate):
            raise ValueError("catalog row endpoint gate changed after screening")


def _catalog_job(result: dict[str, Any], result_path: Path) -> dict[str, Any]:
    return {
        **result,
        "row_result": str(result_path),
        "row_result_sha256": file_digest(result_path),
    }


def _candidate_manifests(root: Path) -> dict[str, Path]:
    result = {}
    for path in sorted(root.glob("**/aligned/manifest.json")):
        manifest = json.loads(path.read_text())
        for entry in manifest["pairs"]:
            pair_id = str(entry["pair_id"])
            if pair_id in result:
                raise ValueError(f"duplicate candidate pair id: {pair_id}")
            result[pair_id] = path
    return result


def _write_immutable_json(path: Path, value: dict[str, Any]) -> None:
    serialized = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_text() != serialized:
        raise ValueError(f"existing catalog artifact differs from current result: {path}")
    path.write_text(serialized)


def _write_summary(
    output: Path,
    plan_path: Path,
    workers: list[WorkerConfig],
    code_commit: str,
    jobs: list[dict[str, Any]],
    progress: dict[str, Any],
    speculative: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "selection_uses_continuation_outcomes": False,
        "selection_uses_action_only_prediction_validity": True,
        "plan": str(plan_path),
        "plan_sha256": file_digest(plan_path),
        "code_commit": code_commit,
        "noise_seed": 0,
        "parallel_workers": [
            {"gpu": worker.gpu, "port": worker.port} for worker in workers
        ],
        "maximum_parallel_rows": len(workers),
        "selection_uses_only_contiguous_completed_prefix": True,
        "speculative_endpoint_rows_excluded_from_selection": speculative,
        **progress,
        "jobs": jobs,
    }
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"screened {progress['processed_rows']}/{progress['planned_rows']} rows: "
        f"{progress['eligible_clusters']}/{progress['minimum_eligible_clusters']} "
        "eligible clusters; "
        f"{progress['valid_prediction_clusters']}/"
        f"{progress['minimum_valid_prediction_clusters']} valid predictions",
        flush=True,
    )
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
