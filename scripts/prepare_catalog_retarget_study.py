#!/usr/bin/env python3
"""Freeze a catalog-screen handoff before any continuation rollout is run."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from action_chunking.pairs import file_digest
from action_chunking.utility_prediction import validate_eligible_retarget_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    catalog = json.loads(args.catalog_summary.read_text())
    gate, candidate_index = prepare_catalog_handoff(catalog, args.catalog_summary)
    args.output.mkdir(parents=True, exist_ok=True)
    _write_immutable(args.output / "gate_summary.json", gate)
    _write_immutable(args.output / "candidate_index.json", candidate_index)
    print(
        f"froze {gate['eligible_directions']} endpoint-eligible directions across "
        f"{gate['eligible_clusters']} independent clusters",
        flush=True,
    )
    return 0


def prepare_catalog_handoff(
    catalog: dict[str, Any], catalog_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    if catalog.get("selection_uses_continuation_outcomes") is not False:
        raise ValueError("catalog screen must explicitly exclude continuation outcomes")
    minimum = _validate_catalog_accounting(catalog)

    rows = []
    manifest_by_pair: dict[str, str] = {}
    manifest_sha256_by_pair: dict[str, str] = {}
    source_gates = []
    for job in catalog["jobs"]:
        if int(job.get("eligible_directions", 0)) <= 0:
            continue
        gate_path = Path(job["gate_summary"])
        if file_digest(gate_path) != job["gate_summary_sha256"]:
            raise ValueError("catalog gate changed after endpoint screening")
        source_gate = json.loads(gate_path.read_text())
        if source_gate.get("selection_uses_continuation_outcomes") is not False:
            raise ValueError("source gate must explicitly exclude continuation outcomes")
        eligible = [row for row in source_gate["rows"] if row["eligible"]]
        if len(eligible) != int(job["eligible_directions"]):
            raise ValueError("catalog job and source gate eligible counts differ")
        if {str(row["pair_id"]) for row in eligible} != {
            str(pair_id) for pair_id in job["eligible_pair_ids"]
        }:
            raise ValueError("catalog job and source gate eligible pair ids differ")
        predictions = job.get("action_only_predictions")
        if not isinstance(predictions, list) or len(predictions) != len(eligible):
            raise ValueError("catalog job lacks one action-only prediction per direction")
        predictions_by_key = {
            (str(item["pair_id"]), str(item["new_side"])): item
            for item in predictions
        }
        if len(predictions_by_key) != len(predictions):
            raise ValueError("catalog job contains duplicate action-only predictions")
        for row in eligible:
            validate_eligible_retarget_row(row)
            if int(row.get("noise_seed", -1)) != 0 or int(row.get("execution_horizon", -1)) != 5:
                raise ValueError("catalog eligibility must use frozen seed zero and horizon five")
            if row.get("source_pair_id") != job.get("source_pair_id"):
                raise ValueError("catalog eligibility row has the wrong source-pair lineage")
        source_gates.append(
            {
                "plan_index": int(job["plan_index"]),
                "screen_id": job["screen_id"],
                "cluster_id": job["cluster_id"],
                "path": str(gate_path),
                "sha256": job["gate_summary_sha256"],
            }
        )
        candidate_root = gate_path.parent.parent / "candidate"
        candidates = _candidate_manifests(candidate_root)
        for row in eligible:
            pair_id = row["pair_id"]
            if pair_id not in candidates:
                raise ValueError(f"eligible pair is absent from catalog candidate root: {pair_id}")
            enriched = {
                **row,
                "cluster_id": job["cluster_id"],
                "catalog_plan_index": int(job["plan_index"]),
                "catalog_screen_id": job["screen_id"],
            }
            prediction_key = (str(pair_id), str(row["new_side"]))
            if prediction_key not in predictions_by_key:
                raise ValueError("eligible direction lacks its frozen action-only prediction")
            prediction = predictions_by_key[prediction_key]
            for path_field, digest_field in (
                ("prediction", "prediction_sha256"),
                ("actions", "actions_sha256"),
                ("manifest", "manifest_sha256"),
            ):
                prediction_path = Path(str(prediction[path_field]))
                if (
                    not prediction_path.is_file()
                    or prediction[digest_field] != file_digest(prediction_path)
                ):
                    raise ValueError("catalog action-only prediction changed before handoff")
            raw_prediction = json.loads(Path(prediction["prediction"]).read_text())
            if (
                raw_prediction.get("pair_id") != prediction.get("pair_id")
                or raw_prediction.get("new_side") != prediction.get("new_side")
                or raw_prediction.get("valid") is not prediction.get("valid")
                or raw_prediction.get("predicted_last_successful_boundary")
                != prediction.get("predicted_last_successful_boundary")
            ):
                raise ValueError("catalog prediction metadata differs from its artifact")
            enriched["action_only_prediction"] = prediction
            rows.append(enriched)
            path = candidates[pair_id]
            if (
                Path(str(prediction["manifest"])) != path
                or prediction["manifest_sha256"] != file_digest(path)
            ):
                raise ValueError("catalog prediction uses the wrong candidate manifest")
            existing = manifest_by_pair.get(pair_id)
            if existing is not None and existing != str(path):
                raise ValueError(f"duplicate catalog pair id has different manifests: {pair_id}")
            manifest_by_pair[pair_id] = str(path)
            manifest_sha256_by_pair[pair_id] = file_digest(path)

    eligible_clusters = len({row["cluster_id"] for row in rows})
    row_keys = [
        (str(row["cluster_id"]), str(row["pair_id"]), str(row["new_side"]))
        for row in rows
    ]
    if len(row_keys) != len(set(row_keys)):
        raise ValueError("catalog handoff contains duplicate eligible directions")
    gate = {
        "schema_version": 1,
        "selection_uses_continuation_outcomes": False,
        "source_catalog_summary": str(catalog_path),
        "source_catalog_summary_sha256": file_digest(catalog_path),
        "catalog_stop_threshold_reached": bool(catalog["stop_threshold_reached"]),
        "catalog_exhausted": bool(catalog["catalog_exhausted"]),
        "minimum_eligible_clusters": minimum,
        "minimum_valid_prediction_clusters": int(
            catalog["minimum_valid_prediction_clusters"]
        ),
        "valid_prediction_clusters": int(catalog["valid_prediction_clusters"]),
        "confirmatory_population_complete": bool(catalog["stop_threshold_reached"]),
        "eligible_clusters": eligible_clusters,
        "eligible_directions": len(rows),
        "source_gates": source_gates,
        "rows": rows,
    }
    candidate_index = {
        "schema_version": 1,
        "selection_uses_continuation_outcomes": False,
        "source_catalog_summary": str(catalog_path),
        "source_catalog_summary_sha256": file_digest(catalog_path),
        "manifest_by_pair": manifest_by_pair,
        "manifest_sha256_by_pair": manifest_sha256_by_pair,
    }
    return gate, candidate_index


def _validate_catalog_accounting(catalog: dict[str, Any]) -> int:
    jobs = catalog.get("jobs", [])
    minimum = int(catalog.get("minimum_eligible_clusters", 0))
    minimum_valid = int(catalog.get("minimum_valid_prediction_clusters", 0))
    planned = int(catalog.get("planned_rows", -1))
    processed = int(catalog.get("processed_rows", -1))
    if (
        minimum <= 0
        or minimum_valid <= 0
        or planned < 0
        or processed != len(jobs)
        or processed > planned
    ):
        raise ValueError("catalog summary has inconsistent progress counts")
    plan = Path(str(catalog.get("plan", "")))
    if (
        re.fullmatch(r"[0-9a-f]{40}", str(catalog.get("code_commit"))) is None
        or not plan.is_file()
        or catalog.get("plan_sha256") != file_digest(plan)
    ):
        raise ValueError("catalog summary lacks a valid code and frozen-plan binding")
    frozen_plan = json.loads(plan.read_text())
    plan_rows = frozen_plan.get("rows")
    stop_rule = frozen_plan.get("stop_rule", {})
    if (
        not isinstance(plan_rows, list)
        or planned != len(plan_rows)
        or frozen_plan.get("selection_uses_continuation_outcomes") is not False
        or frozen_plan.get("selection_uses_action_only_prediction_validity") is not True
        or int(stop_rule.get("minimum_eligible_clusters", 0)) != minimum
        or int(stop_rule.get("minimum_valid_prediction_clusters", 0))
        != minimum_valid
    ):
        raise ValueError("catalog summary differs from its frozen screening plan")
    workers = catalog.get("parallel_workers")
    if (
        not isinstance(workers, list)
        or not 1 <= len(workers) <= 2
        or len({int(worker.get("gpu", -1)) for worker in workers}) != len(workers)
        or len({int(worker.get("port", -1)) for worker in workers}) != len(workers)
        or catalog.get("selection_uses_only_contiguous_completed_prefix") is not True
        or catalog.get("selection_uses_action_only_prediction_validity") is not True
    ):
        raise ValueError("catalog summary has invalid isolated-worker provenance")
    for index, job in enumerate(jobs):
        if int(job.get("plan_index", -1)) != index:
            raise ValueError("catalog jobs are not a contiguous frozen prefix")
        expected_row = plan_rows[index]
        if (
            job.get("screen_id") != expected_row.get("screen_id")
            or job.get("cluster_id") != expected_row.get("cluster_id")
        ):
            raise ValueError("catalog job differs from its frozen plan row")
        _validate_row_result_binding(job)
    if len({str(job.get("screen_id")) for job in jobs}) != len(jobs):
        raise ValueError("catalog jobs contain duplicate screen ids")
    eligible_jobs = [job for job in jobs if int(job.get("eligible_directions", 0)) > 0]
    eligible_clusters = {str(job["cluster_id"]) for job in eligible_jobs}
    primary_by_cluster = {}
    for job in eligible_jobs:
        primary_by_cluster.setdefault(
            str(job["cluster_id"]), job["action_only_predictions"][0]
        )
    valid_prediction_clusters = {
        cluster_id
        for cluster_id, prediction in primary_by_cluster.items()
        if prediction.get("valid") is True
    }
    eligible_directions = sum(int(job.get("eligible_directions", 0)) for job in jobs)
    if (
        int(catalog.get("eligible_clusters", -1)) != len(eligible_clusters)
        or int(catalog.get("eligible_directions", -1)) != eligible_directions
        or sorted(str(value) for value in catalog.get("eligible_cluster_ids", []))
        != sorted(eligible_clusters)
        or int(catalog.get("valid_prediction_clusters", -1))
        != len(valid_prediction_clusters)
        or sorted(
            str(value) for value in catalog.get("valid_prediction_cluster_ids", [])
        )
        != sorted(valid_prediction_clusters)
    ):
        raise ValueError("catalog eligibility totals disagree with its ordered jobs")
    stop_reached = (
        len(eligible_clusters) >= minimum
        and len(valid_prediction_clusters) >= minimum_valid
    )
    exhausted = processed == planned
    if bool(catalog.get("stop_threshold_reached")) != stop_reached:
        raise ValueError("catalog stop flag disagrees with recomputed eligible clusters")
    if bool(catalog.get("catalog_exhausted")) != exhausted:
        raise ValueError("catalog exhaustion flag disagrees with progress counts")
    if not (stop_reached or exhausted):
        raise ValueError("catalog handoff requires the frozen stop rule or catalog exhaustion")
    if stop_reached and jobs:
        prior_progress = _catalog_threshold_counts(jobs[:-1])
        if (
            prior_progress[0] >= minimum
            and prior_progress[1] >= minimum_valid
        ):
            raise ValueError("catalog continued after the first stop-threshold crossing")
    speculative = catalog.get("speculative_endpoint_rows_excluded_from_selection")
    if not isinstance(speculative, list) or len(speculative) > len(workers) - 1:
        raise ValueError("catalog has invalid speculative endpoint accounting")
    expected_speculative_indices = list(
        range(processed, processed + len(speculative))
    )
    if [int(job.get("plan_index", -1)) for job in speculative] != expected_speculative_indices:
        raise ValueError("catalog speculative rows do not immediately follow the frozen prefix")
    for job in speculative:
        expected_row = plan_rows[int(job["plan_index"])]
        if (
            job.get("screen_id") != expected_row.get("screen_id")
            or job.get("cluster_id") != expected_row.get("cluster_id")
        ):
            raise ValueError("catalog speculative row differs from the frozen plan")
        _validate_row_result_binding(job)
    if speculative and not stop_reached:
        raise ValueError("catalog retained speculative rows before its stop threshold")
    return minimum


def _catalog_threshold_counts(jobs: list[dict[str, Any]]) -> tuple[int, int]:
    primary_by_cluster = {}
    for job in jobs:
        if int(job.get("eligible_directions", 0)) <= 0:
            continue
        primary_by_cluster.setdefault(
            str(job["cluster_id"]), job["action_only_predictions"][0]
        )
    return len(primary_by_cluster), sum(
        prediction.get("valid") is True for prediction in primary_by_cluster.values()
    )


def _validate_row_result_binding(job: dict[str, Any]) -> None:
    path = Path(str(job.get("row_result", "")))
    if not path.is_file() or job.get("row_result_sha256") != file_digest(path):
        raise ValueError("catalog row result changed after screening")
    raw = json.loads(path.read_text())
    expected = {
        key: value
        for key, value in job.items()
        if key not in {"row_result", "row_result_sha256"}
    }
    if raw != expected:
        raise ValueError("catalog summary row differs from its bound result artifact")
    source = Path(str(raw.get("source_manifest", "")))
    if not source.is_file() or raw.get("source_manifest_sha256") != file_digest(source):
        raise ValueError("catalog source manifest changed after endpoint screening")


def _candidate_manifests(root: Path) -> dict[str, Path]:
    result = {}
    for path in sorted(root.glob("**/aligned/manifest.json")):
        manifest = json.loads(path.read_text())
        for entry in manifest["pairs"]:
            pair_id = entry["pair_id"]
            if pair_id in result:
                raise ValueError(f"duplicate candidate pair id: {pair_id}")
            result[pair_id] = path
    return result


def _write_immutable(path: Path, payload: dict[str, Any]) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.is_file() and path.read_text() != serialized:
        raise ValueError(f"existing frozen handoff differs from current inputs: {path}")
    path.write_text(serialized)


if __name__ == "__main__":
    raise SystemExit(main())
