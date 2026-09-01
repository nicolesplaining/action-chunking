#!/usr/bin/env python3
"""Freeze a catalog-screen handoff before any continuation rollout is run."""

from __future__ import annotations

import argparse
import json
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
    if not (catalog.get("stop_threshold_reached") or catalog.get("catalog_exhausted")):
        raise ValueError("catalog handoff requires the frozen stop rule or catalog exhaustion")

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
        for row in eligible:
            validate_eligible_retarget_row(row)
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
            rows.append(enriched)
            path = candidates[pair_id]
            existing = manifest_by_pair.get(pair_id)
            if existing is not None and existing != str(path):
                raise ValueError(f"duplicate catalog pair id has different manifests: {pair_id}")
            manifest_by_pair[pair_id] = str(path)
            manifest_sha256_by_pair[pair_id] = file_digest(path)

    eligible_clusters = len({row["cluster_id"] for row in rows})
    minimum = int(catalog["minimum_eligible_clusters"])
    gate = {
        "schema_version": 1,
        "selection_uses_continuation_outcomes": False,
        "source_catalog_summary": str(catalog_path),
        "source_catalog_summary_sha256": file_digest(catalog_path),
        "catalog_stop_threshold_reached": bool(catalog["stop_threshold_reached"]),
        "catalog_exhausted": bool(catalog["catalog_exhausted"]),
        "minimum_eligible_clusters": minimum,
        "confirmatory_population_complete": eligible_clusters >= minimum,
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
