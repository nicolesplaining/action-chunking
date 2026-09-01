"""Outcome-blind public-catalog ordering for retargeting screens."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from action_chunking.pairs import file_digest


def build_retarget_screening_plan(
    catalog_paths: list[Path],
    exclusions_path: Path,
    *,
    initial_states_per_pair: int = 50,
    minimum_eligible_clusters: int = 59,
) -> dict[str, Any]:
    """Build a deterministic target-pair screen without intervention outcomes."""
    if not catalog_paths or initial_states_per_pair <= 0 or minimum_eligible_clusters <= 0:
        raise ValueError("catalogs, initial-state count, and eligible-cluster minimum are required")
    exclusion_payload = json.loads(exclusions_path.read_text())
    exclusions = exclusion_payload.get("exclusions", [])
    pairs = []
    sources = []
    for path in catalog_paths:
        catalog = json.loads(path.read_text())
        suite = str(catalog["suite"])
        sources.append(
            {
                "path": str(path),
                "sha256": file_digest(path),
                "suite": suite,
                "schema_version": int(catalog["schema_version"]),
            }
        )
        for pair in catalog["pairs"]:
            if pair["semantic_role"] != "manipulated_object":
                continue
            pairs.append({**pair, "suite": suite})
    pairs.sort(
        key=lambda row: (
            row["canonical_scene_sha256"],
            row["suite"],
            int(row["base_task_id"]),
            int(row["donor_task_id"]),
        )
    )

    rows = []
    excluded = []
    for pair in pairs:
        for init_index in range(initial_states_per_pair):
            exclusion = _matching_exclusion(pair, init_index, exclusions)
            row = {
                "suite": pair["suite"],
                "canonical_scene_sha256": pair["canonical_scene_sha256"],
                "base_task_id": int(pair["base_task_id"]),
                "donor_task_id": int(pair["donor_task_id"]),
                "base_task": pair["base_task"],
                "donor_task": pair["donor_task"],
                "base_target": pair["base_target"],
                "donor_target": pair["donor_target"],
                "init_index": init_index,
                "cluster_id": (
                    f"{pair['suite']}:{pair['canonical_scene_sha256']}:{init_index:03d}"
                ),
            }
            row["screen_id"] = hashlib.sha256(
                json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            if exclusion is None:
                rows.append(row)
            else:
                excluded.append({**row, "reason": exclusion["reason"]})
    return {
        "schema_version": 1,
        "selection_uses_intervention_outcomes": False,
        "ordering": [
            "canonical_scene_sha256",
            "suite",
            "base_task_id",
            "donor_task_id",
            "init_index",
        ],
        "cluster_unit": "suite_x_canonical_scene_x_init_index",
        "stop_rule": {
            "minimum_eligible_clusters": minimum_eligible_clusters,
            "fallback": "public_catalog_exhaustion",
        },
        "initial_states_per_pair": initial_states_per_pair,
        "source_catalogs": sources,
        "exclusions_source": {
            "path": str(exclusions_path),
            "sha256": file_digest(exclusions_path),
        },
        "target_pair_definitions": len(pairs),
        "candidate_rows": len(rows),
        "excluded_rows": len(excluded),
        "unique_candidate_clusters": len({row["cluster_id"] for row in rows}),
        "rows": rows,
        "excluded": excluded,
    }


def _matching_exclusion(
    pair: dict[str, Any], init_index: int, exclusions: list[dict[str, Any]]
) -> dict[str, Any] | None:
    for exclusion in exclusions:
        if (
            pair["suite"] == exclusion["suite"]
            and pair["canonical_scene_sha256"]
            == exclusion["canonical_scene_sha256"]
            and int(pair["base_task_id"]) == int(exclusion["base_task_id"])
            and int(pair["donor_task_id"]) == int(exclusion["donor_task_id"])
            and int(exclusion["init_index_start"])
            <= init_index
            < int(exclusion["init_index_stop"])
        ):
            return exclusion
    return None
