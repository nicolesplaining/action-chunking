#!/usr/bin/env python3
"""Require the original and hardened confirmation analyses to agree exactly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from action_chunking.pairs import file_digest

REGISTERED_FIELDS = (
    "schema_version",
    "analysis_unit",
    "suite",
    "code_commit",
    "episode_pairs",
    "episodes_per_condition",
    "condition_rollouts",
    "condition_order_counts",
    "per_task",
    "early_exit_successes",
    "early_exit_success_rate",
    "early_exit_success_wilson_ci95",
    "full_control_successes",
    "full_control_success_rate",
    "full_control_success_wilson_ci95",
    "paired_losses",
    "paired_gains",
    "paired_loss_rate",
    "paired_loss_clopper_pearson_upper95",
    "paired_loss_margin",
    "maximum_passing_losses",
    "all_compute_counts_exact",
    "velocity_evaluation_savings_fraction",
    "median_first_replan_latency_savings_fraction",
    "median_first_replan_latency_savings_fraction_bootstrap_ci95",
    "confirmation_positive",
    "confirmation_root",
    "progress_sha256",
    "warmup_sessions",
    "warmup_sessions_sha256",
    "rows",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--hardened", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def compare_confirmation_analyses(
    original: dict[str, Any], hardened: dict[str, Any]
) -> dict[str, Any]:
    missing = {
        label: [field for field in REGISTERED_FIELDS if field not in value]
        for label, value in (("original", original), ("hardened", hardened))
    }
    if any(missing.values()):
        raise ValueError(f"confirmation analysis lacks registered fields: {missing}")
    mismatched = [field for field in REGISTERED_FIELDS if original[field] != hardened[field]]
    if mismatched:
        raise ValueError(f"original and hardened confirmation analyses differ: {mismatched}")
    hardened_only = sorted(set(hardened) - set(original))
    if hardened_only != ["source_artifact_files", "source_artifact_manifest_sha256"]:
        raise ValueError(f"unexpected hardened-only confirmation fields: {hardened_only}")
    if int(hardened["source_artifact_files"]) != 1505:
        raise ValueError("hardened confirmation has the wrong source artifact count")
    return {
        "schema_version": 1,
        "registered_fields_exact": True,
        "registered_fields": list(REGISTERED_FIELDS),
        "hardened_only_fields": hardened_only,
        "episode_pairs": int(hardened["episode_pairs"]),
        "paired_losses": int(hardened["paired_losses"]),
        "confirmation_positive": bool(hardened["confirmation_positive"]),
        "source_artifact_files": int(hardened["source_artifact_files"]),
        "source_artifact_manifest_sha256": hardened[
            "source_artifact_manifest_sha256"
        ],
    }


def main() -> int:
    args = parse_args()
    original = json.loads(args.original.read_text())
    hardened = json.loads(args.hardened.read_text())
    result = compare_confirmation_analyses(original, hardened)
    result.update(
        {
            "original": str(args.original),
            "original_sha256": file_digest(args.original),
            "hardened": str(args.hardened),
            "hardened_sha256": file_digest(args.hardened),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
