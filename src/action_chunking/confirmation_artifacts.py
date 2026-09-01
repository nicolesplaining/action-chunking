"""Promote an audited sealed confirmation into immutable paper artifacts."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from action_chunking.pairs import file_digest


def promote_confirmation_artifacts(
    audit_root: Path,
    figure_root: Path,
    output: Path,
) -> dict[str, Any]:
    """Copy only a mutually consistent audit/figure bundle into the paper tree."""
    if output.exists():
        raise FileExistsError(f"confirmation paper artifact output already exists: {output}")
    original_path = audit_root / "original" / "summary.json"
    hardened_path = audit_root / "hardened" / "summary.json"
    comparison_path = audit_root / "comparison.json"
    analysis_commit_path = audit_root / "analysis_code_commit.txt"
    figure_manifest_path = figure_root / "figure_manifest.json"
    source_paths = {
        "original_summary.json": original_path,
        "hardened_summary.json": hardened_path,
        "audit_comparison.json": comparison_path,
        "analysis_code_commit.txt": analysis_commit_path,
        "figure_manifest.json": figure_manifest_path,
        "fig_early_exit_confirmation.pdf": figure_root / "fig_early_exit_confirmation.pdf",
        "fig_early_exit_confirmation.png": figure_root / "fig_early_exit_confirmation.png",
    }
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"confirmation promotion inputs are missing: {missing}")

    original = _read_json(original_path)
    hardened = _read_json(hardened_path)
    comparison = _read_json(comparison_path)
    figure = _read_json(figure_manifest_path)
    if comparison.get("registered_fields_exact") is not True:
        raise ValueError("confirmation promotion requires exact original/hardened agreement")
    required_comparison = {
        "schema_version": 1,
        "episode_pairs": 500,
        "paired_losses": hardened.get("paired_losses"),
        "confirmation_positive": hardened.get("confirmation_positive"),
        "source_artifact_files": 1505,
        "source_artifact_manifest_sha256": hardened.get(
            "source_artifact_manifest_sha256"
        ),
        "original_sha256": file_digest(original_path),
        "hardened_sha256": file_digest(hardened_path),
    }
    _require_fields(comparison, required_comparison, "confirmation audit comparison")
    if original.get("rows") != hardened.get("rows"):
        raise ValueError("confirmation original and hardened row populations differ")

    required_figure = {
        "schema_version": 1,
        "source_summary_sha256": file_digest(hardened_path),
        "source_artifact_files": 1505,
        "source_artifact_manifest_sha256": hardened.get(
            "source_artifact_manifest_sha256"
        ),
        "confirmation_positive": hardened.get("confirmation_positive"),
        "episode_pairs": 500,
        "early_exit_successes": hardened.get("early_exit_successes"),
        "full_control_successes": hardened.get("full_control_successes"),
        "paired_losses": hardened.get("paired_losses"),
        "paired_gains": hardened.get("paired_gains"),
        "median_latency_savings_fraction": hardened.get(
            "median_first_replan_latency_savings_fraction"
        ),
        "median_latency_savings_ci95": hardened.get(
            "median_first_replan_latency_savings_fraction_bootstrap_ci95"
        ),
        "outputs": [
            "fig_early_exit_confirmation.pdf",
            "fig_early_exit_confirmation.png",
        ],
    }
    _require_fields(figure, required_figure, "confirmation figure manifest")
    for name in figure["outputs"]:
        path = figure_root / name
        if figure.get("output_sha256", {}).get(name) != file_digest(path):
            raise ValueError(f"confirmation figure changed after manifest creation: {name}")

    analysis_commit = analysis_commit_path.read_text().strip()
    if re.fullmatch(r"[0-9a-f]{40}", analysis_commit) is None:
        raise ValueError("confirmation audit analysis commit is invalid")
    run_commit = str(hardened.get("code_commit", ""))
    if re.fullmatch(r"[0-9a-f]{40}", run_commit) is None:
        raise ValueError("confirmation run commit is invalid")

    output.mkdir(parents=True)
    for name, source in source_paths.items():
        shutil.copy2(source, output / name)
    promoted_hashes = {
        name: file_digest(output / name) for name in sorted(source_paths)
    }
    manifest = {
        "schema_version": 1,
        "artifact": "pi05_early_exit_confirmation_seed0",
        "confirmation_positive": hardened["confirmation_positive"],
        "episode_pairs": 500,
        "condition_rollouts": 1000,
        "early_exit_successes": hardened["early_exit_successes"],
        "full_control_successes": hardened["full_control_successes"],
        "paired_losses": hardened["paired_losses"],
        "paired_gains": hardened["paired_gains"],
        "paired_loss_upper95": hardened["paired_loss_clopper_pearson_upper95"],
        "median_latency_savings_fraction": hardened[
            "median_first_replan_latency_savings_fraction"
        ],
        "median_latency_savings_ci95": hardened[
            "median_first_replan_latency_savings_fraction_bootstrap_ci95"
        ],
        "velocity_evaluation_savings_fraction": hardened[
            "velocity_evaluation_savings_fraction"
        ],
        "run_code_commit": run_commit,
        "audit_analysis_code_commit": analysis_commit,
        "source_artifact_files": 1505,
        "source_artifact_manifest_sha256": hardened[
            "source_artifact_manifest_sha256"
        ],
        "files": promoted_hashes,
    }
    (output / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _require_fields(value: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    mismatched = {
        key: {"expected": wanted, "actual": value.get(key)}
        for key, wanted in expected.items()
        if value.get(key) != wanted
    }
    if mismatched:
        raise ValueError(f"{label} is incompatible: {mismatched}")
