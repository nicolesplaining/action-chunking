from __future__ import annotations

import json
from pathlib import Path

import pytest

from action_chunking.confirmation_artifacts import promote_confirmation_artifacts
from action_chunking.pairs import file_digest


def test_promotes_only_a_mutually_consistent_confirmation_bundle(tmp_path: Path) -> None:
    audit, figures = _bundle(tmp_path)
    output = tmp_path / "paper-result"

    manifest = promote_confirmation_artifacts(audit, figures, output)

    assert manifest["confirmation_positive"] is True
    assert manifest["paired_losses"] == 4
    assert manifest["source_artifact_files"] == 1505
    assert set(manifest["files"]) == {
        "analysis_code_commit.txt",
        "audit_comparison.json",
        "fig_early_exit_confirmation.pdf",
        "fig_early_exit_confirmation.png",
        "figure_manifest.json",
        "hardened_summary.json",
        "original_summary.json",
    }
    assert (output / "artifact_manifest.json").is_file()


def test_promotion_rejects_figure_tampering(tmp_path: Path) -> None:
    audit, figures = _bundle(tmp_path)
    (figures / "fig_early_exit_confirmation.png").write_bytes(b"changed")

    with pytest.raises(ValueError, match="figure changed"):
        promote_confirmation_artifacts(audit, figures, tmp_path / "paper-result")


def test_promotion_rejects_figure_with_different_raw_source_digest(
    tmp_path: Path,
) -> None:
    audit, figures = _bundle(tmp_path)
    manifest_path = figures / "figure_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_artifact_manifest_sha256"] = "c" * 64
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="figure manifest is incompatible"):
        promote_confirmation_artifacts(audit, figures, tmp_path / "paper-result")


def _bundle(tmp_path: Path) -> tuple[Path, Path]:
    audit = tmp_path / "audit"
    figures = tmp_path / "figures"
    original_path = audit / "original" / "summary.json"
    hardened_path = audit / "hardened" / "summary.json"
    comparison_path = audit / "comparison.json"
    commit_path = audit / "analysis_code_commit.txt"
    figure_manifest_path = figures / "figure_manifest.json"
    summary = {
        "code_commit": "a" * 40,
        "confirmation_positive": True,
        "early_exit_successes": 496,
        "full_control_successes": 500,
        "paired_losses": 4,
        "paired_gains": 0,
        "paired_loss_clopper_pearson_upper95": 0.018,
        "median_first_replan_latency_savings_fraction": 0.3,
        "median_first_replan_latency_savings_fraction_bootstrap_ci95": [0.29, 0.31],
        "velocity_evaluation_savings_fraction": 0.3,
        "rows": [{"pair_key": "task_00_trial_00"}],
    }
    _write_json(original_path, summary)
    hardened = {
        **summary,
        "source_artifact_files": 1505,
        "source_artifact_manifest_sha256": "b" * 64,
    }
    _write_json(hardened_path, hardened)
    _write_json(
        comparison_path,
        {
            "schema_version": 1,
            "registered_fields_exact": True,
            "episode_pairs": 500,
            "paired_losses": 4,
            "confirmation_positive": True,
            "source_artifact_files": 1505,
            "source_artifact_manifest_sha256": "b" * 64,
            "original_sha256": file_digest(original_path),
            "hardened_sha256": file_digest(hardened_path),
        },
    )
    commit_path.parent.mkdir(parents=True, exist_ok=True)
    commit_path.write_text("c" * 40 + "\n")
    figures.mkdir(parents=True, exist_ok=True)
    (figures / "fig_early_exit_confirmation.pdf").write_bytes(b"pdf")
    (figures / "fig_early_exit_confirmation.png").write_bytes(b"png")
    _write_json(
        figure_manifest_path,
        {
            "schema_version": 1,
            "source_summary_sha256": file_digest(hardened_path),
            "source_artifact_files": 1505,
            "source_artifact_manifest_sha256": "b" * 64,
            "confirmation_positive": True,
            "episode_pairs": 500,
            "early_exit_successes": 496,
            "full_control_successes": 500,
            "paired_losses": 4,
            "paired_gains": 0,
            "median_latency_savings_fraction": 0.3,
            "median_latency_savings_ci95": [0.29, 0.31],
            "outputs": [
                "fig_early_exit_confirmation.pdf",
                "fig_early_exit_confirmation.png",
            ],
            "output_sha256": {
                "fig_early_exit_confirmation.pdf": file_digest(
                    figures / "fig_early_exit_confirmation.pdf"
                ),
                "fig_early_exit_confirmation.png": file_digest(
                    figures / "fig_early_exit_confirmation.png"
                ),
            },
        },
    )
    return audit, figures


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
