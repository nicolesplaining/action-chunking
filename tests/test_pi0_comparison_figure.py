from __future__ import annotations

import csv
import hashlib
import json
import runpy
from pathlib import Path

import pytest

_module = runpy.run_path("scripts/make_pi0_comparison_figure.py")
load = _module["load_audited_comparison"]
make = _module["make_matched_control_figure"]
OUTPUT_FILENAMES = _module["COMPARISON_OUTPUT_FILENAMES"]


def test_matched_control_figure_requires_final_audit_and_exact_grids(
    tmp_path: Path,
) -> None:
    comparison, audit = _inputs(tmp_path)

    data = load(comparison, audit)
    manifest = make(comparison, audit, tmp_path / "figure")

    assert len(data["residual"]) == 180
    assert manifest["primary_metric"] == "all"
    assert (tmp_path / "figure" / "fig_pi0_matched_control.pdf").is_file()
    assert (tmp_path / "figure" / "fig_pi0_matched_control.png").is_file()


def test_matched_control_figure_rejects_wrong_final_audit(tmp_path: Path) -> None:
    comparison, audit = _inputs(tmp_path)
    value = json.loads(audit.read_text())
    value["comparison_summary_sha256"] = "0" * 64
    audit.write_text(json.dumps(value))

    with pytest.raises(ValueError, match="final audit is incompatible"):
        load(comparison, audit)


def test_matched_control_figure_rejects_changed_comparison_csv(tmp_path: Path) -> None:
    comparison, audit = _inputs(tmp_path)
    (comparison / "paired_residual_cells.csv").write_text("tampered\n")

    with pytest.raises(ValueError, match="output changed after analysis"):
        load(comparison, audit)


def _inputs(root: Path) -> tuple[Path, Path]:
    comparison = root / "comparison"
    comparison.mkdir()
    timing_record = {
        "eligible_state_clusters": 12,
        "mean_difference": 0.5,
        "ci95_low": 0.1,
        "ci95_high": 0.9,
        "p_two_sided_sign_flip": 0.1,
    }
    source_files = {}
    for index in range(14):
        path = root / "sources" / f"source-{index}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps({"index": index}))
        source_files[f"source_{index}"] = {
            "path": str(path),
            "sha256": _digest(path),
        }
    _write_csv(
        comparison / "paired_residual_cells.csv",
        [
            _cell(flow_step=step, layer=layer)
            for step in range(10)
            for layer in range(18)
        ],
    )
    _write_csv(
        comparison / "paired_dimension_cells.csv",
        [
            _cell(
                flow_step=step,
                patched_tensor=tensor,
                patched_dimension_group=group,
            )
            for step in range(10)
            for tensor in ("x_t", "v_t")
            for group in ("translation", "rotation", "gripper")
        ],
    )
    _write_csv(
        comparison / "paired_position_normalized_cells.csv",
        [
            _cell(flow_step=step, layer=layer, normalized_position_bin=position)
            for step in (0, 7, 8, 9)
            for layer in (0, 8, 14, 17)
            for position in range(10)
        ],
    )
    for name in OUTPUT_FILENAMES:
        path = comparison / name
        if not path.exists():
            path.write_text(f"artifact,{name}\n")
    summary = {
        "schema_version": 1,
        "analysis_unit": "paired_scene_state",
        "comparison": "pi05_minus_pi0",
        "pi05_action_horizon": 10,
        "pi0_action_horizon": 50,
        "primary_position_window": list(range(10)),
        "normalized_position_bins": 10,
        "bootstrap_replicates": 10_000,
        "timing": {
            "all": {
                "formation_step_difference_pi05_minus_pi0": timing_record,
                "editability_boundary_difference_pi05_minus_pi0": timing_record,
            }
        },
        "source_files": source_files,
        "output_files": {name: _digest(comparison / name) for name in OUTPUT_FILENAMES},
    }
    summary_path = comparison / "summary.json"
    summary_path.write_text(json.dumps(summary))
    audit = root / "final_audit.json"
    audit.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "passed": True,
                "comparison_summary_sha256": _digest(summary_path),
                "comparison_source_files": 14,
                "comparison_output_files": 10,
                "intervention_gpus": 2,
            }
        )
    )
    return comparison, audit


def _cell(**fields: object) -> dict[str, object]:
    return {
        "metric": "all",
        **fields,
        "eligible_state_clusters": 12,
        "mean_difference_pi05_minus_pi0": 0.01,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
