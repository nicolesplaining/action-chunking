from __future__ import annotations

import json
from pathlib import Path

import pytest

import action_chunking.pi0_intervention as module
from action_chunking.pairs import file_digest


def test_completed_pi0_control_is_reconstructed_and_tamper_evident(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output"
    manifest = tmp_path / "manifest.json"
    parity = tmp_path / "parity.json"
    checkpoint = tmp_path / "checkpoint"
    _write_json(manifest, {"pairs": []})
    _write_json(parity, {})
    checkpoint.mkdir()
    binding = {
        "schema_version": 1,
        "passed": True,
        "parity_summary": str(parity),
        "parity_summary_sha256": file_digest(parity),
        "manifest": str(manifest),
        "manifest_sha256": file_digest(manifest),
        "pytorch_checkpoint": str(checkpoint),
        "pytorch_checkpoint_artifact_sha256": {"model.safetensors": "a" * 64},
        "cases": 32,
        "passed_cases": 32,
    }
    monkeypatch.setattr(module, "validate_pi0_intervention_inputs", lambda *_: binding)
    _write_json(output / "intervention_input_binding.json", binding)
    commit = "b" * 40
    (output / "code_commit.txt").parent.mkdir(parents=True, exist_ok=True)
    (output / "code_commit.txt").write_text(commit + "\n")
    (output / "gpu_preflight.csv").write_text(
        "index, uuid, name, driver_version, memory.total [MiB]\n"
        "0, GPU-a, NVIDIA H100 80GB HBM3, 570.0, 81559 MiB\n"
        "1, GPU-b, NVIDIA H100 80GB HBM3, 570.0, 81559 MiB\n"
    )

    clean = tmp_path / "pi0_clean"
    reference = tmp_path / "pi05_clean"
    for root in (clean, reference):
        for index in range(12):
            _write_json(root / f"pair-{index}" / "noise_0" / "summary.json", {"index": index})
    pairs = [f"pair-{index}" for index in range(12)]
    for mode in ("coarse", "population_positions"):
        _write_grid(output, clean, reference, binding, commit, mode, pairs)

    sources = {}
    for index in range(14):
        path = tmp_path / "sources" / f"source-{index}.json"
        _write_json(path, {"index": index})
        sources[f"source_{index}"] = {"path": str(path), "sha256": file_digest(path)}
    comparison_root = output / "comparison"
    for name in module.COMPARISON_OUTPUT_FILENAMES:
        path = comparison_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"artifact,{name}\n")
    comparison = {
        "schema_version": 1,
        "analysis_unit": "paired_scene_state",
        "comparison": "pi05_minus_pi0",
        "pi05_action_horizon": 10,
        "pi0_action_horizon": 50,
        "primary_position_window": list(range(10)),
        "normalized_position_bins": 10,
        "bootstrap_replicates": 10_000,
        "source_files": sources,
        "output_files": {
            name: file_digest(comparison_root / name)
            for name in module.COMPARISON_OUTPUT_FILENAMES
        },
    }
    _write_json(comparison_root / "summary.json", comparison)

    audit = module.audit_pi0_intervention_output(output, parity, checkpoint, manifest)
    assert audit["passed"] is True
    assert audit["common_scene_pair_count"] == 12
    assert audit["intervention_gpus"] == 2
    assert audit["comparison_output_files"] == 10
    assert audit["common_scene_pairs"] == pairs

    first_source = Path(sources["source_0"]["path"])
    _write_json(first_source, {"tampered": True})
    with pytest.raises(ValueError, match="source changed after analysis"):
        module.audit_pi0_intervention_output(output, parity, checkpoint, manifest)

    _write_json(first_source, {"index": 0})
    first_output = comparison_root / module.COMPARISON_OUTPUT_FILENAMES[0]
    original_output = first_output.read_text()
    first_output.write_text("tampered\n")
    with pytest.raises(ValueError, match="output changed after analysis"):
        module.audit_pi0_intervention_output(output, parity, checkpoint, manifest)

    first_output.write_text(original_output)
    (output / "gpu_preflight.csv").write_text(
        "index, uuid, name, driver_version, memory.total [MiB]\n"
        "0, GPU-a, NVIDIA H100 80GB HBM3, 570.0, 81559 MiB\n"
    )
    with pytest.raises(ValueError, match="two distinct H100s"):
        module.audit_pi0_intervention_output(output, parity, checkpoint, manifest)


def _write_grid(
    output: Path,
    clean: Path,
    reference: Path,
    binding: dict,
    commit: str,
    mode: str,
    pairs: list[str],
) -> None:
    root = output / "interventions" / mode
    selection = {
        "schema_version": 1,
        "selection_uses_interventions": False,
        "repo_commit": commit,
        "repo_tracked_clean": True,
        "repo_worktree_clean": True,
        "manifest_sha256": binding["manifest_sha256"],
        "selection_is_clean_eligible_intersection": True,
        "eligibility": "dual_success",
        "mode": mode,
        "minimum_selected_pairs": 12,
        "minimum_selection_passed": True,
        "failure_interpretation": None,
        "noise_seeds": [0],
        "pairs": pairs,
        "model_clean_eligible_pairs": pairs,
        "reference_clean_eligible_pairs": pairs,
        "clean_validation": str(clean),
        "clean_validation_summary_sha256": _summary_hashes(clean),
        "reference_clean_validation": str(reference),
        "reference_clean_validation_summary_sha256": _summary_hashes(reference),
    }
    _write_json(root / "selection.json", selection)
    jobs = []
    for pair_id in pairs:
        metadata_path = root / pair_id / "noise_0" / "metadata.json"
        metadata = {
            "schema_version": 1,
            "pair_id": pair_id,
            "config": "pi0_libero",
            "checkpoint": binding["pytorch_checkpoint"],
            "noise_seed": 0,
            "num_steps": 10,
            "action_horizon": 50,
            "model_action_dim": 32,
            "physical_action_dim": 7,
            "layers": 18,
            "openpi_commit": module.PINNED_OPENPI_REVISION,
            "record_count": 7,
        }
        _write_json(metadata_path, metadata)
        jobs.append(
            {
                "pair_id": pair_id,
                "noise_seed": 0,
                "records": 7,
                "metadata": str(metadata_path),
            }
        )
    _write_json(
        root / "run_summary.json",
        {
            "schema_version": 1,
            "selection_uses_interventions": False,
            "eligibility": "dual_success",
            "mode": mode,
            "expected_jobs": 12,
            "completed_jobs": 12,
            "complete": True,
            "jobs": jobs,
        },
    )
    _write_json(
        output / "analysis" / mode / "summary.json",
        {
            "schema_version": 1,
            "jobs": 12,
            "pairs": 12,
            "state_clusters": 12,
            "noise_seeds": [0],
            "commitment_threshold": 0.8,
            "formation_relative_error_tolerance": 0.2,
        },
    )


def _summary_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): file_digest(path)
        for path in sorted(root.glob("*/noise_*/summary.json"))
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
