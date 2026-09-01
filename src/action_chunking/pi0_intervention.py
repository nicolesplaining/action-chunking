"""Fail-closed input and output audits for the matched-pi0 intervention study."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from action_chunking.conversion import converted_checkpoint_artifact_hashes
from action_chunking.pairs import file_digest

PINNED_OPENPI_REVISION = "215abfb217dbac7d5f1273282331b9b1866c0479"
PRECISION_REPAIR_COMMIT = "e5fe45e2c6784f315ffa59c207457701fb906c05"


def validate_pi0_intervention_inputs(
    parity_summary_path: Path,
    pytorch_checkpoint: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Bind an intervention run to the exact checkpoint that passed parity."""
    parity = json.loads(parity_summary_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    entries = manifest.get("pairs", [])
    pair_ids = [str(entry.get("pair_id")) for entry in entries]
    if len(pair_ids) != 16 or len(set(pair_ids)) != 16:
        raise ValueError("pi0 intervention manifest must contain 16 unique frozen pairs")
    expected_cases = {
        f"{pair_id}:{side}" for pair_id in pair_ids for side in ("base", "donor")
    }
    rows = parity.get("rows", [])
    observed_cases = {str(row.get("case")) for row in rows}
    required = {
        "schema_version": 1,
        "config": "pi0_libero",
        "noise_seed": 0,
        "cases": 32,
        "shape_per_case": [50, 7],
        "max_abs_tolerance": 0.02,
        "minimum_cosine_similarity": 0.999,
        "passed_cases": 32,
        "passed": True,
    }
    mismatched = {
        key: {"expected": expected, "actual": parity.get(key)}
        for key, expected in required.items()
        if parity.get(key) != expected
    }
    if mismatched:
        raise ValueError(f"pi0 parity summary fails the frozen intervention gate: {mismatched}")
    if (
        parity.get("manifest_sha256") != file_digest(manifest_path)
        or len(rows) != 32
        or observed_cases != expected_cases
        or not all(row.get("passed") is True for row in rows)
    ):
        raise ValueError("pi0 parity summary is not bound to the supplied case manifest")

    expected_hashes = parity.get("pytorch_checkpoint_artifact_sha256", {})
    actual_hashes = converted_checkpoint_artifact_hashes(pytorch_checkpoint)
    if expected_hashes != actual_hashes:
        raise ValueError("supplied pi0 checkpoint artifacts differ from the passed parity run")
    if not all(re.fullmatch(r"[0-9a-f]{64}", value) for value in actual_hashes.values()):
        raise ValueError("pi0 checkpoint contains an invalid artifact digest")

    provenance = parity.get("conversion_provenance", {})
    required_provenance = {
        "source_precision_repair_commit": PRECISION_REPAIR_COMMIT,
        "upstream_openpi_revision": PINNED_OPENPI_REVISION,
        "saved_checkpoint_precision": "float32",
    }
    if any(provenance.get(key) != value for key, value in required_provenance.items()):
        raise ValueError("pi0 parity summary has invalid lossless-conversion provenance")
    actual_provenance = json.loads(
        (pytorch_checkpoint / "conversion_provenance.json").read_text()
    )
    if actual_provenance != provenance:
        raise ValueError("pi0 parity provenance differs from the supplied checkpoint")
    prior = parity.get("prior_failed_conversion", {})
    if (
        int(prior.get("cases", -1)) != 32
        or int(prior.get("passed_cases", -1)) != 24
        or re.fullmatch(r"[0-9a-f]{64}", str(prior.get("sha256"))) is None
    ):
        raise ValueError("pi0 parity summary is not bound to the preserved conversion failure")
    identity = parity.get("jax_checkpoint_identity", {})
    if identity.get("finalized") is not True or int(identity.get("optimizer_updates", -1)) != 30_000:
        raise ValueError("pi0 parity summary used the wrong JAX checkpoint identity")
    return {
        "schema_version": 1,
        "passed": True,
        "parity_summary": str(parity_summary_path),
        "parity_summary_sha256": file_digest(parity_summary_path),
        "manifest": str(manifest_path),
        "manifest_sha256": file_digest(manifest_path),
        "pytorch_checkpoint": str(pytorch_checkpoint),
        "pytorch_checkpoint_artifact_sha256": actual_hashes,
        "cases": 32,
        "passed_cases": 32,
    }


def audit_pi0_intervention_output(
    output_root: Path,
    parity_summary_path: Path,
    pytorch_checkpoint: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Reconstruct the complete matched-control lineage after the grid finishes."""
    expected_binding = validate_pi0_intervention_inputs(
        parity_summary_path,
        pytorch_checkpoint,
        manifest_path,
    )
    observed_binding = _read_json(output_root / "intervention_input_binding.json")
    if observed_binding != expected_binding:
        raise ValueError("saved pi0 intervention binding differs from current frozen inputs")

    code_commit = (output_root / "code_commit.txt").read_text().strip()
    if re.fullmatch(r"[0-9a-f]{40}", code_commit) is None:
        raise ValueError("pi0 intervention output has no valid full code commit")

    selections = {
        mode: _audit_grid(output_root, mode, code_commit, expected_binding)
        for mode in ("coarse", "population_positions")
    }
    if selections["coarse"] != selections["population_positions"]:
        raise ValueError("pi0 coarse and position grids used different scene-state intersections")

    comparison = _read_json(output_root / "comparison" / "summary.json")
    required_comparison = {
        "schema_version": 1,
        "analysis_unit": "paired_scene_state",
        "comparison": "pi05_minus_pi0",
        "pi05_action_horizon": 10,
        "pi0_action_horizon": 50,
        "primary_position_window": list(range(10)),
        "normalized_position_bins": 10,
        "bootstrap_replicates": 10_000,
    }
    _require_fields(comparison, required_comparison, "pi0 comparison")
    sources = comparison.get("source_files", {})
    if len(sources) != 14:
        raise ValueError("pi0 comparison does not bind all 14 registered source files")
    for name, source in sources.items():
        path = Path(str(source.get("path", "")))
        if not path.is_file() or source.get("sha256") != file_digest(path):
            raise ValueError(f"pi0 comparison source changed after analysis: {name}")

    return {
        "schema_version": 1,
        "passed": True,
        "code_commit": code_commit,
        "common_scene_pairs": selections["coarse"],
        "common_scene_pair_count": len(selections["coarse"]),
        "input_binding_sha256": file_digest(output_root / "intervention_input_binding.json"),
        "comparison_summary_sha256": file_digest(output_root / "comparison" / "summary.json"),
        "comparison_source_files": len(sources),
    }


def _audit_grid(
    output_root: Path,
    mode: str,
    code_commit: str,
    binding: dict[str, Any],
) -> list[str]:
    root = output_root / "interventions" / mode
    selection = _read_json(root / "selection.json")
    required = {
        "schema_version": 1,
        "selection_uses_interventions": False,
        "repo_commit": code_commit,
        "repo_tracked_clean": True,
        "manifest_sha256": binding["manifest_sha256"],
        "selection_is_clean_eligible_intersection": True,
        "eligibility": "dual_success",
        "mode": mode,
        "minimum_selected_pairs": 12,
        "minimum_selection_passed": True,
        "failure_interpretation": None,
        "noise_seeds": [0],
    }
    _require_fields(selection, required, f"pi0 {mode} selection")
    pairs = [str(value) for value in selection.get("pairs", [])]
    model_pairs = [str(value) for value in selection.get("model_clean_eligible_pairs", [])]
    reference_pairs = [str(value) for value in selection.get("reference_clean_eligible_pairs", [])]
    expected_pairs = [pair_id for pair_id in model_pairs if pair_id in set(reference_pairs)]
    if len(pairs) < 12 or len(pairs) != len(set(pairs)) or pairs != expected_pairs:
        raise ValueError(f"pi0 {mode} selection is not the frozen clean intersection")
    _audit_validation_hashes(
        Path(str(selection["clean_validation"])),
        selection.get("clean_validation_summary_sha256"),
    )
    _audit_validation_hashes(
        Path(str(selection["reference_clean_validation"])),
        selection.get("reference_clean_validation_summary_sha256"),
    )

    run = _read_json(root / "run_summary.json")
    _require_fields(
        run,
        {
            "schema_version": 1,
            "selection_uses_interventions": False,
            "eligibility": "dual_success",
            "mode": mode,
            "expected_jobs": len(pairs),
            "completed_jobs": len(pairs),
            "complete": True,
        },
        f"pi0 {mode} run summary",
    )
    jobs = run.get("jobs", [])
    observed_jobs = {(str(job.get("pair_id")), int(job.get("noise_seed", -1))) for job in jobs}
    if len(jobs) != len(pairs) or observed_jobs != {(pair_id, 0) for pair_id in pairs}:
        raise ValueError(f"pi0 {mode} run has incomplete or duplicate jobs")
    for job in jobs:
        metadata_path = Path(str(job["metadata"]))
        expected_metadata_path = root / str(job["pair_id"]) / "noise_0" / "metadata.json"
        if metadata_path.resolve() != expected_metadata_path.resolve():
            raise ValueError(f"pi0 {mode} job metadata path is outside its registered cell")
        metadata = _read_json(metadata_path)
        _require_fields(
            metadata,
            {
                "schema_version": 1,
                "pair_id": str(job["pair_id"]),
                "config": "pi0_libero",
                "checkpoint": binding["pytorch_checkpoint"],
                "noise_seed": 0,
                "num_steps": 10,
                "action_horizon": 50,
                "model_action_dim": 32,
                "physical_action_dim": 7,
                "layers": 18,
                "openpi_commit": PINNED_OPENPI_REVISION,
            },
            f"pi0 {mode} job metadata",
        )
        if int(job.get("records", -1)) != int(metadata.get("record_count", -2)):
            raise ValueError(f"pi0 {mode} job record count differs from metadata")

    analysis = _read_json(output_root / "analysis" / mode / "summary.json")
    _require_fields(
        analysis,
        {
            "schema_version": 1,
            "jobs": len(pairs),
            "pairs": len(pairs),
            "noise_seeds": [0],
            "commitment_threshold": 0.8,
            "formation_relative_error_tolerance": 0.2,
        },
        f"pi0 {mode} analysis",
    )
    if int(analysis.get("state_clusters", 0)) < 12:
        raise ValueError(f"pi0 {mode} analysis has fewer than 12 independent state clusters")
    return pairs


def _audit_validation_hashes(root: Path, expected: Any) -> None:
    paths = sorted(root.glob("*/noise_*/summary.json"))
    actual = {str(path.relative_to(root)): file_digest(path) for path in paths}
    if not paths or actual != expected:
        raise ValueError(f"clean-validation summaries changed after selection: {root}")


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
        raise ValueError(f"{label} fails the frozen audit: {mismatched}")
