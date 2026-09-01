"""Parity checks for JAX-to-PyTorch policy conversion."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from action_chunking.pairs import file_digest
from action_chunking.pi0_checkpoint import validate_pi0_final_checkpoint

PRIOR_FAILURE_MAXIMUM_ABS_ERROR = 2.0130362831905284
PRIOR_FAILURE_MINIMUM_COSINE = 0.805807150674655
PARITY_ARTIFACT_NAMES = (
    "actions_jax.npy",
    "actions_pytorch.npy",
    "identifiers_jax.json",
    "identifiers_pytorch.json",
)


def converted_checkpoint_artifact_hashes(checkpoint: Path) -> dict[str, str]:
    """Hash all runtime-defining files in a lossless converted checkpoint."""
    required = ("config.json", "conversion_provenance.json", "model.safetensors")
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing:
        raise FileNotFoundError(f"converted checkpoint artifacts are missing: {missing}")
    assets = sorted(path for path in (checkpoint / "assets").rglob("*") if path.is_file())
    if not assets:
        raise FileNotFoundError("converted checkpoint has no normalization assets")
    paths = [*(checkpoint / name for name in required), *assets]
    return {str(path.relative_to(checkpoint)): file_digest(path) for path in paths}


def conversion_parity_summary(
    identifiers: list[str],
    reference: np.ndarray,
    converted: np.ndarray,
    *,
    max_abs_tolerance: float = 0.02,
    minimum_cosine_similarity: float = 0.999,
) -> dict[str, Any]:
    """Evaluate frozen per-case physical-action conversion tolerances."""
    if reference.shape != converted.shape or reference.ndim != 3:
        raise ValueError("conversion arrays must share [case, action, dimension] shape")
    if len(identifiers) != reference.shape[0] or len(set(identifiers)) != len(identifiers):
        raise ValueError("conversion case identifiers must be unique and complete")
    if max_abs_tolerance <= 0.0 or not 0.0 < minimum_cosine_similarity <= 1.0:
        raise ValueError("conversion tolerances are invalid")
    rows = []
    for identifier, source, destination in zip(
        identifiers, reference, converted, strict=True
    ):
        difference = destination - source
        source_flat = source.ravel()
        destination_flat = destination.ravel()
        denominator = np.linalg.norm(source_flat) * np.linalg.norm(destination_flat)
        cosine = float(np.dot(source_flat, destination_flat) / denominator)
        row = {
            "case": identifier,
            "max_abs_error": float(np.max(np.abs(difference))),
            "mean_abs_error": float(np.mean(np.abs(difference))),
            "rmse": float(np.sqrt(np.mean(np.square(difference)))),
            "cosine_similarity": cosine,
        }
        row["passed"] = bool(
            row["max_abs_error"] <= max_abs_tolerance
            and row["cosine_similarity"] >= minimum_cosine_similarity
        )
        rows.append(row)
    return {
        "schema_version": 1,
        "cases": len(rows),
        "shape_per_case": list(reference.shape[1:]),
        "max_abs_tolerance": max_abs_tolerance,
        "minimum_cosine_similarity": minimum_cosine_similarity,
        "maximum_case_abs_error": max(row["max_abs_error"] for row in rows),
        "minimum_case_cosine_similarity": min(row["cosine_similarity"] for row in rows),
        "passed_cases": sum(row["passed"] for row in rows),
        "passed": all(row["passed"] for row in rows),
        "rows": rows,
    }


def validate_saved_conversion_parity(summary_path: Path) -> dict[str, Any]:
    """Reconstruct a saved parity decision from its immutable worker artifacts."""
    if not summary_path.is_file():
        raise FileNotFoundError("conversion parity summary is missing")
    summary = json.loads(summary_path.read_text())
    root = summary_path.parent
    missing = [name for name in PARITY_ARTIFACT_NAMES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"conversion parity worker artifacts are missing: {missing}")
    actual_hashes = {
        name: file_digest(root / name) for name in PARITY_ARTIFACT_NAMES
    }
    if summary.get("parity_artifact_sha256") != actual_hashes:
        raise ValueError("conversion parity worker artifact hashes differ from the summary")

    jax_identifiers = json.loads((root / "identifiers_jax.json").read_text())
    pytorch_identifiers = json.loads((root / "identifiers_pytorch.json").read_text())
    if jax_identifiers != pytorch_identifiers:
        raise ValueError("saved conversion workers have different case identifiers")
    reconstructed = conversion_parity_summary(
        jax_identifiers,
        np.load(root / "actions_jax.npy", allow_pickle=False),
        np.load(root / "actions_pytorch.npy", allow_pickle=False),
        max_abs_tolerance=float(summary.get("max_abs_tolerance", -1.0)),
        minimum_cosine_similarity=float(
            summary.get("minimum_cosine_similarity", -1.0)
        ),
    )
    mismatched = {
        key: {"expected": value, "actual": summary.get(key)}
        for key, value in reconstructed.items()
        if summary.get(key) != value
    }
    if mismatched:
        raise ValueError(
            f"conversion parity summary differs from reconstructed worker outputs: {mismatched}"
        )
    return summary


def validate_prior_conversion_failure(
    summary_path: Path,
    jax_checkpoint: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Bind a lossless rerun to the preserved 24/32 bfloat16 failure."""
    if not summary_path.is_file():
        raise FileNotFoundError("prior failed conversion summary is missing")
    summary = json.loads(summary_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    expected_cases = {
        f"{entry['pair_id']}:{side}"
        for entry in manifest["pairs"]
        for side in ("base", "donor")
    }
    observed_cases = {str(row.get("case")) for row in summary.get("rows", [])}
    required = {
        "schema_version": 1,
        "config": "pi0_libero",
        "noise_seed": 0,
        "cases": 32,
        "shape_per_case": [50, 7],
        "max_abs_tolerance": 0.02,
        "minimum_cosine_similarity": 0.999,
        "passed_cases": 24,
        "passed": False,
    }
    mismatched = {
        key: {"expected": expected, "actual": summary.get(key)}
        for key, expected in required.items()
        if summary.get(key) != expected
    }
    if mismatched:
        raise ValueError(f"prior conversion failure differs from frozen audit: {mismatched}")
    if (
        Path(summary.get("jax_checkpoint", "")).resolve() != jax_checkpoint.resolve()
        or summary.get("jax_checkpoint_identity")
        != validate_pi0_final_checkpoint(jax_checkpoint)
    ):
        raise ValueError("prior conversion failure used a different JAX checkpoint")
    if (
        summary.get("manifest_sha256") != file_digest(manifest_path)
        or observed_cases != expected_cases
        or len(summary.get("rows", [])) != len(expected_cases)
        or len(expected_cases) != 32
    ):
        raise ValueError("prior conversion failure used a different case manifest")
    if sum(bool(row.get("passed")) for row in summary["rows"]) != 24:
        raise ValueError("prior conversion failure rows disagree with the pass count")
    if (
        float(summary.get("maximum_case_abs_error", -1.0))
        != PRIOR_FAILURE_MAXIMUM_ABS_ERROR
        or float(summary.get("minimum_case_cosine_similarity", -1.0))
        != PRIOR_FAILURE_MINIMUM_COSINE
    ):
        raise ValueError("prior conversion failure metrics differ from the preserved result")
    artifact_hashes = summary.get("pytorch_checkpoint_artifact_sha256", {})
    if set(artifact_hashes) != {"config.json", "model.safetensors"} or not all(
        re.fullmatch(r"[0-9a-f]{64}", str(value)) for value in artifact_hashes.values()
    ):
        raise ValueError("prior conversion failure has invalid checkpoint hashes")
    return {
        "path": str(summary_path),
        "sha256": file_digest(summary_path),
        "passed_cases": 24,
        "cases": 32,
        "maximum_case_abs_error": PRIOR_FAILURE_MAXIMUM_ABS_ERROR,
        "minimum_case_cosine_similarity": PRIOR_FAILURE_MINIMUM_COSINE,
        "pytorch_checkpoint_artifact_sha256": artifact_hashes,
    }
