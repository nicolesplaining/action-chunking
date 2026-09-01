#!/usr/bin/env python3
"""Bind a matched-pi0 intervention run to its exact passed parity artifacts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from action_chunking.conversion import converted_checkpoint_artifact_hashes
from action_chunking.pairs import file_digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parity-summary", type=Path, required=True)
    parser.add_argument("--pytorch-checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def validate_pi0_intervention_inputs(
    parity_summary_path: Path,
    pytorch_checkpoint: Path,
    manifest_path: Path,
) -> dict[str, Any]:
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
        "source_precision_repair_commit": "e5fe45e2c6784f315ffa59c207457701fb906c05",
        "upstream_openpi_revision": "215abfb217dbac7d5f1273282331b9b1866c0479",
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


def main() -> int:
    args = parse_args()
    result = validate_pi0_intervention_inputs(
        args.parity_summary,
        args.pytorch_checkpoint,
        args.manifest,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
