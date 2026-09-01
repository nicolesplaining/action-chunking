from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path

import pytest

_module = runpy.run_path("scripts/validate_pi0_intervention_inputs.py")
validate = _module["validate_pi0_intervention_inputs"]

_PROVENANCE = {
    "source_precision_repair_commit": "e5fe45e2c6784f315ffa59c207457701fb906c05",
    "upstream_openpi_revision": "215abfb217dbac7d5f1273282331b9b1866c0479",
    "saved_checkpoint_precision": "float32",
}


def test_intervention_inputs_bind_checkpoint_manifest_and_assets(tmp_path: Path) -> None:
    checkpoint, hashes = _checkpoint(tmp_path)
    manifest = _manifest(tmp_path)
    parity = _parity(tmp_path, manifest, hashes)

    result = validate(parity, checkpoint, manifest)

    assert result["passed"] is True
    assert result["pytorch_checkpoint_artifact_sha256"] == hashes

    (checkpoint / "assets" / "norm_stats.json").write_text("changed")
    with pytest.raises(ValueError, match="differ from the passed parity"):
        validate(parity, checkpoint, manifest)


def test_intervention_inputs_reject_different_manifest(tmp_path: Path) -> None:
    checkpoint, hashes = _checkpoint(tmp_path)
    manifest = _manifest(tmp_path)
    parity = _parity(tmp_path, manifest, hashes)
    changed = json.loads(manifest.read_text())
    changed["pairs"][0]["pair_id"] = "changed"
    manifest.write_text(json.dumps(changed))

    with pytest.raises(ValueError, match="supplied case manifest"):
        validate(parity, checkpoint, manifest)


def _checkpoint(root: Path) -> tuple[Path, dict[str, str]]:
    checkpoint = root / "checkpoint"
    files = {
        "config.json": b"config",
        "conversion_provenance.json": json.dumps(_PROVENANCE).encode(),
        "model.safetensors": b"weights",
        "assets/norm_stats.json": b"norms",
    }
    for name, content in files.items():
        path = checkpoint / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return checkpoint, {
        name: hashlib.sha256(content).hexdigest() for name, content in files.items()
    }


def _manifest(root: Path) -> Path:
    path = root / "manifest.json"
    path.write_text(json.dumps({"pairs": [{"pair_id": f"pair-{index:02d}"} for index in range(16)]}))
    return path


def _parity(root: Path, manifest: Path, hashes: dict[str, str]) -> Path:
    cases = [
        f"pair-{index:02d}:{side}"
        for index in range(16)
        for side in ("base", "donor")
    ]
    path = root / "parity.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "config": "pi0_libero",
                "noise_seed": 0,
                "cases": 32,
                "shape_per_case": [50, 7],
                "max_abs_tolerance": 0.02,
                "minimum_cosine_similarity": 0.999,
                "passed_cases": 32,
                "passed": True,
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "rows": [{"case": case, "passed": True} for case in cases],
                "pytorch_checkpoint_artifact_sha256": hashes,
                "jax_checkpoint_identity": {"finalized": True, "optimizer_updates": 30_000},
                "conversion_provenance": _PROVENANCE,
                "prior_failed_conversion": {
                    "cases": 32,
                    "passed_cases": 24,
                    "sha256": "a" * 64,
                },
            }
        )
    )
    return path
