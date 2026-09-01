from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import pytest

from action_chunking.conversion import (
    PARITY_ARTIFACT_NAMES,
    conversion_parity_summary,
)
from action_chunking.pairs import file_digest

_module = runpy.run_path("scripts/validate_pi0_intervention_inputs.py")
validate = _module["validate_pi0_intervention_inputs"]

_PROVENANCE = {
    "source_precision_repair_commit": "e5fe45e2c6784f315ffa59c207457701fb906c05",
    "upstream_openpi_revision": "215abfb217dbac7d5f1273282331b9b1866c0479",
    "saved_checkpoint_precision": "float32",
    "prior_failed_summary_sha256": "a" * 64,
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


def test_intervention_inputs_reconstruct_parity_worker_outputs(tmp_path: Path) -> None:
    checkpoint, hashes = _checkpoint(tmp_path)
    manifest = _manifest(tmp_path)
    parity = _parity(tmp_path, manifest, hashes)
    actions = np.load(tmp_path / "actions_pytorch.npy")
    actions[0, 0, 0] += 0.5
    np.save(tmp_path / "actions_pytorch.npy", actions)

    with pytest.raises(ValueError, match="worker artifact hashes differ"):
        validate(parity, checkpoint, manifest)


def test_intervention_inputs_require_same_prior_failure_digest(tmp_path: Path) -> None:
    checkpoint, hashes = _checkpoint(tmp_path)
    manifest = _manifest(tmp_path)
    parity = _parity(tmp_path, manifest, hashes)
    summary = json.loads(parity.read_text())
    summary["conversion_provenance"]["prior_failed_summary_sha256"] = "b" * 64
    parity.write_text(json.dumps(summary))
    (checkpoint / "conversion_provenance.json").write_text(
        json.dumps(summary["conversion_provenance"])
    )
    summary["pytorch_checkpoint_artifact_sha256"] = {
        name: file_digest(path)
        for name, path in (
            ("config.json", checkpoint / "config.json"),
            ("conversion_provenance.json", checkpoint / "conversion_provenance.json"),
            ("model.safetensors", checkpoint / "model.safetensors"),
            ("assets/norm_stats.json", checkpoint / "assets" / "norm_stats.json"),
        )
    }
    parity.write_text(json.dumps(summary))

    with pytest.raises(ValueError, match="preserved conversion failure"):
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
    actions = np.stack(
        [np.full((50, 7), index + 1.0, dtype=np.float64) for index in range(32)]
    )
    np.save(root / "actions_jax.npy", actions)
    np.save(root / "actions_pytorch.npy", actions)
    for backend in ("jax", "pytorch"):
        (root / f"identifiers_{backend}.json").write_text(json.dumps(cases))
    summary = conversion_parity_summary(cases, actions, actions)
    summary.update(
        {
            "config": "pi0_libero",
            "noise_seed": 0,
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "pytorch_checkpoint_artifact_sha256": hashes,
            "jax_checkpoint_identity": {"finalized": True, "optimizer_updates": 30_000},
            "conversion_provenance": _PROVENANCE,
            "prior_failed_conversion": {
                "cases": 32,
                "passed_cases": 24,
                "sha256": "a" * 64,
            },
            "parity_artifact_sha256": {
                name: file_digest(root / name) for name in PARITY_ARTIFACT_NAMES
            },
        }
    )
    path = root / "summary.json"
    path.write_text(json.dumps(summary))
    return path
