from __future__ import annotations

import hashlib
import json
import runpy
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from action_chunking.conversion import (
    conversion_parity_summary,
    validate_prior_conversion_failure,
)

_manifest_script = runpy.run_path("scripts/validate_conversion_manifest.py")
_lossless_script = runpy.run_path("scripts/convert_pi0_checkpoint_lossless.py")


def test_conversion_launcher_passes_preserved_failure_to_final_parity() -> None:
    launcher = Path("scripts/run_pi0_conversion.sh").read_text()

    assert '--prior-failed-summary "$prior_failed_parity"' in launcher
    assert '[[ -e "$pytorch_output" || -e "$parity_output" ]]' in launcher
    assert 'checkpoint_assets="$jax_checkpoint/assets"' in launcher
    assert 'sibling_assets="$jax_checkpoint/../assets"' in launcher
    assert 'diff -qr "$jax_assets" "$pytorch_output/assets"' in launcher
    assert 'cp -a "$jax_assets" "$pytorch_output/assets"' in launcher
    assert "status --porcelain=v1 --untracked-files=all" in launcher
    assert '--action-chunking-commit "$action_chunking_commit"' in launcher
    assert launcher.count('--action-chunking-commit "$action_chunking_commit"') == 2


def test_lossless_adapter_accepts_preserved_failure_argument(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prior = tmp_path / "failed.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "convert_pi0_checkpoint_lossless.py",
            "--checkpoint-dir",
            str(tmp_path / "29999"),
            "--output-path",
            str(tmp_path / "converted"),
            "--upstream-converter",
            str(tmp_path / "convert.py"),
            "--prior-failed-summary",
            str(prior),
            "--upstream-revision",
            "openpi-commit",
            "--action-chunking-commit",
            "a" * 40,
        ],
    )

    args = _lossless_script["parse_args"]()

    assert args.prior_failed_summary == prior


@dataclass(frozen=True)
class _FakeModelConfig:
    dtype: str
    action_dim: int = 32


def test_lossless_conversion_forces_float32_without_mutating_source() -> None:
    source = _FakeModelConfig(dtype="bfloat16")

    converted = _lossless_script["float32_conversion_config"](source)

    assert source.dtype == "bfloat16"
    assert converted.dtype == "float32"
    assert converted.action_dim == source.action_dim


def test_lossless_conversion_rejects_non_dataclass_config() -> None:
    with pytest.raises(TypeError, match="dataclass"):
        _lossless_script["float32_conversion_config"]({"dtype": "bfloat16"})


def test_lossless_conversion_records_immutable_public_provenance(tmp_path) -> None:
    converter = tmp_path / "convert.py"
    converter.write_text("# pinned converter\n")
    prior = tmp_path / "failed.json"
    prior.write_text('{"passed": false}\n')

    result = _lossless_script["conversion_provenance"](
        converter,
        "openpi-commit",
        prior,
        "a" * 40,
    )

    assert result["source_precision_repair_commit"] == (
        "e5fe45e2c6784f315ffa59c207457701fb906c05"
    )
    assert result["upstream_openpi_revision"] == "openpi-commit"
    assert result["action_chunking_commit"] == "a" * 40
    assert result["upstream_converter_sha256"] == hashlib.sha256(
        converter.read_bytes()
    ).hexdigest()
    assert result["prior_failed_summary_sha256"] == hashlib.sha256(
        prior.read_bytes()
    ).hexdigest()
    assert result["saved_checkpoint_precision"] == "float32"

    with pytest.raises(ValueError, match="full lowercase"):
        _lossless_script["conversion_provenance"](
            converter,
            "openpi-commit",
            prior,
            "short",
        )


def test_conversion_parity_requires_every_case() -> None:
    reference = np.ones((2, 10, 7), dtype=np.float64)
    converted = reference.copy()
    converted[1, 0, 0] += 0.03

    result = conversion_parity_summary(["a", "b"], reference, converted)

    assert result["passed_cases"] == 1
    assert result["passed"] is False


def test_conversion_parity_accepts_small_error() -> None:
    reference = np.ones((1, 10, 7), dtype=np.float64)
    converted = reference + 0.001

    result = conversion_parity_summary(["a"], reference, converted)

    assert result["passed"] is True


def test_converted_checkpoint_hashes_required_artifacts(tmp_path) -> None:
    expected = {}
    for name, content in (
        ("config.json", b"config"),
        ("conversion_provenance.json", b"provenance"),
        ("model.safetensors", b"weights"),
    ):
        (tmp_path / name).write_bytes(content)
        expected[name] = hashlib.sha256(content).hexdigest()
    asset = tmp_path / "assets" / "norm_stats.json"
    asset.parent.mkdir()
    asset.write_bytes(b"norms")
    expected["assets/norm_stats.json"] = hashlib.sha256(b"norms").hexdigest()

    assert _manifest_script["_checkpoint_hashes"](tmp_path) == expected


def test_converted_checkpoint_hashes_reject_missing_weights(tmp_path) -> None:
    (tmp_path / "config.json").write_text("config")

    with pytest.raises(FileNotFoundError, match=r"model\.safetensors"):
        _manifest_script["_checkpoint_hashes"](tmp_path)


def test_converted_checkpoint_hashes_reject_missing_assets(tmp_path) -> None:
    for name in ("config.json", "conversion_provenance.json", "model.safetensors"):
        (tmp_path / name).write_text(name)

    with pytest.raises(FileNotFoundError, match="normalization assets"):
        _manifest_script["_checkpoint_hashes"](tmp_path)


def test_conversion_provenance_binds_upstream_converter(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    converter = tmp_path / "convert.py"
    converter.write_text("# pinned converter\n")
    prior = tmp_path / "failed.json"
    prior.write_text('{"passed": false}\n')
    provenance = _lossless_script["conversion_provenance"](
        converter,
        "215abfb217dbac7d5f1273282331b9b1866c0479",
        prior,
        "a" * 40,
    )
    (checkpoint / "conversion_provenance.json").write_text(
        __import__("json").dumps(provenance)
    )

    assert _manifest_script["_validate_conversion_provenance"](
        checkpoint,
        converter,
        prior,
        "a" * 40,
    ) == provenance

    missing_commit = dict(provenance)
    missing_commit.pop("action_chunking_commit")
    (checkpoint / "conversion_provenance.json").write_text(json.dumps(missing_commit))
    with pytest.raises(ValueError, match="action-chunking commit"):
        _manifest_script["_validate_conversion_provenance"](
            checkpoint,
            converter,
            prior,
            "a" * 40,
        )
    (checkpoint / "conversion_provenance.json").write_text(json.dumps(provenance))

    with pytest.raises(ValueError, match="wrong action-chunking commit"):
        _manifest_script["_validate_conversion_provenance"](
            checkpoint,
            converter,
            prior,
            "b" * 40,
        )

    converter.write_text("# changed converter\n")
    with pytest.raises(ValueError, match="wrong upstream converter digest"):
        _manifest_script["_validate_conversion_provenance"](
            checkpoint,
            converter,
            prior,
            "a" * 40,
        )

    converter.write_text("# pinned converter\n")
    prior.write_text('{"changed": true}\n')
    with pytest.raises(ValueError, match="conversion provenance mismatch"):
        _manifest_script["_validate_conversion_provenance"](
            checkpoint,
            converter,
            prior,
            "a" * 40,
        )


def test_lossless_rerun_binds_the_preserved_failed_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "29999"
    checkpoint.mkdir()
    manifest = tmp_path / "manifest.json"
    entries = [{"pair_id": f"pair-{index:02d}"} for index in range(16)]
    manifest.write_text(json.dumps({"pairs": entries}))
    identity = {"finalized": True, "optimizer_updates": 30_000}
    rows = [
        {"case": f"pair-{index:02d}:{side}", "passed": row_index < 24}
        for row_index, (index, side) in enumerate(
            (index, side) for index in range(16) for side in ("base", "donor")
        )
    ]
    summary_path = tmp_path / "failed.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "config": "pi0_libero",
                "noise_seed": 0,
                "cases": 32,
                "shape_per_case": [50, 7],
                "max_abs_tolerance": 0.02,
                "minimum_cosine_similarity": 0.999,
                "passed_cases": 24,
                "passed": False,
                "jax_checkpoint": str(checkpoint),
                "jax_checkpoint_identity": identity,
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "maximum_case_abs_error": 2.0130362831905284,
                "minimum_case_cosine_similarity": 0.805807150674655,
                "pytorch_checkpoint_artifact_sha256": {
                    "config.json": "a" * 64,
                    "model.safetensors": "b" * 64,
                },
                "rows": rows,
            }
        )
    )
    monkeypatch.setitem(
        validate_prior_conversion_failure.__globals__,
        "validate_pi0_final_checkpoint",
        lambda _path: identity,
    )

    result = validate_prior_conversion_failure(
        summary_path, checkpoint, manifest
    )

    assert result["passed_cases"] == 24
    assert result["cases"] == 32
    assert result["sha256"] == hashlib.sha256(summary_path.read_bytes()).hexdigest()

    changed = json.loads(summary_path.read_text())
    changed["rows"][0]["case"] = "different:base"
    summary_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="different case manifest"):
        validate_prior_conversion_failure(
            summary_path, checkpoint, manifest
        )
