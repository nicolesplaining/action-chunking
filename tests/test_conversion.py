from __future__ import annotations

import hashlib
import runpy

import numpy as np
import pytest

from action_chunking.conversion import conversion_parity_summary

_manifest_script = runpy.run_path("scripts/validate_conversion_manifest.py")


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
        ("model.safetensors", b"weights"),
    ):
        (tmp_path / name).write_bytes(content)
        expected[name] = hashlib.sha256(content).hexdigest()

    assert _manifest_script["_checkpoint_hashes"](tmp_path) == expected


def test_converted_checkpoint_hashes_reject_missing_weights(tmp_path) -> None:
    (tmp_path / "config.json").write_text("config")

    with pytest.raises(FileNotFoundError, match=r"model\.safetensors"):
        _manifest_script["_checkpoint_hashes"](tmp_path)
