from __future__ import annotations

import hashlib
import runpy
from dataclasses import dataclass

import numpy as np
import pytest

from action_chunking.conversion import conversion_parity_summary

_manifest_script = runpy.run_path("scripts/validate_conversion_manifest.py")
_lossless_script = runpy.run_path("scripts/convert_pi0_checkpoint_lossless.py")


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
