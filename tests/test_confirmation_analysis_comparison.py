from __future__ import annotations

import copy
import runpy

import pytest

compare_confirmation_analyses = runpy.run_path(
    "scripts/compare_confirmation_analyses.py"
)["compare_confirmation_analyses"]
REGISTERED_FIELDS = runpy.run_path("scripts/compare_confirmation_analyses.py")[
    "REGISTERED_FIELDS"
]


def test_original_and_hardened_confirmation_must_match_exactly() -> None:
    original = _summary()
    hardened = {
        **copy.deepcopy(original),
        "source_artifact_files": 1505,
        "source_artifact_manifest_sha256": "a" * 64,
    }

    result = compare_confirmation_analyses(original, hardened)

    assert result["registered_fields_exact"] is True
    assert result["episode_pairs"] == 500
    assert result["source_artifact_files"] == 1505


def test_comparison_rejects_changed_outcome() -> None:
    original = _summary()
    hardened = {
        **copy.deepcopy(original),
        "paired_losses": 1,
        "source_artifact_files": 1505,
        "source_artifact_manifest_sha256": "a" * 64,
    }

    with pytest.raises(ValueError, match="paired_losses"):
        compare_confirmation_analyses(original, hardened)


def test_comparison_rejects_unregistered_hardened_field() -> None:
    original = _summary()
    hardened = {
        **copy.deepcopy(original),
        "source_artifact_files": 1505,
        "source_artifact_manifest_sha256": "a" * 64,
        "post_hoc_result": True,
    }

    with pytest.raises(ValueError, match="unexpected hardened-only"):
        compare_confirmation_analyses(original, hardened)


def _summary() -> dict:
    value = {field: None for field in REGISTERED_FIELDS}
    value.update(
        {
            "schema_version": 1,
            "episode_pairs": 500,
            "paired_losses": 0,
            "confirmation_positive": True,
            "rows": [{"pair_key": "pair"}],
        }
    )
    return value
