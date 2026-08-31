from __future__ import annotations

import pytest

from action_chunking.screening import summarize_clean_screen

SCREEN = {
    "base_affinity_maximum": -0.005,
    "donor_affinity_minimum": 0.005,
    "translation_l2_minimum": 0.01,
}


def _record(seed: int, *, passes: bool) -> dict:
    return {
        "pair_id": "pair_0",
        "scene_state_sha256": "state",
        "manifest": "manifest.json",
        "fixture_sha256": "fixture",
        "init_index": 0,
        "base_target": "mug_a",
        "donor_target": "mug_b",
        "noise_seed": seed,
        "base_target_direction_affinity": -0.02 if passes else 0.0,
        "donor_target_direction_affinity": 0.03,
        "direction_contrast": 0.05 if passes else 0.03,
        "endpoint_group_l2_contrasts": {"translation": 0.1},
        "direction_screen_pass": passes,
    }


def test_summarize_clean_screen_recomputes_pair_and_contrast_tables() -> None:
    pair_rows, contrast_rows = summarize_clean_screen(
        [_record(0, passes=True), _record(1, passes=False)],
        [0, 1],
        SCREEN,
    )
    assert pair_rows[0]["passing_seeds"] == 1
    assert pair_rows[0]["passes_all_seeds"] is False
    assert pair_rows[0]["base_affinity_failures"] == 1
    assert contrast_rows[0]["pairs_passing_any_seed"] == 1
    assert contrast_rows[0]["pairs_passing_all_seeds"] == 0


def test_summarize_clean_screen_rejects_missing_seed() -> None:
    with pytest.raises(ValueError, match="expected noise seeds"):
        summarize_clean_screen([_record(0, passes=True)], [0, 1], SCREEN)


def test_summarize_clean_screen_rejects_tampered_pass_flag() -> None:
    record = _record(0, passes=True)
    record["direction_screen_pass"] = False
    with pytest.raises(ValueError, match="pass flags disagree"):
        summarize_clean_screen([record], [0], SCREEN)
