from __future__ import annotations

import json

import numpy as np

from action_chunking.pairs import load_action_chunk, load_replan_input


def test_saved_replan_chunk_lookup_is_exact(tmp_path) -> None:
    chunks = [np.full((10, 7), index, dtype=np.float64).tolist() for index in range(6)]
    path = tmp_path / "actions.json"
    path.write_text(json.dumps(chunks))

    np.testing.assert_array_equal(load_action_chunk(path, 5), np.full((10, 7), 5.0))


def test_saved_replan_input_lookup_is_exact(tmp_path) -> None:
    path = tmp_path / "replans.npz"
    np.savez_compressed(
        path,
        replan_indices=np.arange(3),
        images=np.arange(3 * 4 * 5 * 3, dtype=np.uint8).reshape(3, 4, 5, 3),
        wrist_images=np.zeros((3, 4, 5, 3), dtype=np.uint8),
        states=np.arange(24, dtype=np.float64).reshape(3, 8),
    )

    result = load_replan_input(path, 2)

    np.testing.assert_array_equal(result["state"], np.arange(16, 24, dtype=np.float64))
    assert result["image"].shape == (4, 5, 3)
