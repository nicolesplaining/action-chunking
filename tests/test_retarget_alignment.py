from __future__ import annotations

import json

import numpy as np

from action_chunking.pairs import load_action_chunk


def test_saved_replan_chunk_lookup_is_exact(tmp_path) -> None:
    chunks = [np.full((10, 7), index, dtype=np.float64).tolist() for index in range(6)]
    path = tmp_path / "actions.json"
    path.write_text(json.dumps(chunks))

    np.testing.assert_array_equal(load_action_chunk(path, 5), np.full((10, 7), 5.0))
