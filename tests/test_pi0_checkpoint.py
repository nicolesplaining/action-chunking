from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from action_chunking.pi0_checkpoint import validate_pi0_final_checkpoint


def test_validates_final_checkpoint_identity(tmp_path: Path) -> None:
    checkpoint, hashes = _checkpoint(tmp_path)

    result = validate_pi0_final_checkpoint(
        checkpoint,
        expected_hashes=hashes,
        expected_label="final",
        expected_experiment="experiment",
    )

    assert result["optimizer_updates"] == 30_000
    assert result["finalized"] is True


def test_rejects_wrong_checkpoint_label(tmp_path: Path) -> None:
    checkpoint, hashes = _checkpoint(tmp_path)

    with pytest.raises(ValueError, match="checkpoint label"):
        validate_pi0_final_checkpoint(
            checkpoint,
            expected_hashes=hashes,
            expected_label="other",
            expected_experiment="experiment",
        )


def test_rejects_changed_manifest(tmp_path: Path) -> None:
    checkpoint, hashes = _checkpoint(tmp_path)
    (checkpoint / "params/manifest.ocdbt").write_text("changed")

    with pytest.raises(ValueError, match="hash mismatch"):
        validate_pi0_final_checkpoint(
            checkpoint,
            expected_hashes=hashes,
            expected_label="final",
            expected_experiment="experiment",
        )


def _checkpoint(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    checkpoint = tmp_path / "experiment" / "final"
    (checkpoint / "params").mkdir(parents=True)
    (checkpoint / "train_state").mkdir()
    metadata = {
        "item_handlers": {"assets": "a", "params": "p", "train_state": "t"},
        "init_timestamp_nsecs": 1,
        "commit_timestamp_nsecs": 2,
    }
    (checkpoint / "_CHECKPOINT_METADATA").write_text(json.dumps(metadata))
    (checkpoint / "params/manifest.ocdbt").write_text("params")
    (checkpoint / "train_state/manifest.ocdbt").write_text("state")
    paths = ("_CHECKPOINT_METADATA", "params/manifest.ocdbt", "train_state/manifest.ocdbt")
    hashes = {
        relative: hashlib.sha256((checkpoint / relative).read_bytes()).hexdigest()
        for relative in paths
    }
    return checkpoint, hashes
