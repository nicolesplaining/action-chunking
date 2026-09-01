"""Fail-closed identity checks for the frozen matched pi0 checkpoint."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

PI0_FINAL_UPDATE_COUNT = 30_000
PI0_FINAL_CHECKPOINT_LABEL = "29999"
PI0_FINAL_EXPERIMENT = "pi0_libero_seed42_30000"
PI0_FINAL_ARTIFACT_SHA256 = {
    "_CHECKPOINT_METADATA": "cf454e8412c1e734cec099baf1a0638b56ad07b9e70219e17ea0a8b780d5fae8",
    "params/manifest.ocdbt": "1a338f26792c1614a315c0e9bac7bb532c21f9bc2ac6b196be0fcad862fe0894",
    "train_state/manifest.ocdbt": "f77c5744e270d94646d1fde33e2f4ff29fabf801b4ff5b6eefce9a77a02acfe1",
}


def validate_pi0_final_checkpoint(
    checkpoint: Path,
    *,
    expected_hashes: Mapping[str, str] = PI0_FINAL_ARTIFACT_SHA256,
    expected_label: str = PI0_FINAL_CHECKPOINT_LABEL,
    expected_experiment: str = PI0_FINAL_EXPERIMENT,
) -> dict[str, object]:
    """Validate identity and finalized Orbax metadata without loading model tensors."""
    checkpoint = checkpoint.resolve()
    if checkpoint.name != expected_label:
        raise ValueError(
            f"expected zero-based final checkpoint label {expected_label}, found {checkpoint.name}"
        )
    if checkpoint.parent.name != expected_experiment:
        raise ValueError(
            f"expected experiment {expected_experiment}, found {checkpoint.parent.name}"
        )
    observed = {}
    for relative, expected in expected_hashes.items():
        path = checkpoint / relative
        if not path.is_file():
            raise FileNotFoundError(f"final checkpoint artifact is missing: {path}")
        observed[relative] = _digest(path)
        if observed[relative] != expected:
            raise ValueError(f"final checkpoint artifact hash mismatch: {relative}")
    metadata = json.loads((checkpoint / "_CHECKPOINT_METADATA").read_text())
    handlers = metadata.get("item_handlers", {})
    if set(handlers) != {"assets", "params", "train_state"}:
        raise ValueError("final checkpoint metadata has unexpected item handlers")
    initialized = int(metadata.get("init_timestamp_nsecs", 0))
    committed = int(metadata.get("commit_timestamp_nsecs", 0))
    if initialized <= 0 or committed < initialized:
        raise ValueError("final checkpoint metadata is not committed")
    return {
        "schema_version": 1,
        "model": "pi0_libero",
        "optimizer_updates": PI0_FINAL_UPDATE_COUNT,
        "orbax_zero_based_checkpoint_label": expected_label,
        "experiment": expected_experiment,
        "artifact_sha256": observed,
        "finalized": True,
    }


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
