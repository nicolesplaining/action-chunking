#!/usr/bin/env python3
"""Create an explicit clean-only selection artifact from a pair manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from action_chunking.pairs import file_digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--pair-ids", help="Comma-separated subset; default is every manifest pair")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    requested = set(args.pair_ids.split(",")) if args.pair_ids else None
    selected = []
    for entry in manifest["pairs"]:
        if requested is not None and entry["pair_id"] not in requested:
            continue
        fixture = args.manifest.parent / entry["fixture"]
        if file_digest(fixture) != entry["fixture_sha256"]:
            raise ValueError(f"fixture hash mismatch: {fixture}")
        selected.append(
            {
                "pair_id": entry["pair_id"],
                "manifest": str(args.manifest),
                "fixture_sha256": entry["fixture_sha256"],
                "scene_state_sha256": entry["identity_hashes"]["sim_state"],
                "init_index": entry["init_index"],
                "base_target": entry["base_target"],
                "donor_target": entry["donor_target"],
            }
        )
    if requested is not None and {pair["pair_id"] for pair in selected} != requested:
        missing = sorted(requested - {pair["pair_id"] for pair in selected})
        raise ValueError(f"requested pair IDs absent from manifest: {missing}")
    payload = {
        "schema_version": 1,
        "selection_source": str(args.manifest),
        "selection_reason": args.reason,
        "selection_uses_interventions": False,
        "selected_pairs": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"selected_pairs": len(selected), "reason": args.reason}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
