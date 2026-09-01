#!/usr/bin/env python3
"""Sample an action-only retarget curve and freeze its utility prediction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from openpi_client import websocket_client_policy

from action_chunking.metrics import target_direction_affinity
from action_chunking.pairs import action_noise_shape, advance_action_noise, load_instruction_pair
from action_chunking.utility_prediction import (
    PI05_ACTION_NOISE_SHAPE,
    predict_last_successful_boundary,
    validate_pi05_prediction_arrays,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--new-side", choices=("base", "donor"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--minimum-target-contrast", type=float, default=0.01)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    entry = _manifest_entry(manifest, args.pair_id)
    old_side = "donor" if args.new_side == "base" else "base"
    if entry.get("origin_side") is not None and entry["origin_side"] != old_side:
        raise ValueError("new side does not match the clean-screened pre-contact direction")
    direction = f"{old_side}_to_{args.new_side}"
    pair = load_instruction_pair(args.manifest.parent / entry["fixture"])
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    metadata = client.get_server_metadata()
    if "dynamic_retarget" not in metadata.get("causal_intervention_families", []):
        raise ValueError("server does not advertise dynamic retargeting")
    noise_shape = action_noise_shape(metadata)
    if noise_shape != PI05_ACTION_NOISE_SHAPE:
        raise ValueError(
            "pi0.5 retarget prediction requires action noise shape "
            f"{PI05_ACTION_NOISE_SHAPE}, got {noise_shape}"
        )
    noise_rng = np.random.default_rng(args.noise_seed)
    source_replan_index = int(entry.get("source_replan_index") or 0)
    advance_action_noise(noise_rng, source_replan_index, noise_shape)
    noise = noise_rng.standard_normal(noise_shape, dtype=np.float32)

    actions_by_boundary = {}
    records = []
    source_position = np.asarray(entry[f"{old_side}_target_position"], dtype=np.float64)
    destination_position = np.asarray(entry[f"{args.new_side}_target_position"], dtype=np.float64)
    for boundary in range(11):
        response = _infer(client, pair, args.new_side, boundary, "continue", noise)
        actions = np.asarray(response["actions"])
        actions_by_boundary[boundary] = actions
        records.append(
            {
                "family": "flow_switch",
                "direction": direction,
                "switch_after_steps": boundary,
                "target_direction_affinity": target_direction_affinity(
                    actions,
                    entry["end_effector_position"],
                    source_position,
                    destination_position,
                    executed_horizon=5,
                ),
            }
        )
    restart = np.asarray(_infer(client, pair, args.new_side, 0, "restart", noise)["actions"])
    validate_pi05_prediction_arrays(actions_by_boundary, restart, noise)
    if not np.array_equal(restart, actions_by_boundary[0]):
        raise ValueError("boundary-zero continuation differs from clean restart")
    try:
        prediction = predict_last_successful_boundary(
            records,
            direction,
            threshold=args.threshold,
            minimum_target_contrast=args.minimum_target_contrast,
        )
        prediction["valid"] = True
        prediction["invalid_reason"] = None
    except ValueError as error:
        prediction = {
            "schema_version": 1,
            "direction": direction,
            "metric": "target_direction_affinity",
            "threshold": args.threshold,
            "minimum_target_contrast": args.minimum_target_contrast,
            "valid": False,
            "invalid_reason": str(error),
            "editability_boundary": None,
            "predicted_last_successful_boundary": None,
            "raw_target_direction_affinity": [
                float(record["target_direction_affinity"]) for record in records
            ],
        }
    prediction.update(
        {
            "pair_id": args.pair_id,
            "noise_seed": args.noise_seed,
            "noise_start_index": source_replan_index,
            "action_noise_shape": list(noise_shape),
            "action_shape": list(restart.shape),
            "old_side": old_side,
            "new_side": args.new_side,
            "executed_action_horizon": 5,
            "boundary_zero_restart_exact": True,
            "action_sha256_by_boundary": {
                str(boundary): hashlib.sha256(actions.tobytes()).hexdigest()
                for boundary, actions in actions_by_boundary.items()
            },
            "restart_action_sha256": hashlib.sha256(restart.tobytes()).hexdigest(),
            "action_noise_sha256": hashlib.sha256(noise.tobytes()).hexdigest(),
        }
    )
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "actions.npz",
        **{f"continue_after_{boundary}": actions for boundary, actions in actions_by_boundary.items()},
        restart=restart,
        noise=noise,
    )
    (args.output / "prediction.json").write_text(
        json.dumps(prediction, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(prediction, indent=2, sort_keys=True))
    return 0


def _infer(
    client: Any,
    pair: Any,
    new_side: str,
    boundary: int,
    strategy: str,
    noise: np.ndarray,
) -> dict[str, Any]:
    old_side = "donor" if new_side == "base" else "base"
    response = client.infer(
        {
            "observation/image": getattr(pair, f"{new_side}_image"),
            "observation/wrist_image": getattr(pair, f"{new_side}_wrist_image"),
            "observation/state": getattr(pair, f"{new_side}_state"),
            "prompt": getattr(pair, f"{old_side}_prompt"),
            "_donor_prompt": getattr(pair, f"{new_side}_prompt"),
            "_action_noise": noise,
            "_intervention": {
                "family": "dynamic_retarget",
                "strategy": strategy,
                "switch_after_steps": boundary,
            },
        }
    )
    if response.get("retarget_diagnostics") is None:
        raise ValueError("retarget prediction response omitted diagnostics")
    return response


def _manifest_entry(manifest: dict[str, Any], pair_id: str) -> dict[str, Any]:
    matches = [entry for entry in manifest["pairs"] if entry["pair_id"] == pair_id]
    if len(matches) != 1:
        raise ValueError(f"expected one manifest entry for {pair_id!r}, found {len(matches)}")
    return matches[0]


if __name__ == "__main__":
    raise SystemExit(main())
