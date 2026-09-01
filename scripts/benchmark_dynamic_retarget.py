#!/usr/bin/env python3
"""Benchmark post-update latency for continuation versus full restart."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from openpi_client import websocket_client_policy

from action_chunking.pairs import load_instruction_pair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--boundaries", default="0,7,8,9,10")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repeats <= 0 or args.warmup < 0:
        raise ValueError("repeats must be positive and warmup nonnegative")
    boundaries = _boundaries(args.boundaries)
    manifest = json.loads(args.manifest.read_text())
    entry = _manifest_entry(manifest, args.pair_id)
    pair = load_instruction_pair(args.manifest.parent / entry["fixture"])
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    metadata = client.get_server_metadata()
    if "dynamic_retarget" not in metadata.get("causal_intervention_families", []):
        raise ValueError("server does not advertise dynamic retargeting")

    jobs = [
        (side, boundary, strategy)
        for side in ("base", "donor")
        for boundary in boundaries
        for strategy in ("continue", "restart")
    ]
    warmup_noise = np.zeros((10, 32), dtype=np.float32)
    for side, boundary, strategy in jobs:
        for _ in range(args.warmup):
            _infer(client, pair, side, boundary, strategy, warmup_noise)

    trials = [(*job, repeat) for repeat in range(args.repeats) for job in jobs]
    random.Random(args.noise_seed).shuffle(trials)
    rows = []
    for side, boundary, strategy, repeat in trials:
        noise = np.random.default_rng(args.noise_seed + repeat).standard_normal(
            (10, 32), dtype=np.float32
        )
        response, roundtrip_ms = _infer(client, pair, side, boundary, strategy, noise)
        actions = np.asarray(response["actions"])
        diagnostics = response["retarget_diagnostics"]
        rows.append(
            {
                "pair_id": args.pair_id,
                "side": side,
                "repeat": repeat,
                "switch_after_steps": boundary,
                "strategy": strategy,
                "action_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
                "post_event_velocity_evaluations": int(
                    diagnostics["post_event_velocity_evaluations"]
                ),
                "donor_condition_ms": float(diagnostics["donor_condition_ms"]),
                "post_event_integration_ms": float(diagnostics["post_event_integration_ms"]),
                "post_event_total_ms": float(diagnostics["post_event_total_ms"]),
                "client_roundtrip_ms_including_sunk_prefix": roundtrip_ms,
            }
        )
    rows.sort(
        key=lambda row: (
            row["switch_after_steps"],
            row["side"],
            row["repeat"],
            row["strategy"],
        )
    )
    _validate_action_controls(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "latency_trials.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = _summarize(rows, boundaries, args)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _infer(
    client: Any,
    pair: Any,
    new_side: str,
    boundary: int,
    strategy: str,
    noise: np.ndarray,
) -> tuple[dict[str, Any], float]:
    old_side = "donor" if new_side == "base" else "base"
    request = {
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
    start = time.perf_counter()
    response = client.infer(request)
    elapsed_ms = 1000.0 * (time.perf_counter() - start)
    if response.get("retarget_diagnostics") is None:
        raise ValueError("dynamic-retarget benchmark response omitted diagnostics")
    return response, elapsed_ms


def _validate_action_controls(rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["side"], int(row["repeat"]))].append(row)
    for key, selected in grouped.items():
        restart_hashes = {row["action_sha256"] for row in selected if row["strategy"] == "restart"}
        if len(restart_hashes) != 1:
            raise ValueError(f"restart actions differ across event boundaries for {key}")
        continue_zero = {
            row["action_sha256"]
            for row in selected
            if row["strategy"] == "continue" and row["switch_after_steps"] == 0
        }
        if continue_zero != restart_hashes:
            raise ValueError(f"boundary-zero continuation differs from restart for {key}")


def _summarize(
    rows: list[dict[str, Any]], boundaries: list[int], args: argparse.Namespace
) -> dict[str, Any]:
    by_boundary = []
    for boundary in boundaries:
        selected = [row for row in rows if row["switch_after_steps"] == boundary]
        strategies = {
            strategy: [row for row in selected if row["strategy"] == strategy]
            for strategy in ("continue", "restart")
        }
        continue_times = np.asarray(
            [row["post_event_total_ms"] for row in strategies["continue"]], dtype=np.float64
        )
        restart_times = np.asarray(
            [row["post_event_total_ms"] for row in strategies["restart"]], dtype=np.float64
        )
        by_boundary.append(
            {
                "switch_after_steps": boundary,
                "continue_post_event_velocity_evaluations": int(
                    strategies["continue"][0]["post_event_velocity_evaluations"]
                ),
                "restart_post_event_velocity_evaluations": int(
                    strategies["restart"][0]["post_event_velocity_evaluations"]
                ),
                "continue_median_post_event_ms": float(np.median(continue_times)),
                "restart_median_post_event_ms": float(np.median(restart_times)),
                "median_post_event_ms_saved": float(np.median(restart_times - continue_times)),
                "median_post_event_latency_ratio": float(np.median(continue_times / restart_times)),
            }
        )
    return {
        "schema_version": 1,
        "pair_id": args.pair_id,
        "noise_seed": args.noise_seed,
        "repeats_per_side_strategy_boundary": args.repeats,
        "warmup_requests_per_side_strategy_boundary": args.warmup,
        "action_controls_exact": True,
        "primary_latency_excludes_sunk_source_prefix": True,
        "boundaries": by_boundary,
    }


def _boundaries(value: str) -> list[int]:
    try:
        result = sorted({int(item) for item in value.split(",")})
    except ValueError as error:
        raise ValueError("boundaries must be comma-separated integers") from error
    if not result or result[0] != 0 or result[-1] > 10:
        raise ValueError("benchmark boundaries must include zero and lie within [0, 10]")
    return result


def _manifest_entry(manifest: dict[str, Any], pair_id: str) -> dict[str, Any]:
    matches = [entry for entry in manifest["pairs"] if entry["pair_id"] == pair_id]
    if len(matches) != 1:
        raise ValueError(f"expected one manifest entry for {pair_id!r}, found {len(matches)}")
    return matches[0]


if __name__ == "__main__":
    raise SystemExit(main())
