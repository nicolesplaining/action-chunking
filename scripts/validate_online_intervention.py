#!/usr/bin/env python3
"""Verify online clean endpoints and identity interventions against offline traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from openpi_client import websocket_client_policy

from action_chunking.pairs import load_instruction_pair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--clean-trace", type=Path, required=True)
    parser.add_argument("--noise", type=Path, required=True)
    parser.add_argument("--offline-records", type=Path)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    entry = _manifest_entry(manifest, args.pair_id)
    pair = load_instruction_pair(args.manifest.parent / entry["fixture"])
    with np.load(args.clean_trace) as trace:
        expected = {"base": trace["base_actions"], "donor": trace["donor_actions"]}
    noise = np.load(args.noise)
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    metadata = client.get_server_metadata()
    if not metadata.get("accepts_causal_intervention"):
        raise ValueError("server does not advertise causal interventions")

    records = []
    for side, other in (("base", "donor"), ("donor", "base")):
        request = {**pair.raw_observation(side), "_action_noise": noise}
        clean = _actions(client.infer(request))
        records.append(_record(side, "clean", clean, expected[side]))
        records.append(
            _record(
                side,
                "flow_source_identity",
                _intervene(
                    client,
                    request,
                    getattr(pair, f"{other}_prompt"),
                    {"family": "flow_switch", "switch_after_steps": 10},
                ),
                expected[side],
            )
        )
        records.append(
            _record(
                side,
                "flow_donor_endpoint",
                _intervene(
                    client,
                    request,
                    getattr(pair, f"{other}_prompt"),
                    {"family": "flow_switch", "switch_after_steps": 0},
                ),
                expected[other],
            )
        )
        records.append(
            _record(
                side,
                "residual_identity",
                _intervene(
                    client,
                    request,
                    getattr(pair, f"{side}_prompt"),
                    {"family": "residual_patch", "flow_step": 0, "layer": 0, "action_positions": "all"},
                ),
                expected[side],
            )
        )
        records.append(
            _record(
                side,
                "dimension_identity",
                _intervene(
                    client,
                    request,
                    getattr(pair, f"{side}_prompt"),
                    {
                        "family": "action_dimension_patch",
                        "flow_step": 0,
                        "patched_tensor": "x_t",
                        "action_dimensions": [0, 1, 2],
                    },
                ),
                expected[side],
            )
        )

    if args.offline_records is not None:
        offline = [json.loads(line) for line in args.offline_records.read_text().splitlines()]
        for side, other, direction in (
            ("base", "donor", "base_to_donor"),
            ("donor", "base", "donor_to_base"),
        ):
            request = {**pair.raw_observation(side), "_action_noise": noise}
            residual = _unique_record(
                offline,
                family="residual_patch",
                direction=direction,
                flow_step=9,
                layer=17,
                action_positions="all",
            )
            records.append(
                _record(
                    side,
                    "residual_nonidentity",
                    _intervene(
                        client,
                        request,
                        getattr(pair, f"{other}_prompt"),
                        {"family": "residual_patch", "flow_step": 9, "layer": 17, "action_positions": "all"},
                    ),
                    np.asarray(residual["actions"]),
                )
            )
            dimension = _unique_record(
                offline,
                family="action_dimension_patch",
                direction=direction,
                flow_step=9,
                patched_tensor="x_t",
                action_dimension_group="translation",
            )
            records.append(
                _record(
                    side,
                    "dimension_nonidentity",
                    _intervene(
                        client,
                        request,
                        getattr(pair, f"{other}_prompt"),
                        {
                            "family": "action_dimension_patch",
                            "flow_step": 9,
                            "patched_tensor": "x_t",
                            "action_dimensions": [0, 1, 2],
                        },
                    ),
                    np.asarray(dimension["actions"]),
                )
            )
    summary = {
        "schema_version": 1,
        "pair_id": args.pair_id,
        "server_metadata": {
            key: metadata[key]
            for key in ("flow_steps", "action_expert_layers", "causal_intervention_families")
        },
        "all_exact": all(record["max_abs_error"] == 0.0 for record in records),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["all_exact"] else 1


def _intervene(client: Any, request: dict[str, Any], donor_prompt: str, spec: dict[str, Any]) -> np.ndarray:
    return _actions(client.infer({**request, "_donor_prompt": donor_prompt, "_intervention": spec}))


def _actions(response: dict[str, Any]) -> np.ndarray:
    return np.asarray(response["actions"])


def _record(side: str, control: str, observed: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    return {
        "side": side,
        "control": control,
        "max_abs_error": float(np.max(np.abs(observed - expected))),
    }


def _manifest_entry(manifest: dict[str, Any], pair_id: str) -> dict[str, Any]:
    matches = [entry for entry in manifest["pairs"] if entry["pair_id"] == pair_id]
    if len(matches) != 1:
        raise ValueError(f"expected one manifest entry for {pair_id!r}, found {len(matches)}")
    return matches[0]


def _unique_record(records: list[dict[str, Any]], **fields: Any) -> dict[str, Any]:
    matches = [record for record in records if all(record.get(key) == value for key, value in fields.items())]
    if len(matches) != 1:
        raise ValueError(f"expected one offline record matching {fields}, found {len(matches)}")
    return matches[0]


if __name__ == "__main__":
    raise SystemExit(main())
