#!/usr/bin/env python3
"""Cache exact endpoint rollouts without classifying or selecting candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import run_retarget_eligibility_screen as gate

from action_chunking.pairs import file_digest
from action_chunking.retarget_eligibility import controller_replay_summary_exact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--gate-output", type=Path, required=True)
    parser.add_argument("--source-pair-ids", required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--execution-horizon", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    requested = {value.strip() for value in args.source_pair_ids.split(",") if value.strip()}
    if not requested:
        raise ValueError("source pair ids must be nonempty")
    manifests = sorted(args.candidate_root.glob("**/aligned/manifest.json"))
    selected = []
    observed = set()
    for path in manifests:
        manifest = json.loads(path.read_text())
        entries = manifest.get("pairs", [])
        source_ids = {entry.get("source_pair_id") for entry in entries}
        if len(source_ids) != 1:
            raise ValueError(f"candidate manifest must contain one source pair: {path}")
        source_id = next(iter(source_ids))
        if source_id in requested:
            selected.append((path, manifest))
            observed.add(source_id)
    if observed != requested:
        raise ValueError(f"requested source manifests are incomplete: {sorted(requested - observed)}")

    args.gate_output.mkdir(parents=True, exist_ok=True)
    endpoint_args = SimpleNamespace(
        gpu=args.gpu,
        port=args.port,
        noise_seed=args.noise_seed,
    )
    launcher = Path(__file__).with_name("run_pair_validation.sh")
    records = []
    for manifest_path, manifest in selected:
        if manifest.get("pair_family") != "instruction_target_precontact":
            raise ValueError(f"unexpected pair family in {manifest_path}")
        relative_parent = manifest_path.parent.relative_to(args.candidate_root)
        for entry in manifest["pairs"]:
            event_output = (
                args.gate_output / relative_parent / entry["pair_id"] / "event_horizon"
            )
            summary = gate._run_endpoint(
                launcher,
                manifest_path,
                entry["pair_id"],
                event_output,
                endpoint_args,
                stop_after_contact=True,
                max_steps=args.execution_horizon,
                noise_start_index=int(entry.get("source_replan_index") or 0),
            )
            replay_exact = controller_replay_summary_exact(
                summary, bool(entry.get("controller_replay_required"))
            )
            chunk_exact = gate._source_chunk_exact(entry, event_output)
            input_exact = gate._source_input_exact(
                entry, manifest_path.parent / entry["fixture"]
            )
            if not (replay_exact and chunk_exact and input_exact):
                raise ValueError(f"precomputed endpoint failed exactness controls: {entry['pair_id']}")
            records.append(
                {
                    "pair_id": entry["pair_id"],
                    "source_pair_id": entry["source_pair_id"],
                    "manifest": str(manifest_path),
                    "manifest_sha256": file_digest(manifest_path),
                    "event_summary": str(event_output / "summary.json"),
                    "event_summary_sha256": file_digest(event_output / "summary.json"),
                    "controller_replay_exact": replay_exact,
                    "source_chunk_exact": chunk_exact,
                    "source_input_exact": input_exact,
                }
            )
            _write_audit(args, requested, records)
    return 0


def _write_audit(args: argparse.Namespace, requested: set[str], records: list[dict]) -> None:
    payload = {
        "schema_version": 1,
        "purpose": "outcome_blind_endpoint_cache",
        "selection_uses_continuation_outcomes": False,
        "classifies_behavioral_eligibility": False,
        "requested_source_pair_ids": sorted(requested),
        "completed_endpoints": len(records),
        "noise_seed": args.noise_seed,
        "execution_horizon": args.execution_horizon,
        "records": records,
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
