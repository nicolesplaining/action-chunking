#!/usr/bin/env python3
"""Generate pre-contact retargeting candidates from clean dual-success rollouts."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--offsets", default="1,2,3,4,5")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    offsets = sorted({int(value) for value in args.offsets.split(",")})
    if not offsets or offsets[0] <= 0:
        raise ValueError("offsets must be unique positive integers")
    manifest = json.loads(args.source_manifest.read_text())
    entries = manifest.get("pairs", [])
    if not entries:
        raise ValueError("source manifest contains no pairs")
    args.output.mkdir(parents=True, exist_ok=True)
    launcher = Path(__file__).with_name("run_precontact_pair_generation.sh")
    jobs = []
    for entry in entries:
        pair_id = entry["pair_id"]
        rollout = args.rollout_root / pair_id
        summary_path = rollout / "summary.json"
        if not summary_path.is_file():
            jobs.append({"pair_id": pair_id, "status": "missing_clean_rollout"})
            _write_summary(args.output, jobs, len(entries), offsets)
            continue
        summary = json.loads(summary_path.read_text())
        if not _dual_success_target_first(entry, summary):
            jobs.append({"pair_id": pair_id, "status": "clean_endpoint_ineligible"})
            _write_summary(args.output, jobs, len(entries), offsets)
            continue
        generated = []
        for offset in offsets:
            output = args.output / pair_id / f"offset_{offset}"
            output_manifest = output / "manifest.json"
            if not output_manifest.is_file():
                completed = subprocess.run(
                    [
                        str(launcher),
                        str(args.source_manifest),
                        pair_id,
                        str(args.gpu),
                        str(rollout),
                        str(output),
                        str(offset),
                    ],
                    check=False,
                )
                if completed.returncode != 0 or not output_manifest.is_file():
                    raise RuntimeError(f"pre-contact generation failed: {pair_id}, offset {offset}")
            generated.append(str(output_manifest))
        jobs.append(
            {
                "pair_id": pair_id,
                "status": "generated",
                "candidate_manifests": generated,
            }
        )
        _write_summary(args.output, jobs, len(entries), offsets)
    return 0


def _dual_success_target_first(entry: dict, summary: dict) -> bool:
    if summary.get("pair_id") != entry.get("pair_id") or not summary.get("both_successful"):
        return False
    by_side = {result["side"]: result for result in summary["results"]}
    if set(by_side) != {"base", "donor"}:
        return False
    for side in ("base", "donor"):
        contacts = by_side[side].get("first_contact_step_by_object", {})
        if not contacts or min(contacts, key=contacts.get) != entry[f"{side}_target"]:
            return False
    return True


def _write_summary(output: Path, jobs: list[dict], expected: int, offsets: list[int]) -> None:
    payload = {
        "schema_version": 1,
        "selection_uses_intervention_outcomes": False,
        "offsets": offsets,
        "expected_source_pairs": expected,
        "processed_source_pairs": len(jobs),
        "generated_source_pairs": sum(job["status"] == "generated" for job in jobs),
        "jobs": jobs,
    }
    (output / "generation_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"processed {len(jobs)}/{expected}: {payload['generated_source_pairs']} generated",
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
