#!/usr/bin/env python3
"""Catalog exact-scene instruction-target pairs in a public LIBERO suite."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path

from libero.libero import benchmark, get_libero_path

from action_chunking.pairs import (
    canonicalize_bddl_scene,
    file_digest,
    instruction_difference_role,
    instruction_target_difference,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_90")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    task_suite = benchmark.get_benchmark_dict()[args.suite]()
    tasks = []
    groups: dict[str, list[dict]] = defaultdict(list)
    for task_id in range(task_suite.n_tasks):
        task = task_suite.get_task(task_id)
        path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        text = path.read_text()
        scene_hash = hashlib.sha256(canonicalize_bddl_scene(text).encode()).hexdigest()
        record = {
            "task_id": task_id,
            "task": task.bddl_file,
            "language": str(task.language),
            "bddl": str(path),
            "bddl_sha256": file_digest(path),
            "canonical_scene_sha256": scene_hash,
            "text": text,
        }
        tasks.append(record)
        groups[scene_hash].append(record)

    pairs = []
    for scene_hash, group in sorted(groups.items()):
        for base, donor in itertools.combinations(sorted(group, key=lambda row: row["task_id"]), 2):
            try:
                base_target, donor_target = instruction_target_difference(base["text"], donor["text"])
            except ValueError:
                continue
            base_role = instruction_difference_role(base["text"], base_target)
            donor_role = instruction_difference_role(donor["text"], donor_target)
            semantic_role = base_role if base_role == donor_role else "mixed"
            pairs.append(
                {
                    "canonical_scene_sha256": scene_hash,
                    "base_task_id": base["task_id"],
                    "donor_task_id": donor["task_id"],
                    "base_task": base["task"],
                    "donor_task": donor["task"],
                    "base_language": base["language"],
                    "donor_language": donor["language"],
                    "base_target": base_target,
                    "donor_target": donor_target,
                    "semantic_role": semantic_role,
                    "base_bddl_sha256": base["bddl_sha256"],
                    "donor_bddl_sha256": donor["bddl_sha256"],
                }
            )

    roles = sorted({pair["semantic_role"] for pair in pairs})
    role_counts = {role: sum(pair["semantic_role"] == role for pair in pairs) for role in roles}
    payload = {
        "schema_version": 2,
        "suite": args.suite,
        "tasks": len(tasks),
        "canonical_scenes": len(groups),
        "scenes_with_multiple_tasks": sum(len(group) > 1 for group in groups.values()),
        "exact_single_variable_pairs": len(pairs),
        "exact_instruction_target_pairs": role_counts.get("manipulated_object", 0),
        "exact_instruction_destination_pairs": role_counts.get("destination", 0),
        "semantic_role_counts": role_counts,
        "pairs": pairs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key != "pairs"},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
