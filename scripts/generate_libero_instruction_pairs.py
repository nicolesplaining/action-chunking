#!/usr/bin/env python3
"""Generate byte-checked prompt-only target pairs from public LIBERO tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools

from action_chunking.pairs import InstructionPair, array_digest, canonicalize_bddl_scene, file_digest

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_90")
    parser.add_argument("--base-task", required=True, help="BDD L filename, with or without .bddl")
    parser.add_argument("--donor-task", required=True, help="BDD L filename, with or without .bddl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--resize", type=int, default=224)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.count <= 0 or args.start_index < 0 or args.settle_steps < 0:
        raise ValueError("count must be positive; indices and settle steps must be nonnegative")

    suite_type = benchmark.get_benchmark_dict()[args.suite]
    task_suite = suite_type()
    base_id, base_task = _resolve_task(task_suite, args.base_task)
    _, donor_task = _resolve_task(task_suite, args.donor_task)
    base_bddl = _bddl_path(base_task)
    donor_bddl = _bddl_path(donor_task)
    base_text = base_bddl.read_text()
    donor_text = donor_bddl.read_text()
    base_scene = canonicalize_bddl_scene(base_text)
    donor_scene = canonicalize_bddl_scene(donor_text)
    if base_scene != donor_scene:
        raise ValueError("BDD L files differ outside language, obj_of_interest, or goal clauses")

    base_targets = _task_specific_targets(base_text, donor_text)
    initial_states = task_suite.get_task_init_states(base_id)
    stop_index = args.start_index + args.count
    if stop_index > len(initial_states):
        raise IndexError(f"requested initial state {stop_index - 1}, but only {len(initial_states)} exist")

    args.output.mkdir(parents=True, exist_ok=True)
    pending = _collect_base_samples(base_task, initial_states, args)
    entries = _validate_with_donor_and_save(donor_task, pending, base_targets, args)
    manifest = {
        "schema_version": 1,
        "pair_family": "instruction_target",
        "suite": args.suite,
        "seed": args.seed,
        "settle_steps": args.settle_steps,
        "source": {
            "base_bddl": str(base_bddl),
            "base_bddl_sha256": file_digest(base_bddl),
            "donor_bddl": str(donor_bddl),
            "donor_bddl_sha256": file_digest(donor_bddl),
            "canonical_scene_sha256": hashlib.sha256(base_scene.encode()).hexdigest(),
        },
        "registered_difference": {
            "field": "prompt_target",
            "base_target": base_targets[0],
            "donor_target": base_targets[1],
        },
        "pairs": entries,
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(manifest_path), "pairs": len(entries)}, indent=2))
    return 0


def _collect_base_samples(task: Any, initial_states: np.ndarray, args: argparse.Namespace) -> list[dict[str, Any]]:
    env = _make_env(task, args.resolution, args.seed)
    samples = []
    try:
        for init_index in range(args.start_index, args.start_index + args.count):
            env.reset()
            obs = env.set_init_state(initial_states[init_index])
            for _ in range(args.settle_steps):
                obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
            sim_state = np.asarray(env.get_sim_state()).copy()
            obs = env.regenerate_obs_from_state(sim_state)
            samples.append(
                {
                    "init_index": init_index,
                    "sim_state": sim_state,
                    "input": _model_input(obs, task.language, args.resize),
                    "object_poses": _object_poses(env),
                }
            )
    finally:
        env.close()
    return samples


def _validate_with_donor_and_save(
    task: Any,
    samples: list[dict[str, Any]],
    targets: tuple[str, str],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    env = _make_env(task, args.resolution, args.seed)
    entries = []
    try:
        env.reset()
        for sample in samples:
            donor_obs = env.regenerate_obs_from_state(sample["sim_state"])
            donor_input = _model_input(donor_obs, task.language, args.resize)
            donor_sim_state = np.asarray(env.get_sim_state()).copy()
            pair = InstructionPair(
                base_image=sample["input"]["image"],
                base_wrist_image=sample["input"]["wrist_image"],
                base_state=sample["input"]["state"],
                base_sim_state=sample["sim_state"],
                base_prompt=sample["input"]["prompt"],
                donor_image=donor_input["image"],
                donor_wrist_image=donor_input["wrist_image"],
                donor_state=donor_input["state"],
                donor_sim_state=donor_sim_state,
                donor_prompt=donor_input["prompt"],
            )
            pair.validate()
            donor_poses = _object_poses(env)
            _assert_pose_maps_equal(sample["object_poses"], donor_poses)
            for target in targets:
                if target not in donor_poses:
                    raise KeyError(f"target {target!r} is absent from object pose map")

            pair_id = f"{args.suite}_{sample['init_index']:03d}_{targets[0]}_to_{targets[1]}"
            pair_path = args.output / f"{pair_id}.npz"
            np.savez_compressed(
                pair_path,
                base_image=pair.base_image,
                base_wrist_image=pair.base_wrist_image,
                base_state=pair.base_state,
                base_sim_state=pair.base_sim_state,
                base_prompt=np.asarray(pair.base_prompt),
                donor_image=pair.donor_image,
                donor_wrist_image=pair.donor_wrist_image,
                donor_state=pair.donor_state,
                donor_sim_state=pair.donor_sim_state,
                donor_prompt=np.asarray(pair.donor_prompt),
            )
            entries.append(
                {
                    "pair_id": pair_id,
                    "init_index": sample["init_index"],
                    "fixture": pair_path.name,
                    "fixture_sha256": file_digest(pair_path),
                    "base_prompt": pair.base_prompt,
                    "donor_prompt": pair.donor_prompt,
                    "base_target": targets[0],
                    "donor_target": targets[1],
                    "end_effector_position": pair.base_state[:3].tolist(),
                    "base_target_position": donor_poses[targets[0]]["pos"],
                    "donor_target_position": donor_poses[targets[1]]["pos"],
                    "identity_hashes": {
                        "image": array_digest(pair.base_image),
                        "wrist_image": array_digest(pair.base_wrist_image),
                        "state": array_digest(pair.base_state),
                        "sim_state": array_digest(pair.base_sim_state),
                        "object_poses": _json_digest(donor_poses),
                    },
                }
            )
    finally:
        env.close()
    return entries


def _resolve_task(task_suite: Any, requested: str) -> tuple[int, Any]:
    requested_name = requested if requested.endswith(".bddl") else requested + ".bddl"
    matches = [(index, task_suite.get_task(index)) for index in range(task_suite.n_tasks)]
    matches = [(index, task) for index, task in matches if task.bddl_file == requested_name]
    if len(matches) != 1:
        raise ValueError(f"expected one task named {requested_name!r}, found {len(matches)}")
    return matches[0]


def _bddl_path(task: Any) -> Path:
    return Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file


def _make_env(task: Any, resolution: int, seed: int) -> OffScreenRenderEnv:
    env = OffScreenRenderEnv(
        bddl_file_name=_bddl_path(task),
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env


def _model_input(obs: dict[str, np.ndarray], prompt: str, resize: int) -> dict[str, Any]:
    image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    return {
        "image": image_tools.convert_to_uint8(image_tools.resize_with_pad(image, resize, resize)),
        "wrist_image": image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, resize, resize)),
        "state": np.concatenate(
            (
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"].copy()),
                obs["robot0_gripper_qpos"],
            )
        ),
        "prompt": str(prompt),
    }


def _object_poses(env: OffScreenRenderEnv) -> dict[str, dict[str, list[float]]]:
    poses = {}
    for name, object_state in env.env.object_states_dict.items():
        try:
            geom = object_state.get_geom_state()
        except (AttributeError, NotImplementedError):
            continue
        if not isinstance(geom, dict) or "pos" not in geom or "quat" not in geom:
            continue
        poses[name] = {
            "pos": np.asarray(geom["pos"], dtype=np.float64).tolist(),
            "quat": np.asarray(geom["quat"], dtype=np.float64).tolist(),
        }
    return poses


def _assert_pose_maps_equal(base: dict[str, Any], donor: dict[str, Any]) -> None:
    if base.keys() != donor.keys():
        raise ValueError("object inventory differs after simulator-state interchange")
    for name in base:
        for field in ("pos", "quat"):
            if not np.array_equal(np.asarray(base[name][field]), np.asarray(donor[name][field])):
                raise ValueError(f"object {name} differs in {field}")


def _task_specific_targets(base_text: str, donor_text: str) -> tuple[str, str]:
    base_interest = _clause_atoms(base_text, ":obj_of_interest")
    donor_interest = _clause_atoms(donor_text, ":obj_of_interest")
    base_only = sorted(set(base_interest) - set(donor_interest))
    donor_only = sorted(set(donor_interest) - set(base_interest))
    if len(base_only) != 1 or len(donor_only) != 1:
        raise ValueError("tasks must designate exactly one different object of interest")
    return base_only[0], donor_only[0]


def _clause_atoms(text: str, clause: str) -> list[str]:
    marker = text.find(f"({clause}")
    if marker < 0:
        raise ValueError(f"missing BDDL clause {clause}")
    depth = 0
    for end in range(marker, len(text)):
        if text[end] == "(":
            depth += 1
        elif text[end] == ")":
            depth -= 1
            if depth == 0:
                body = text[marker + len(clause) + 1 : end]
                return body.split()
    raise ValueError(f"unbalanced BDDL clause {clause}")


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(denominator, 0.0):
        return np.zeros(3)
    return quat[:3] * 2.0 * math.acos(quat[3]) / denominator


def _json_digest(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
