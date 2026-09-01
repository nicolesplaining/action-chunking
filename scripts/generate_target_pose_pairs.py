#!/usr/bin/env python3
"""Generate clean-only target-pose grids from an exact pre-contact state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import generate_postgrasp_instruction_pairs as phase
import numpy as np

from action_chunking.pairs import InstructionPair, array_digest, file_digest, load_instruction_pair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--origin", choices=("base", "donor"), default="base")
    parser.add_argument("--snapshot-step", type=int, required=True)
    parser.add_argument("--target-side", choices=("base", "donor"), default="base")
    parser.add_argument("--offsets", default="0.02,0.04,0.06")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--resize", type=int, default=224)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    offsets = _offsets(args.offsets)
    manifest = json.loads(args.source_manifest.read_text())
    entry = phase._manifest_entry(manifest, args.pair_id)
    source_pair = load_instruction_pair(args.source_manifest.parent / entry["fixture"])
    target = entry[f"{args.target_side}_target"]
    prompt = getattr(source_pair, f"{args.target_side}_prompt")
    bddl_path = Path(manifest["source"][f"{args.target_side}_bddl"])
    state = _trace_state(args.rollout / f"{args.origin}_sim_states.npz", args.snapshot_step)

    base = _restore(bddl_path, state, prompt, target, entry, args)
    direction = np.asarray(base["target_position"][:2]) - np.asarray(base["input"]["state"][:2])
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 1e-12:
        raise ValueError("planar target-pose offset direction is undefined")
    direction /= direction_norm
    args.output.mkdir(parents=True, exist_ok=True)

    output_entries = []
    for offset in offsets:
        donor = _shift_target(
            bddl_path,
            state,
            prompt,
            target,
            direction * offset,
            entry,
            args,
        )
        _validate_configuration(base["configuration"], donor["configuration"])
        pair = InstructionPair(
            base_image=base["input"]["image"],
            base_wrist_image=base["input"]["wrist_image"],
            base_state=base["input"]["state"],
            base_sim_state=base["sim_state"],
            base_prompt=prompt,
            donor_image=donor["input"]["image"],
            donor_wrist_image=donor["input"]["wrist_image"],
            donor_state=donor["input"]["state"],
            donor_sim_state=donor["sim_state"],
            donor_prompt=prompt,
            registered_variable="target_pose",
        )
        pair.validate()
        if base["target_contact"] or donor["target_contact"]:
            raise ValueError("target-pose grid starts after gripper-target contact")

        offset_mm = round(offset * 1000)
        pair_id = f"{entry['pair_id']}_pose_{args.origin}_{args.snapshot_step:03d}_plus_{offset_mm:03d}mm"
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
            registered_variable=np.asarray(pair.registered_variable),
        )
        output_entries.append(
            {
                "pair_id": pair_id,
                "source_pair_id": entry["pair_id"],
                "init_index": entry["init_index"],
                "origin_side": args.origin,
                "snapshot_step": args.snapshot_step,
                "fixture": pair_path.name,
                "fixture_sha256": file_digest(pair_path),
                "registered_variable": "target_pose",
                "base_prompt": prompt,
                "donor_prompt": prompt,
                "base_target": target,
                "donor_target": target,
                "semantic_role": "target_pose",
                "base_manipulated_object": target,
                "donor_manipulated_object": target,
                "pose_offset_m": offset,
                "pose_offset_xy": (direction * offset).tolist(),
                "end_effector_position": pair.base_state[:3].tolist(),
                "base_target_position": base["target_position"],
                "donor_target_position": donor["target_position"],
                "identity_hashes": {
                    "sim_state": array_digest(pair.base_sim_state),
                    "base_sim_state": array_digest(pair.base_sim_state),
                    "donor_sim_state": array_digest(pair.donor_sim_state),
                    "robot_state": array_digest(pair.base_state),
                },
            }
        )

    source = {
        "base_bddl": str(bddl_path),
        "base_bddl_sha256": file_digest(bddl_path),
        "donor_bddl": str(bddl_path),
        "donor_bddl_sha256": file_digest(bddl_path),
        "parent_manifest": str(args.source_manifest),
    }
    output_manifest = {
        "schema_version": 1,
        "pair_family": "target_pose",
        "suite": manifest["suite"],
        "seed": args.seed,
        "source": source,
        "registered_difference": {
            "field": "target_pose",
            "target": target,
            "offset_axis": "planar_target_minus_end_effector",
            "offsets_m": offsets,
            "selection_uses_interventions": False,
        },
        "pairs": output_entries,
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(output_manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(manifest_path), "pairs": len(output_entries)}, indent=2))
    return 0


def _restore(
    bddl_path: Path,
    state: np.ndarray,
    prompt: str,
    target: str,
    entry: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    env = phase._make_env(bddl_path, args)
    try:
        phase._replay_resets(env, entry["init_index"])
        obs = env.regenerate_obs_from_state(state)
        restored = np.asarray(env.get_sim_state()).copy()
        if not np.array_equal(restored, state):
            raise ValueError("base target-pose state changed during restoration")
        return _sample(env, obs, restored, prompt, target, args)
    finally:
        env.close()


def _shift_target(
    bddl_path: Path,
    state: np.ndarray,
    prompt: str,
    target: str,
    offset_xy: np.ndarray,
    entry: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    env = phase._make_env(bddl_path, args)
    try:
        phase._replay_resets(env, entry["init_index"])
        env.regenerate_obs_from_state(state)
        target_object = env.env.get_object(target)
        free_joint = target_object.joints[-1]
        qpos = np.asarray(env.sim.data.get_joint_qpos(free_joint), dtype=np.float64).copy()
        if qpos.shape != (7,):
            raise ValueError(f"target joint {free_joint!r} is not a free joint")
        qpos[:2] += offset_xy
        env.sim.data.set_joint_qpos(free_joint, qpos)
        env.sim.forward()
        donor_state = np.asarray(env.get_sim_state()).copy()
        obs = env.regenerate_obs_from_state(donor_state)
        return _sample(env, obs, donor_state, prompt, target, args)
    finally:
        env.close()


def _sample(
    env: Any,
    obs: dict[str, np.ndarray],
    sim_state: np.ndarray,
    prompt: str,
    target: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "sim_state": sim_state,
        "input": phase._model_input(obs, prompt, args.resize),
        "target_position": phase._object_position(env, target).tolist(),
        "target_contact": phase._gripper_object_contact(env, target),
        "configuration": _configuration(env, target),
    }


def _configuration(env: Any, target: str) -> dict[str, np.ndarray]:
    target_object = env.env.get_object(target)
    free_joint = target_object.joints[-1]
    address = env.sim.model.get_joint_qpos_addr(free_joint)
    if not isinstance(address, tuple) or address[1] - address[0] != 7:
        raise ValueError(f"target joint {free_joint!r} is not a free joint")
    actuator_state = env.sim.data.act
    return {
        "qpos": np.asarray(env.sim.data.qpos, dtype=np.float64).copy(),
        "qvel": np.asarray(env.sim.data.qvel, dtype=np.float64).copy(),
        "act": (
            np.asarray(actuator_state, dtype=np.float64).copy()
            if actuator_state is not None
            else np.empty(0, dtype=np.float64)
        ),
        "target_qpos_indices": np.arange(address[0], address[1], dtype=np.int64),
    }


def _validate_configuration(
    base: dict[str, np.ndarray],
    donor: dict[str, np.ndarray],
) -> None:
    if not np.array_equal(base["target_qpos_indices"], donor["target_qpos_indices"]):
        raise ValueError("target free-joint address differs in target-pose pair")
    target_indices = base["target_qpos_indices"]
    changed_indices = np.flatnonzero(base["qpos"] != donor["qpos"])
    if len(changed_indices) == 0:
        raise ValueError("registered target pose did not change")
    if not np.all(np.isin(changed_indices, target_indices[:2])):
        raise ValueError("generalized positions differ outside the registered planar target pose")
    if not np.array_equal(base["qpos"][target_indices[2:]], donor["qpos"][target_indices[2:]]):
        raise ValueError("target height or orientation changed in target-pose pair")
    for field in ("qvel", "act"):
        if not np.array_equal(base[field], donor[field]):
            raise ValueError(f"simulator {field} differs in target-pose pair")


def _trace_state(path: Path, snapshot_step: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as trace:
        steps = np.asarray(trace["step_indices"], dtype=np.int64)
        states = np.asarray(trace["sim_states"])
    if len(steps) != len(states) or not np.array_equal(steps, np.arange(len(states))):
        raise ValueError("simulator-state trace must contain contiguous zero-based steps")
    if snapshot_step < 0 or snapshot_step >= len(states):
        raise IndexError(f"snapshot step {snapshot_step} is absent from {path}")
    return states[snapshot_step].copy()


def _offsets(value: str) -> list[float]:
    try:
        offsets = sorted({float(item) for item in value.split(",")})
    except ValueError as error:
        raise ValueError("offsets must be comma-separated numbers") from error
    if not offsets or offsets[0] <= 0.0:
        raise ValueError("target-pose offsets must be positive")
    return offsets


if __name__ == "__main__":
    raise SystemExit(main())
