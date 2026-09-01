#!/usr/bin/env python3
"""Generate exact same-task pairs differing only in a distractor obstacle pose."""

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
    parser.add_argument("--target-side", choices=("base", "donor"), default="base")
    parser.add_argument("--obstacle-side", choices=("base", "donor"), default="donor")
    parser.add_argument("--fractions", default="0.35,0.50,0.65")
    parser.add_argument("--lateral-offsets", default="0.00,-0.05,0.05")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--resize", type=int, default=224)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--minimum-object-clearance", type=float, default=0.01)
    parser.add_argument("--minimum-gripper-clearance", type=float, default=0.04)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.target_side == args.obstacle_side:
        raise ValueError("target and obstacle must be different registered objects")
    fractions = _numbers(args.fractions, "fractions")
    laterals = _numbers(args.lateral_offsets, "lateral offsets")
    if any(value <= 0.0 or value >= 1.0 for value in fractions):
        raise ValueError("path fractions must lie strictly between zero and one")
    manifest = json.loads(args.source_manifest.read_text())
    entry = phase._manifest_entry(manifest, args.pair_id)
    if entry.get("semantic_role", "manipulated_object") != "manipulated_object":
        raise ValueError("obstacle-pose generation requires a manipulated-object pair")
    source_pair = load_instruction_pair(args.source_manifest.parent / entry["fixture"])
    target = entry[f"{args.target_side}_target"]
    obstacle = entry[f"{args.obstacle_side}_target"]
    if target == obstacle:
        raise ValueError("target and obstacle object names must differ")
    prompt = getattr(source_pair, f"{args.target_side}_prompt")
    state = np.asarray(getattr(source_pair, f"{args.target_side}_sim_state")).copy()
    bddl_path = Path(manifest["source"][f"{args.target_side}_bddl"])
    base = _restore(bddl_path, state, prompt, target, obstacle, entry, args)

    direction = np.asarray(base["target_position"][:2]) - np.asarray(base["eef_position"][:2])
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        raise ValueError("end-effector-to-target path direction is undefined")
    direction /= norm
    perpendicular = np.asarray([-direction[1], direction[0]])
    args.output.mkdir(parents=True, exist_ok=True)

    output_entries = []
    exclusions = []
    for fraction in fractions:
        for lateral in laterals:
            desired_xy = (
                np.asarray(base["eef_position"][:2])
                + fraction * (
                    np.asarray(base["target_position"][:2])
                    - np.asarray(base["eef_position"][:2])
                )
                + lateral * perpendicular
            )
            candidate_id = _candidate_id(entry["pair_id"], fraction, lateral)
            reason = _geometric_exclusion(base, desired_xy, args)
            if reason is not None:
                exclusions.append({"pair_id": candidate_id, "reason": reason})
                continue
            donor = _place_obstacle(
                bddl_path,
                state,
                prompt,
                target,
                obstacle,
                desired_xy,
                entry,
                args,
            )
            _validate_configuration(base["configuration"], donor["configuration"])
            if donor["gripper_obstacle_contact"]:
                exclusions.append({"pair_id": candidate_id, "reason": "initial_gripper_obstacle_contact"})
                continue
            if donor["target_obstacle_contact"]:
                exclusions.append({"pair_id": candidate_id, "reason": "initial_target_obstacle_contact"})
                continue
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
                registered_variable="obstacle_pose",
            )
            pair.validate()
            pair_path = args.output / f"{candidate_id}.npz"
            _save_pair(pair_path, pair)
            output_entries.append(
                {
                    "pair_id": candidate_id,
                    "source_pair_id": entry["pair_id"],
                    "init_index": int(entry["init_index"]),
                    "fixture": pair_path.name,
                    "fixture_sha256": file_digest(pair_path),
                    "registered_variable": "obstacle_pose",
                    "semantic_role": "obstacle_pose",
                    "base_prompt": prompt,
                    "donor_prompt": prompt,
                    "base_target": target,
                    "donor_target": target,
                    "obstacle": obstacle,
                    "path_fraction": fraction,
                    "lateral_offset_m": lateral,
                    "end_effector_position": base["eef_position"],
                    "base_target_position": base["target_position"],
                    "donor_target_position": donor["target_position"],
                    "base_obstacle_position": base["obstacle_position"],
                    "donor_obstacle_position": donor["obstacle_position"],
                    "obstacle_bounding_radius_m": base["obstacle_bounding_radius_m"],
                    "target_bounding_radius_m": base["target_bounding_radius_m"],
                    "identity_hashes": {
                        "sim_state": array_digest(pair.base_sim_state),
                        "base_sim_state": array_digest(pair.base_sim_state),
                        "donor_sim_state": array_digest(pair.donor_sim_state),
                        "robot_state": array_digest(pair.base_state),
                    },
                }
            )
    output_manifest = {
        "schema_version": 1,
        "pair_family": "obstacle_pose",
        "suite": manifest["suite"],
        "source": {
            "reset_seed": args.seed,
            "base_bddl": str(bddl_path),
            "base_bddl_sha256": file_digest(bddl_path),
            "donor_bddl": str(bddl_path),
            "donor_bddl_sha256": file_digest(bddl_path),
            "parent_manifest": str(args.source_manifest),
            "parent_manifest_sha256": file_digest(args.source_manifest),
        },
        "registered_difference": {
            "field": "obstacle_pose",
            "target": target,
            "obstacle": obstacle,
            "placement_axis": "eef_to_target_path_with_planar_lateral_offset",
            "fractions": fractions,
            "lateral_offsets_m": laterals,
            "minimum_object_clearance_m": args.minimum_object_clearance,
            "minimum_gripper_clearance_m": args.minimum_gripper_clearance,
            "selection_uses_interventions": False,
        },
        "exclusions": exclusions,
        "geometry_exhausted": not output_entries,
        "pairs": output_entries,
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(output_manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "pairs": len(output_entries),
                "geometric_exclusions": len(exclusions),
            },
            indent=2,
        )
    )
    return 0


def _restore(
    bddl_path: Path,
    state: np.ndarray,
    prompt: str,
    target: str,
    obstacle: str,
    entry: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    env = phase._make_env(bddl_path, args)
    try:
        phase._replay_resets(env, entry["init_index"])
        obs = env.regenerate_obs_from_state(state)
        restored = np.asarray(env.get_sim_state()).copy()
        if not np.array_equal(restored, state):
            raise ValueError("base obstacle-pose state changed during restoration")
        return _sample(env, obs, restored, prompt, target, obstacle, args)
    finally:
        env.close()


def _place_obstacle(
    bddl_path: Path,
    state: np.ndarray,
    prompt: str,
    target: str,
    obstacle: str,
    desired_xy: np.ndarray,
    entry: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    env = phase._make_env(bddl_path, args)
    try:
        phase._replay_resets(env, entry["init_index"])
        env.regenerate_obs_from_state(state)
        obstacle_object = env.env.get_object(obstacle)
        free_joint = obstacle_object.joints[-1]
        qpos = np.asarray(env.sim.data.get_joint_qpos(free_joint), dtype=np.float64).copy()
        if qpos.shape != (7,):
            raise ValueError(f"obstacle joint {free_joint!r} is not a free joint")
        qpos[:2] = desired_xy
        env.sim.data.set_joint_qpos(free_joint, qpos)
        env.sim.forward()
        donor_state = np.asarray(env.get_sim_state()).copy()
        obs = env.regenerate_obs_from_state(donor_state)
        return _sample(env, obs, donor_state, prompt, target, obstacle, args)
    finally:
        env.close()


def _sample(
    env: Any,
    obs: dict[str, np.ndarray],
    sim_state: np.ndarray,
    prompt: str,
    target: str,
    obstacle: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "sim_state": sim_state,
        "input": phase._model_input(obs, prompt, args.resize),
        "eef_position": np.asarray(obs["robot0_eef_pos"]).tolist(),
        "target_position": phase._object_position(env, target).tolist(),
        "obstacle_position": phase._object_position(env, obstacle).tolist(),
        "obstacle_bounding_radius_m": _bounding_radius(env, obstacle),
        "target_bounding_radius_m": _bounding_radius(env, target),
        "gripper_obstacle_contact": phase._gripper_object_contact(env, obstacle),
        "target_obstacle_contact": _objects_contact(env, target, obstacle),
        "configuration": _configuration(env, obstacle),
    }


def _configuration(env: Any, obstacle: str) -> dict[str, np.ndarray]:
    obstacle_object = env.env.get_object(obstacle)
    free_joint = obstacle_object.joints[-1]
    address = env.sim.model.get_joint_qpos_addr(free_joint)
    if not isinstance(address, tuple) or address[1] - address[0] != 7:
        raise ValueError(f"obstacle joint {free_joint!r} is not a free joint")
    actuator_state = env.sim.data.act
    return {
        "qpos": np.asarray(env.sim.data.qpos, dtype=np.float64).copy(),
        "qvel": np.asarray(env.sim.data.qvel, dtype=np.float64).copy(),
        "act": (
            np.asarray(actuator_state, dtype=np.float64).copy()
            if actuator_state is not None
            else np.empty(0, dtype=np.float64)
        ),
        "obstacle_qpos_indices": np.arange(address[0], address[1], dtype=np.int64),
    }


def _validate_configuration(base: dict[str, np.ndarray], donor: dict[str, np.ndarray]) -> None:
    indices = base["obstacle_qpos_indices"]
    if not np.array_equal(indices, donor["obstacle_qpos_indices"]):
        raise ValueError("obstacle free-joint address differs across the pose pair")
    changed = np.flatnonzero(base["qpos"] != donor["qpos"])
    if len(changed) == 0 or not np.all(np.isin(changed, indices[:2])):
        raise ValueError("generalized positions differ outside obstacle planar coordinates")
    if not np.array_equal(base["qpos"][indices[2:]], donor["qpos"][indices[2:]]):
        raise ValueError("obstacle height or orientation changed")
    for field in ("qvel", "act"):
        if not np.array_equal(base[field], donor[field]):
            raise ValueError(f"simulator {field} differs in obstacle-pose pair")


def _geometric_exclusion(base: dict[str, Any], desired_xy: np.ndarray, args: argparse.Namespace) -> str | None:
    target_distance = float(
        np.linalg.norm(desired_xy - np.asarray(base["target_position"][:2]))
    )
    required_target = (
        float(base["obstacle_bounding_radius_m"])
        + float(base["target_bounding_radius_m"])
        + args.minimum_object_clearance
    )
    if target_distance <= required_target:
        return "target_obstacle_bounding_spheres_overlap"
    gripper_distance = float(
        np.linalg.norm(desired_xy - np.asarray(base["eef_position"][:2]))
    )
    if gripper_distance <= float(base["obstacle_bounding_radius_m"]) + args.minimum_gripper_clearance:
        return "gripper_obstacle_clearance_below_minimum"
    return None


def _objects_contact(env: Any, first: str, second: str) -> bool:
    first_geoms = set(env.env.get_object(first).contact_geoms)
    second_geoms = set(env.env.get_object(second).contact_geoms)
    for contact in env.sim.data.contact[: env.sim.data.ncon]:
        names = {
            env.sim.model.geom_id2name(contact.geom1),
            env.sim.model.geom_id2name(contact.geom2),
        }
        if names & first_geoms and names & second_geoms:
            return True
    return False


def _bounding_radius(env: Any, name: str) -> float:
    geom_ids = [env.sim.model.geom_name2id(geom) for geom in env.env.get_object(name).contact_geoms]
    return float(np.max(np.asarray(env.sim.model.geom_rbound)[geom_ids]))


def _save_pair(path: Path, pair: InstructionPair) -> None:
    np.savez_compressed(
        path,
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


def _numbers(value: str, name: str) -> list[float]:
    try:
        values = [float(item) for item in value.split(",")]
    except ValueError as error:
        raise ValueError(f"{name} must be comma-separated numbers") from error
    if not values or len(values) != len(set(values)) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite and unique")
    return values


def _candidate_id(source: str, fraction: float, lateral: float) -> str:
    lateral_mm = round(abs(lateral) * 1000)
    sign = "zero" if lateral == 0.0 else ("left" if lateral < 0.0 else "right")
    return f"{source}_obstacle_f{round(fraction * 100):02d}_{sign}_{lateral_mm:03d}mm"


if __name__ == "__main__":
    raise SystemExit(main())
