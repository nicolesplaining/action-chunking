#!/usr/bin/env python3
"""Generate prompt-only destination pairs at deterministic post-grasp states."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools

from action_chunking.pairs import InstructionPair, array_digest, file_digest, load_instruction_pair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--origins", default="base,donor")
    parser.add_argument("--lift-threshold", type=float, default=0.02)
    parser.add_argument("--persistence-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--resize", type=int, default=224)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.lift_threshold <= 0.0 or args.persistence_steps <= 0:
        raise ValueError("lift threshold and persistence steps must be positive")
    origins = [value.strip() for value in args.origins.split(",") if value.strip()]
    if not origins or len(origins) != len(set(origins)) or not set(origins) <= {"base", "donor"}:
        raise ValueError("origins must be a nonempty unique subset of base,donor")

    manifest = json.loads(args.source_manifest.read_text())
    if manifest.get("pair_family") != "instruction_destination":
        raise ValueError("post-grasp generation requires an instruction_destination manifest")
    entry = _manifest_entry(manifest, args.pair_id)
    if entry.get("semantic_role") != "destination":
        raise ValueError("source pair is not registered as a destination substitution")
    manipulated = entry["base_manipulated_object"]
    if manipulated != entry["donor_manipulated_object"]:
        raise ValueError("destination pair must preserve the manipulated object")
    source_pair = load_instruction_pair(args.source_manifest.parent / entry["fixture"])
    bddl_paths = {
        side: Path(manifest["source"][f"{side}_bddl"])
        for side in ("base", "donor")
    }

    args.output.mkdir(parents=True, exist_ok=True)
    selections = {
        origin: _select_snapshot(
            bddl_paths[origin],
            source_pair,
            entry,
            manipulated,
            origin,
            args,
        )
        for origin in origins
    }
    output_entries = []
    for origin in origins:
        selection = selections[origin]
        state = selection.pop("sim_state")
        restored = {
            side: _restore_side(
                side,
                bddl_paths[side],
                state,
                getattr(source_pair, f"{side}_prompt"),
                entry,
                manipulated,
                args,
            )
            for side in ("base", "donor")
        }
        pair = InstructionPair(
            base_image=restored["base"]["input"]["image"],
            base_wrist_image=restored["base"]["input"]["wrist_image"],
            base_state=restored["base"]["input"]["state"],
            base_sim_state=restored["base"]["sim_state"],
            base_prompt=source_pair.base_prompt,
            donor_image=restored["donor"]["input"]["image"],
            donor_wrist_image=restored["donor"]["input"]["wrist_image"],
            donor_state=restored["donor"]["input"]["state"],
            donor_sim_state=restored["donor"]["sim_state"],
            donor_prompt=source_pair.donor_prompt,
        )
        pair.validate()
        _assert_pose_maps_equal(restored["base"]["object_poses"], restored["donor"]["object_poses"])
        if not restored["base"]["manipulated_contact"] or not restored["donor"]["manipulated_contact"]:
            raise ValueError("selected post-grasp state does not preserve gripper-object contact on both sides")

        pair_id = f"{entry['pair_id']}_postgrasp_{origin}_{selection['snapshot_step']:03d}"
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
        output_entries.append(
            {
                "pair_id": pair_id,
                "source_pair_id": entry["pair_id"],
                "init_index": entry["init_index"],
                "origin_side": origin,
                "snapshot_step": selection["snapshot_step"],
                "snapshot_diagnostics": selection,
                "fixture": pair_path.name,
                "fixture_sha256": file_digest(pair_path),
                "base_prompt": pair.base_prompt,
                "donor_prompt": pair.donor_prompt,
                "base_target": entry["base_target"],
                "donor_target": entry["donor_target"],
                "semantic_role": "destination",
                "base_manipulated_object": manipulated,
                "donor_manipulated_object": manipulated,
                "end_effector_position": pair.base_state[:3].tolist(),
                "base_target_position": restored["base"]["target_position"],
                "donor_target_position": restored["donor"]["target_position"],
                "identity_hashes": {
                    "image": array_digest(pair.base_image),
                    "wrist_image": array_digest(pair.base_wrist_image),
                    "state": array_digest(pair.base_state),
                    "sim_state": array_digest(pair.base_sim_state),
                    "object_poses": _json_digest(restored["base"]["object_poses"]),
                },
            }
        )

    output_manifest = {
        "schema_version": 1,
        "pair_family": "instruction_destination_postgrasp",
        "suite": manifest["suite"],
        "seed": args.seed,
        "source": manifest["source"],
        "registered_difference": manifest["registered_difference"],
        "snapshot_rule": {
            "selection_uses_interventions": False,
            "source_rollout": str(args.rollout),
            "origins": origins,
            "earliest_qualifying_state": True,
            "lift_threshold_m": args.lift_threshold,
            "persistence_steps": args.persistence_steps,
            "requires_gripper_object_contact": True,
        },
        "pairs": output_entries,
    }
    output_path = args.output / "manifest.json"
    output_path.write_text(json.dumps(output_manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(output_path), "pairs": len(output_entries)}, indent=2))
    return 0


def _select_snapshot(
    bddl_path: Path,
    pair: InstructionPair,
    entry: dict[str, Any],
    manipulated: str,
    origin: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    trace_path = args.rollout / f"{origin}_sim_states.npz"
    with np.load(trace_path, allow_pickle=False) as trace:
        step_indices = np.asarray(trace["step_indices"], dtype=np.int64)
        states = np.asarray(trace["sim_states"])
    if len(step_indices) != len(states) or not np.array_equal(step_indices, np.arange(len(states))):
        raise ValueError(f"{origin} simulator-state trace must contain contiguous zero-based steps")

    env = _make_env(bddl_path, args)
    try:
        _replay_resets(env, entry["init_index"])
        env.regenerate_obs_from_state(getattr(pair, f"{origin}_sim_state"))
        initial_position = _object_position(env, manipulated)
        diagnostics = []
        for index in range(len(states)):
            step = step_indices[index]
            state = states[index]
            env.regenerate_obs_from_state(state)
            position = _object_position(env, manipulated)
            diagnostics.append(
                {
                    "step": int(step),
                    "object_position": position.tolist(),
                    "object_lift_m": float(position[2] - initial_position[2]),
                    "gripper_object_contact": _gripper_object_contact(env, manipulated),
                }
            )
        qualifies = [
            row["gripper_object_contact"] and row["object_lift_m"] >= args.lift_threshold
            for row in diagnostics
        ]
        diagnostics_path = args.output / f"{origin}_snapshot_diagnostics.jsonl"
        with diagnostics_path.open("w") as stream:
            for row in diagnostics:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        start = next(
            (
                index
                for index in range(len(qualifies) - args.persistence_steps + 1)
                if all(qualifies[index : index + args.persistence_steps])
            ),
            None,
        )
        if start is None:
            raise ValueError(f"{origin} rollout has no persistent lifted-contact post-grasp state")
        selected = diagnostics[start]
        return {
            "origin_side": origin,
            "snapshot_step": selected["step"],
            "object_lift_m": selected["object_lift_m"],
            "object_position": selected["object_position"],
            "persistence_verified_through_step": diagnostics[start + args.persistence_steps - 1]["step"],
            "sim_state": states[start].copy(),
        }
    finally:
        env.close()


def _restore_side(
    side: str,
    bddl_path: Path,
    state: np.ndarray,
    prompt: str,
    entry: dict[str, Any],
    manipulated: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    env = _make_env(bddl_path, args)
    try:
        _replay_resets(env, entry["init_index"])
        obs = env.regenerate_obs_from_state(state)
        restored_state = np.asarray(env.get_sim_state()).copy()
        if not np.array_equal(restored_state, state):
            raise ValueError("post-grasp simulator state changed during restoration")
        poses = _object_poses(env)
        return {
            "sim_state": restored_state,
            "input": _model_input(obs, prompt, args.resize),
            "object_poses": poses,
            "manipulated_contact": _gripper_object_contact(env, manipulated),
            "target_position": _named_position(env, entry[f"{side}_target"], poses),
        }
    finally:
        env.close()


def _manifest_entry(manifest: dict[str, Any], pair_id: str) -> dict[str, Any]:
    matches = [entry for entry in manifest["pairs"] if entry["pair_id"] == pair_id]
    if len(matches) != 1:
        raise ValueError(f"expected one manifest entry for {pair_id!r}, found {len(matches)}")
    return matches[0]


def _make_env(bddl_path: Path, args: argparse.Namespace) -> OffScreenRenderEnv:
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=args.resolution,
        camera_widths=args.resolution,
    )
    env.seed(args.seed)
    return env


def _replay_resets(env: OffScreenRenderEnv, init_index: int) -> None:
    for _ in range(init_index + 1):
        env.reset()


def _gripper_object_contact(env: OffScreenRenderEnv, object_name: str) -> bool:
    gripper_geoms = set(env.robots[0].gripper.contact_geoms)
    object_geoms = set(env.env.get_object(object_name).contact_geoms)
    for contact in env.sim.data.contact[: env.sim.data.ncon]:
        geoms = {
            env.sim.model.geom_id2name(contact.geom1),
            env.sim.model.geom_id2name(contact.geom2),
        }
        if geoms & gripper_geoms and geoms & object_geoms:
            return True
    return False


def _object_position(env: OffScreenRenderEnv, object_name: str) -> np.ndarray:
    geom_names = env.env.get_object(object_name).contact_geoms
    geom_ids = [env.sim.model.geom_name2id(name) for name in geom_names]
    if not geom_ids:
        raise ValueError(f"object {object_name!r} has no contact geoms")
    return np.asarray(env.sim.data.geom_xpos[geom_ids], dtype=np.float64).mean(axis=0)


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


def _named_position(
    env: OffScreenRenderEnv,
    name: str,
    object_poses: dict[str, dict[str, list[float]]],
) -> list[float]:
    if name in object_poses:
        return object_poses[name]["pos"]
    for kind, positions in (
        ("site", env.sim.data.site_xpos),
        ("geom", env.sim.data.geom_xpos),
        ("body", env.sim.data.body_xpos),
    ):
        lookup = getattr(env.sim.model, f"{kind}_name2id")
        try:
            index = lookup(name)
        except (KeyError, TypeError, ValueError):
            continue
        return np.asarray(positions[index], dtype=np.float64).tolist()
    raise KeyError(f"semantic atom {name!r} has no object, site, geom, or body position")


def _assert_pose_maps_equal(base: dict[str, Any], donor: dict[str, Any]) -> None:
    if base.keys() != donor.keys():
        raise ValueError("object inventory differs after post-grasp state interchange")
    for name in base:
        for field in ("pos", "quat"):
            if not np.array_equal(np.asarray(base[name][field]), np.asarray(donor[name][field])):
                raise ValueError(f"object {name} differs in {field}")


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
