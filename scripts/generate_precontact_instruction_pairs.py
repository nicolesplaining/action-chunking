#!/usr/bin/env python3
"""Generate prompt-only target pairs one action horizon before clean contact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import generate_postgrasp_instruction_pairs as phase
import numpy as np

from action_chunking.pairs import (
    InstructionPair,
    array_digest,
    file_digest,
    load_action_chunk,
    load_instruction_pair,
    load_replan_input,
    replan_snapshot_step,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--origins", default="base,donor")
    parser.add_argument("--precontact-offset", type=int, default=10)
    parser.add_argument("--replan-steps", type=int)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--resize", type=int, default=224)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.precontact_offset <= 0:
        raise ValueError("precontact offset must be positive")
    if args.replan_steps is not None and args.replan_steps <= 0:
        raise ValueError("replan steps must be positive")
    origins = [value.strip() for value in args.origins.split(",") if value.strip()]
    if not origins or len(origins) != len(set(origins)) or not set(origins) <= {"base", "donor"}:
        raise ValueError("origins must be a nonempty unique subset of base,donor")

    manifest = json.loads(args.source_manifest.read_text())
    if manifest.get("pair_family") != "instruction_target":
        raise ValueError("pre-contact generation requires an instruction_target manifest")
    entry = phase._manifest_entry(manifest, args.pair_id)
    if entry.get("semantic_role", "manipulated_object") != "manipulated_object":
        raise ValueError("source pair is not registered as a manipulated-object substitution")
    source_pair = load_instruction_pair(args.source_manifest.parent / entry["fixture"])
    rollout_summary = json.loads((args.rollout / "summary.json").read_text())
    if rollout_summary["pair_id"] != args.pair_id or int(rollout_summary["noise_seed"]) != 0:
        raise ValueError("source rollout does not match the requested seed-0 pair")
    rollout_results = {result["side"]: result for result in rollout_summary["results"]}
    bddl_paths = {
        side: Path(manifest["source"][f"{side}_bddl"])
        for side in ("base", "donor")
    }

    args.output.mkdir(parents=True, exist_ok=True)
    output_entries = []
    for origin in origins:
        target = entry[f"{origin}_target"]
        contacts = rollout_results[origin]["first_contact_step_by_object"]
        if not contacts or min(contacts, key=contacts.get) != target:
            raise ValueError(f"{origin} clean rollout does not first contact its instructed target")
        contact_step = int(contacts[target])
        snapshot_step, source_replan_index = (
            replan_snapshot_step(contact_step, args.replan_steps)
            if args.replan_steps is not None
            else (contact_step - args.precontact_offset, None)
        )
        precontact_offset = contact_step - snapshot_step
        if snapshot_step < 0:
            raise ValueError(f"{origin} contact occurs before the requested pre-contact horizon")
        state = _trace_state(args.rollout / f"{origin}_sim_states.npz", snapshot_step)
        source_action_chunk_sha256 = (
            array_digest(
                load_action_chunk(
                    args.rollout / f"{origin}_actions.json", source_replan_index
                )
            )
            if source_replan_index is not None
            else None
        )
        controller_replay = (
            _controller_replay_trace(
                args.rollout,
                origin,
                snapshot_step,
                source_replan_index,
                args.replan_steps,
            )
            if source_replan_index is not None
            else None
        )
        source_replan_input = (
            load_replan_input(
                args.rollout / f"{origin}_replan_inputs.npz", source_replan_index
            )
            if source_replan_index is not None
            else None
        )
        restored = {
            side: _restore_side(
                side,
                bddl_paths[side],
                state,
                getattr(source_pair, f"{side}_prompt"),
                entry,
                args,
            )
            for side in ("base", "donor")
        }
        pair = InstructionPair(
            base_image=(
                source_replan_input["image"]
                if source_replan_input is not None
                else restored["base"]["input"]["image"]
            ),
            base_wrist_image=(
                source_replan_input["wrist_image"]
                if source_replan_input is not None
                else restored["base"]["input"]["wrist_image"]
            ),
            base_state=(
                source_replan_input["state"]
                if source_replan_input is not None
                else restored["base"]["input"]["state"]
            ),
            base_sim_state=restored["base"]["sim_state"],
            base_prompt=source_pair.base_prompt,
            donor_image=(
                source_replan_input["image"]
                if source_replan_input is not None
                else restored["donor"]["input"]["image"]
            ),
            donor_wrist_image=(
                source_replan_input["wrist_image"]
                if source_replan_input is not None
                else restored["donor"]["input"]["wrist_image"]
            ),
            donor_state=(
                source_replan_input["state"]
                if source_replan_input is not None
                else restored["donor"]["input"]["state"]
            ),
            donor_sim_state=restored["donor"]["sim_state"],
            donor_prompt=source_pair.donor_prompt,
        )
        pair.validate()
        phase._assert_pose_maps_equal(restored["base"]["object_poses"], restored["donor"]["object_poses"])
        if any(restored[side]["registered_target_contact"] for side in ("base", "donor")):
            raise ValueError("pre-contact snapshot already has gripper contact with a registered target")

        pair_id = f"{entry['pair_id']}_precontact_{origin}_{snapshot_step:03d}"
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
            **(
                {
                    "controller_replay_actions": controller_replay["actions"],
                    "controller_replay_sim_states": controller_replay["sim_states"],
                }
                if controller_replay is not None
                else {}
            ),
        )
        output_entries.append(
            {
                "pair_id": pair_id,
                "source_pair_id": entry["pair_id"],
                "init_index": entry["init_index"],
                "origin_side": origin,
                "snapshot_step": snapshot_step,
                "source_contact_step": contact_step,
                "precontact_offset_steps": precontact_offset,
                "source_replan_index": source_replan_index,
                "source_action_chunk_sha256": source_action_chunk_sha256,
                "controller_replay_required": controller_replay is not None,
                "controller_replay_steps": (
                    len(controller_replay["actions"])
                    if controller_replay is not None
                    else 0
                ),
                "controller_replay_actions_sha256": (
                    array_digest(controller_replay["actions"])
                    if controller_replay is not None
                    else None
                ),
                "controller_replay_sim_states_sha256": (
                    array_digest(controller_replay["sim_states"])
                    if controller_replay is not None
                    else None
                ),
                "source_replan_input_sha256": (
                    {
                        key: array_digest(value)
                        for key, value in source_replan_input.items()
                    }
                    if source_replan_input is not None
                    else None
                ),
                "fixture": pair_path.name,
                "fixture_sha256": file_digest(pair_path),
                "base_prompt": pair.base_prompt,
                "donor_prompt": pair.donor_prompt,
                "base_target": entry["base_target"],
                "donor_target": entry["donor_target"],
                "semantic_role": "manipulated_object",
                "base_manipulated_object": entry["base_target"],
                "donor_manipulated_object": entry["donor_target"],
                "end_effector_position": pair.base_state[:3].tolist(),
                "base_target_position": restored["base"]["target_position"],
                "donor_target_position": restored["donor"]["target_position"],
                "identity_hashes": {
                    "image": array_digest(pair.base_image),
                    "wrist_image": array_digest(pair.base_wrist_image),
                    "state": array_digest(pair.base_state),
                    "sim_state": array_digest(pair.base_sim_state),
                    "object_poses": phase._json_digest(restored["base"]["object_poses"]),
                },
            }
        )

    output_manifest = {
        "schema_version": 1,
        "pair_family": "instruction_target_precontact",
        "suite": manifest["suite"],
        "seed": args.seed,
        "source": manifest["source"],
        "registered_difference": manifest["registered_difference"],
        "snapshot_rule": {
            "selection_uses_interventions": False,
            "source_rollout": str(args.rollout),
            "origins": origins,
            "precontact_offset_steps": (
                None if args.replan_steps is not None else args.precontact_offset
            ),
            "replan_steps": args.replan_steps,
            "reference": (
                "latest receding-horizon replan boundary before first instructed-object contact"
                if args.replan_steps is not None
                else "one fixed offset before first instructed-object contact"
            ),
        },
        "pairs": output_entries,
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(output_manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(manifest_path), "pairs": len(output_entries)}, indent=2))
    return 0


def _trace_state(path: Path, snapshot_step: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as trace:
        steps = np.asarray(trace["step_indices"], dtype=np.int64)
        states = np.asarray(trace["sim_states"])
    if len(steps) != len(states) or not np.array_equal(steps, np.arange(len(states))):
        raise ValueError("simulator-state trace must contain contiguous zero-based steps")
    if snapshot_step >= len(states):
        raise IndexError(f"snapshot step {snapshot_step} is absent from {path}")
    return states[snapshot_step].copy()


def _controller_replay_trace(
    rollout: Path,
    side: str,
    snapshot_step: int,
    source_replan_index: int,
    replan_steps: int | None,
) -> dict[str, np.ndarray]:
    """Load the exact prefix needed to reconstruct non-MuJoCo controller state."""
    if replan_steps is None or replan_steps <= 0:
        raise ValueError("controller replay requires positive replan steps")
    chunks = json.loads((rollout / f"{side}_actions.json").read_text())
    if source_replan_index >= len(chunks):
        raise IndexError("source replan index is absent from the clean action trace")
    source_chunk = np.asarray(chunks[source_replan_index], dtype=np.float64)
    if source_chunk.ndim != 2 or source_chunk.shape[0] < replan_steps:
        raise ValueError("clean action chunks must be rank-2 and cover the execution horizon")
    prefixes = [
        np.asarray(chunk, dtype=np.float64)[:replan_steps]
        for chunk in chunks[:source_replan_index]
    ]
    actions = (
        np.concatenate(prefixes, axis=0)
        if prefixes
        else np.empty((0, source_chunk.shape[1]), dtype=np.float64)
    )
    if len(actions) != snapshot_step:
        raise ValueError("controller replay action count must equal the snapshot step")
    with np.load(rollout / f"{side}_sim_states.npz", allow_pickle=False) as trace:
        steps = np.asarray(trace["step_indices"], dtype=np.int64)
        states = np.asarray(trace["sim_states"])
    if not np.array_equal(steps, np.arange(len(states))):
        raise ValueError("simulator-state trace must contain contiguous zero-based steps")
    if snapshot_step >= len(states):
        raise IndexError("controller replay endpoint is absent from simulator-state trace")
    sim_states = states[: snapshot_step + 1].copy()
    return {"actions": actions, "sim_states": sim_states}


def _restore_side(
    side: str,
    bddl_path: Path,
    state: np.ndarray,
    prompt: str,
    entry: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    env = phase._make_env(bddl_path, args)
    try:
        phase._replay_resets(env, entry["init_index"])
        obs = env.regenerate_obs_from_state(state)
        restored_state = np.asarray(env.get_sim_state()).copy()
        if not np.array_equal(restored_state, state):
            raise ValueError("pre-contact simulator state changed during restoration")
        target = entry[f"{side}_target"]
        registered_targets = (entry["base_target"], entry["donor_target"])
        return {
            "sim_state": restored_state,
            "input": phase._model_input(obs, prompt, args.resize),
            "object_poses": phase._object_poses(env),
            "target_position": phase._object_position(env, target).tolist(),
            "registered_target_contact": any(
                phase._gripper_object_contact(env, candidate) for candidate in registered_targets
            ),
        }
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
