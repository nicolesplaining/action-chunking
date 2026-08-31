#!/usr/bin/env python3
"""Validate both sides of an instruction pair in closed-loop LIBERO rollouts."""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
from typing import Any

import imageio
import numpy as np
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools, websocket_client_policy

from action_chunking.pairs import load_instruction_pair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--resize", type=int, default=224)
    parser.add_argument("--expected-clean-trace", type=Path)
    parser.add_argument("--expected-clean-screen", type=Path)
    parser.add_argument("--initial-input-mode", choices=("strict", "fixture"), default="strict")
    parser.add_argument("--save-sim-states", action="store_true")
    parser.add_argument("--intervention", type=Path, help="JSON intervention specification")
    parser.add_argument("--intervene-replans", default="0", help="Comma-separated zero-based replans or all")
    parser.add_argument("--stop-after-first-task-contact", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    entry = _manifest_entry(manifest, args.pair_id)
    pair = load_instruction_pair(args.manifest.parent / entry["fixture"])
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    server_metadata = client.get_server_metadata()
    if not server_metadata.get("accepts_action_noise"):
        raise ValueError("server does not advertise explicit action-noise support")
    intervention = json.loads(args.intervention.read_text()) if args.intervention is not None else None
    if intervention is not None and not server_metadata.get("accepts_causal_intervention"):
        raise ValueError("server does not advertise causal-intervention support")
    intervention_replans = _replan_selection(args.intervene_replans)

    source_paths = manifest["source"]
    bddl_paths = {
        "base": Path(source_paths["base_bddl"]),
        "donor": Path(source_paths["donor_bddl"]),
    }
    if args.expected_clean_trace is not None and args.expected_clean_screen is not None:
        raise ValueError("provide at most one clean endpoint source")
    if intervention is not None and (args.expected_clean_trace is not None or args.expected_clean_screen is not None):
        raise ValueError("clean endpoint equality cannot be required for an intervened first chunk")
    expected = None
    if args.expected_clean_trace is not None:
        with np.load(args.expected_clean_trace) as trace:
            expected = {"base": trace["base_actions"], "donor": trace["donor_actions"]}
    elif args.expected_clean_screen is not None:
        expected = _clean_screen_actions(args.expected_clean_screen, args.pair_id, args.noise_seed)

    results = []
    for side in ("base", "donor"):
        result = _rollout(
            side,
            bddl_paths[side],
            pair,
            entry,
            client,
            expected,
            intervention,
            intervention_replans,
            args,
        )
        results.append(result)
    summary = {
        "schema_version": 1,
        "pair_id": args.pair_id,
        "noise_seed": args.noise_seed,
        "shared_noise_by_replan_index": True,
        "both_successful": all(result["success"] for result in results),
        "intervention": intervention,
        "intervene_replans": args.intervene_replans if intervention is not None else None,
        "stop_after_first_task_contact": args.stop_after_first_task_contact,
        "results": results,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["both_successful"] else 1


def _rollout(
    side: str,
    bddl_path: Path,
    pair: Any,
    entry: dict[str, Any],
    client: Any,
    expected: dict[str, np.ndarray] | None,
    intervention: dict[str, Any] | None,
    intervention_replans: set[int] | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=args.resolution,
        camera_widths=args.resolution,
    )
    env.seed(7)
    target = entry[f"{side}_target"]
    prompt = getattr(pair, f"{side}_prompt")
    rng = np.random.default_rng(args.noise_seed)
    frames = []
    action_chunks = []
    trajectory_records = []
    simulator_states = []
    contacts: dict[str, int] = {}
    first_chunk_error = None
    applied_replans = []
    terminated_after_first_task_contact = False
    try:
        # Fixture generation advances the seeded environment reset sequence once
        # per initialization index. Replay that sequence because observation-
        # relevant task state is not fully represented by MuJoCo's flat state.
        for _ in range(entry["init_index"] + 1):
            env.reset()
        obs = env.regenerate_obs_from_state(getattr(pair, f"{side}_sim_state"))
        restored_sim_state = np.asarray(env.get_sim_state())
        expected_sim_state = np.asarray(getattr(pair, f"{side}_sim_state"))
        if not np.array_equal(restored_sim_state, expected_sim_state):
            raise ValueError(f"restored {side} rollout differs from fixture in simulator state")
        live_initial = _model_input(obs, prompt, args.resize)
        initial_input_diagnostics = _initial_input_diagnostics(pair, side, live_initial)
        if args.initial_input_mode == "strict" and not all(
            field["array_equal"] for field in initial_input_diagnostics.values()
        ):
            mismatched = [key for key, field in initial_input_diagnostics.items() if not field["array_equal"]]
            raise ValueError(f"restored {side} rollout differs from fixture in {mismatched}")
        fixture_initial = _fixture_model_input(pair, side, prompt)
        if args.save_sim_states:
            simulator_states.append(restored_sim_state.copy())
        action_plan = collections.deque()
        success = False
        replans = 0
        steps = 0
        while steps < args.max_steps:
            model_input = _model_input(obs, prompt, args.resize)
            frames.append(model_input["observation/image"])
            if not action_plan:
                noise = rng.standard_normal((10, 32), dtype=np.float32)
                policy_input = fixture_initial if replans == 0 and args.initial_input_mode == "fixture" else model_input
                request = {**policy_input, "_action_noise": noise}
                if intervention is not None and (intervention_replans is None or replans in intervention_replans):
                    other_side = "donor" if side == "base" else "base"
                    request["_donor_prompt"] = getattr(pair, f"{other_side}_prompt")
                    request["_intervention"] = intervention
                    applied_replans.append(replans)
                response = client.infer(request)
                chunk = np.asarray(response["actions"])
                if replans == 0 and expected is not None:
                    first_chunk_error = float(np.max(np.abs(chunk - expected[side])))
                    if first_chunk_error != 0.0:
                        raise ValueError(f"{side} first chunk differs from offline clean endpoint")
                action_chunks.append(chunk.tolist())
                action_plan.extend(chunk[: args.replan_steps])
                replans += 1
            action = np.asarray(action_plan.popleft())
            obs, _, done, _ = env.step(action.tolist())
            steps += 1
            if args.save_sim_states:
                simulator_states.append(np.asarray(env.get_sim_state()).copy())
            _update_contacts(env, contacts, steps)
            trajectory_records.append(
                {
                    "episode_num": 0 if side == "base" else 1,
                    "task_id": -1,
                    "task_episode_idx": entry["init_index"],
                    "task_description": prompt,
                    "prompt_task_description": prompt,
                    "step_in_episode": steps,
                    "eef_pos": np.asarray(obs["robot0_eef_pos"]).tolist(),
                    "eef_quat": np.asarray(obs["robot0_eef_quat"]).tolist(),
                    "gripper_action": float(action[6]),
                    "gripper_qpos": np.asarray(obs["robot0_gripper_qpos"]).tolist(),
                    "done": bool(done),
                }
            )
            if done:
                success = True
                break
            if args.stop_after_first_task_contact and contacts:
                terminated_after_first_task_contact = True
                break
        imageio.mimwrite(args.output / f"{side}.mp4", frames, fps=10)
        (args.output / f"{side}_actions.json").write_text(json.dumps(action_chunks) + "\n")
        with (args.output / f"{side}_trajectory_records.jsonl").open("w") as stream:
            for record in trajectory_records:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
        if args.save_sim_states:
            np.savez_compressed(
                args.output / f"{side}_sim_states.npz",
                step_indices=np.arange(len(simulator_states), dtype=np.int64),
                sim_states=np.stack(simulator_states),
            )
        return {
            "side": side,
            "target": target,
            "prompt": prompt,
            "success": success,
            "steps": steps,
            "replans": replans,
            "first_chunk_max_abs_error": first_chunk_error,
            "restored_sim_state_max_abs_error": 0.0,
            "initial_input_mode": args.initial_input_mode,
            "live_initial_input_diagnostics": initial_input_diagnostics,
            "saved_simulator_states": len(simulator_states),
            "intervention_replans_applied": applied_replans,
            "terminated_after_first_task_contact": terminated_after_first_task_contact,
            "first_contact_step_by_object": contacts,
            "target_contacted": target in contacts,
        }
    finally:
        env.close()


def _model_input(obs: dict[str, np.ndarray], prompt: str, resize: int) -> dict[str, Any]:
    image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    return {
        "observation/image": image_tools.convert_to_uint8(image_tools.resize_with_pad(image, resize, resize)),
        "observation/wrist_image": image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, resize, resize)),
        "observation/state": np.concatenate(
            (
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"].copy()),
                obs["robot0_gripper_qpos"],
            )
        ),
        "prompt": prompt,
    }


def _initial_input_diagnostics(pair: Any, side: str, model_input: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "observation/image": getattr(pair, f"{side}_image"),
        "observation/wrist_image": getattr(pair, f"{side}_wrist_image"),
        "observation/state": getattr(pair, f"{side}_state"),
    }
    diagnostics = {}
    for key, value in expected.items():
        live = np.asarray(model_input[key])
        reference = np.asarray(value)
        if live.shape != reference.shape or live.dtype != reference.dtype:
            raise ValueError(f"restored {side} rollout has incompatible {key} shape or dtype")
        difference = np.abs(live.astype(np.float64) - reference.astype(np.float64))
        diagnostics[key] = {
            "array_equal": bool(np.array_equal(live, reference)),
            "maximum_absolute_error": float(np.max(difference)),
            "differing_elements": int(np.count_nonzero(difference)),
            "elements": int(difference.size),
        }
    return diagnostics


def _fixture_model_input(pair: Any, side: str, prompt: str) -> dict[str, Any]:
    return {
        "observation/image": getattr(pair, f"{side}_image"),
        "observation/wrist_image": getattr(pair, f"{side}_wrist_image"),
        "observation/state": getattr(pair, f"{side}_state"),
        "prompt": prompt,
    }


def _replan_selection(value: str) -> set[int] | None:
    if value == "all":
        return None
    try:
        replans = {int(item) for item in value.split(",")}
    except ValueError as error:
        raise ValueError("intervene-replans must be all or comma-separated integers") from error
    if not replans or min(replans) < 0:
        raise ValueError("intervene-replans must contain nonnegative integers")
    return replans


def _update_contacts(env: OffScreenRenderEnv, contacts: dict[str, int], step: int) -> None:
    gripper_geoms = set(env.robots[0].gripper.contact_geoms)
    touching_gripper = set()
    for contact in env.sim.data.contact[: env.sim.data.ncon]:
        first = env.sim.model.geom_id2name(contact.geom1)
        second = env.sim.model.geom_id2name(contact.geom2)
        if first in gripper_geoms:
            touching_gripper.add(second)
        if second in gripper_geoms:
            touching_gripper.add(first)
    for name in env.env.objects_dict:
        if name in contacts:
            continue
        object_geoms = set(env.env.get_object(name).contact_geoms)
        if touching_gripper & object_geoms:
            contacts[name] = step


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(denominator, 0.0):
        return np.zeros(3)
    return quat[:3] * 2.0 * math.acos(quat[3]) / denominator


def _manifest_entry(manifest: dict[str, Any], pair_id: str) -> dict[str, Any]:
    matches = [entry for entry in manifest["pairs"] if entry["pair_id"] == pair_id]
    if len(matches) != 1:
        raise ValueError(f"expected one manifest entry for {pair_id!r}, found {len(matches)}")
    return matches[0]


def _clean_screen_actions(path: Path, pair_id: str, noise_seed: int) -> dict[str, np.ndarray]:
    matches = []
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if record["pair_id"] == pair_id and int(record["noise_seed"]) == noise_seed:
            matches.append(record)
    if len(matches) != 1:
        raise ValueError(
            f"expected one clean-screen record for pair {pair_id!r}, seed {noise_seed}; found {len(matches)}"
        )
    return {
        "base": np.asarray(matches[0]["base_actions"]),
        "donor": np.asarray(matches[0]["donor_actions"]),
    }


if __name__ == "__main__":
    raise SystemExit(main())
