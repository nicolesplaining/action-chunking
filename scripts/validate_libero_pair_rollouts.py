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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    entry = _manifest_entry(manifest, args.pair_id)
    pair = load_instruction_pair(args.manifest.parent / entry["fixture"])
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    if not client.get_server_metadata().get("accepts_action_noise"):
        raise ValueError("server does not advertise explicit action-noise support")

    source_paths = manifest["source"]
    bddl_paths = {
        "base": Path(source_paths["base_bddl"]),
        "donor": Path(source_paths["donor_bddl"]),
    }
    expected = None
    if args.expected_clean_trace is not None:
        with np.load(args.expected_clean_trace) as trace:
            expected = {"base": trace["base_actions"], "donor": trace["donor_actions"]}

    results = []
    for side in ("base", "donor"):
        result = _rollout(side, bddl_paths[side], pair, entry, client, expected, args)
        results.append(result)
    summary = {
        "schema_version": 1,
        "pair_id": args.pair_id,
        "noise_seed": args.noise_seed,
        "shared_noise_by_replan_index": True,
        "both_successful": all(result["success"] for result in results),
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
    contacts: dict[str, int] = {}
    first_chunk_error = None
    try:
        env.reset()
        obs = env.regenerate_obs_from_state(getattr(pair, f"{side}_sim_state"))
        initial = _model_input(obs, prompt, args.resize)
        _assert_initial_input(pair, side, initial)
        action_plan = collections.deque()
        success = False
        replans = 0
        steps = 0
        while steps < args.max_steps:
            model_input = _model_input(obs, prompt, args.resize)
            frames.append(model_input["observation/image"])
            if not action_plan:
                noise = rng.standard_normal((10, 32), dtype=np.float32)
                response = client.infer({**model_input, "_action_noise": noise})
                chunk = np.asarray(response["actions"])
                if replans == 0 and expected is not None:
                    first_chunk_error = float(np.max(np.abs(chunk - expected[side])))
                    if first_chunk_error != 0.0:
                        raise ValueError(f"{side} first chunk differs from offline clean endpoint")
                action_chunks.append(chunk.tolist())
                action_plan.extend(chunk[: args.replan_steps])
                replans += 1
            obs, _, done, _ = env.step(action_plan.popleft().tolist())
            steps += 1
            _update_contacts(env, contacts, steps)
            if done:
                success = True
                break
        imageio.mimwrite(args.output / f"{side}.mp4", frames, fps=10)
        (args.output / f"{side}_actions.json").write_text(json.dumps(action_chunks) + "\n")
        return {
            "side": side,
            "target": target,
            "prompt": prompt,
            "success": success,
            "steps": steps,
            "replans": replans,
            "first_chunk_max_abs_error": first_chunk_error,
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


def _assert_initial_input(pair: Any, side: str, model_input: dict[str, Any]) -> None:
    expected = {
        "observation/image": getattr(pair, f"{side}_image"),
        "observation/wrist_image": getattr(pair, f"{side}_wrist_image"),
        "observation/state": getattr(pair, f"{side}_state"),
    }
    for key, value in expected.items():
        if not np.array_equal(model_input[key], value):
            raise ValueError(f"restored {side} rollout differs from fixture in {key}")


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


if __name__ == "__main__":
    raise SystemExit(main())
