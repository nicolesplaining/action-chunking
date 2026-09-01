#!/usr/bin/env python3
"""Run the frozen paired 500-episode pi0.5 early-exit confirmation.

The environment loop is adapted from OpenPI's public ``examples/libero/main.py``
at the repository-pinned revision. This wrapper adds paired condition order,
shared replan-indexed action noise, exact input hashes, and pair-atomic resume.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

SUITE = "libero_goal"
TASKS = 10
TRIALS_PER_TASK = 50
TOTAL_PAIRS = TASKS * TRIALS_PER_TASK
WAIT_STEPS = 10
MAX_ACTION_STEPS = 300
REPLAN_STEPS = 5
RESIZE = 224
NOISE_SHAPE = (10, 32)
DUMMY_ACTION = np.asarray([0.0] * 6 + [-1.0], dtype=np.float64)
CONDITIONS = {"early_exit_7": 7, "full_control_10": 10}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--code-commit", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.port <= 0 or args.seed < 0 or args.noise_seed < 0:
        raise ValueError("port must be positive and seeds must be nonnegative")
    if re.fullmatch(r"[0-9a-f]{40}", args.code_commit) is None:
        raise ValueError("code commit must be a full lowercase Git SHA-1")

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools, websocket_client_policy

    args.output.mkdir(parents=True, exist_ok=True)
    suite = benchmark.get_benchmark_dict()[SUITE]()
    if int(suite.n_tasks) != TASKS:
        raise ValueError("frozen LIBERO Goal confirmation requires exactly ten tasks")
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    jobs: list[dict[str, Any]] = []
    warmup_complete = False
    warmup_sessions = _existing_warmup_sessions(args.output, args.code_commit)

    for task_id in range(TASKS):
        task = suite.get_task(task_id)
        initial_states = suite.get_task_init_states(task_id)
        if len(initial_states) < TRIALS_PER_TASK:
            raise ValueError(f"task {task_id} has fewer than 50 initial states")
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(
            bddl_file_name=bddl,
            camera_heights=256,
            camera_widths=256,
        )
        try:
            if not warmup_complete:
                prepared = _prepare_initial(
                    env,
                    initial_states[0],
                    str(task.language),
                    args.seed,
                    image_tools,
                )
                warmup = _warmup(client, prepared["model_input"], args.noise_seed)
                warmup["session_index"] = len(warmup_sessions)
                warmup["code_commit"] = args.code_commit
                with (args.output / "warmup_sessions.jsonl").open("a") as stream:
                    stream.write(json.dumps(warmup, sort_keys=True) + "\n")
                warmup_sessions.append(warmup)
                warmup_complete = True

            for trial_index in range(TRIALS_PER_TASK):
                pair_key = _pair_key(task_id, trial_index)
                pair_root = args.output / "pairs" / pair_key
                pair_summary_path = pair_root / "pair_summary.json"
                if pair_summary_path.is_file():
                    pair_summary = json.loads(pair_summary_path.read_text())
                    _validate_existing_pair(pair_summary, task_id, trial_index, args.code_commit)
                else:
                    pair_root.mkdir(parents=True, exist_ok=True)
                    order = _condition_order(task_id, trial_index)
                    results = {}
                    for condition in order:
                        result = _run_condition(
                            env=env,
                            initial_state=initial_states[trial_index],
                            task_description=str(task.language),
                            task_id=task_id,
                            trial_index=trial_index,
                            condition=condition,
                            env_seed=args.seed,
                            noise_seed=args.noise_seed,
                            code_commit=args.code_commit,
                            client=client,
                            image_tools=image_tools,
                        )
                        results[condition] = result
                        (pair_root / f"{condition}.json").write_text(
                            json.dumps(result, indent=2, sort_keys=True) + "\n"
                        )
                    pair_summary = _pair_summary(
                        task_id,
                        trial_index,
                        str(task.language),
                        order,
                        results,
                        args.code_commit,
                    )
                    (pair_root / "pair_summary.json").write_text(
                        json.dumps(pair_summary, indent=2, sort_keys=True) + "\n"
                    )
                jobs.append(_job_record(pair_summary, pair_summary_path))
                _write_progress(args, jobs, len(warmup_sessions))
        finally:
            env.close()
    return 0


def _run_condition(
    *,
    env: Any,
    initial_state: Any,
    task_description: str,
    task_id: int,
    trial_index: int,
    condition: str,
    env_seed: int,
    noise_seed: int,
    code_commit: str,
    client: Any,
    image_tools: Any,
) -> dict[str, Any]:
    after_steps = CONDITIONS[condition]
    prepared = _prepare_initial(
        env,
        initial_state,
        task_description,
        env_seed,
        image_tools,
    )
    obs = prepared["observation"]
    action_plan: collections.deque[np.ndarray] = collections.deque()
    diagnostics = []
    noise_hashes = []
    action_hashes = []
    replans = 0
    action_steps = 0
    success = False
    while action_steps < MAX_ACTION_STEPS:
        if not action_plan:
            model_input = _model_input(obs, task_description, image_tools)
            noise = _noise_for_replan(noise_seed, task_id, trial_index, replans)
            response = client.infer(
                {
                    **model_input,
                    "_action_noise": noise,
                    "_intervention": {
                        "after_steps": after_steps,
                        "family": "early_exit",
                        "schema_version": 1,
                        "total_flow_steps": 10,
                    },
                }
            )
            if response.get("intervention_family") != "early_exit":
                raise ValueError("confirmation server returned the wrong intervention family")
            diagnostic = response.get("early_exit_diagnostics")
            _validate_diagnostic(diagnostic, after_steps)
            chunk = np.asarray(response["actions"], dtype=np.float64)
            if chunk.shape != (10, 7) or np.any(~np.isfinite(chunk)):
                raise ValueError("confirmation policy returned an invalid physical action chunk")
            diagnostics.append({"replan_index": replans, **diagnostic})
            noise_hashes.append(_array_digest(noise))
            action_hashes.append(_array_digest(chunk))
            action_plan.extend(chunk[:REPLAN_STEPS])
            replans += 1
        action = np.asarray(action_plan.popleft())
        obs, _, done, _ = env.step(action.tolist())
        action_steps += 1
        if done:
            success = True
            break
    if len(diagnostics) != replans or len(noise_hashes) != replans:
        raise ValueError("confirmation replan accounting is inconsistent")
    return {
        "schema_version": 1,
        "suite": SUITE,
        "task_id": task_id,
        "trial_index": trial_index,
        "task_description": task_description,
        "condition": condition,
        "code_commit": code_commit,
        "environment_seed": env_seed,
        "noise_seed": noise_seed,
        "after_steps": after_steps,
        "total_flow_steps": 10,
        "success": success,
        "action_steps": action_steps,
        "replans": replans,
        "initial_input_sha256": prepared["input_hashes"],
        "initial_sim_state_sha256": prepared["sim_state_sha256"],
        "initial_state_fixture_sha256": _array_digest(np.asarray(initial_state)),
        "noise_sha256_by_replan": noise_hashes,
        "action_sha256_by_replan": action_hashes,
        "early_exit_diagnostics": diagnostics,
    }


def _prepare_initial(
    env: Any,
    initial_state: Any,
    task_description: str,
    env_seed: int,
    image_tools: Any,
) -> dict[str, Any]:
    env.seed(env_seed)
    env.reset()
    obs = env.set_init_state(initial_state)
    for _ in range(WAIT_STEPS):
        obs, _, done, _ = env.step(DUMMY_ACTION.tolist())
        if done:
            raise ValueError("confirmation task succeeded during the fixed wait prefix")
    model_input = _model_input(obs, task_description, image_tools)
    hashes = {key: _array_digest(np.asarray(value)) for key, value in model_input.items() if key != "prompt"}
    hashes["prompt"] = hashlib.sha256(task_description.encode()).hexdigest()
    return {
        "observation": obs,
        "model_input": model_input,
        "input_hashes": hashes,
        "sim_state_sha256": _array_digest(np.asarray(env.get_sim_state())),
    }


def _model_input(obs: dict[str, Any], prompt: str, image_tools: Any) -> dict[str, Any]:
    image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    image = image_tools.convert_to_uint8(image_tools.resize_with_pad(image, RESIZE, RESIZE))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, RESIZE, RESIZE))
    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            _quat2axisangle(np.asarray(obs["robot0_eef_quat"]).copy()),
            obs["robot0_gripper_qpos"],
        )
    )
    return {
        "observation/image": image,
        "observation/wrist_image": wrist,
        "observation/state": state,
        "prompt": prompt,
    }


def _pair_summary(
    task_id: int,
    trial_index: int,
    task_description: str,
    order: list[str],
    results: dict[str, dict[str, Any]],
    code_commit: str,
) -> dict[str, Any]:
    if set(results) != set(CONDITIONS):
        raise ValueError("confirmation pair lacks one condition")
    early = results["early_exit_7"]
    full = results["full_control_10"]
    if early["initial_input_sha256"] != full["initial_input_sha256"]:
        raise ValueError("paired confirmation initial model inputs are not exact")
    if early["initial_sim_state_sha256"] != full["initial_sim_state_sha256"]:
        raise ValueError("paired confirmation simulator states are not exact")
    if early["initial_state_fixture_sha256"] != full["initial_state_fixture_sha256"]:
        raise ValueError("paired confirmation initial-state fixtures are not exact")
    common = min(early["replans"], full["replans"])
    if early["noise_sha256_by_replan"][:common] != full["noise_sha256_by_replan"][:common]:
        raise ValueError("paired confirmation action noise is not shared")
    return {
        "schema_version": 1,
        "suite": SUITE,
        "task_id": task_id,
        "trial_index": trial_index,
        "pair_key": _pair_key(task_id, trial_index),
        "task_description": task_description,
        "code_commit": code_commit,
        "condition_order": order,
        "order_digest_sha256": _order_digest(task_id, trial_index),
        "initial_inputs_exact": True,
        "initial_sim_state_exact": True,
        "shared_noise_common_replans": common,
        "shared_noise_exact": True,
        "early_exit_7": early,
        "full_control_10": full,
        "paired_loss": bool(full["success"] and not early["success"]),
    }


def _warmup(client: Any, model_input: dict[str, Any], noise_seed: int) -> dict[str, Any]:
    records = []
    for condition in ("full_control_10", "early_exit_7"):
        after_steps = CONDITIONS[condition]
        noise = _noise_for_replan(noise_seed, -1, -1, after_steps)
        response = client.infer(
            {
                **model_input,
                "_action_noise": noise,
                "_intervention": {
                    "after_steps": after_steps,
                    "family": "early_exit",
                    "schema_version": 1,
                    "total_flow_steps": 10,
                },
            }
        )
        diagnostic = response.get("early_exit_diagnostics")
        _validate_diagnostic(diagnostic, after_steps)
        records.append({"condition": condition, "diagnostic": diagnostic})
    return {"schema_version": 1, "scored": False, "records": records}


def _write_progress(args: argparse.Namespace, jobs: list[dict[str, Any]], warmup_sessions: int) -> None:
    payload = {
        "schema_version": 1,
        "suite": SUITE,
        "expected_pairs": TOTAL_PAIRS,
        "completed_pairs": len(jobs),
        "seed": args.seed,
        "noise_seed": args.noise_seed,
        "conditions": CONDITIONS,
        "code_commit": args.code_commit,
        "warmup_sessions": warmup_sessions,
        "paired_losses_so_far": sum(job["paired_loss"] for job in jobs),
        "early_exit_successes_so_far": sum(job["early_exit_success"] for job in jobs),
        "full_control_successes_so_far": sum(job["full_control_success"] for job in jobs),
        "jobs": jobs,
    }
    (args.output / "progress.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        f"confirmed {len(jobs)}/{TOTAL_PAIRS}: "
        f"losses={payload['paired_losses_so_far']}, "
        f"k7={payload['early_exit_successes_so_far']}, "
        f"k10={payload['full_control_successes_so_far']}",
        flush=True,
    )


def _job_record(pair: dict[str, Any], path: Path) -> dict[str, Any]:
    return {
        "pair_key": pair["pair_key"],
        "task_id": int(pair["task_id"]),
        "trial_index": int(pair["trial_index"]),
        "condition_order": list(pair["condition_order"]),
        "early_exit_success": bool(pair["early_exit_7"]["success"]),
        "full_control_success": bool(pair["full_control_10"]["success"]),
        "paired_loss": bool(pair["paired_loss"]),
        "pair_summary": str(path),
    }


def _validate_existing_pair(pair: dict[str, Any], task_id: int, trial_index: int, code_commit: str) -> None:
    if (
        pair.get("schema_version") != 1
        or pair.get("suite") != SUITE
        or pair.get("code_commit") != code_commit
        or int(pair.get("task_id", -1)) != task_id
        or int(pair.get("trial_index", -1)) != trial_index
        or pair.get("condition_order") != _condition_order(task_id, trial_index)
        or pair.get("order_digest_sha256") != _order_digest(task_id, trial_index)
        or pair.get("initial_inputs_exact") is not True
        or pair.get("initial_sim_state_exact") is not True
        or pair.get("shared_noise_exact") is not True
    ):
        raise ValueError(f"existing confirmation pair is incompatible: {_pair_key(task_id, trial_index)}")
    for condition, after_steps in CONDITIONS.items():
        result = pair.get(condition)
        if (
            not isinstance(result, dict)
            or result.get("code_commit") != pair.get("code_commit")
            or int(result.get("environment_seed", -1)) != 7
            or int(result.get("noise_seed", -1)) != 0
            or int(result.get("after_steps", -1)) != after_steps
        ):
            raise ValueError("existing confirmation pair has the wrong condition")
        diagnostics = result.get("early_exit_diagnostics", [])
        if len(diagnostics) != int(result.get("replans", -1)) or not diagnostics:
            raise ValueError("existing confirmation pair has incomplete diagnostics")
        for diagnostic in diagnostics:
            _validate_diagnostic(diagnostic, after_steps)
        noise_hashes = result.get("noise_sha256_by_replan", [])
        action_hashes = result.get("action_sha256_by_replan", [])
        if len(noise_hashes) != int(result["replans"]) or len(action_hashes) != int(result["replans"]):
            raise ValueError("existing confirmation pair has incomplete action/noise hashes")
        for replan_index, observed in enumerate(noise_hashes):
            expected = _array_digest(_noise_for_replan(0, task_id, trial_index, replan_index))
            if observed != expected:
                raise ValueError("existing confirmation pair has the wrong action noise")


def _existing_warmup_sessions(output: Path, code_commit: str) -> list[dict[str, Any]]:
    path = output / "warmup_sessions.jsonl"
    if not path.is_file():
        return []
    sessions = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for index, session in enumerate(sessions):
        if (
            int(session.get("session_index", -1)) != index
            or session.get("scored") is not False
            or session.get("code_commit") != code_commit
        ):
            raise ValueError("existing warm-up session log is invalid")
        records = session.get("records", [])
        if [record.get("condition") for record in records] != [
            "full_control_10",
            "early_exit_7",
        ]:
            raise ValueError("existing warm-up session conditions are invalid")
        for record in records:
            _validate_diagnostic(record.get("diagnostic"), CONDITIONS[record["condition"]])
    return sessions


def _validate_diagnostic(diagnostic: Any, after_steps: int) -> None:
    if not isinstance(diagnostic, dict):
        raise ValueError("early-exit diagnostic is missing")
    expected_savings = 10 - after_steps
    if (
        int(diagnostic.get("after_steps", -1)) != after_steps
        or int(diagnostic.get("total_flow_steps", -1)) != 10
        or int(diagnostic.get("velocity_field_evaluations", -1)) != after_steps
        or int(diagnostic.get("velocity_field_evaluation_savings", -1)) != expected_savings
        or float(diagnostic.get("velocity_field_evaluation_savings_fraction", -1.0)) != expected_savings / 10
        or float(diagnostic.get("integration_ms", 0.0)) <= 0.0
    ):
        raise ValueError("early-exit diagnostic fails exact compute or latency accounting")


def _condition_order(task_id: int, trial_index: int) -> list[str]:
    ranked_trials = sorted(
        range(TRIALS_PER_TASK),
        key=lambda value: _order_digest(task_id, value),
    )
    if trial_index in set(ranked_trials[: TRIALS_PER_TASK // 2]):
        return ["early_exit_7", "full_control_10"]
    return ["full_control_10", "early_exit_7"]


def _order_digest(task_id: int, trial_index: int) -> str:
    return hashlib.sha256(f"{SUITE}:{task_id}:{trial_index}".encode()).hexdigest()


def _pair_key(task_id: int, trial_index: int) -> str:
    return f"task_{task_id:02d}_trial_{trial_index:02d}"


def _noise_for_replan(noise_seed: int, task_id: int, trial_index: int, replan_index: int) -> np.ndarray:
    material = f"{noise_seed}:{SUITE}:{task_id}:{trial_index}:{replan_index}".encode()
    seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    return rng.standard_normal(NOISE_SHAPE, dtype=np.float32)


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(denominator), 0.0):
        return np.zeros(3)
    return quat[:3] * 2.0 * math.acos(float(quat[3])) / denominator


if __name__ == "__main__":
    raise SystemExit(main())
