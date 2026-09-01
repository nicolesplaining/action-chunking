from __future__ import annotations

import runpy
from pathlib import Path

_rollout_endpoint = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts" / "run_online_flow_sweep.py")
)["_rollout_endpoint"]


def test_rollout_endpoint_uses_requested_mode_not_success() -> None:
    assert _rollout_endpoint([{"stop_after_first_task_contact": False, "stop_after_registered_destination": False}]) == "full"
    assert (
        _rollout_endpoint([{"stop_after_first_task_contact": True, "stop_after_registered_destination": False}])
        == "first_contact"
    )
    assert (
        _rollout_endpoint([{"stop_after_first_task_contact": False, "stop_after_registered_destination": True}])
        == "destination"
    )
    assert (
        _rollout_endpoint(
            [
                {"stop_after_first_task_contact": False, "stop_after_registered_destination": False},
                {"stop_after_first_task_contact": True, "stop_after_registered_destination": False},
            ]
        )
        == "mixed"
    )
