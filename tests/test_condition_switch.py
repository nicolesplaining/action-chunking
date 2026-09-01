from __future__ import annotations

import pytest

from action_chunking.condition_switch import pop_donor_observation


def test_pop_donor_observation_requires_complete_override() -> None:
    request = {
        "prompt": "move safely",
        "_donor_image": "image",
        "_donor_wrist_image": "wrist",
        "_donor_state": "state",
    }

    donor = pop_donor_observation(request)

    assert donor == {
        "observation/image": "image",
        "observation/wrist_image": "wrist",
        "observation/state": "state",
    }
    assert request == {"prompt": "move safely"}

    with pytest.raises(ValueError, match="requires image"):
        pop_donor_observation({"_donor_image": "image"})


def test_pop_donor_observation_allows_prompt_only_switch() -> None:
    request = {"prompt": "new target"}
    assert pop_donor_observation(request) is None
    assert request == {"prompt": "new target"}
