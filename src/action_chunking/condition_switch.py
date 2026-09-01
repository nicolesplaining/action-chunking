"""Validation helpers for paired full-condition switches."""

from __future__ import annotations

from typing import Any


def pop_donor_observation(request: dict[str, Any]) -> dict[str, Any] | None:
    """Remove an optional complete donor observation from a policy request."""
    internal = {
        "_donor_image": "observation/image",
        "_donor_wrist_image": "observation/wrist_image",
        "_donor_state": "observation/state",
    }
    present = [key in request for key in internal]
    if any(present) and not all(present):
        raise ValueError("donor observation override requires image, wrist image, and state")
    if not any(present):
        return None
    return {external: request.pop(private) for private, external in internal.items()}
