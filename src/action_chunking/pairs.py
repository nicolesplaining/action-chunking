"""Schemas and invariants for minimally different intervention pairs."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def advance_reset_sequence(env: Any, start_index: int) -> None:
    """Match task-local renderer state when generating a nonzero index range."""
    if start_index < 0:
        raise ValueError("start_index must be nonnegative")
    for _ in range(start_index):
        env.reset()


def replan_snapshot_step(contact_step: int, replan_steps: int) -> tuple[int, int]:
    """Return the latest replan boundary before a one-indexed contact step."""
    if contact_step <= 0 or replan_steps <= 0:
        raise ValueError("contact_step and replan_steps must be positive")
    snapshot_step = ((contact_step - 1) // replan_steps) * replan_steps
    return snapshot_step, snapshot_step // replan_steps


def advance_action_noise(rng: Any, replan_index: int, shape: tuple[int, int]) -> None:
    """Advance a NumPy generator to the noise draw at ``replan_index``."""
    if replan_index < 0 or any(value <= 0 for value in shape):
        raise ValueError("replan_index must be nonnegative and noise shape positive")
    for _ in range(replan_index):
        rng.standard_normal(shape, dtype=np.float32)


def load_action_chunk(path: Path, replan_index: int) -> NDArray[np.float64]:
    """Load one saved clean action chunk without changing its numeric values."""
    if replan_index < 0:
        raise ValueError("replan index must be nonnegative")
    chunks = json.loads(path.read_text())
    if replan_index >= len(chunks):
        raise IndexError(f"replan index {replan_index} is absent from {path}")
    chunk = np.asarray(chunks[replan_index], dtype=np.float64)
    if chunk.ndim != 2:
        raise ValueError(f"saved action chunk has invalid shape in {path}")
    return chunk


def load_replan_input(path: Path, replan_index: int) -> dict[str, NDArray[Any]]:
    """Load the exact model observation used for one saved clean replan."""
    if replan_index < 0:
        raise ValueError("replan index must be nonnegative")
    with np.load(path, allow_pickle=False) as trace:
        indices = np.asarray(trace["replan_indices"], dtype=np.int64)
        if not np.array_equal(indices, np.arange(len(indices))):
            raise ValueError("saved replan inputs must have contiguous zero-based indices")
        if replan_index >= len(indices):
            raise IndexError(f"replan index {replan_index} is absent from {path}")
        result = {
            "image": np.asarray(trace["images"][replan_index]).copy(),
            "wrist_image": np.asarray(trace["wrist_images"][replan_index]).copy(),
            "state": np.asarray(trace["states"][replan_index]).copy(),
        }
    if result["image"].ndim != 3 or result["wrist_image"].ndim != 3:
        raise ValueError(f"saved replan input has invalid image shape in {path}")
    if result["state"].ndim != 1:
        raise ValueError(f"saved replan input has invalid state shape in {path}")
    return result


@dataclasses.dataclass(frozen=True)
class InstructionPair:
    """Model-ready inputs for one explicitly registered paired intervention."""

    base_image: NDArray[np.uint8]
    base_wrist_image: NDArray[np.uint8]
    base_state: NDArray[np.floating]
    base_sim_state: NDArray[np.floating]
    base_prompt: str
    donor_image: NDArray[np.uint8]
    donor_wrist_image: NDArray[np.uint8]
    donor_state: NDArray[np.floating]
    donor_sim_state: NDArray[np.floating]
    donor_prompt: str
    registered_variable: str = "instruction"

    def validate(self) -> None:
        """Raise unless inputs obey the registered variable's equality contract."""

        if self.registered_variable == "instruction":
            if self.base_prompt == self.donor_prompt:
                raise ValueError("instruction pair prompts must differ")
            equal_fields = ("image", "wrist_image", "state", "sim_state")
        elif self.registered_variable == "target_pose":
            if self.base_prompt != self.donor_prompt:
                raise ValueError("target-pose pair prompts must match")
            equal_fields = ("state",)
        else:
            raise ValueError(f"unsupported registered pair variable {self.registered_variable!r}")
        for field in equal_fields:
            base = np.asarray(getattr(self, f"base_{field}"))
            donor = np.asarray(getattr(self, f"donor_{field}"))
            if base.dtype != donor.dtype or base.shape != donor.shape or not np.array_equal(base, donor):
                raise ValueError(f"instruction pair differs in {field}")

    def raw_observation(self, side: str) -> dict[str, Any]:
        """Return an official OpenPI LIBERO policy input for one side."""

        if side not in {"base", "donor"}:
            raise ValueError("side must be 'base' or 'donor'")
        return {
            "observation/image": getattr(self, f"{side}_image"),
            "observation/wrist_image": getattr(self, f"{side}_wrist_image"),
            "observation/state": getattr(self, f"{side}_state"),
            "prompt": getattr(self, f"{side}_prompt"),
        }


def array_digest(array: NDArray[Any]) -> str:
    """Hash array identity including dtype and shape, not bytes alone."""

    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode())
    digest.update(json.dumps(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def file_digest(path: Path) -> str:
    """Return the SHA-256 hash of a file without loading it all at once."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_instruction_pair(path: Path) -> InstructionPair:
    """Load and validate a registered pair saved by a LIBERO generator."""

    with np.load(path, allow_pickle=False) as fixture:
        pair = InstructionPair(
            base_image=fixture["base_image"],
            base_wrist_image=fixture["base_wrist_image"],
            base_state=fixture["base_state"],
            base_sim_state=fixture["base_sim_state"],
            base_prompt=str(fixture["base_prompt"].item()),
            donor_image=fixture["donor_image"],
            donor_wrist_image=fixture["donor_wrist_image"],
            donor_state=fixture["donor_state"],
            donor_sim_state=fixture["donor_sim_state"],
            donor_prompt=str(fixture["donor_prompt"].item()),
            registered_variable=(
                str(fixture["registered_variable"].item())
                if "registered_variable" in fixture
                else "instruction"
            ),
        )
    pair.validate()
    return pair


def canonicalize_bddl_scene(text: str) -> str:
    """Remove task semantics from a BDDL problem and canonicalize its scene.

    The language, object-of-interest, and goal clauses may vary in an
    instruction-target pair. Everything else, including region bounds, object
    inventory, and initialization predicates, must be identical.
    """

    scene = text
    for clause in (":language", ":obj_of_interest", ":goal"):
        scene = _remove_balanced_clause(scene, clause)
    return " ".join(scene.split())


def instruction_target_difference(base_text: str, donor_text: str) -> tuple[str, str]:
    """Return the single object-of-interest substitution across task semantics."""

    base_interest = clause_atoms(base_text, ":obj_of_interest")
    donor_interest = clause_atoms(donor_text, ":obj_of_interest")
    base_only = sorted(set(base_interest) - set(donor_interest))
    donor_only = sorted(set(donor_interest) - set(base_interest))
    if len(base_only) != 1 or len(donor_only) != 1:
        raise ValueError("tasks must designate exactly one different object of interest")
    for clause in (":obj_of_interest", ":goal"):
        base_normalized = _replace_atom(_balanced_clause(base_text, clause), base_only[0], "__target__")
        donor_normalized = _replace_atom(_balanced_clause(donor_text, clause), donor_only[0], "__target__")
        if " ".join(base_normalized.split()) != " ".join(donor_normalized.split()):
            raise ValueError(f"tasks differ beyond one target substitution in {clause}")
    return base_only[0], donor_only[0]


def instruction_difference_role(text: str, atom: str) -> str:
    """Classify a substituted goal atom by its argument position."""

    goal = _balanced_clause(text, ":goal")
    positions = []
    for match in re.finditer(r"\(([^()]+)\)", goal):
        tokens = match.group(1).split()
        positions.extend(index for index, token in enumerate(tokens[1:]) if token == atom)
    if not positions:
        raise ValueError(f"goal does not contain substituted atom {atom!r}")
    if all(position == 0 for position in positions):
        return "manipulated_object"
    if all(position > 0 for position in positions):
        return "destination"
    return "mixed"


def goal_argument_atoms(text: str, argument_index: int) -> list[str]:
    """Return unique atoms occupying one zero-based goal-predicate argument position."""

    if argument_index < 0:
        raise ValueError("goal argument index must be nonnegative")
    goal = _balanced_clause(text, ":goal")
    atoms = []
    for match in re.finditer(r"\(([^()]+)\)", goal):
        arguments = match.group(1).split()[1:]
        if argument_index < len(arguments):
            atoms.append(arguments[argument_index])
    return sorted(set(atoms))


def clause_atoms(text: str, clause: str) -> list[str]:
    """Extract whitespace-separated atoms from one balanced BDDL clause."""

    balanced = _balanced_clause(text, clause)
    return balanced[len(clause) + 2 : -1].split()


def _balanced_clause(text: str, clause: str) -> str:
    marker = text.find(f"({clause}")
    if marker < 0:
        raise ValueError(f"missing BDDL clause {clause}")
    depth = 0
    for end in range(marker, len(text)):
        if text[end] == "(":
            depth += 1
        elif text[end] == ")":
            depth -= 1
            if depth == 0:
                return text[marker : end + 1]
    raise ValueError(f"unbalanced BDDL clause {clause}")


def _replace_atom(text: str, atom: str, replacement: str) -> str:
    return re.sub(rf"(?<![\w]){re.escape(atom)}(?![\w])", replacement, text)


def _remove_balanced_clause(text: str, clause: str) -> str:
    search_from = 0
    while True:
        marker = text.find(f"({clause}", search_from)
        if marker < 0:
            return text
        depth = 0
        end = None
        for index in range(marker, len(text)):
            if text[index] == "(":
                depth += 1
            elif text[index] == ")":
                depth -= 1
                if depth == 0:
                    end = index + 1
                    break
        if end is None:
            raise ValueError(f"unbalanced BDDL clause {clause}")
        text = text[:marker] + text[end:]
        search_from = marker
