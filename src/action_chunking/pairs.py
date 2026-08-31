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


@dataclasses.dataclass(frozen=True)
class InstructionPair:
    """Model-ready inputs for a prompt-only target intervention."""

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

    def validate(self) -> None:
        """Raise unless language is the only model-input or simulator difference."""

        if self.base_prompt == self.donor_prompt:
            raise ValueError("instruction pair prompts must differ")
        equal_fields = (
            "image",
            "wrist_image",
            "state",
            "sim_state",
        )
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
    """Load and validate an instruction pair saved by the LIBERO generator."""

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
