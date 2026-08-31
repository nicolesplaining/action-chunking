from __future__ import annotations

import numpy as np
import pytest

from action_chunking.pairs import InstructionPair, canonicalize_bddl_scene


def make_pair() -> InstructionPair:
    image = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    state = np.arange(8, dtype=np.float32)
    sim_state = np.arange(20, dtype=np.float64)
    return InstructionPair(
        base_image=image,
        base_wrist_image=image.copy(),
        base_state=state,
        base_sim_state=sim_state,
        base_prompt="pick up alpha",
        donor_image=image.copy(),
        donor_wrist_image=image.copy(),
        donor_state=state.copy(),
        donor_sim_state=sim_state.copy(),
        donor_prompt="pick up beta",
    )


def test_instruction_pair_requires_prompt_only_difference() -> None:
    pair = make_pair()
    pair.validate()
    changed = pair.donor_image.copy()
    changed[0, 0, 0] += 1
    with pytest.raises(ValueError, match="image"):
        InstructionPair(**{**pair.__dict__, "donor_image": changed}).validate()


def test_canonical_bddl_scene_ignores_only_task_semantics() -> None:
    template = """
    (define (problem same)
      (:language {language})
      (:regions (r (:target table) (:ranges (({x} 0 1 1)))))
      (:objects alpha beta - thing)
      (:obj_of_interest {target})
      (:init (On alpha r))
      (:goal (And (In {target} bin))))
    """
    alpha = template.format(language="pick alpha", target="alpha", x=0)
    beta = template.format(language="pick beta", target="beta", x=0)
    moved = template.format(language="pick beta", target="beta", x=0.1)
    assert canonicalize_bddl_scene(alpha) == canonicalize_bddl_scene(beta)
    assert canonicalize_bddl_scene(alpha) != canonicalize_bddl_scene(moved)


def test_canonical_bddl_rejects_unbalanced_clause() -> None:
    with pytest.raises(ValueError, match="unbalanced"):
        canonicalize_bddl_scene("(define (:goal (And x)")
