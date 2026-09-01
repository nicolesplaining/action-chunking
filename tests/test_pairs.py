from __future__ import annotations

import numpy as np
import pytest

from action_chunking.pairs import (
    InstructionPair,
    canonicalize_bddl_scene,
    goal_argument_atoms,
    instruction_difference_role,
    instruction_target_difference,
)


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


def test_target_pose_pair_requires_same_prompt_and_robot_state() -> None:
    pair = make_pair()
    changed_image = pair.donor_image.copy()
    changed_image[0, 0, 0] += 1
    changed_sim_state = pair.donor_sim_state.copy()
    changed_sim_state[0] += 0.02
    pose_pair = InstructionPair(
        **{
            **pair.__dict__,
            "donor_image": changed_image,
            "donor_sim_state": changed_sim_state,
            "donor_prompt": pair.base_prompt,
            "registered_variable": "target_pose",
        }
    )
    pose_pair.validate()
    changed_robot_state = pose_pair.donor_state.copy()
    changed_robot_state[0] += 0.01
    with pytest.raises(ValueError, match="state"):
        InstructionPair(**{**pose_pair.__dict__, "donor_state": changed_robot_state}).validate()
    with pytest.raises(ValueError, match="prompts must match"):
        InstructionPair(**{**pose_pair.__dict__, "donor_prompt": "different"}).validate()


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


def test_instruction_target_difference_requires_one_atom_per_side() -> None:
    alpha = "(define (:obj_of_interest basket alpha) (:goal (And (In alpha basket_region))))"
    beta = "(define (:obj_of_interest basket beta) (:goal (And (In beta basket_region))))"
    assert instruction_target_difference(alpha, beta) == ("alpha", "beta")
    with pytest.raises(ValueError, match="exactly one"):
        instruction_target_difference(
            alpha,
            "(define (:obj_of_interest gamma delta) (:goal (And (In gamma delta_region))))",
        )


def test_instruction_target_difference_rejects_other_goal_changes() -> None:
    alpha = "(define (:obj_of_interest basket alpha) (:goal (And (In alpha basket_region))))"
    beta = "(define (:obj_of_interest basket beta) (:goal (And (On beta basket_region))))"
    with pytest.raises(ValueError, match="beyond one target substitution"):
        instruction_target_difference(alpha, beta)


def test_instruction_difference_role_distinguishes_object_and_destination() -> None:
    target = "(define (:goal (And (In mug basket_region))))"
    destination = "(define (:goal (And (In bowl plate))))"
    assert instruction_difference_role(target, "mug") == "manipulated_object"
    assert instruction_difference_role(destination, "plate") == "destination"


def test_goal_argument_atoms_extracts_goal_roles() -> None:
    text = "(:goal (And (On akita_black_bowl_1 plate_1)))"
    assert goal_argument_atoms(text, 0) == ["akita_black_bowl_1"]
    assert goal_argument_atoms(text, 1) == ["plate_1"]
