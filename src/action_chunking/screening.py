"""Validation and tabulation of clean paired-chunk screens."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def summarize_clean_screen(
    records: list[dict[str, Any]],
    expected_noise_seeds: list[int],
    screen_definition: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build pair- and contrast-level tables without inspecting interventions."""

    if not records:
        raise ValueError("clean screen contains no records")
    expected = set(expected_noise_seeds)
    if not expected or len(expected) != len(expected_noise_seeds):
        raise ValueError("expected noise seeds must be nonempty and unique")

    base_maximum = float(screen_definition["base_affinity_maximum"])
    donor_minimum = float(screen_definition["donor_affinity_minimum"])
    translation_minimum = float(screen_definition["translation_l2_minimum"])
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_pair[record["pair_id"]].append(record)

    pair_rows = []
    invariant_fields = (
        "scene_state_sha256",
        "manifest",
        "fixture_sha256",
        "init_index",
        "base_target",
        "donor_target",
    )
    for pair_id, group in sorted(by_pair.items()):
        seeds = [int(record["noise_seed"]) for record in group]
        if len(seeds) != len(set(seeds)) or set(seeds) != expected:
            raise ValueError(f"{pair_id} does not contain exactly the expected noise seeds")
        for field in invariant_fields:
            if len({record[field] for record in group}) != 1:
                raise ValueError(f"{pair_id} has inconsistent {field}")

        first = group[0]
        base_affinities = [float(record["base_target_direction_affinity"]) for record in group]
        donor_affinities = [float(record["donor_target_direction_affinity"]) for record in group]
        direction_contrasts = [float(record["direction_contrast"]) for record in group]
        translation_contrasts = [
            float(record["endpoint_group_l2_contrasts"]["translation"]) for record in group
        ]
        recomputed_passes = [
            base <= base_maximum and donor >= donor_minimum and translation >= translation_minimum
            for base, donor, translation in zip(
                base_affinities,
                donor_affinities,
                translation_contrasts,
                strict=True,
            )
        ]
        recorded_passes = [bool(record["direction_screen_pass"]) for record in group]
        if recomputed_passes != recorded_passes:
            raise ValueError(f"{pair_id} pass flags disagree with the recorded screen definition")
        pair_rows.append(
            {
                "pair_id": pair_id,
                "scene_state_sha256": first["scene_state_sha256"],
                "manifest": first["manifest"],
                "fixture_sha256": first["fixture_sha256"],
                "init_index": first["init_index"],
                "base_target": first["base_target"],
                "donor_target": first["donor_target"],
                "noise_seeds": ",".join(str(seed) for seed in sorted(seeds)),
                "passing_seeds": sum(recorded_passes),
                "passes_all_seeds": all(recorded_passes),
                "base_affinity_max": max(base_affinities),
                "donor_affinity_min": min(donor_affinities),
                "direction_contrast_mean": sum(direction_contrasts) / len(direction_contrasts),
                "translation_l2_min": min(translation_contrasts),
                "base_affinity_failures": sum(value > base_maximum for value in base_affinities),
                "donor_affinity_failures": sum(value < donor_minimum for value in donor_affinities),
                "translation_l2_failures": sum(value < translation_minimum for value in translation_contrasts),
            }
        )

    by_contrast: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        by_contrast[(row["base_target"], row["donor_target"])].append(row)
    contrast_rows = []
    for (base_target, donor_target), group in sorted(by_contrast.items()):
        contrast_rows.append(
            {
                "base_target": base_target,
                "donor_target": donor_target,
                "pairs": len(group),
                "independent_serialized_states": len({row["scene_state_sha256"] for row in group}),
                "pairs_passing_all_seeds": sum(row["passes_all_seeds"] for row in group),
                "pairs_passing_any_seed": sum(row["passing_seeds"] > 0 for row in group),
                "mean_passing_seeds": sum(row["passing_seeds"] for row in group) / len(group),
            }
        )
    return pair_rows, contrast_rows
