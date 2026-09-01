"""Paired scene-cluster comparisons between pi0.5 and matched pi0."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from action_chunking.analysis import benjamini_hochberg, commitment_step


def paired_timing_rows(
    pi05_rows: list[dict[str, Any]], pi0_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Match clean-eligible timing units and collapse repeated modes within scene."""
    fields = ("formation_step", "commitment_step")
    left = _eligible_lookup(pi05_rows, fields)
    right = _eligible_lookup(pi0_rows, fields)
    grouped: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    for key in sorted(set(left) & set(right)):
        pair_id, cluster, _seed, metric = key
        record = {
            "pi05_formation_step": float(left[key]["formation_step"]),
            "pi0_formation_step": float(right[key]["formation_step"]),
            "pi05_editability_boundary": float(left[key]["commitment_step"]),
            "pi0_editability_boundary": float(right[key]["commitment_step"]),
        }
        record["pi05_formation_editability_gap"] = (
            record["pi05_editability_boundary"] - record["pi05_formation_step"]
        )
        record["pi0_formation_editability_gap"] = (
            record["pi0_editability_boundary"] - record["pi0_formation_step"]
        )
        record["pair_id"] = pair_id
        grouped[(cluster, metric)].append(record)

    output = []
    for (cluster, metric), records in sorted(grouped.items()):
        row: dict[str, Any] = {
            "scene_state_sha256": cluster,
            "metric": metric,
            "matched_pair_ids": sorted({str(record["pair_id"]) for record in records}),
        }
        for field in (
            "pi05_formation_step",
            "pi0_formation_step",
            "pi05_editability_boundary",
            "pi0_editability_boundary",
            "pi05_formation_editability_gap",
            "pi0_formation_editability_gap",
        ):
            row[field] = float(np.mean([float(record[field]) for record in records]))
        row["formation_step_difference_pi05_minus_pi0"] = (
            row["pi05_formation_step"] - row["pi0_formation_step"]
        )
        row["editability_boundary_difference_pi05_minus_pi0"] = (
            row["pi05_editability_boundary"] - row["pi0_editability_boundary"]
        )
        row["gap_difference_pi05_minus_pi0"] = (
            row["pi05_formation_editability_gap"] - row["pi0_formation_editability_gap"]
        )
        output.append(row)
    return output


def paired_timing_summary(
    rows: list[dict[str, Any]], *, bootstrap_replicates: int = 10_000, seed: int = 0
) -> dict[str, Any]:
    output = {}
    for metric in sorted({str(row["metric"]) for row in rows}):
        selected = [row for row in rows if row["metric"] == metric]
        output[metric] = {
            field: paired_difference_summary(
                [float(row[field]) for row in selected],
                bootstrap_replicates=bootstrap_replicates,
                seed=seed + offset,
            )
            for offset, field in enumerate(
                (
                    "formation_step_difference_pi05_minus_pi0",
                    "editability_boundary_difference_pi05_minus_pi0",
                    "gap_difference_pi05_minus_pi0",
                )
            )
        }
    return output


def paired_flow_shape_rows(
    pi05_rows: list[dict[str, Any]], pi0_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Compare late weighting, transition width, and swap asymmetry by scene."""
    left = _state_flow_curves(pi05_rows)
    right = _state_flow_curves(pi0_rows)
    output = []
    for key in sorted(set(left) & set(right)):
        cluster, metric = key
        pi05 = _flow_shape(left[key])
        pi0 = _flow_shape(right[key])
        row = {
            "scene_state_sha256": cluster,
            "metric": metric,
        }
        for field in (
            "retention_auc",
            "late_weighting_index",
            "transition_width_10_to_90",
            "directional_asymmetry_auc",
        ):
            row[f"pi05_{field}"] = pi05[field]
            row[f"pi0_{field}"] = pi0[field]
            row[f"{field}_difference_pi05_minus_pi0"] = pi05[field] - pi0[field]
        output.append(row)
    return output


def paired_flow_shape_summary(
    rows: list[dict[str, Any]], *, bootstrap_replicates: int = 10_000, seed: int = 0
) -> dict[str, Any]:
    fields = (
        "retention_auc_difference_pi05_minus_pi0",
        "late_weighting_index_difference_pi05_minus_pi0",
        "transition_width_10_to_90_difference_pi05_minus_pi0",
        "directional_asymmetry_auc_difference_pi05_minus_pi0",
    )
    return {
        metric: {
            field: paired_difference_summary(
                [float(row[field]) for row in rows if row["metric"] == metric],
                bootstrap_replicates=bootstrap_replicates,
                seed=seed + offset,
            )
            for offset, field in enumerate(fields)
        }
        for metric in sorted({str(row["metric"]) for row in rows})
    }


def paired_cell_rows(
    pi05_rows: list[dict[str, Any]],
    pi0_rows: list[dict[str, Any]],
    cell_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Return paired state-level causal effects at common model cells."""
    left = _state_cell_means(pi05_rows, cell_fields)
    right = _state_cell_means(pi0_rows, cell_fields)
    output = []
    for key in sorted(set(left) & set(right)):
        cluster, metric, *cell = key
        output.append(
            {
                "scene_state_sha256": cluster,
                "metric": metric,
                **dict(zip(cell_fields, cell, strict=True)),
                "pi05_symmetric_ncte": left[key],
                "pi0_symmetric_ncte": right[key],
                "difference_pi05_minus_pi0": left[key] - right[key],
            }
        )
    return output


def aggregate_paired_cells(
    rows: list[dict[str, Any]],
    cell_fields: tuple[str, ...],
    *,
    bootstrap_replicates: int = 10_000,
    seed: int = 0,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["metric"], *(row[field] for field in cell_fields))].append(row)
    output = []
    for key, selected in sorted(grouped.items()):
        clusters = [str(row["scene_state_sha256"]) for row in selected]
        if len(clusters) != len(set(clusters)):
            raise ValueError("paired model cell contains duplicate scene clusters")
        differences = np.asarray(
            [float(row["difference_pi05_minus_pi0"]) for row in selected],
            dtype=np.float64,
        )
        indices = rng.integers(
            0, len(differences), size=(bootstrap_replicates, len(differences))
        )
        bootstrap = differences[indices].mean(axis=1)
        record = {
            "metric": key[0],
            **dict(zip(cell_fields, key[1:], strict=True)),
            "eligible_state_clusters": len(differences),
            "mean_pi05_symmetric_ncte": float(
                np.mean([float(row["pi05_symmetric_ncte"]) for row in selected])
            ),
            "mean_pi0_symmetric_ncte": float(
                np.mean([float(row["pi0_symmetric_ncte"]) for row in selected])
            ),
            "mean_difference_pi05_minus_pi0": float(differences.mean()),
            "difference_ci95_low": float(np.quantile(bootstrap, 0.025)),
            "difference_ci95_high": float(np.quantile(bootstrap, 0.975)),
            "p_two_sided_sign_flip": _sign_flip_p(differences, rng, bootstrap_replicates),
        }
        output.append(record)
    for metric in sorted({str(row["metric"]) for row in output}):
        selected = [row for row in output if row["metric"] == metric]
        q_values = benjamini_hochberg(
            [float(row["p_two_sided_sign_flip"]) for row in selected]
        )
        for row, q_value in zip(selected, q_values, strict=True):
            row["q_bh_within_metric_family"] = float(q_value)
    return output


def normalized_position_rows(
    rows: list[dict[str, Any]], action_horizon: int, *, bins: int = 10
) -> list[dict[str, Any]]:
    if action_horizon <= 0 or bins <= 0:
        raise ValueError("action horizon and normalized-position bins must be positive")
    output = []
    for row in rows:
        position = int(row["action_position"])
        if position < 0 or position >= action_horizon:
            raise ValueError("action position lies outside the registered native horizon")
        output.append(
            {
                **row,
                "normalized_position_bin": min(
                    bins - 1, int(position * bins / action_horizon)
                ),
            }
        )
    return output


def paired_difference_summary(
    values: list[float], *, bootstrap_replicates: int = 10_000, seed: int = 0
) -> dict[str, Any]:
    if not values:
        return {
            "eligible_state_clusters": 0,
            "mean_difference": None,
            "ci95_low": None,
            "ci95_high": None,
            "p_two_sided_sign_flip": None,
        }
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(bootstrap_replicates, len(array)))
    bootstrap = array[indices].mean(axis=1)
    return {
        "eligible_state_clusters": len(array),
        "mean_difference": float(array.mean()),
        "ci95_low": float(np.quantile(bootstrap, 0.025)),
        "ci95_high": float(np.quantile(bootstrap, 0.975)),
        "p_two_sided_sign_flip": _sign_flip_p(array, rng, bootstrap_replicates),
    }


def _eligible_lookup(
    rows: list[dict[str, Any]], fields: tuple[str, ...]
) -> dict[tuple[str, str, int, str], dict[str, Any]]:
    result = {}
    for row in rows:
        if not _boolean(row["eligible"]) or any(_missing(row.get(field)) for field in fields):
            continue
        key = (
            str(row["pair_id"]),
            str(row["scene_state_sha256"]),
            int(row["noise_seed"]),
            str(row["metric"]),
        )
        if key in result:
            raise ValueError("duplicate model timing unit")
        result[key] = row
    return result


def _state_cell_means(
    rows: list[dict[str, Any]], cell_fields: tuple[str, ...]
) -> dict[tuple[Any, ...], float]:
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        if not _boolean(row["eligible"]):
            continue
        key = (
            str(row["scene_state_sha256"]),
            str(row["metric"]),
            *(row[field] for field in cell_fields),
        )
        grouped[key].append(float(row["symmetric_ncte"]))
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def _state_flow_curves(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if _boolean(row["eligible"]):
            grouped[
                (
                    str(row["scene_state_sha256"]),
                    str(row["metric"]),
                    int(row["switch_after_steps"]),
                )
            ].append(row)
    by_state: dict[tuple[str, str], dict[int, tuple[float, float]]] = defaultdict(dict)
    for (cluster, metric, boundary), selected in grouped.items():
        by_state[(cluster, metric)][boundary] = (
            float(np.mean([float(row["symmetric_retention"]) for row in selected])),
            float(np.mean([float(row["directional_asymmetry"]) for row in selected])),
        )
    output = {}
    for key, values in by_state.items():
        if set(values) != set(range(11)):
            continue
        output[key] = {
            "retention": np.asarray([values[boundary][0] for boundary in range(11)]),
            "asymmetry": np.asarray([values[boundary][1] for boundary in range(11)]),
        }
    return output


def _flow_shape(curves: dict[str, np.ndarray]) -> dict[str, float]:
    retention = curves["retention"]
    low, _ = commitment_step(retention, 0.1)
    high, _ = commitment_step(retention, 0.9)
    if low is None or high is None:
        raise ValueError("complete eligible flow curve lacks a 10--90% transition")
    auc = float(np.trapz(retention, dx=0.1))
    return {
        "retention_auc": auc,
        "late_weighting_index": 0.5 - auc,
        "transition_width_10_to_90": float(high - low),
        "directional_asymmetry_auc": float(np.trapz(curves["asymmetry"], dx=0.1)),
    }


def _sign_flip_p(
    values: np.ndarray, rng: np.random.Generator, monte_carlo_replicates: int
) -> float:
    if len(values) <= 20:
        assignments = np.arange(1 << len(values), dtype=np.uint64)[:, None]
        bits = (assignments >> np.arange(len(values), dtype=np.uint64)[None, :]) & 1
        signs = 1.0 - 2.0 * bits.astype(np.float64)
        null = (signs * values[None, :]).mean(axis=1)
        return float(np.mean(np.abs(null) >= abs(float(values.mean()))))
    signs = rng.choice((-1.0, 1.0), size=(monte_carlo_replicates, len(values)))
    null = (signs * values[None, :]).mean(axis=1)
    exceedances = int(np.sum(np.abs(null) >= abs(float(values.mean()))))
    return (exceedances + 1) / (monte_carlo_replicates + 1)


def _boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value in {"True", "true", "1", 1}:
        return True
    if value in {"False", "false", "0", 0}:
        return False
    raise ValueError(f"invalid serialized boolean: {value!r}")


def _missing(value: Any) -> bool:
    return value is None or str(value).strip() in {"", "None", "nan"}
