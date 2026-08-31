#!/usr/bin/env python3
"""Aggregate closed-loop categorical commitment curves with state-clustered uncertainty."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=17)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.bootstrap_replicates <= 0:
        raise ValueError("bootstrap-replicates must be positive")
    manifest = json.loads(args.manifest.read_text())
    cluster_by_pair = {entry["pair_id"]: entry["identity_hashes"]["sim_state"] for entry in manifest["pairs"]}
    rows = _read_rows(args.input, cluster_by_pair)
    _validate_jobs(rows)
    pair_curves = _pair_curves(rows)
    curve = _curve_summary(pair_curves, args.bootstrap_replicates, args.bootstrap_seed)
    commitments = _commitment_rows(rows)
    summary = _summary(rows, pair_curves, curve, commitments, args)

    args.output.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output / "rollouts.csv", rows)
    _write_csv(args.output / "pair_curves.csv", pair_curves)
    _write_csv(args.output / "commitment_steps.csv", commitments)
    _write_csv(args.output / "curve.csv", curve)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _plot_curve(curve, args.output / "categorical_commitment")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _read_rows(root: Path, cluster_by_pair: dict[str, str]) -> list[dict[str, Any]]:
    rows = []
    paths = sorted(root.glob("*/noise_*/rollouts.csv"))
    if (root / "rollouts.csv").exists():
        paths.append(root / "rollouts.csv")
    for path in paths:
        with path.open(newline="") as stream:
            for raw in csv.DictReader(stream):
                pair_id = raw["pair_id"]
                if pair_id not in cluster_by_pair:
                    raise ValueError(f"pair {pair_id!r} is absent from the manifest")
                rows.append(
                    {
                        "pair_id": pair_id,
                        "scene_state_sha256": cluster_by_pair[pair_id],
                        "noise_seed": int(raw["noise_seed"]),
                        "switch_after_steps": int(raw["switch_after_steps"]),
                        "side": raw["side"],
                        "source_target": raw["source_target"],
                        "donor_target": raw["donor_target"],
                        "first_contact_object": raw["first_contact_object"] or None,
                        "first_contact_step": int(raw["first_contact_step"]) if raw["first_contact_step"] else None,
                        "first_contact_is_source": _boolean(raw["first_contact_is_source"]),
                        "first_contact_is_donor": _boolean(raw["first_contact_is_donor"]),
                        "initial_input_exact": _boolean(raw["initial_input_exact"]),
                        "simulator_state_exact": _boolean(raw["simulator_state_exact"]),
                    }
                )
    if not rows:
        raise ValueError("no completed online flow sweep tables found")
    return rows


def _validate_jobs(rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["pair_id"], row["noise_seed"])].append(row)
    expected = {(boundary, side) for boundary in range(11) for side in ("base", "donor")}
    for job, selected in grouped.items():
        observed = {(row["switch_after_steps"], row["side"]) for row in selected}
        if observed != expected or len(selected) != len(expected):
            raise ValueError(f"flow job {job} does not contain one row per boundary and direction")
        if not all(row["initial_input_exact"] and row["simulator_state_exact"] for row in selected):
            raise ValueError(f"flow job {job} failed exact restoration controls")
        by_side = defaultdict(dict)
        for row in selected:
            by_side[row["side"]][row["switch_after_steps"]] = row
        for side, curve in by_side.items():
            if not curve[0]["first_contact_is_donor"]:
                raise ValueError(f"flow job {job} side {side} failed the full-donor positive control")
            if not curve[10]["first_contact_is_source"]:
                raise ValueError(f"flow job {job} side {side} failed the full-source identity control")


def _pair_curves(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["pair_id"], row["scene_state_sha256"], row["switch_after_steps"])].append(row)
    result = []
    for (pair_id, cluster, boundary), selected in sorted(grouped.items()):
        result.append(
            {
                "pair_id": pair_id,
                "scene_state_sha256": cluster,
                "switch_after_steps": boundary,
                "side_seed_observations": len(selected),
                "source_retention": float(np.mean([row["first_contact_is_source"] for row in selected])),
                "donor_transfer": float(np.mean([row["first_contact_is_donor"] for row in selected])),
                "neither_target": float(
                    np.mean(
                        [
                            not row["first_contact_is_source"] and not row["first_contact_is_donor"]
                            for row in selected
                        ]
                    )
                ),
            }
        )
    return result


def _curve_summary(pair_curves: list[dict[str, Any]], replicates: int, seed: int) -> list[dict[str, Any]]:
    clusters = sorted({row["scene_state_sha256"] for row in pair_curves})
    rng = np.random.default_rng(seed)
    output = []
    for boundary in range(11):
        selected = [row for row in pair_curves if row["switch_after_steps"] == boundary]
        by_cluster = {row["scene_state_sha256"]: row for row in selected}
        if set(by_cluster) != set(clusters):
            raise ValueError(f"boundary {boundary} is missing one or more state clusters")
        source = np.asarray([by_cluster[cluster]["source_retention"] for cluster in clusters])
        donor = np.asarray([by_cluster[cluster]["donor_transfer"] for cluster in clusters])
        sample_indices = rng.integers(0, len(clusters), size=(replicates, len(clusters)))
        source_bootstrap = source[sample_indices].mean(axis=1)
        donor_bootstrap = donor[sample_indices].mean(axis=1)
        output.append(
            {
                "switch_after_steps": boundary,
                "normalized_switch_time": boundary / 10.0,
                "source_retention": float(source.mean()),
                "source_retention_ci95_low": float(np.quantile(source_bootstrap, 0.025)),
                "source_retention_ci95_high": float(np.quantile(source_bootstrap, 0.975)),
                "donor_transfer": float(donor.mean()),
                "donor_transfer_ci95_low": float(np.quantile(donor_bootstrap, 0.025)),
                "donor_transfer_ci95_high": float(np.quantile(donor_bootstrap, 0.975)),
            }
        )
    return output


def _commitment_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["pair_id"], row["scene_state_sha256"], row["noise_seed"], row["side"])
        grouped[key].append(row)
    result = []
    for (pair_id, cluster, noise_seed, side), selected in sorted(grouped.items()):
        curve = {row["switch_after_steps"]: row["first_contact_is_source"] for row in selected}
        commitment = next(
            boundary for boundary in range(11) if all(curve[later] for later in range(boundary, 11))
        )
        result.append(
            {
                "pair_id": pair_id,
                "scene_state_sha256": cluster,
                "noise_seed": noise_seed,
                "side": side,
                "categorical_commitment_step": commitment,
                "monotonic_source_retention": all(
                    not curve[boundary] or curve[boundary + 1] for boundary in range(10)
                ),
            }
        )
    return result


def _summary(
    rows: list[dict[str, Any]],
    pair_curves: list[dict[str, Any]],
    curve: list[dict[str, Any]],
    commitments: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    commitment_values = np.asarray([row["categorical_commitment_step"] for row in commitments])
    source_curve = np.asarray([row["source_retention"] for row in curve])
    return {
        "schema_version": 1,
        "pairs": len({row["pair_id"] for row in rows}),
        "state_clusters": len({row["scene_state_sha256"] for row in rows}),
        "noise_seeds": sorted({row["noise_seed"] for row in rows}),
        "direction_seed_curves": len(commitments),
        "full_donor_positive_control_rate": curve[0]["donor_transfer"],
        "full_source_identity_control_rate": curve[-1]["source_retention"],
        "categorical_commitment_step_median": float(np.median(commitment_values)),
        "categorical_commitment_step_q25": float(np.quantile(commitment_values, 0.25)),
        "categorical_commitment_step_q75": float(np.quantile(commitment_values, 0.75)),
        "monotonic_direction_seed_curves": sum(row["monotonic_source_retention"] for row in commitments),
        "source_retention_auc": float(np.trapz(source_curve, dx=0.1)),
        "bootstrap": {
            "unit": "scene_state_sha256",
            "replicates": args.bootstrap_replicates,
            "seed": args.bootstrap_seed,
        },
        "pair_curve_rows": len(pair_curves),
    }


def _plot_curve(rows: list[dict[str, Any]], stem: Path) -> None:
    x = np.asarray([row["switch_after_steps"] for row in rows])
    source = np.asarray([row["source_retention"] for row in rows])
    source_low = np.asarray([row["source_retention_ci95_low"] for row in rows])
    source_high = np.asarray([row["source_retention_ci95_high"] for row in rows])
    donor = np.asarray([row["donor_transfer"] for row in rows])
    donor_low = np.asarray([row["donor_transfer_ci95_low"] for row in rows])
    donor_high = np.asarray([row["donor_transfer_ci95_high"] for row in rows])
    fig, axis = plt.subplots(figsize=(6.2, 4.1))
    axis.plot(x, source, marker="o", color="#2166ac", label="source target retained")
    axis.fill_between(x, source_low, source_high, color="#2166ac", alpha=0.18)
    axis.plot(x, donor, marker="s", color="#b2182b", label="donor target transferred")
    axis.fill_between(x, donor_low, donor_high, color="#b2182b", alpha=0.18)
    axis.set(xlabel="flow updates under source condition before switch", ylabel="first-contact probability", ylim=(-0.03, 1.03))
    axis.set_xticks(range(11))
    axis.legend(frameon=False)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=240)
    plt.close(fig)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _boolean(value: str) -> bool:
    if value not in {"True", "False"}:
        raise ValueError(f"expected serialized boolean, got {value!r}")
    return value == "True"


if __name__ == "__main__":
    raise SystemExit(main())
