#!/usr/bin/env python3
"""Run OpenPI's converter without silently downcasting source weights.

This is a narrow adapter for the public fix proposed in OpenPI PR #978. The
upstream conversion function remains responsible for every parameter mapping;
we only construct its intermediate PI0 config in float32 and request a float32
checkpoint so that the policy loader can recreate OpenPI's mixed-precision
inference layout without first losing source precision.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

OPENPI_PRECISION_REPAIR_COMMIT = "e5fe45e2c6784f315ffa59c207457701fb906c05"
OPENPI_PRECISION_REPAIR_URL = (
    "https://github.com/Greyman-Seu/openpi/commit/"
    f"{OPENPI_PRECISION_REPAIR_COMMIT}"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--config-name", default="pi0_libero")
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--upstream-converter", type=Path, required=True)
    parser.add_argument("--upstream-revision", required=True)
    return parser.parse_args()


def float32_conversion_config(model_config):
    """Return the public PI0 config with lossless conversion precision."""
    if not dataclasses.is_dataclass(model_config):
        raise TypeError("model config must be a dataclass instance")
    if not hasattr(model_config, "dtype"):
        raise TypeError("model config does not expose dtype")
    return dataclasses.replace(model_config, dtype="float32")


def load_upstream_converter(path: Path) -> ModuleType:
    if not path.is_file():
        raise FileNotFoundError(f"upstream converter is missing: {path}")
    spec = importlib.util.spec_from_file_location("openpi_jax_to_pytorch", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load upstream converter: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "convert_pi0_checkpoint", None)):
        raise AttributeError("upstream converter lacks convert_pi0_checkpoint")
    return module


def conversion_provenance(converter: Path, upstream_revision: str) -> dict:
    if not converter.is_file():
        raise FileNotFoundError(f"upstream converter is missing: {converter}")
    return {
        "schema_version": 1,
        "adapter": "openpi_pr978_float32_intermediate",
        "source_precision_repair_commit": OPENPI_PRECISION_REPAIR_COMMIT,
        "source_precision_repair_url": OPENPI_PRECISION_REPAIR_URL,
        "upstream_openpi_revision": upstream_revision,
        "upstream_converter_sha256": hashlib.sha256(converter.read_bytes()).hexdigest(),
        "intermediate_model_config_dtype": "float32",
        "saved_checkpoint_precision": "float32",
        "policy_loader_precision_behavior": "unchanged_openpi_mixed_precision",
    }


def main() -> None:
    args = parse_args()

    from openpi.models import pi0_config
    from openpi.training import config as training_config

    model_config = training_config.get_config(args.config_name).model
    if not isinstance(model_config, pi0_config.Pi0Config):
        raise TypeError(f"config {args.config_name!r} is not a Pi0Config")
    model_config = float32_conversion_config(model_config)

    converter = load_upstream_converter(args.upstream_converter)
    converter.convert_pi0_checkpoint(
        str(args.checkpoint_dir),
        "float32",
        str(args.output_path),
        model_config,
    )
    provenance = conversion_provenance(
        args.upstream_converter,
        args.upstream_revision,
    )
    (args.output_path / "conversion_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
