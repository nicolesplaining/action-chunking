#!/usr/bin/env python3
"""Serve an official OpenPI policy while accepting explicit per-request noise."""

from __future__ import annotations

import argparse
import dataclasses
import logging

import numpy as np
from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as training_config
from openpi_client import base_policy
from typing_extensions import override


class NoiseAwarePolicy(base_policy.BasePolicy):
    """Minimal transport adapter; model and transforms remain upstream-owned."""

    def __init__(self, policy):
        self.policy = policy

    @override
    def infer(self, obs):
        request = dict(obs)
        noise = request.pop("_action_noise", None)
        if noise is None:
            raise ValueError("deterministic research server requires _action_noise")
        return self.policy.infer(request, noise=np.asarray(noise, dtype=np.float32))

    @property
    def metadata(self):
        return {**self.policy.metadata, "accepts_action_noise": True}

    @override
    def reset(self) -> None:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="pi05_libero")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--port", type=int, default=8002)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = training_config.get_config(args.config)
    config = dataclasses.replace(config, model=dataclasses.replace(config.model, pytorch_compile_mode=None))
    policy = policy_config.create_trained_policy(config, args.checkpoint, pytorch_device=args.device)
    wrapped = NoiseAwarePolicy(policy)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=wrapped,
        host="0.0.0.0",
        port=args.port,
        metadata=wrapped.metadata,
    )
    logging.info("serving explicit-noise policy on port %d", args.port)
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
