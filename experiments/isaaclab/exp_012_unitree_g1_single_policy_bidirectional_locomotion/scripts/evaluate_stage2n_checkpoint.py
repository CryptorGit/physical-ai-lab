"""Reuse the audited Stage 2K evaluator for Stage 2N checkpoints."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
from torch import nn

SCRIPT = Path(__file__).resolve()
CORE = SCRIPT.with_name("evaluate_stage2k_gait_latent.py")

pre = argparse.ArgumentParser(add_help=False)
pre.add_argument("--checkpoint", required=True)
pre.add_argument("--eval-mode", choices=("endpoints", "toggleA", "toggleB"), required=True)
pre.add_argument("--action-mode", choices=("deterministic", "stochastic"), required=True)
pre.add_argument("--episodes", type=int, default=100)
pre.add_argument("--output-dir", required=True)
known, remaining = pre.parse_known_args()
sys.argv = [sys.argv[0], "--mode", known.eval_mode, *remaining]
spec = importlib.util.spec_from_file_location("stage2n_eval_core", CORE)
core = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(core)


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        payload = torch.load(known.checkpoint, map_location="cpu", weights_only=False)
        state = payload["actor_state_dict"]
        self.first_base_weight = nn.Parameter(state["first_base_weight"], requires_grad=False)
        self.first_gait_column = nn.Parameter(state["first_gait_column"], requires_grad=False)
        self.first_bias = nn.Parameter(state["first_bias"], requires_grad=False)
        self.hidden = nn.Sequential(
            nn.ELU(), nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 37),
        )
        self.hidden.load_state_dict({
            key.removeprefix("hidden."): value
            for key, value in state.items() if key.startswith("hidden.")
        })
        self.register_buffer("log_std_walk", state["distribution.log_std_walk"])
        self.register_buffer("log_std_run", state["distribution.log_std_run"])

    def load_state_dict(self, state_dict, strict=True):
        return nn.modules.module._IncompatibleKeys([], [])

    def forward(self, observation, gait):
        first = nn.functional.linear(observation, self.first_base_weight, self.first_bias)
        mean = self.hidden(first + gait.reshape(-1, 1) * self.first_gait_column.T)
        if known.action_mode == "deterministic":
            return mean
        g = gait.reshape(-1, 1)
        std = torch.exp((1 - g) * self.log_std_walk + g * self.log_std_run)
        return mean + torch.randn_like(mean) * std


original_conditions = core.conditions


def conditions(mode):
    values = original_conditions(mode)
    for value in values:
        value["episodes"] = known.episodes
    return values


def main():
    output = Path(known.output_dir)
    raw = output / f"{known.action_mode}_{known.eval_mode}"
    raw.mkdir(parents=True, exist_ok=True)
    core.OUT = output
    core.RAW = raw
    proxy = output / "_stage2n_eval_proxy.pt"
    torch.save({"model_state_dict": {}}, proxy)
    core.STUDENT = proxy
    core.Student = Policy
    core.conditions = conditions
    torch.manual_seed(20268121)
    core.main()


if __name__ == "__main__":
    main()
