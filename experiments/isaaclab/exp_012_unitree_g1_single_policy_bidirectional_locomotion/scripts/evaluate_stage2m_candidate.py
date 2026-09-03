"""Evaluate the frozen Stage 2K mean with the calibrated gait-specific std pair."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2m_stochastic_gait_endpoint_robustness"
K = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2k_gait_latent_preflight"
L = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2l_gait_conditioned_gaussian_std_preflight"
CORE = SCRIPT.with_name("evaluate_stage2k_gait_latent.py")
ALPHA_WALK = 0.30
ALPHA_RUN = 0.65

pre = argparse.ArgumentParser(add_help=False)
pre.add_argument("--candidate-mode", choices=("authority0", "authority1", "toggleA", "toggleB"), required=True)
known, remaining = pre.parse_known_args()
sys.argv = [sys.argv[0], "--mode", known.candidate_mode, *remaining]
spec = importlib.util.spec_from_file_location("stage2m_candidate_core", CORE)
core = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(core)


class CandidatePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        student = torch.load(K / "student/selected_gait_latent_student.pt", map_location="cpu", weights_only=False)
        state = student["model_state_dict"]
        self.first_base_weight = nn.Parameter(state["first_base_weight"].clone(), requires_grad=False)
        self.first_gait_column = nn.Parameter(state["first_gait_column"].clone(), requires_grad=False)
        self.first_bias = nn.Parameter(state["first_bias"].clone(), requires_grad=False)
        self.hidden = nn.Sequential(
            nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 37),
        )
        self.hidden.load_state_dict(OrderedDict(
            (key.removeprefix("hidden."), value) for key, value in state.items() if key.startswith("hidden.")
        ), strict=True)
        dist = torch.load(
            L / "student/stage2l_gait_conditioned_std_student.pt", map_location="cpu", weights_only=False
        )["model_state_dict"]
        self.register_buffer("log_std_walk", dist["log_std_walk"].clone())
        self.register_buffer("log_std_run", dist["log_std_run"].clone())

    def load_state_dict(self, state_dict, strict=True):
        return nn.modules.module._IncompatibleKeys([], [])

    def forward(self, observation, gait):
        first = nn.functional.linear(observation, self.first_base_weight, self.first_bias)
        mean = self.hidden(first + gait.reshape(-1, 1) * self.first_gait_column.T)
        g = gait.reshape(-1, 1).double()
        log_effective_std = (
            (1 - g) * (self.log_std_walk + torch.log(torch.tensor(ALPHA_WALK, dtype=torch.float64, device=g.device)))
            + g * (self.log_std_run + torch.log(torch.tensor(ALPHA_RUN, dtype=torch.float64, device=g.device)))
        )
        std = log_effective_std.exp().to(mean.dtype)
        return mean + torch.randn_like(mean) * std


def main():
    torch.manual_seed(20268221)
    raw = OUT / "raw" / "candidate"
    raw.mkdir(parents=True, exist_ok=True)
    core.OUT = OUT
    core.RAW = raw
    core.STUDENT = L / "student/stage2l_gait_conditioned_std_student.pt"
    core.Student = CandidatePolicy
    core.main()


if __name__ == "__main__":
    main()
