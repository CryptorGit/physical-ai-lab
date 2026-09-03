"""Reuse the Stage 2K physical evaluator with selectable Stage 2L Gaussian policies."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
K = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2k_gait_latent_preflight"
OUT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2l_gait_conditioned_gaussian_std_preflight"
L = OUT / "student/stage2l_gait_conditioned_std_student.pt"
WALK = REPO / "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_21-40-39_stage2_1024_750/model_4246.pt"
RUN = REPO / "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-18_00-44-32_stage4_1024_500/model_5244.pt"
EVALUATOR = SCRIPT.with_name("evaluate_stage2k_gait_latent.py")

pre = argparse.ArgumentParser(add_help=False)
pre.add_argument("--policy", choices=("deterministic", "teacher", "shared", "conditioned"), required=True)
known, remaining = pre.parse_known_args()
sys.argv = [sys.argv[0], *remaining]

spec = importlib.util.spec_from_file_location("stage2k_eval_core", EVALUATOR)
core = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(core)


def actor_from_state(state):
    actor = nn.Sequential(
        nn.Linear(123, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(),
        nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 37),
    )
    actor.load_state_dict(OrderedDict(
        (key.removeprefix("mlp."), value) for key, value in state.items() if key.startswith("mlp.")
    ), strict=True)
    return actor


class RuntimeStudent(nn.Module):
    mean_trace_bitwise = True
    mean_trace_samples = 0

    def __init__(self):
        super().__init__()
        stage2l = torch.load(L, map_location="cpu", weights_only=False)["model_state_dict"]
        stage2k = torch.load(K / "student/selected_gait_latent_student.pt", map_location="cpu", weights_only=False)
        self.first_base_weight = nn.Parameter(stage2l["first_base_weight"].clone(), requires_grad=False)
        self.first_gait_column = nn.Parameter(stage2l["first_gait_column"].clone(), requires_grad=False)
        self.first_bias = nn.Parameter(stage2l["first_bias"].clone(), requires_grad=False)
        self.hidden = nn.Sequential(
            nn.ELU(), nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 37),
        )
        hidden = OrderedDict((key.removeprefix("hidden."), value) for key, value in stage2l.items() if key.startswith("hidden."))
        self.hidden.load_state_dict(hidden, strict=True)
        self.register_buffer("log_std_walk", stage2l["log_std_walk"].clone())
        self.register_buffer("log_std_run", stage2l["log_std_run"].clone())
        self.register_buffer("shared_std", stage2k["model_state_dict"]["std"].clone())
        walk_state = torch.load(WALK, map_location="cpu", weights_only=False)["actor_state_dict"]
        run_state = torch.load(RUN, map_location="cpu", weights_only=False)["actor_state_dict"]
        self.walk_actor = actor_from_state(walk_state)
        self.run_actor = actor_from_state(run_state)
        self.register_buffer("walk_std", walk_state["distribution.std_param"].clone())
        self.register_buffer("run_std", run_state["distribution.std_param"].clone())
        self.reference_first_base = stage2k["model_state_dict"]["first_base_weight"]
        self.reference_first_gait = stage2k["model_state_dict"]["first_gait_column"]
        self.reference_first_bias = stage2k["model_state_dict"]["first_bias"]
        self.reference_hidden = OrderedDict(
            (key.removeprefix("hidden."), value) for key, value in stage2k["model_state_dict"].items()
            if key.startswith("hidden.")
        )

    def load_state_dict(self, state_dict, strict=True):
        # The core evaluator expects to load its selected checkpoint. This instance already
        # loaded the immutable Stage 2L/teacher payloads selected by --policy.
        return nn.modules.module._IncompatibleKeys([], [])

    def student_mean(self, observation, gait):
        first = nn.functional.linear(observation, self.first_base_weight, self.first_bias)
        first = first + gait.reshape(-1, 1) * self.first_gait_column.T
        return self.hidden(first)

    def forward(self, observation, gait):
        student_mean = self.student_mean(observation, gait)
        if known.policy == "teacher":
            mean = torch.where(
                gait.reshape(-1, 1) < .5, self.walk_actor(observation), self.run_actor(observation)
            )
            std = torch.where(gait.reshape(-1, 1) < .5, self.walk_std, self.run_std)
        else:
            mean = student_mean
            if known.policy == "shared":
                std = self.shared_std.expand_as(mean)
            else:
                log_std = ((1 - gait.reshape(-1, 1)) * self.log_std_walk
                           + gait.reshape(-1, 1) * self.log_std_run)
                std = log_std.exp().to(mean.dtype)
        if known.policy == "deterministic":
            # Equality is checked on every physical evaluation observation, not only a static sample.
            RuntimeStudent.mean_trace_samples += len(observation)
            RuntimeStudent.mean_trace_bitwise &= bool(torch.equal(student_mean, mean))
            return mean
        return mean + torch.randn_like(mean) * std


def main():
    torch.manual_seed(20267121)
    policy_raw = OUT / "raw" / known.policy
    policy_raw.mkdir(parents=True, exist_ok=True)
    core.OUT = OUT
    core.RAW = policy_raw
    core.STUDENT = L
    core.Student = RuntimeStudent
    core.main()
    result_path = policy_raw / f"{core.args.mode}_evaluation.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["runtime_policy"] = known.policy
    result["action_sampling"] = "mean" if known.policy == "deterministic" else "diagonal Gaussian S100"
    if known.policy == "deterministic":
        result["mean_action_trace_bitwise_stage2k"] = RuntimeStudent.mean_trace_bitwise
        result["mean_action_trace_samples"] = RuntimeStudent.mean_trace_samples
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
