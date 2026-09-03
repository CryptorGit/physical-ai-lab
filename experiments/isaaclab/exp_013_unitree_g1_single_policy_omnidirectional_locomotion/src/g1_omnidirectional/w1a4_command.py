"""Fixed W1A4 retention/expansion command sampler."""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from g1_omnidirectional.w1a_command import W1AContinuousTranslationCommand


class W1A4RetentionCommand(W1AContinuousTranslationCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        repo = Path(__file__).resolve().parents[5]
        path = repo / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a4_low_speed_retention_consolidation/iteration80_failed_0p6_sector_manifest.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        self.failed = torch.tensor(data["failed_directions_deg"], device=self.device) * math.pi / 180

    @property
    def phase(self):
        return "W1A4_RETENTION_CONSOLIDATION"

    def _resample_command(self, env_ids):
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        n = len(ids)
        if not n:
            return
        group = torch.multinomial(torch.tensor((.30, .20, .40, .10), device=self.device), n, replacement=True)
        # Keep the stress set exactly mirror-balanced in every resampling batch.
        b_ids = torch.where(group == 1)[0]
        if len(b_ids) % 2:
            group[b_ids[-1]] = 0
        theta = torch.empty(n, device=self.device)
        speed = torch.empty(n, device=self.device)
        a, b, c, d = (group == index for index in range(4))
        theta[a] = torch.rand(int(a.sum()), device=self.device) * 2 * math.pi - math.pi
        speed[a] = .25 + torch.rand(int(a.sum()), device=self.device) * .10
        if b.any():
            rear = torch.tensor((213.75, 225, 236.25, 247.5, 258.75), device=self.device)
            mirror = torch.tensor((146.25, 135, 123.75, 112.5, 101.25), device=self.device)
            count = int(b.sum())
            pair = torch.arange(count, device=self.device) % 2
            index = torch.arange(count, device=self.device) // 2 % 5
            centers = torch.where(pair == 0, rear[index], mirror[index])
            theta[b] = torch.deg2rad(centers + (torch.rand(count, device=self.device) - .5) * 11.25)
            speed[b] = .25 + torch.rand(count, device=self.device) * .15
        if c.any():
            count = int(c.sum())
            index = torch.randint(len(self.failed), (count,), device=self.device)
            theta[c] = self.failed[index] + (torch.rand(count, device=self.device) - .5) * (math.pi / 8)
            speed[c] = .45 + torch.rand(count, device=self.device) * .15
        if d.any():
            count = int(d.sum())
            diagonal = torch.rand(count, device=self.device) >= .5
            signs = torch.where(torch.rand(count, device=self.device) < .5, -1., 1.)
            theta[d] = torch.where(diagonal, signs * math.pi / 4, torch.zeros(count, device=self.device))
            speed[d] = torch.where(diagonal, .6 + torch.rand(count, device=self.device) * .4,
                                   .6 + torch.rand(count, device=self.device) * .6)
        self.sampled_theta[ids] = theta
        self.sampled_speed[ids] = speed
        self.vel_command_b[ids, 0] = speed * torch.cos(theta)
        self.vel_command_b[ids, 1] = speed * torch.sin(theta)
        self.vel_command_b[ids, 2] = 0
        self.is_standing_env[ids] = False
        self.is_heading_env[ids] = False
