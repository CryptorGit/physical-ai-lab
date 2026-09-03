"""Mirror-paired yaw-conditioned WALK curriculum for Phase W1B."""
from __future__ import annotations

import math
import torch

from g1_omnidirectional.w1a_command import W1AContinuousTranslationCommand


class W1BYawConditionedCommand(W1AContinuousTranslationCommand):
    @property
    def phase(self):
        if self.training_iteration <= 40:
            return "Y1_FORWARD_MOVING_TURNS"
        if self.training_iteration <= 100:
            return "Y2_ALL_DIRECTION_MOVING_TURNS"
        if self.training_iteration <= 150:
            return "Y3_TURN_IN_PLACE_ACQUISITION"
        return "Y4_BALANCED_CONSOLIDATION"

    @staticmethod
    def _away_from_zero(count, low, high, device):
        magnitude = torch.empty(count, device=device).uniform_(low, high)
        sign = torch.where(torch.rand(count, device=device) < .5, -1., 1.)
        return magnitude * sign

    def _resample_command(self, env_ids):
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if not len(ids):
            return
        if len(ids) % 2:
            raise RuntimeError("W1B requires an even environment population for exact mirror pairing")
        half = len(ids) // 2
        first, mirror = ids[:half], ids[half:]
        phase = self.phase
        if phase == "Y1_FORWARD_MOVING_TURNS":
            weights = (.45, .45, .10)
        elif phase == "Y2_ALL_DIRECTION_MOVING_TURNS":
            weights = (.40, .50, .10)
        elif phase == "Y3_TURN_IN_PLACE_ACQUISITION":
            weights = (.35, .40, .25)
        else:
            weights = (.35, .45, .20)
        group = torch.multinomial(torch.tensor(weights, device=self.device), half, replacement=True)
        theta = torch.zeros(half, device=self.device)
        speed = torch.zeros(half, device=self.device)
        yaw = torch.zeros(half, device=self.device)
        zero, moving, pure = group == 0, group == 1, group == 2
        if zero.any():
            n = int(zero.sum())
            theta[zero] = torch.empty(n, device=self.device).uniform_(-math.pi, math.pi)
            speed[zero] = torch.empty(n, device=self.device).uniform_(.25, .35)
        if moving.any():
            n = int(moving.sum())
            if phase == "Y1_FORWARD_MOVING_TURNS":
                theta[moving] = torch.empty(n, device=self.device).uniform_(-math.pi / 4, math.pi / 4)
                speed[moving] = torch.empty(n, device=self.device).uniform_(.25, .60)
                yaw[moving] = self._away_from_zero(n, .05, .30, self.device)
            elif phase == "Y2_ALL_DIRECTION_MOVING_TURNS":
                theta[moving] = torch.empty(n, device=self.device).uniform_(-math.pi, math.pi)
                speed[moving] = torch.empty(n, device=self.device).uniform_(.20, .40)
                yaw[moving] = self._away_from_zero(n, .05, .35, self.device)
            elif phase == "Y3_TURN_IN_PLACE_ACQUISITION":
                theta[moving] = torch.empty(n, device=self.device).uniform_(-math.pi, math.pi)
                speed[moving] = torch.empty(n, device=self.device).uniform_(.20, .50)
                yaw[moving] = self._away_from_zero(n, .05, .40, self.device)
            else:
                theta[moving] = torch.empty(n, device=self.device).uniform_(-math.pi, math.pi)
                max_speed = torch.where(theta[moving].abs() <= math.pi / 4, .8, .6)
                speed[moving] = .20 + torch.rand(n, device=self.device) * (max_speed - .20)
                yaw[moving] = self._away_from_zero(n, .05, .50, self.device)
        if pure.any():
            n = int(pure.sum())
            if phase == "Y1_FORWARD_MOVING_TURNS":
                speed[pure] = torch.empty(n, device=self.device).uniform_(0, .10)
                theta[pure] = torch.empty(n, device=self.device).uniform_(-math.pi, math.pi)
                yaw[pure] = self._away_from_zero(n, .15, .25, self.device)
            elif phase == "Y2_ALL_DIRECTION_MOVING_TURNS":
                speed[pure] = torch.empty(n, device=self.device).uniform_(0, .08)
                theta[pure] = torch.empty(n, device=self.device).uniform_(-math.pi, math.pi)
                yaw[pure] = self._away_from_zero(n, .05, .30, self.device)
            elif phase == "Y3_TURN_IN_PLACE_ACQUISITION":
                yaw[pure] = self._away_from_zero(n, .15, .45, self.device)
            else:
                yaw[pure] = self._away_from_zero(n, .15, .50, self.device)
        vx, vy = speed * torch.cos(theta), speed * torch.sin(theta)
        self.vel_command_b[first, 0] = vx
        self.vel_command_b[first, 1] = vy
        self.vel_command_b[first, 2] = yaw
        self.vel_command_b[mirror, 0] = vx
        self.vel_command_b[mirror, 1] = -vy
        self.vel_command_b[mirror, 2] = -yaw
        self.sampled_theta[first], self.sampled_theta[mirror] = theta, -theta
        self.sampled_speed[first], self.sampled_speed[mirror] = speed, speed
        self.is_standing_env[ids] = False
        self.is_heading_env[ids] = False

    def _update_command(self):
        if self.external_override_enabled:
            self.vel_command_b.copy_(self.external_override)
