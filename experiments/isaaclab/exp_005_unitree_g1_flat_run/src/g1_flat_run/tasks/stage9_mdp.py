"""Stage 9 command sampling for stable periodic running around 5.0 m/s."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .stage3_mdp import FocusedRunVelocityCommand
from .stage8_mdp import Stage8VelocityCommand


class Stage9VelocityCommand(Stage8VelocityCommand):
    """Focus on 4.85--5.00 m/s and keep 5.10 m/s as a small upper probe."""

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        super(FocusedRunVelocityCommand, self)._resample_command(env_ids)
        count = len(env_ids)
        if count == 0:
            return
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        draw = torch.rand(count, device=self.device)
        focus = torch.empty(count, device=self.device).uniform_(4.85, 5.00)
        support = torch.empty(count, device=self.device).uniform_(4.70, 4.90)
        upper_probe = torch.empty(count, device=self.device).uniform_(5.00, 5.10)
        self.vel_command_b[ids, 0] = torch.where(
            draw < 0.60,
            focus,
            torch.where(draw < 0.85, support, upper_probe),
        )
