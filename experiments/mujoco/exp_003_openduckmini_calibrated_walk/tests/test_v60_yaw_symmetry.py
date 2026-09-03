import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from v60_yaw_objective import bounded_yaw_progress, yaw_related_total  # noqa: E402


def test_bounded_objective_and_total_are_left_right_symmetric():
    for command in (0.3, 0.6):
        for ratio in np.linspace(-1.0, 4.0, 101):
            positive = bounded_yaw_progress(
                command, command * ratio, xp=np
            )
            negative = bounded_yaw_progress(
                -command, -command * ratio, xp=np
            )
            assert np.isclose(positive, negative, atol=1e-12, rtol=0.0)
            positive_total = yaw_related_total(
                command,
                command * ratio,
                objective="bounded_command_centered_gaussian",
                xp=np,
            )
            negative_total = yaw_related_total(
                -command,
                -command * ratio,
                objective="bounded_command_centered_gaussian",
                xp=np,
            )
            assert np.isclose(
                positive_total, negative_total, atol=1e-12, rtol=0.0
            )
