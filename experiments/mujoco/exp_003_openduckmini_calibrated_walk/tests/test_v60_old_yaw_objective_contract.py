import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from v60_yaw_objective import old_yaw_progress  # noqa: E402


def test_old_objective_is_exact_unbounded_dot_product():
    for command in (-0.6, -0.3, 0.0, 0.3, 0.6):
        for ratio in (-1.0, 0.0, 0.5, 1.0, 2.0, 3.5, 4.0):
            actual = command * ratio
            assert old_yaw_progress(command, actual, xp=np) == command * actual
