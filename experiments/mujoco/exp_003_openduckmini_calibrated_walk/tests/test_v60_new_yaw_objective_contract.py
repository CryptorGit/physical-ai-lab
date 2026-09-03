import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from v60_yaw_objective import bounded_yaw_progress  # noqa: E402


def test_new_objective_peaks_at_exact_tracking():
    ratios = np.linspace(-1.0, 4.0, 1001)
    for command in (-0.6, -0.3, 0.3, 0.6):
        values = bounded_yaw_progress(command, command * ratios, xp=np)
        assert ratios[int(np.argmax(values))] == 1.0


def test_zero_command_has_zero_finite_contribution():
    values = bounded_yaw_progress(0.0, np.linspace(-1.0, 1.0, 101), xp=np)
    assert np.all(np.isfinite(values))
    assert np.all(values == 0.0)
