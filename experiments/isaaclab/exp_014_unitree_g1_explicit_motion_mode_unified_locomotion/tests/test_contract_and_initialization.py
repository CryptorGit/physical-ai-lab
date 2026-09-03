from pathlib import Path
import sys

import torch

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[2]
sys.path.insert(0, str(EXP / "src"))
sys.path.insert(0, str(REPO / "experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/src"))

from g1_explicit_motion_mode.contract import ExplicitMotionModeCommand, MotionMode, build_observation_141
from g1_explicit_motion_mode.student import initialize_s0_from_w1b, widen_student
from g1_omnidirectional.policy import FrozenGaitActor

W1B = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1b_r2_pending_mirror_queue_repair_rerun/checkpoints/model_200.pt"


def test_zero_velocity_modes_are_distinct_and_causal():
    base = torch.zeros(2, 123)
    state = ExplicitMotionModeCommand.zeros(2)
    state.request(torch.tensor([MotionMode.STAND, MotionMode.WALK]))
    out = build_observation_141(base, state)
    assert out.shape == (2, 141)
    assert torch.equal(out[0, :124], out[1, :124])
    assert not torch.equal(out[0], out[1])
    assert torch.equal(out[0, 127:130], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.equal(out[1, 127:130], torch.tensor([0.0, 1.0, 0.0]))


def test_s0_is_bitwise_w1b_at_initialization():
    torch.manual_seed(1401)
    old123 = torch.randn(4096, 123)
    gait = torch.randint(0, 2, (4096,), dtype=torch.float32)
    tail = torch.randn(4096, 17)
    x = torch.cat((old123, gait[:, None], tail), dim=1)
    teacher = FrozenGaitActor(W1B).eval()
    student = initialize_s0_from_w1b(W1B).eval()
    with torch.inference_mode():
        expected = teacher(old123, gait)
        actual = student(x)
    assert torch.equal(expected, actual)


def test_net2wider_numerical_gap_is_bounded_but_above_registered_threshold():
    torch.manual_seed(1402)
    x = torch.randn(1024, 141)
    s0 = initialize_s0_from_w1b(W1B).eval()
    gaps = []
    for name in ("S1", "S2"):
        widened = widen_student(s0, name).eval()
        with torch.inference_mode():
            difference = (s0(x) - widened(x)).abs().max().item()
        gaps.append(difference)
        assert difference < 2e-5, (name, difference)
    assert any(gap > 1e-7 for gap in gaps), "update the widening availability audit if kernels change"
