from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from safe_gait_experts.contract import (
    ACTUATOR_JOINT_ORDER,
    HEAD_JOINTS,
    SAFE_INIT_POS,
    SAFE_JOINT_LIMITS,
)
from safe_gait_experts.safe_randomization import (
    actuator_name_to_index,
    build_qpos_noise_scale,
    clip_reset_qpos_to_physical_safe_limits,
    make_domain_randomizer,
    resolve_randomization_targets,
    scale_body_masses_with_payload,
)


@dataclass
class _NamedView:
    id: int
    name: str = ""


class _FakeCpuModel:
    def __init__(self) -> None:
        # Deliberately make every relevant ID differ from the historical 0/1
        # assumptions and scramble the actuator ordering.
        self._geom_ids = {"decor": 0, "floor": 3}
        self._body_ids = {"world": 0, "trunk_assembly": 4, "floor": 6}
        self.geom_bodyid = np.array([0, 1, 2, 6])
        self.body_mass = np.array([0.0, 0.1, 0.2, 0.3, 1.5, 0.4, 0.0])

        self._actuator_names = list(reversed(ACTUATOR_JOINT_ORDER))
        self.nu = len(self._actuator_names)
        self.actuator_trnid = np.column_stack(
            [np.arange(14, dtype=np.int32), -np.ones(14, dtype=np.int32)]
        )
        self.jnt_dofadr = np.arange(7, 35, 2, dtype=np.int32)
        self.jnt_qposadr = np.arange(8, 36, 2, dtype=np.int32)

    def geom(self, name: str) -> _NamedView:
        if name not in self._geom_ids:
            raise KeyError(name)
        return _NamedView(self._geom_ids[name], name)

    def body(self, name: str) -> _NamedView:
        if name not in self._body_ids:
            raise KeyError(name)
        return _NamedView(self._body_ids[name], name)

    def actuator(self, index: int) -> _NamedView:
        return _NamedView(index, self._actuator_names[index])


def test_floor_and_root_are_resolved_by_name_not_fixed_id() -> None:
    targets = resolve_randomization_targets(_FakeCpuModel())
    assert targets.floor_geom_id == 3
    assert targets.floor_body_id == 6
    assert targets.root_body_id == 4
    assert targets.floor_body_has_mass is False
    assert targets.root_body_has_mass is True


def test_all_14_noise_scales_follow_scrambled_actuator_names() -> None:
    model = _FakeCpuModel()
    indices = actuator_name_to_index(model)
    scale = build_qpos_noise_scale(indices)
    assert scale.shape == (14,)

    for name, index in indices.items():
        if name in HEAD_JOINTS:
            expected = 0.0
        elif name.endswith("_knee"):
            expected = 0.05
        elif name.endswith("_ankle"):
            expected = 0.08
        else:
            expected = 0.03
        assert scale[index] == expected, name

    # Explicit regressions for the side that the historical list-index code
    # could silently leave at zero.
    assert scale[indices["right_hip_pitch"]] == 0.03
    assert scale[indices["right_knee"]] == 0.05
    assert scale[indices["right_ankle"]] == 0.08


def test_reset_qpos_preserves_exact_home_and_clips_noise_only_to_physical_safe() -> None:
    home = np.asarray([SAFE_INIT_POS[name] for name in ACTUATOR_JOINT_ORDER])
    np.testing.assert_array_equal(
        clip_reset_qpos_to_physical_safe_limits(home),
        home,
    )

    noisy = home.copy()
    left_knee = ACTUATOR_JOINT_ORDER.index("left_knee")
    right_knee = ACTUATOR_JOINT_ORDER.index("right_knee")
    head_yaw = ACTUATOR_JOINT_ORDER.index("head_yaw")
    noisy[left_knee] = 99.0
    noisy[right_knee] = -99.0
    noisy[head_yaw] = 1.0
    clipped = clip_reset_qpos_to_physical_safe_limits(noisy)

    noisy_clipped = clip_reset_qpos_to_physical_safe_limits(
        noisy, noise_applied=True
    )
    assert noisy_clipped[left_knee] == pytest.approx(
        SAFE_JOINT_LIMITS["left_knee"][1] - 0.005
    )
    assert noisy_clipped[right_knee] == pytest.approx(
        SAFE_JOINT_LIMITS["right_knee"][0] + 0.005
    )
    assert noisy_clipped[head_yaw] == 0.0

    # Without reset noise, exact physical SAFE clipping remains margin-free.
    assert clipped[left_knee] == SAFE_JOINT_LIMITS["left_knee"][1]
    assert clipped[right_knee] == SAFE_JOINT_LIMITS["right_knee"][0]
    assert clipped[head_yaw] == 0.0
    # Reset is not silently teleported to the inward desired-target envelope.
    assert clipped[left_knee] > SAFE_JOINT_LIMITS["left_knee"][1] - 0.050


def test_domain_randomizer_captures_cpu_resolved_metadata() -> None:
    model = _FakeCpuModel()
    randomizer = make_domain_randomizer(model)
    assert randomizer.targets.floor_geom_id == 3
    assert randomizer.targets.root_body_id == 4
    assert len(randomizer.actuated_dof_addresses) == 14
    assert len(randomizer.actuated_qpos_addresses) == 14
    assert len(randomizer.qpos_noise_scale) == 14
    indices = actuator_name_to_index(model)
    assert randomizer.qpos_noise_scale[indices["head_yaw"]] == 0.0
    assert randomizer.qpos_noise_scale[indices["right_knee"]] == 0.05


def test_massless_bodies_remain_massless_and_receive_no_payload() -> None:
    masses = np.array([0.0, 0.0, 1.0, 0.5])
    factors = np.array([1.1, 0.9, 1.1, 0.9])
    result = scale_body_masses_with_payload(
        masses,
        factors,
        root_body_id=1,
        payload_delta=0.2,
    )
    assert result[0] == 0.0
    assert result[1] == 0.0
    assert result[2] == pytest.approx(1.1)
    assert result[3] == pytest.approx(0.45)


def test_positive_root_payload_is_applied_without_allowing_negative_mass() -> None:
    masses = np.array([0.0, 0.05, 1.0])
    result = scale_body_masses_with_payload(
        masses,
        np.ones(3),
        root_body_id=1,
        payload_delta=-1.0,
        minimum_positive_mass=1e-6,
    )
    assert result[0] == 0.0
    assert result[1] == pytest.approx(1e-6)


def test_missing_actuator_is_a_hard_error() -> None:
    incomplete = {name: index for index, name in enumerate(ACTUATOR_JOINT_ORDER[:-1])}
    with pytest.raises(ValueError, match="missing"):
        build_qpos_noise_scale(incomplete)


def test_missing_named_model_target_is_a_hard_error() -> None:
    with pytest.raises(ValueError, match="does not exist"):
        resolve_randomization_targets(_FakeCpuModel(), floor_geom_name="ground")
