from __future__ import annotations

from qmini_population_bwm.qmini_asset import QMINI_JOINT_ORDER, load_qmini_contract, validate_qmini_contract


def test_official_qmini_joint_contract() -> None:
    contract = load_qmini_contract()
    assert validate_qmini_contract(contract) == []
    assert len(contract.joints) == 10
    assert contract.joint_names == QMINI_JOINT_ORDER
    assert contract.transmission_count == 0
    assert contract.foot_collision_links == ("LL_ankle", "RL_ankle")


def test_joint_limits_are_from_official_urdf() -> None:
    contract = load_qmini_contract()
    assert contract.joints[0].lower == -0.349
    assert contract.joints[0].upper == 0.525
    assert contract.joints[1].effort == 60.0
    assert contract.joints[1].velocity == 0.3
    assert contract.joints[5].lower == -0.525
    assert contract.joints[5].upper == 0.349
