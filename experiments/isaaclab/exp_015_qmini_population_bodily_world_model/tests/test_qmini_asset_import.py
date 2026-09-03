from __future__ import annotations

from qmini_population_bwm.qmini_asset import (
    OFFICIAL_URDF_PATH,
    load_qmini_contract,
    official_link_mass_total,
    validate_qmini_contract,
)


def test_vendored_official_asset_is_parseable() -> None:
    assert OFFICIAL_URDF_PATH.exists()
    contract = load_qmini_contract()
    assert not validate_qmini_contract(contract)
    assert official_link_mass_total(contract) == 11.145138000000001
    assert all(path.endswith(".STL") for path in contract.mesh_files)


def test_visual_and_collision_meshes_are_not_synthesized() -> None:
    contract = load_qmini_contract()
    by_name = {link.name: link for link in contract.links}
    assert by_name["LL_ankle"].visual_meshes == ("meshes/LL_ankle.STL",)
    assert by_name["LL_ankle"].collision_meshes == ("meshes/LL_ankle.STL",)
    assert by_name["LL_hip_yaw"].collision_meshes == ()
    assert by_name["RL_hip_roll"].collision_meshes == ()
