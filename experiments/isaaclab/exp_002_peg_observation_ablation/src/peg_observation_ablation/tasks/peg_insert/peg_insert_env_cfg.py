"""Observation configurations for peg-insertion ablation experiments."""

from isaaclab.utils.configclass import configclass
from isaaclab_tasks.direct.factory.factory_env_cfg import FactoryTaskPegInsertCfg


@configclass
class PegObservationBaselineEnvCfg(FactoryTaskPegInsertCfg):
    """Full policy observation.

    Observation dimension:
    3 position
    + 4 quaternion
    + 3 linear velocity
    + 3 angular velocity
    + 6 previous action
    = 19
    """

    pass


@configclass
class PegObservationNoAngvelEnvCfg(FactoryTaskPegInsertCfg):
    """Remove end-effector angular velocity."""

    obs_order = [
        "fingertip_pos_rel_fixed",
        "fingertip_quat",
        "ee_linvel",
    ]


@configclass
class PegObservationNoLinvelEnvCfg(FactoryTaskPegInsertCfg):
    """Remove end-effector linear velocity."""

    obs_order = [
        "fingertip_pos_rel_fixed",
        "fingertip_quat",
        "ee_angvel",
    ]


@configclass
class PegObservationNoVelocityEnvCfg(FactoryTaskPegInsertCfg):
    """Remove both linear and angular velocity."""

    obs_order = [
        "fingertip_pos_rel_fixed",
        "fingertip_quat",
    ]


@configclass
class PegObservationPositionOnlyEnvCfg(FactoryTaskPegInsertCfg):
    """Use relative position only, in addition to previous action."""

    obs_order = [
        "fingertip_pos_rel_fixed",
    ]


@configclass
class PegObservationPoseOnlyEnvCfg(FactoryTaskPegInsertCfg):
    """Use relative position and orientation, but no velocities."""

    obs_order = [
        "fingertip_pos_rel_fixed",
        "fingertip_quat",
    ]