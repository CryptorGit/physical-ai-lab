import importlib.util
import sys
from pathlib import Path

import jax
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_v60_bounded_yaw_pilot.py"


def _module():
    spec = importlib.util.spec_from_file_location("v60_train_reward", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_initial_state_and_controller_inputs_match_between_arms():
    module = _module()
    control = module.V60YawPilotJoystick(
        objective_mode="old_unbounded_dot"
    )
    treatment = module.V60YawPilotJoystick(
        objective_mode="bounded_command_centered_gaussian"
    )
    key = jax.random.PRNGKey(module.SEED)
    control_state = control.reset(key)
    treatment_state = treatment.reset(key)
    np.testing.assert_array_equal(control_state.data.qpos, treatment_state.data.qpos)
    np.testing.assert_array_equal(control_state.data.qvel, treatment_state.data.qvel)
    np.testing.assert_array_equal(
        control_state.info["command"], treatment_state.info["command"]
    )
    np.testing.assert_array_equal(
        control_state.obs["state"], treatment_state.obs["state"]
    )
