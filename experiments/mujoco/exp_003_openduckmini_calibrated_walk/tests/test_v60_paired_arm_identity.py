import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_v60_bounded_yaw_pilot.py"


def _module():
    spec = importlib.util.spec_from_file_location("v60_train", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_resolved_configs_only_have_allowed_differences():
    module = _module()
    control = module.resolved_config("control")
    treatment = module.resolved_config("treatment")
    assert module.arm_identity_view(control) == module.arm_identity_view(
        treatment
    )
    differing = {
        key for key in control if control.get(key) != treatment.get(key)
    }
    assert differing == {"objective_name", "run_name", "output_path"}
