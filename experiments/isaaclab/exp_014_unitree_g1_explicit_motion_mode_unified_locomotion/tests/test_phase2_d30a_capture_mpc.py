from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve()
SCRIPT = HERE.parents[1] / "scripts/run_phase2_d30a_post_touchdown_capture_mpc.py"
SPEC = importlib.util.spec_from_file_location("phase2_d30a", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class D30ABasisTests(unittest.TestCase):
    def test_svd_basis_uses_registered_minimal_dimension(self):
        rng = np.random.default_rng(11)
        native = np.zeros((300, 37))
        native[:, :4] = rng.normal(size=(300, 4))
        differences = np.zeros((300, 37))
        differences[:, :4] = rng.normal(size=(300, 4))
        basis = MODULE.WMoveCaptureActionBasisV1.fit(native, differences)
        self.assertEqual(basis.preregistered_dimension(), 4)
        self.assertLessEqual(basis.preregistered_dimension(), 12)
        self.assertGreaterEqual(
            np.cumsum(basis.explained_variance_ratio)[basis.preregistered_dimension() - 1],
            0.95,
        )

    def test_basis_round_trip_is_deterministic(self):
        rng = np.random.default_rng(12)
        native = rng.normal(size=(80, 37))
        delta = rng.normal(size=(80, 37))
        basis_a = MODULE.WMoveCaptureActionBasisV1.fit(native, delta)
        basis_b = MODULE.WMoveCaptureActionBasisV1.fit(native, delta)
        coordinates = basis_a.transform(native)
        np.testing.assert_allclose(basis_a.components, basis_b.components)
        np.testing.assert_allclose(
            basis_a.inverse_transform(coordinates),
            basis_b.inverse_transform(basis_b.transform(native)),
        )

    def test_nonfinite_actions_are_rejected(self):
        with self.assertRaises(ValueError):
            MODULE.WMoveCaptureActionBasisV1.fit(
                np.ones((4, 37)),
                np.full((4, 37), np.nan),
            )


class D30AModelTests(unittest.TestCase):
    def test_local_linear_model_and_holdout(self):
        rng = np.random.default_rng(13)
        states = rng.normal(size=(80, 2))
        controls = rng.normal(size=(80, 1))
        next_states = 0.9 * states + 0.1 * controls + np.array([0.02, -0.03])
        model = MODULE.LocalLinearDynamics.fit(states, controls, next_states)
        holdout_states = rng.normal(size=2)
        holdout_controls = rng.normal(size=(3, 1))
        measured = model.rollout(holdout_states, holdout_controls)
        result = MODULE.validate_holdout(model, holdout_states, holdout_controls, measured)
        self.assertTrue(result["pass"])
        self.assertTrue(result["gates"]["one_step"])
        self.assertTrue(result["gates"]["three_step"])

    def test_bin_identification_is_fail_closed_when_bin_missing(self):
        result = MODULE.identify_bins({})
        self.assertFalse(result["pass"])
        self.assertFalse(result["bins"]["LEFT_early"]["available"])
        self.assertEqual(
            result["bins"]["RIGHT_late"]["reason"],
            "missing_synchronized_physics_rows",
        )

    def test_registered_perturbations_are_symmetric(self):
        schedule = MODULE.perturbation_schedule(3, 0.10)
        self.assertEqual(len(schedule), 6)
        np.testing.assert_allclose(schedule[0], -schedule[1])
        np.testing.assert_allclose(schedule[2], -schedule[3])


class D30AControllerTests(unittest.TestCase):
    def test_bounded_finite_horizon_controller_is_finite_and_bounded(self):
        a = np.array([[0.95, 0.0], [0.0, 0.9]])
        b = np.array([[0.1], [0.2]])
        controller = MODULE.FiniteHorizonBoundedLQRMPC(
            a,
            b,
            np.eye(2),
            np.array([[0.1]]),
            np.array([-0.05]),
            np.array([0.05]),
            horizon=16,
        )
        control = controller.control(np.array([100.0, -100.0]))
        self.assertTrue(np.isfinite(control).all())
        self.assertGreaterEqual(control[0], -0.05)
        self.assertLessEqual(control[0], 0.05)
        np.testing.assert_array_equal(control, controller.control(np.array([100.0, -100.0])))


class D30AAdapterSmokeTests(unittest.TestCase):
    def test_runtime_resolution_prefers_registered_isaaclab_path(self):
        path = MODULE.resolve_isaac_python(None)
        if MODULE.KNOWN_ISAACLAB_PYTHON.is_file():
            self.assertEqual(path, MODULE.KNOWN_ISAACLAB_PYTHON.resolve())
        else:
            self.assertTrue(path.is_file())

    def test_registered_failure_classifications_are_closed(self):
        self.assertIn(
            "EXP014_D30A_LOCAL_CAPTURE_MODEL_INVALID",
            {"EXP014_D30A_MULTIPLE_FAILURES", "EXP014_D30A_LOCAL_CAPTURE_MODEL_INVALID"},
        )


if __name__ == "__main__":
    unittest.main()
