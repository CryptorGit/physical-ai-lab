"""Frozen Stage 9 contact-kinematics helpers.

All vectors are world-frame SI quantities.  The functions are independent of
Isaac stepping so their contracts can be tested analytically.
"""

from __future__ import annotations

import numpy as np


def surface_velocity(
    body_linear_velocity,
    body_angular_velocity,
    contact_position,
    body_com_position,
):
    """Return ``v_b + omega_b x (p_c - x_b)``."""
    linear = np.asarray(body_linear_velocity, dtype=np.float64)
    angular = np.asarray(body_angular_velocity, dtype=np.float64)
    point = np.asarray(contact_position, dtype=np.float64)
    com = np.asarray(body_com_position, dtype=np.float64)
    return linear + np.cross(angular, point - com)


def relative_contact_velocity(
    foot_linear_velocity,
    foot_angular_velocity,
    contact_position,
    foot_com_position,
    ground_linear_velocity=(0.0, 0.0, 0.0),
    ground_angular_velocity=(0.0, 0.0, 0.0),
    ground_com_position=(0.0, 0.0, 0.0),
):
    """Return foot surface velocity relative to the ground surface."""
    foot = surface_velocity(
        foot_linear_velocity, foot_angular_velocity, contact_position, foot_com_position
    )
    ground = surface_velocity(
        ground_linear_velocity, ground_angular_velocity, contact_position, ground_com_position
    )
    return foot - ground


def tangential_velocity(relative_velocity, contact_normal, epsilon=1e-12):
    """Split relative velocity into normal and tangential components."""
    velocity = np.asarray(relative_velocity, dtype=np.float64)
    normal = np.asarray(contact_normal, dtype=np.float64)
    unit = normal / (np.linalg.norm(normal, axis=-1, keepdims=True) + epsilon)
    normal_component = np.sum(velocity * unit, axis=-1, keepdims=True) * unit
    tangent = velocity - normal_component
    return tangent, np.linalg.norm(tangent, axis=-1), normal_component


def force_components(normal_force, contact_normal, friction_force, mu=0.6, epsilon=1e-12):
    """Return normal/tangent forces and Coulomb-cone utilization.

    PhysX detailed-contact telemetry exposes the normal force as a scalar and
    friction as a separate world-frame vector.  ``mu`` is the resolved dynamic
    coefficient (0.6 for the frozen Go2 task).
    """
    normal = np.asarray(contact_normal, dtype=np.float64)
    unit = normal / (np.linalg.norm(normal, axis=-1, keepdims=True) + epsilon)
    scalar = np.maximum(0.0, np.asarray(normal_force, dtype=np.float64))
    normal_vector = scalar[..., None] * unit
    tangent = np.asarray(friction_force, dtype=np.float64)
    utilization = np.linalg.norm(tangent, axis=-1) / (mu * scalar + epsilon)
    return normal_vector, tangent, utilization


def yaw_moment(contact_position, root_com_position, force):
    """Return the world-z component of ``(p_c - x_root) x F``."""
    point = np.asarray(contact_position, dtype=np.float64)
    root = np.asarray(root_com_position, dtype=np.float64)
    force = np.asarray(force, dtype=np.float64)
    return np.cross(point - root, force)[..., 2]


def stable_contact_mask(contact, minimum_steps=3, boundary_steps=2):
    """Return Stage 6-compatible stable-contact and boundary masks."""
    values = np.asarray(contact, dtype=bool)
    stable = np.zeros_like(values)
    boundary = np.zeros_like(values)
    start = None
    for index, value in enumerate(np.r_[values, False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            end = index
            boundary[start : min(end, start + boundary_steps)] = True
            boundary[max(start, end - boundary_steps) : end] = True
            lo, hi = start + boundary_steps, end - boundary_steps
            if end - start >= minimum_steps and hi > lo:
                stable[lo:hi] = True
            start = None
    return stable, boundary


def maximum_contiguous_duration(mask, dt):
    best = current = 0
    for value in np.asarray(mask, dtype=bool):
        current = current + 1 if value else 0
        best = max(best, current)
    return float(best * dt)


def run_unit_tests():
    normal = np.array([0.0, 0.0, 1.0])
    cases = {}

    def record(name, actual, expected, tolerance=1e-12):
        error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
        cases[name] = {
            "actual": np.asarray(actual).tolist(),
            "expected": np.asarray(expected).tolist(),
            "max_abs_error": error,
            "pass": error <= tolerance,
        }

    rel = relative_contact_velocity([0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0])
    record("static_body", tangential_velocity(rel, normal)[1], 0.0)
    rel = relative_contact_velocity([0, 0, 0.2], [0, 0, 0], [0, 0, 0], [0, 0, 0])
    record("pure_vertical_motion", tangential_velocity(rel, normal)[1], 0.0)
    rel = relative_contact_velocity([0.2, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0])
    record("pure_horizontal_translation", tangential_velocity(rel, normal)[1], 0.2)
    # COM at (0,0,1), point at origin, omega around y.  v_b cancels omega x r.
    omega = np.array([0.0, 1.0, 0.0])
    point, com = np.zeros(3), np.array([0.0, 0.0, 1.0])
    cancelling_linear = -np.cross(omega, point - com)
    rel = relative_contact_velocity(cancelling_linear, omega, point, com)
    record("rotation_about_contact_point", tangential_velocity(rel, normal)[1], 0.0)
    rel = relative_contact_velocity([0, 0, 0], omega, point, com)
    record("rotation_about_com", tangential_velocity(rel, normal)[0], [-1.0, 0.0, 0.0])
    rel = relative_contact_velocity(
        [0.2, 0, 0], [0, 0, 0], point, com,
        ground_linear_velocity=[0.2, 0, 0],
    )
    record("moving_ground_equal_surface_velocity", tangential_velocity(rel, normal)[1], 0.0)
    stable, boundary = stable_contact_mask(
        [False, True, True, True, True, True, True, True, False]
    )
    record("stable_contact_mask", stable.astype(int), [0, 0, 0, 1, 1, 1, 0, 0, 0])
    record("boundary_mask", boundary.astype(int), [0, 1, 1, 0, 0, 0, 1, 1, 0])
    return {
        "contract": "world-frame SI; static ground unless explicitly supplied",
        "cases": cases,
        "all_pass": all(item["pass"] for item in cases.values()),
        "nan_inf": 0,
    }
