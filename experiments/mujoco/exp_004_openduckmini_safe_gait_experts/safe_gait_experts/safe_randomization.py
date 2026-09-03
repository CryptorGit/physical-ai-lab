"""Name-resolved, head-safe domain randomization for MuJoCo/MJX.

MuJoCo's CPU model owns names; MJX intentionally does not.  This module
resolves model-specific IDs and actuator addresses once on the CPU, then
captures only integer metadata in the callable handed to Brax/MJX.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .contract import (
    ACTUATOR_JOINT_ORDER,
    CONTRACT,
    HEAD_JOINTS,
    RESET_NOISE_MARGIN_RAD,
    SAFE_JOINT_LIMITS,
)


@dataclass(frozen=True)
class RandomizationTargets:
    """IDs resolved from one concrete MuJoCo CPU model."""

    floor_geom_id: int
    floor_body_id: int
    root_body_id: int
    floor_body_has_mass: bool
    root_body_has_mass: bool


@dataclass(frozen=True)
class RandomizationConfig:
    """Conservative ranges frozen by ``contract.json``."""

    floor_friction_range: tuple[float, float] = (0.5, 1.0)
    dof_friction_scale_range: tuple[float, float] = (0.9, 1.1)
    armature_scale_range: tuple[float, float] = (1.0, 1.05)
    root_com_delta_range_m: tuple[float, float] = (-0.05, 0.05)
    body_mass_scale_range: tuple[float, float] = (0.9, 1.1)
    root_payload_delta_range_kg: tuple[float, float] = (-0.1, 0.1)
    actuator_gain_scale_range: tuple[float, float] = (0.9, 1.1)
    mass_epsilon: float = 1e-12
    minimum_positive_mass: float = 1e-6

    def __post_init__(self) -> None:
        for field_name in (
            "floor_friction_range",
            "dof_friction_scale_range",
            "armature_scale_range",
            "root_com_delta_range_m",
            "body_mass_scale_range",
            "root_payload_delta_range_kg",
            "actuator_gain_scale_range",
        ):
            lower, upper = getattr(self, field_name)
            if not lower <= upper:
                raise ValueError(f"invalid range for {field_name}: {lower}, {upper}")
        if self.floor_friction_range[0] <= 0.0:
            raise ValueError("floor friction must remain positive")
        if self.body_mass_scale_range[0] <= 0.0:
            raise ValueError("body mass scaling must remain positive")
        if self.mass_epsilon < 0.0 or self.minimum_positive_mass <= 0.0:
            raise ValueError("mass thresholds must be positive")


def _named_id(cpu_model: Any, kind: str, name: str) -> int:
    getter = getattr(cpu_model, kind, None)
    if getter is None or not callable(getter):
        raise TypeError(f"CPU model does not expose {kind}(name)")
    try:
        named_view = getter(name)
    except (KeyError, ValueError) as exc:
        raise ValueError(f"MuJoCo {kind} named {name!r} does not exist") from exc
    if named_view is None or not hasattr(named_view, "id"):
        raise ValueError(f"MuJoCo {kind} named {name!r} does not exist")
    object_id = int(named_view.id)
    if object_id < 0:
        raise ValueError(f"MuJoCo {kind} named {name!r} has invalid id {object_id}")
    return object_id


def resolve_randomization_targets(
    cpu_model: Any,
    *,
    floor_geom_name: str = "floor",
    root_body_name: str = "trunk_assembly",
    mass_epsilon: float = 1e-12,
) -> RandomizationTargets:
    """Resolve floor/root identifiers using names from a MuJoCo CPU model."""

    floor_geom_id = _named_id(cpu_model, "geom", floor_geom_name)
    root_body_id = _named_id(cpu_model, "body", root_body_name)

    geom_bodyid = np.asarray(cpu_model.geom_bodyid)
    body_mass = np.asarray(cpu_model.body_mass, dtype=np.float64)
    if floor_geom_id >= geom_bodyid.shape[0]:
        raise ValueError("resolved floor geom id is outside geom_bodyid")
    floor_body_id = int(geom_bodyid[floor_geom_id])
    if not 0 <= floor_body_id < body_mass.shape[0]:
        raise ValueError("resolved floor body id is outside body_mass")
    if not 0 <= root_body_id < body_mass.shape[0]:
        raise ValueError("resolved root body id is outside body_mass")
    if np.any(body_mass < 0.0):
        raise ValueError("MuJoCo body masses must not be negative")

    return RandomizationTargets(
        floor_geom_id=floor_geom_id,
        floor_body_id=floor_body_id,
        root_body_id=root_body_id,
        floor_body_has_mass=bool(body_mass[floor_body_id] > mass_epsilon),
        root_body_has_mass=bool(body_mass[root_body_id] > mass_epsilon),
    )


def actuator_name_to_index(cpu_model: Any) -> dict[str, int]:
    """Build a duplicate-checked actuator-name-to-index mapping."""

    result: dict[str, int] = {}
    for index in range(int(cpu_model.nu)):
        actuator = cpu_model.actuator(index)
        name = str(actuator.name)
        if not name:
            raise ValueError(f"actuator {index} is unnamed")
        if name in result:
            raise ValueError(f"duplicate actuator name: {name}")
        result[name] = index
    return result


def build_qpos_noise_scale(
    actuator_indices: Mapping[str, int],
    *,
    hip_scale: float = 0.03,
    knee_scale: float = 0.05,
    ankle_scale: float = 0.08,
) -> np.ndarray:
    """Return all 14 qpos noise scales in the model's actuator order.

    Exact names, not left-leg list positions, determine every assignment.
    Consequently both right-leg joints and arbitrarily reordered actuators are
    handled correctly.  Head entries are contractually fixed to exact zero.
    """

    expected = set(ACTUATOR_JOINT_ORDER)
    actual = set(actuator_indices)
    if actual != expected:
        raise ValueError(
            "actuator set mismatch: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    if any(float(value) < 0.0 for value in (hip_scale, knee_scale, ankle_scale)):
        raise ValueError("noise scales must be non-negative")

    indices = [int(actuator_indices[name]) for name in ACTUATOR_JOINT_ORDER]
    if len(set(indices)) != 14 or set(indices) != set(range(14)):
        raise ValueError("actuator indices must be a permutation of range(14)")

    scale_by_name = {
        name: (
            0.0
            if name in HEAD_JOINTS
            else float(knee_scale)
            if name.endswith("_knee")
            else float(ankle_scale)
            if name.endswith("_ankle")
            else float(hip_scale)
        )
        for name in ACTUATOR_JOINT_ORDER
    }
    result = np.empty(14, dtype=np.float64)
    for name, index in actuator_indices.items():
        result[int(index)] = scale_by_name[name]
    return result


def clip_reset_qpos_to_physical_safe_limits(
    actuator_qpos: Any,
    joint_names: tuple[str, ...] = ACTUATOR_JOINT_ORDER,
    *,
    noise_applied: bool = False,
    reset_noise_margin_rad: float = RESET_NOISE_MARGIN_RAD,
) -> np.ndarray:
    """Apply the reset-only qpos contract without an inner-target teleport.

    Zero noise preserves exact ``SAFE_INIT_POS``.  When noise is enabled, leg
    qpos is clipped 0.005 rad inside the physical SAFE bounds.  This reset
    margin is deliberately independent of the much larger desired-target
    margin.  Head qpos is always forced to exact zero.
    """

    names = tuple(joint_names)
    if len(names) != 14 or set(names) != set(ACTUATOR_JOINT_ORDER):
        raise ValueError("reset qpos requires the exact 14 actuator names")
    values = np.asarray(actuator_qpos, dtype=np.float64)
    if values.shape != (14,) or not np.all(np.isfinite(values)):
        raise ValueError("reset actuator qpos must be one finite 14-axis vector")
    margin = float(reset_noise_margin_rad)
    if not np.isfinite(margin) or margin != RESET_NOISE_MARGIN_RAD:
        raise ValueError(
            f"reset noise margin must remain exactly {RESET_NOISE_MARGIN_RAD}"
        )
    inward_margin = margin if bool(noise_applied) else 0.0
    result = values.copy()
    for index, name in enumerate(names):
        if name in HEAD_JOINTS:
            result[index] = 0.0
        else:
            lower, upper = SAFE_JOINT_LIMITS[name]
            result[index] = np.clip(
                result[index], lower + inward_margin, upper - inward_margin
            )
    return result


def _resolve_actuated_joint_addresses(
    cpu_model: Any, actuator_indices: Mapping[str, int]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    transmission = np.asarray(cpu_model.actuator_trnid)
    if transmission.shape != (14, 2):
        raise ValueError(
            "expected 14 direct joint actuator transmissions, "
            f"got shape {transmission.shape}"
        )
    joint_ids = transmission[:, 0].astype(np.int64)
    if np.any(joint_ids < 0):
        raise ValueError("every actuator must reference a joint")

    jnt_dofadr = np.asarray(cpu_model.jnt_dofadr)
    jnt_qposadr = np.asarray(cpu_model.jnt_qposadr)
    if np.any(joint_ids >= len(jnt_dofadr)) or np.any(joint_ids >= len(jnt_qposadr)):
        raise ValueError("actuator transmission references an invalid joint")

    # Reconstruct in actuator-index order so model XML ordering cannot leak
    # into semantic joint grouping.
    names_by_index = [""] * 14
    for name, index in actuator_indices.items():
        names_by_index[int(index)] = name
    if any(not name for name in names_by_index):
        raise ValueError("actuator index mapping is incomplete")

    dof_addresses = tuple(int(jnt_dofadr[joint_ids[index]]) for index in range(14))
    qpos_addresses = tuple(
        int(jnt_qposadr[joint_ids[index]]) for index in range(14)
    )
    if len(set(dof_addresses)) != 14 or len(set(qpos_addresses)) != 14:
        raise ValueError("actuated joints must have unique dof and qpos addresses")
    return dof_addresses, qpos_addresses


def scale_body_masses_with_payload(
    body_mass: Any,
    mass_scale: Any,
    *,
    root_body_id: int,
    payload_delta: Any,
    mass_epsilon: float = 1e-12,
    minimum_positive_mass: float = 1e-6,
    xp: Any = np,
) -> Any:
    """Scale masses and add payload only when the root was already massive.

    Multiplication keeps every exactly massless body at zero.  The conditional
    root update prevents the historical failure mode where additive payload
    turns a world, floor, or grouping body into a physical link.
    """

    body_mass_array = xp.asarray(body_mass)
    mass_scale_array = xp.asarray(mass_scale, dtype=body_mass_array.dtype)
    scaled = body_mass_array * mass_scale_array
    root_base_mass = body_mass_array[root_body_id]
    positive_candidate = xp.maximum(
        scaled[root_body_id] + payload_delta,
        xp.asarray(minimum_positive_mass, dtype=body_mass_array.dtype),
    )
    replacement = xp.where(
        root_base_mass > mass_epsilon,
        positive_candidate,
        scaled[root_body_id],
    )
    if hasattr(scaled, "at"):
        return scaled.at[root_body_id].set(replacement)
    result = np.array(scaled, copy=True)
    result[root_body_id] = replacement
    return result


@dataclass(frozen=True)
class ResolvedDomainRandomizer:
    """Callable randomizer ready for Brax's DomainRandomizationVmapWrapper."""

    targets: RandomizationTargets
    actuated_dof_addresses: tuple[int, ...]
    actuated_qpos_addresses: tuple[int, ...]
    qpos_noise_scale: tuple[float, ...]
    config: RandomizationConfig = RandomizationConfig()

    def __call__(self, model: Any, rng: Any) -> tuple[Any, Any]:
        return domain_randomize(
            model,
            rng,
            targets=self.targets,
            actuated_dof_addresses=self.actuated_dof_addresses,
            actuated_qpos_addresses=self.actuated_qpos_addresses,
            qpos_noise_scale=self.qpos_noise_scale,
            config=self.config,
        )


def make_domain_randomizer(
    cpu_model: Any,
    *,
    floor_geom_name: str | None = None,
    root_body_name: str | None = None,
    config: RandomizationConfig | None = None,
) -> ResolvedDomainRandomizer:
    """Resolve CPU-model metadata and create the MJX randomizer callable."""

    config = config or RandomizationConfig()
    names = CONTRACT["model_names"]
    targets = resolve_randomization_targets(
        cpu_model,
        floor_geom_name=floor_geom_name or str(names["floor_geom"]),
        root_body_name=root_body_name or str(names["root_body"]),
        mass_epsilon=config.mass_epsilon,
    )
    actuator_indices = actuator_name_to_index(cpu_model)
    noise = CONTRACT["qpos_noise_scale_rad"]
    qpos_noise_scale = build_qpos_noise_scale(
        actuator_indices,
        hip_scale=float(noise["hip"]),
        knee_scale=float(noise["knee"]),
        ankle_scale=float(noise["ankle"]),
    )
    dof_addresses, qpos_addresses = _resolve_actuated_joint_addresses(
        cpu_model, actuator_indices
    )
    return ResolvedDomainRandomizer(
        targets=targets,
        actuated_dof_addresses=dof_addresses,
        actuated_qpos_addresses=qpos_addresses,
        qpos_noise_scale=tuple(float(value) for value in qpos_noise_scale),
        config=config,
    )


def domain_randomize(
    model: Any,
    rng: Any,
    *,
    targets: RandomizationTargets,
    actuated_dof_addresses: tuple[int, ...],
    actuated_qpos_addresses: tuple[int, ...],
    qpos_noise_scale: tuple[float, ...],
    config: RandomizationConfig = RandomizationConfig(),
) -> tuple[Any, Any]:
    """Randomize an MJX model batch using CPU-resolved identifiers."""

    try:
        import jax
        import jax.numpy as jp
    except ModuleNotFoundError as exc:  # pragma: no cover - host dependent
        raise RuntimeError("domain_randomize requires JAX/MJX at training time") from exc

    if len(actuated_dof_addresses) != 14:
        raise ValueError("expected 14 actuated dof addresses")
    if len(actuated_qpos_addresses) != 14 or len(qpos_noise_scale) != 14:
        raise ValueError("expected 14 actuated qpos addresses and noise scales")

    dof_addresses = jp.asarray(actuated_dof_addresses, dtype=jp.int32)
    qpos_addresses = jp.asarray(actuated_qpos_addresses, dtype=jp.int32)
    qpos_scale = jp.asarray(qpos_noise_scale, dtype=model.qpos0.dtype)
    if getattr(rng, "ndim", None) == 1:
        rng = rng[None, :]

    @jax.vmap
    def randomize_one(key: Any) -> tuple[Any, ...]:
        keys = jax.random.split(key, 8)

        floor_friction = jax.random.uniform(
            keys[0],
            minval=config.floor_friction_range[0],
            maxval=config.floor_friction_range[1],
            dtype=model.geom_friction.dtype,
        )
        geom_friction = model.geom_friction.at[targets.floor_geom_id, 0].set(
            floor_friction
        )

        friction_factor = jax.random.uniform(
            keys[1],
            shape=(14,),
            minval=config.dof_friction_scale_range[0],
            maxval=config.dof_friction_scale_range[1],
            dtype=model.dof_frictionloss.dtype,
        )
        dof_frictionloss = model.dof_frictionloss.at[dof_addresses].set(
            model.dof_frictionloss[dof_addresses] * friction_factor
        )

        armature_factor = jax.random.uniform(
            keys[2],
            shape=(14,),
            minval=config.armature_scale_range[0],
            maxval=config.armature_scale_range[1],
            dtype=model.dof_armature.dtype,
        )
        dof_armature = model.dof_armature.at[dof_addresses].set(
            model.dof_armature[dof_addresses] * armature_factor
        )

        root_com_delta = jax.random.uniform(
            keys[3],
            shape=(3,),
            minval=config.root_com_delta_range_m[0],
            maxval=config.root_com_delta_range_m[1],
            dtype=model.body_ipos.dtype,
        )
        root_ipos = jp.where(
            model.body_mass[targets.root_body_id] > config.mass_epsilon,
            model.body_ipos[targets.root_body_id] + root_com_delta,
            model.body_ipos[targets.root_body_id],
        )
        body_ipos = model.body_ipos.at[targets.root_body_id].set(root_ipos)

        body_mass_factor = jax.random.uniform(
            keys[4],
            shape=model.body_mass.shape,
            minval=config.body_mass_scale_range[0],
            maxval=config.body_mass_scale_range[1],
            dtype=model.body_mass.dtype,
        )
        payload_delta = jax.random.uniform(
            keys[5],
            minval=config.root_payload_delta_range_kg[0],
            maxval=config.root_payload_delta_range_kg[1],
            dtype=model.body_mass.dtype,
        )
        body_mass = scale_body_masses_with_payload(
            model.body_mass,
            body_mass_factor,
            root_body_id=targets.root_body_id,
            payload_delta=payload_delta,
            mass_epsilon=config.mass_epsilon,
            minimum_positive_mass=config.minimum_positive_mass,
            xp=jp,
        )

        qpos_delta = jax.random.uniform(
            keys[6],
            shape=(14,),
            minval=-1.0,
            maxval=1.0,
            dtype=model.qpos0.dtype,
        ) * qpos_scale
        qpos0 = model.qpos0.at[qpos_addresses].set(
            model.qpos0[qpos_addresses] + qpos_delta
        )

        gain_factor = jax.random.uniform(
            keys[7],
            shape=(14,),
            minval=config.actuator_gain_scale_range[0],
            maxval=config.actuator_gain_scale_range[1],
            dtype=model.actuator_gainprm.dtype,
        )
        current_kp = model.actuator_gainprm[:, 0]
        randomized_kp = current_kp * gain_factor
        actuator_gainprm = model.actuator_gainprm.at[:, 0].set(randomized_kp)
        actuator_biasprm = model.actuator_biasprm.at[:, 1].set(-randomized_kp)

        return (
            geom_friction,
            body_ipos,
            dof_frictionloss,
            dof_armature,
            body_mass,
            qpos0,
            actuator_gainprm,
            actuator_biasprm,
        )

    (
        geom_friction,
        body_ipos,
        dof_frictionloss,
        dof_armature,
        body_mass,
        qpos0,
        actuator_gainprm,
        actuator_biasprm,
    ) = randomize_one(rng)

    replacements = {
        "geom_friction": geom_friction,
        "body_ipos": body_ipos,
        "dof_frictionloss": dof_frictionloss,
        "dof_armature": dof_armature,
        "body_mass": body_mass,
        "qpos0": qpos0,
        "actuator_gainprm": actuator_gainprm,
        "actuator_biasprm": actuator_biasprm,
    }
    in_axes = jax.tree_util.tree_map(lambda _: None, model).tree_replace(
        {name: 0 for name in replacements}
    )
    return model.tree_replace(replacements), in_axes
