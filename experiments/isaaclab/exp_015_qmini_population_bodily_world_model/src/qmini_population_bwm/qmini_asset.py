"""Official Qmini URDF audit and Isaac Lab asset configuration.

The XML parser is intentionally independent of Isaac Sim so that the source
contract can be checked in a normal Python environment. Isaac Lab is imported
only by build_qmini_articulation_cfg.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import math
from typing import Any
import xml.etree.ElementTree as ET


EXP_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_ASSET_ROOT = EXP_ROOT / "assets" / "qmini_official"
OFFICIAL_URDF_PATH = OFFICIAL_ASSET_ROOT / "urdf" / "Qmini.urdf"

QMINI_JOINT_ORDER: tuple[str, ...] = (
    "LL-joint1",
    "LL-joint2",
    "LL-joint3",
    "LL-joint4",
    "LL-joint5",
    "RL-joint1",
    "RL-joint2",
    "RL-joint3",
    "RL-joint4",
    "RL-joint5",
)
LEFT_JOINTS = QMINI_JOINT_ORDER[:5]
RIGHT_JOINTS = QMINI_JOINT_ORDER[5:]
EXPECTED_LOCOMOTION_JOINT_COUNT = 10

# The current official URDF has no neck joint. The README/DIY material
# documents an 11th physical 8010 motor reserved for expansion, so we do not
# invent a neck joint in the simulator contract.
NECK_JOINT_NAME: str | None = None


def _float_attr(element: ET.Element | None, name: str, default: float | None = None) -> float | None:
    if element is None or name not in element.attrib:
        return default
    return float(element.attrib[name])


def _vec(element: ET.Element | None, name: str, length: int = 3) -> tuple[float, ...] | None:
    if element is None or name not in element.attrib:
        return None
    values = tuple(float(value) for value in element.attrib[name].split())
    if len(values) != length:
        raise ValueError(f"expected {length} values in {name!r}, got {values!r}")
    return values


@dataclass(frozen=True)
class JointSpec:
    name: str
    joint_type: str
    parent: str
    child: str
    axis_xyz: tuple[float, ...] | None
    origin_xyz: tuple[float, ...] | None
    origin_rpy: tuple[float, ...] | None
    lower: float | None
    upper: float | None
    effort: float | None
    velocity: float | None

    @property
    def midpoint(self) -> float:
        if self.lower is None or self.upper is None:
            raise ValueError(f"joint {self.name} has no finite position limits")
        return (self.lower + self.upper) / 2.0

    @property
    def half_range(self) -> float:
        if self.lower is None or self.upper is None:
            raise ValueError(f"joint {self.name} has no finite position limits")
        return (self.upper - self.lower) / 2.0


@dataclass(frozen=True)
class InertialSpec:
    mass: float
    origin_xyz: tuple[float, ...] | None
    origin_rpy: tuple[float, ...] | None
    ixx: float
    ixy: float
    ixz: float
    iyy: float
    iyz: float
    izz: float


@dataclass(frozen=True)
class LinkSpec:
    name: str
    inertial: InertialSpec | None
    visual_meshes: tuple[str, ...]
    collision_meshes: tuple[str, ...]


@dataclass(frozen=True)
class QminiContract:
    urdf_path: str
    root_link: str
    joints: tuple[JointSpec, ...]
    links: tuple[LinkSpec, ...]
    transmission_count: int
    mesh_files: tuple[str, ...]
    foot_collision_links: tuple[str, ...]
    commented_collision_links: tuple[str, ...]

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(joint.name for joint in self.joints)

    @property
    def joint_limits(self) -> dict[str, dict[str, float | None]]:
        return {
            joint.name: {
                "lower": joint.lower,
                "upper": joint.upper,
                "effort": joint.effort,
                "velocity": joint.velocity,
            }
            for joint in self.joints
        }

    @property
    def default_joint_pose_midpoint(self) -> dict[str, float]:
        return {joint.name: joint.midpoint for joint in self.joints}

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["joint_names"] = list(self.joint_names)
        payload["joint_limits"] = self.joint_limits
        payload["default_joint_pose_midpoint"] = self.default_joint_pose_midpoint
        return payload


def _parse_inertial(link: ET.Element) -> InertialSpec | None:
    inertial = link.find("inertial")
    if inertial is None:
        return None
    mass_element = inertial.find("mass")
    inertia_element = inertial.find("inertia")
    if mass_element is None or inertia_element is None:
        raise ValueError(f"link {link.attrib.get('name')} has incomplete inertial block")
    required = ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")
    if any(name not in inertia_element.attrib for name in required):
        raise ValueError(f"link {link.attrib.get('name')} has incomplete inertia block")
    origin = inertial.find("origin")
    return InertialSpec(
        mass=float(mass_element.attrib["value"]),
        origin_xyz=_vec(origin, "xyz"),
        origin_rpy=_vec(origin, "rpy"),
        ixx=float(inertia_element.attrib["ixx"]),
        ixy=float(inertia_element.attrib["ixy"]),
        ixz=float(inertia_element.attrib["ixz"]),
        iyy=float(inertia_element.attrib["iyy"]),
        iyz=float(inertia_element.attrib["iyz"]),
        izz=float(inertia_element.attrib["izz"]),
    )


def _mesh_names(link: ET.Element, kind: str) -> tuple[str, ...]:
    values: list[str] = []
    for element in link.findall(kind):
        mesh = element.find("geometry/mesh")
        if mesh is not None and "filename" in mesh.attrib:
            values.append(mesh.attrib["filename"])
    return tuple(values)


def load_qmini_contract(urdf_path: str | Path = OFFICIAL_URDF_PATH) -> QminiContract:
    """Parse the official Qmini URDF without normalizing or reordering it."""

    path = Path(urdf_path).resolve()
    root = ET.parse(path).getroot()
    links = tuple(
        LinkSpec(
            name=element.attrib["name"],
            inertial=_parse_inertial(element),
            visual_meshes=_mesh_names(element, "visual"),
            collision_meshes=_mesh_names(element, "collision"),
        )
        for element in root.findall("link")
    )
    joints: list[JointSpec] = []
    for element in root.findall("joint"):
        origin = element.find("origin")
        axis = element.find("axis")
        limit = element.find("limit")
        parent = element.find("parent")
        child = element.find("child")
        if parent is None or child is None:
            raise ValueError(f"joint {element.attrib.get('name')} has no parent/child")
        joints.append(
            JointSpec(
                name=element.attrib["name"],
                joint_type=element.attrib["type"],
                parent=parent.attrib["link"],
                child=child.attrib["link"],
                axis_xyz=_vec(axis, "xyz"),
                origin_xyz=_vec(origin, "xyz"),
                origin_rpy=_vec(origin, "rpy"),
                lower=_float_attr(limit, "lower"),
                upper=_float_attr(limit, "upper"),
                effort=_float_attr(limit, "effort"),
                velocity=_float_attr(limit, "velocity"),
            )
        )
    all_meshes = sorted(
        {mesh for link in links for mesh in (*link.visual_meshes, *link.collision_meshes)}
    )
    foot_collision_links = tuple(
        link.name
        for link in links
        if link.name in {"LL_ankle", "RL_ankle"} and link.collision_meshes
    )
    expected_collision_links = {link.name for link in links if link.collision_meshes}
    known_mechanical_links = {
        "base_link",
        "LL_hip_yaw",
        "LL_hip_roll",
        "LL_hip_pitch",
        "LL_knee",
        "LL_ankle",
        "RL_hip_yaw",
        "RL_hip_roll",
        "RL_hip_pitch",
        "RL_knee",
        "RL_ankle",
    }
    commented_collision_links = tuple(sorted(known_mechanical_links - expected_collision_links))
    return QminiContract(
        urdf_path=str(path),
        root_link=links[0].name if links else "UNKNOWN",
        joints=tuple(joints),
        links=links,
        transmission_count=len(root.findall("transmission")),
        mesh_files=tuple(all_meshes),
        foot_collision_links=foot_collision_links,
        commented_collision_links=commented_collision_links,
    )


def validate_qmini_contract(contract: QminiContract) -> list[str]:
    """Return human-readable contract failures; an empty list means PASS."""

    failures: list[str] = []
    if contract.root_link != "base_link":
        failures.append(f"root link is {contract.root_link!r}, expected 'base_link'")
    if len(contract.joints) != EXPECTED_LOCOMOTION_JOINT_COUNT:
        failures.append(f"joint count {len(contract.joints)} != {EXPECTED_LOCOMOTION_JOINT_COUNT}")
    if contract.joint_names != QMINI_JOINT_ORDER:
        failures.append(f"joint order {contract.joint_names!r} != official contract")
    if any(joint.joint_type != "revolute" for joint in contract.joints):
        failures.append("all locomotion joints must be revolute")
    if contract.transmission_count != 0:
        failures.append("official URDF unexpectedly contains transmissions")
    for joint in contract.joints:
        if any(value is None or not math.isfinite(value) for value in (joint.lower, joint.upper)):
            failures.append(f"joint {joint.name} lacks finite position limits")
        elif joint.lower >= joint.upper:
            failures.append(f"joint {joint.name} has inverted position limits")
        if any(value is None or not math.isfinite(value) for value in (joint.effort, joint.velocity)):
            failures.append(f"joint {joint.name} lacks finite URDF effort/velocity limits")
    for link in contract.links:
        if link.inertial is None:
            failures.append(f"link {link.name} lacks inertial data")
        else:
            inertia_values = (
                link.inertial.mass,
                link.inertial.ixx,
                link.inertial.ixy,
                link.inertial.ixz,
                link.inertial.iyy,
                link.inertial.iyz,
                link.inertial.izz,
            )
            if any(not math.isfinite(value) for value in inertia_values):
                failures.append(f"link {link.name} has non-finite inertial data")
    urdf_parent = Path(contract.urdf_path).parent
    for mesh in contract.mesh_files:
        if not (urdf_parent / mesh).resolve().exists():
            failures.append(f"missing mesh {mesh!r}")
    if set(contract.foot_collision_links) != {"LL_ankle", "RL_ankle"}:
        failures.append(f"foot collision links are {contract.foot_collision_links!r}")
    return failures


def _gain_by_joint(pattern_values: tuple[float, ...]) -> dict[str, float]:
    return dict(zip(QMINI_JOINT_ORDER, pattern_values, strict=True))


# The official URDF does not specify actuator gains. These values are kept as
# an explicitly tagged RoboTamer reference, never as Qmini motor ground truth.
ROBOTAMER_REFERENCE_KP = _gain_by_joint((55.0, 105.0, 75.0, 45.0, 30.0) * 2)
ROBOTAMER_REFERENCE_KD = _gain_by_joint((0.3, 2.5, 0.3, 0.5, 0.25) * 2)


def isaaclab_imported_joint_pattern(official_joint_name: str) -> str:
    """Return the observed Isaac Sim 3.x sanitized-name pattern.

    The official URDF names contain hyphens. The Isaac URDF importer replaces
    them and adds importer-specific prefixes/suffixes. This is an import-layer
    name mapping, not a change to the source joint order.
    """

    if official_joint_name not in QMINI_JOINT_ORDER:
        raise ValueError(f"unknown official Qmini joint {official_joint_name!r}")
    token = official_joint_name.replace("-", "")
    return rf".*{token}.*_ZC$"


def canonicalize_isaaclab_joint_name(imported_joint_name: str) -> str | None:
    """Map an imported Isaac joint name back to the official Qmini name."""

    for official_name in QMINI_JOINT_ORDER:
        token = official_name.replace("-", "")
        if token in imported_joint_name and imported_joint_name.endswith("_ZC"):
            return official_name
    return None


def official_to_isaaclab_joint_indices(imported_joint_names: tuple[str, ...] | list[str]) -> tuple[int, ...]:
    """Return the raw Isaac articulation indices in official Qmini order."""

    canonical = tuple(canonicalize_isaaclab_joint_name(name) for name in imported_joint_names)
    if any(name is None for name in canonical):
        raise ValueError(f"unrecognized imported Qmini joint names: {imported_joint_names!r}")
    if set(canonical) != set(QMINI_JOINT_ORDER) or len(canonical) != len(QMINI_JOINT_ORDER):
        raise ValueError(f"imported Qmini joint set mismatch: {canonical!r}")
    return tuple(canonical.index(name) for name in QMINI_JOINT_ORDER)


def build_qmini_articulation_cfg(*, prim_path: str = "/World/Qmini") -> Any:
    """Build an Isaac Lab 3.x ArticulationCfg for the vendored official URDF.

    The URDF effort/velocity limits are joint limits from the official file,
    not claims about the 8010 motor's nominal continuous limits. The PD gains
    and simulation timing are the separately labelled RoboTamer reference
    values audited in manifests/actuator_contract.json.
    """

    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    from isaaclab.sim.spawners.from_files.from_files_cfg import UrdfFileCfg

    contract = load_qmini_contract()
    failures = validate_qmini_contract(contract)
    if failures:
        raise ValueError("cannot build Qmini asset from invalid contract: " + "; ".join(failures))

    effort_limits = {
        isaaclab_imported_joint_pattern(joint.name): float(joint.effort)
        for joint in contract.joints
    }
    velocity_limits = {
        isaaclab_imported_joint_pattern(joint.name): float(joint.velocity)
        for joint in contract.joints
    }
    default_pose = {
        isaaclab_imported_joint_pattern(joint.name): joint.midpoint
        for joint in contract.joints
    }
    zero_velocity = {
        isaaclab_imported_joint_pattern(joint.name): 0.0
        for joint in contract.joints
    }
    stiffness = {
        isaaclab_imported_joint_pattern(joint.name): ROBOTAMER_REFERENCE_KP[joint.name]
        for joint in contract.joints
    }
    damping = {
        isaaclab_imported_joint_pattern(joint.name): ROBOTAMER_REFERENCE_KD[joint.name]
        for joint in contract.joints
    }
    spawn = UrdfFileCfg(
        asset_path=str(OFFICIAL_URDF_PATH),
        fix_base=False,
        merge_fixed_joints=False,
        make_instanceable=False,
        activate_contact_sensors=True,
        # The importer-level joint-drive override is disabled because Isaac
        # Sim resolves its dict keys before sanitizing hyphenated URDF names.
        # The explicit Articulation actuator below applies the same labelled
        # RoboTamer reference gains using the observed importer-name patterns.
        joint_drive=None,
    )
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=spawn,
        init_state=ArticulationCfg.InitialStateCfg(
            # The official source does not publish a simulator root height.
            # 0.45 m is retained only as a provisional RoboTamer reference.
            pos=(0.0, 0.0, 0.45),
            joint_pos=default_pose,
            joint_vel=zero_velocity,
        ),
        actuators={
            "qmini_implicit_pd": ImplicitActuatorCfg(
                joint_names_expr=[isaaclab_imported_joint_pattern(name) for name in QMINI_JOINT_ORDER],
                effort_limit_sim=effort_limits,
                velocity_limit_sim=velocity_limits,
                stiffness=stiffness,
                damping=damping,
            )
        },
    )


def qmini_action_bounds(contract: QminiContract | None = None) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return midpoint/half-range action map derived only from URDF limits."""

    contract = contract or load_qmini_contract()
    return (
        tuple(joint.midpoint for joint in contract.joints),
        tuple(joint.half_range for joint in contract.joints),
    )


def official_link_mass_total(contract: QminiContract | None = None) -> float:
    contract = contract or load_qmini_contract()
    return sum(link.inertial.mass for link in contract.links if link.inertial is not None)
