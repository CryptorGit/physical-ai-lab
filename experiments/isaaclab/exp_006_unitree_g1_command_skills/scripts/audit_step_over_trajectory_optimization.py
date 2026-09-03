"""Limited contact-constrained whole-body trajectory feasibility audit.

This is deliberately an offline audit.  It never writes policy parameters and
does not connect its candidate trajectory to the production action route.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.spatial.transform import Rotation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", default="shared/models/mujoco_menagerie/unitree_g1/scene.xml")
parser.add_argument("--isaac-state", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--max-ik-evals", type=int, default=500)
args = parser.parse_args()

KEYPOINTS = {
    "toe": np.array((0.06383880963290349, 0.0, -0.025807180037281774)),
    "sole": np.array((0.04321213238651294, 0.0, -0.025807180037281774)),
    "heel": np.array((0.022585455140122387, 0.0, -0.025807180037281774)),
}
OBSTACLE = {"height_m": .05, "depth_m": .06, "width_m": 2.20, "front_x_m": .29,
            "rear_x_m": .35, "landing_target_x_m": .41, "clearance_m": .07,
            "landing_margin_m": .03, "optimized_conservative_landing_x_m": .50}
PHASE_DURATIONS = {"double_support_start": .6, "right_support_left_swing": 1.8,
                   "double_support_transfer": .6, "left_support_right_swing": 1.8,
                   "double_support_finish": .8, "standing_recovery": 1.0}


def point(data: mujoco.MjData, body_id: int, local: np.ndarray) -> np.ndarray:
    return data.xpos[body_id] + data.xmat[body_id].reshape(3, 3) @ local


def smoothstep5(t: float) -> float:
    t = np.clip(t, 0.0, 1.0)
    return t**3 * (10 - 15*t + 6*t*t)


def sha256(path: Path) -> str:
    digest = hashlib.sha256(); digest.update(path.read_bytes()); return digest.hexdigest()


class Audit:
    def __init__(self, model_path: Path, isaac: dict):
        spec = mujoco.MjSpec.from_file(str(model_path))
        spec.worldbody.add_geom(name="audit_obstacle", type=mujoco.mjtGeom.mjGEOM_BOX,
                                pos=(.32, 0, .025), size=(.03, 1.10, .025),
                                friction=(1.0, .005, .0001), rgba=(.8, .2, .1, 1))
        self.model = spec.compile(); self.data = mujoco.MjData(self.model); self.isaac = isaac
        self.left_body = self.model.body("left_ankle_roll_link").id
        self.right_body = self.model.body("right_ankle_roll_link").id
        self.pelvis_body = self.model.body("pelvis").id
        self.q0 = np.zeros(self.model.nq); mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.q0[:] = self.data.qpos
        self.q0[:3] = (0, 0, float(isaac["root_pos_w"][2]))
        # The captured Isaac buffers expose xyzw.  The feasibility problem is
        # robot-local, so remove arbitrary world yaw and use MuJoCo wxyz identity.
        self.q0[3:7] = (1, 0, 0, 0)
        isaac_pos = dict(zip(isaac["joint_names"], isaac["joint_pos"]))
        aliases = {"waist_yaw_joint": "torso_joint", "left_elbow_joint": "left_elbow_pitch_joint",
                   "right_elbow_joint": "right_elbow_pitch_joint"}
        for joint_id in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if name == "floating_base_joint": continue
            source = aliases.get(name, name)
            if source in isaac_pos: self.q0[self.model.jnt_qposadr[joint_id]] = isaac_pos[source]
        # Align the authoritative keypoint bottoms with the floor without
        # changing the captured joint configuration.
        self.forward(self.q0)
        pre = self.feet(self.q0)
        self.q0[0] -= .5 * (pre["left"]["sole"][0] + pre["right"]["sole"][0])
        self.q0[1] -= .5 * (pre["left"]["sole"][1] + pre["right"]["sole"][1])
        self.forward(self.q0)
        bottom = min(point(self.data, body, KEYPOINTS[k])[2]
                     for body in (self.left_body, self.right_body) for k in KEYPOINTS)
        self.q0[2] -= bottom
        self.forward(self.q0)
        self.initial = self.feet(self.q0)
        self.variable_joint_names = [
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
            "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint",
            "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
            "right_ankle_pitch_joint", "right_ankle_roll_joint", "waist_yaw_joint",
            "waist_roll_joint", "waist_pitch_joint",
        ]
        self.qadr = [self.model.jnt_qposadr[self.model.joint(name).id] for name in self.variable_joint_names]
        self.dofadr = [self.model.jnt_dofadr[self.model.joint(name).id] for name in self.variable_joint_names]
        self.lo = np.array([self.model.jnt_range[self.model.joint(name).id, 0] for name in self.variable_joint_names])
        self.hi = np.array([self.model.jnt_range[self.model.joint(name).id, 1] for name in self.variable_joint_names])
        margin = .05 * (self.hi - self.lo); self.lo += margin; self.hi -= margin
        self.x0 = self.encode(self.q0)
        self.lower = np.r_[[-.20, -.16, .48], [-.30]*3, self.lo]
        self.upper = np.r_[[.60, .16, .90], [.30]*3, self.hi]
        self.effort = np.asarray(isaac["joint_effort_limits"])
        self.velocity = np.asarray(isaac["joint_velocity_limits"])
        self.isaac_names = isaac["joint_names"]
        self.mj_dof_by_isaac = []
        for name in self.isaac_names:
            mj_name = "waist_yaw_joint" if name == "torso_joint" else name.replace("_elbow_pitch_joint", "_elbow_joint")
            try: self.mj_dof_by_isaac.append(self.model.jnt_dofadr[self.model.joint(mj_name).id])
            except KeyError: self.mj_dof_by_isaac.append(-1)

    def forward(self, q: np.ndarray) -> None:
        self.data.qpos[:] = q; self.data.qvel[:] = 0; mujoco.mj_forward(self.model, self.data)

    def feet(self, q: np.ndarray) -> dict:
        self.forward(q)
        return {side: {name: point(self.data, body, local).copy() for name, local in KEYPOINTS.items()}
                for side, body in (("left", self.left_body), ("right", self.right_body))}

    def encode(self, q: np.ndarray) -> np.ndarray:
        quat_xyzw = np.r_[q[4:7], q[3]]
        return np.r_[q[:3], Rotation.from_quat(quat_xyzw).as_rotvec(), q[self.qadr]]

    def decode(self, x: np.ndarray) -> np.ndarray:
        q = self.q0.copy(); q[:3] = x[:3]
        quat_xyzw = Rotation.from_rotvec(x[3:6]).as_quat(); q[3:7] = np.r_[quat_xyzw[3], quat_xyzw[:3]]
        q[self.qadr] = x[6:]; return q

    def solve_pose(self, seed: np.ndarray, support: tuple[str, ...], swing: str | None,
                   swing_target: np.ndarray | None, com_target: np.ndarray, stage: str) -> tuple[np.ndarray, dict]:
        support_targets = {side: self.initial[side] for side in support}
        q_seed = seed.copy(); x_seed = np.clip(self.encode(q_seed), self.lower, self.upper)

        def residual(x):
            q = self.decode(x); self.forward(q); out = []
            for side in support:
                body = self.left_body if side == "left" else self.right_body
                for name in KEYPOINTS:
                    out.extend(120.0 * (point(self.data, body, KEYPOINTS[name]) - support_targets[side][name]))
            if swing is not None:
                body = self.left_body if swing == "left" else self.right_body
                out.extend(120.0 * (point(self.data, body, KEYPOINTS["sole"]) - swing_target))
                rot = self.data.xmat[body].reshape(3, 3)
                out.extend(8.0 * np.array((rot[2, 0], rot[2, 1])))
            com = self.data.subtree_com[self.pelvis_body]
            out.extend(35.0 * (com[:2] - com_target[:2]))
            out.extend(4.0 * x[3:5])
            out.extend(.12 * (x[6:] - self.x0[6:]))
            return np.asarray(out)

        result = least_squares(residual, x_seed, bounds=(self.lower, self.upper), max_nfev=args.max_ik_evals,
                               xtol=1e-10, ftol=1e-10, gtol=1e-10, verbose=0)
        q = self.decode(result.x); feet = self.feet(q)
        support_error = max((np.linalg.norm(feet[s][k] - support_targets[s][k])
                             for s in support for k in KEYPOINTS), default=0.0)
        swing_error = np.linalg.norm(feet[swing]["sole"] - swing_target) if swing else 0.0
        diagnostics = {"stage": stage, "success": bool(result.success), "status": int(result.status),
                       "message": result.message, "nfev": int(result.nfev), "cost": float(result.cost),
                       "support_keypoint_error_max_m": float(support_error), "swing_target_error_m": float(swing_error),
                       "pelvis_pos": q[:3].tolist(), "com": self.data.subtree_com[self.pelvis_body].tolist(),
                       "feet": {s: {k: v.tolist() for k, v in values.items()} for s, values in feet.items()},
                       "soft_joint_limit_violation_rad": float(max(np.max(self.lo-result.x[6:]), np.max(result.x[6:]-self.hi), 0))}
        return q, diagnostics

    def contact_force_solution(self, q: np.ndarray, qv: np.ndarray, qa: np.ndarray,
                               supports: tuple[str, ...]) -> dict:
        self.data.qpos[:] = q; self.data.qvel[:] = qv; self.data.qacc[:] = qa
        old_disable = self.model.opt.disableflags
        self.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
        mujoco.mj_inverse(self.model, self.data)
        self.model.opt.disableflags = old_disable
        required = self.data.qfrc_inverse.copy()
        columns = []
        # Four real collision support points per foot; forces are the decision variables.
        corners = [np.array((-.05, -.025, -.03)), np.array((-.05, .025, -.03)),
                   np.array((.12, -.03, -.03)), np.array((.12, .03, -.03))]
        for side in supports:
            body = self.left_body if side == "left" else self.right_body
            for local in corners:
                p = point(self.data, body, local); jp = np.zeros((3, self.model.nv)); jr = np.zeros((3, self.model.nv))
                mujoco.mj_jac(self.model, self.data, jp, jr, p, body)
                columns.append(jp.T)
        a = np.concatenate(columns, axis=1) if columns else np.zeros((self.model.nv, 0))
        base_a = a[:6]; target = required[:6]; count = a.shape[1] // 3
        mu = .6
        normal0 = max(mujoco.mj_getTotalmass(self.model)*9.81/max(count, 1), 1)
        # Parameterization enforces fz>0 and a circular friction cone exactly.
        def unpack(z):
            force = np.zeros(3*count)
            for i in range(count):
                ax, ay, log_fz = z[3*i:3*i+3]; fz = math.exp(log_fz)
                denom = math.sqrt(1.0 + ax*ax + ay*ay)
                force[3*i:3*i+3] = (mu*fz*ax/denom, mu*fz*ay/denom, fz)
            return force
        def force_residual(z):
            force = unpack(z)
            return np.r_[(base_a @ force - target) / 100.0, 1e-5*force]
        z0 = np.tile((0.0, 0.0, math.log(normal0)), count)
        lower = np.tile((-20.0, -20.0, math.log(1e-4)), count)
        upper = np.tile((20.0, 20.0, math.log(5000.0)), count)
        result = least_squares(force_residual, z0, bounds=(lower, upper), max_nfev=250) if count else None
        force = unpack(result.x) if result is not None else np.zeros(0); residual = required - a @ force
        torque_ratios = []
        velocity_ratios = []
        ratio_names = []
        for i, dof in enumerate(self.mj_dof_by_isaac):
            if dof < 0: continue
            torque_ratios.append(abs(residual[dof]) / max(self.effort[i], 1e-6))
            velocity_ratios.append(abs(qv[dof]) / max(self.velocity[i], 1e-6))
            ratio_names.append(self.isaac_names[i])
        normals = force[2::3]
        friction_util = [math.hypot(force[3*i], force[3*i+1]) / max(mu*normals[i], 1e-9) for i in range(count)]
        return {"solver_success": bool(result.success) if result else False,
                "base_dynamics_residual_norm": float(np.linalg.norm(residual[:6])),
                "max_effort_utilization": max(torque_ratios, default=math.inf),
                "max_effort_joint": ratio_names[int(np.argmax(torque_ratios))] if torque_ratios else None,
                "max_velocity_utilization": max(velocity_ratios, default=math.inf),
                "max_velocity_joint": ratio_names[int(np.argmax(velocity_ratios))] if velocity_ratios else None,
                "max_friction_cone_utilization": max(friction_util, default=math.inf),
                "min_normal_force_n": float(min(normals, default=-math.inf))}


def main() -> None:
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model).resolve(strict=True); isaac_path = Path(args.isaac_state).resolve(strict=True)
    isaac = json.loads(isaac_path.read_text(encoding="utf-8")); audit = Audit(model_path, isaac)
    q0 = audit.q0.copy(); init = audit.initial
    right_xy = init["right"]["sole"][:2]; left_xy = init["left"]["sole"][:2]
    diagnostics = []
    poses: list[tuple[str, np.ndarray, tuple[str, ...]]] = [("initial_double_support", q0, ("left", "right"))]

    # A: fixed right support, lead path with explicit front/apex/rear knots.
    q, d = audit.solve_pose(q0, ("left", "right"), None, None, np.r_[right_xy, 0], "A_weight_shift_right")
    diagnostics.append(d); poses.append((d["stage"], q, ("left", "right")))
    for label, target in (("A_lead_front_clear", np.array((.27, left_xy[1], .10))),
                          ("A_lead_apex", np.array((.32, left_xy[1], .12))),
                          ("A_lead_rear_clear", np.array((.37, left_xy[1], .10))),
                          ("A_lead_land", np.array((.50, left_xy[1], 0.0)))):
        q, d = audit.solve_pose(q, ("right",), "left", target, np.r_[right_xy, 0], label)
        diagnostics.append(d); poses.append((label, q, ("right",)))
    # B/transfer: fix both at their achieved placements and move COM to left.
    audit.initial["left"] = audit.feet(q)["left"]
    q, d = audit.solve_pose(q, ("left", "right"), None, None, np.r_[[.50, left_xy[1]], 0], "B_double_support_transfer")
    diagnostics.append(d); poses.append((d["stage"], q, ("left", "right")))
    # C: left support, trail path.
    for label, target in (("C_trail_front_clear", np.array((.27, right_xy[1], .10))),
                          ("C_trail_apex", np.array((.32, right_xy[1], .12))),
                          ("C_trail_rear_clear", np.array((.37, right_xy[1], .10))),
                          ("C_trail_land", np.array((.50, right_xy[1], 0.0)))):
        q, d = audit.solve_pose(q, ("left",), "right", target, np.r_[[.50, left_xy[1]], 0], label)
        diagnostics.append(d); poses.append((label, q, ("left",)))
    audit.initial["right"] = audit.feet(q)["right"]
    q, d = audit.solve_pose(q, ("left", "right"), None, None, np.array((.50, 0, 0)), "D_final_recovery")
    diagnostics.append(d); poses.append((d["stage"], q, ("left", "right")))

    # Minimum-jerk interpolation; phase duration bounds remain 0.3--2.0 s.
    durations = [.8, .5, .5, .5, .5, .8, .5, .5, .5, .5, 1.0]
    rows = []; previous_t = 0.0
    for segment in range(len(poses)-1):
        qa, qb = poses[segment][1], poses[segment+1][1]; duration = durations[segment]
        samples = max(4, round(duration/.04))
        for k in range(samples):
            u = k/samples; s = smoothstep5(u); q = (1-s)*qa+s*qb
            q[3:7] /= np.linalg.norm(q[3:7])
            # analytic finite differences are evaluated after assembling all positions
            rows.append({"time_s": previous_t+u*duration, "stage": poses[segment+1][0],
                         "supports": list(poses[segment+1][2]), "qpos": q.tolist()})
        previous_t += duration
    rows.append({"time_s": previous_t, "stage": poses[-1][0], "supports": list(poses[-1][2]), "qpos": poses[-1][1].tolist()})
    times = np.array([row["time_s"] for row in rows]); qmat = np.array([row["qpos"] for row in rows])
    # Differentiate configuration coordinates conservatively; quaternion rates are
    # mapped by MuJoCo's differentiatePos.
    qvel = np.zeros((len(rows), audit.model.nv))
    for i in range(1, len(rows)):
        mujoco.mj_differentiatePos(audit.model, qvel[i], times[i]-times[i-1], qmat[i-1], qmat[i])
    qvel[0] = qvel[1]; qacc = np.gradient(qvel, times, axis=0)
    path_failures = []; dynamics = []
    obstacle_id = audit.model.geom("audit_obstacle").id
    for i, row in enumerate(rows):
        q = qmat[i]; feet = audit.feet(q); stage = row["stage"]
        swing = "left" if stage.startswith("A_lead") else "right" if stage.startswith("C_trail") else None
        clearance = math.inf
        if swing:
            for name in ("toe", "sole", "heel"):
                p = feet[swing][name]
                if OBSTACLE["front_x_m"] <= p[0] <= OBSTACLE["rear_x_m"]:
                    clearance = min(clearance, p[2])
                    if p[2] < OBSTACLE["clearance_m"]: path_failures.append({"row": i, "type": "clearance", "keypoint": name, "value": float(p[2])})
        # MuJoCo mesh/sphere collision query, independent of link origins.
        audit.forward(q)
        obstacle_collision = False; collision_pairs = []
        for cidx in range(audit.data.ncon):
            contact = audit.data.contact[cidx]
            if obstacle_id in (contact.geom1, contact.geom2) and contact.dist <= 0:
                obstacle_collision = True
                other = contact.geom2 if contact.geom1 == obstacle_id else contact.geom1
                collision_pairs.append({"geom": mujoco.mj_id2name(audit.model, mujoco.mjtObj.mjOBJ_GEOM, other),
                                        "body": mujoco.mj_id2name(audit.model, mujoco.mjtObj.mjOBJ_BODY, int(audit.model.geom_bodyid[other])),
                                        "distance_m": float(contact.dist)})
        if obstacle_collision: path_failures.append({"row": i, "type": "obstacle_collision", "pairs": collision_pairs})
        dyn = audit.contact_force_solution(q, qvel[i], qacc[i], tuple(row["supports"])); dyn["row"] = i; dynamics.append(dyn)
        row.update({"qvel": qvel[i].tolist(), "qacc": qacc[i].tolist(), "feet": {s: {k:v.tolist() for k,v in f.items()} for s,f in feet.items()},
                    "clearance_m": None if math.isinf(clearance) else clearance, "obstacle_collision": obstacle_collision,
                    "dynamics": dyn})

    kinematic_feasible = all(d["support_keypoint_error_max_m"] <= .01 and d["swing_target_error_m"] <= .01
                             and d["soft_joint_limit_violation_rad"] <= 1e-6 for d in diagnostics) and not path_failures
    dynamic_feasible = kinematic_feasible and all(d["solver_success"] and d["base_dynamics_residual_norm"] <= 5.0
                                                   and d["max_effort_utilization"] <= 1.0
                                                   and d["max_velocity_utilization"] <= 1.0
                                                   and d["max_friction_cone_utilization"] <= 1.0
                                                   and d["min_normal_force_n"] >= 0 for d in dynamics)
    if not kinematic_feasible:
        classification = "OPTIMIZATION_FAILURE" if any(d["swing_target_error_m"] <= .01 for d in diagnostics) else "KINEMATICALLY_INFEASIBLE"
    elif not dynamic_feasible:
        # A single direct-collocation parameterization cannot prove physical
        # infeasibility.  Preserve the distinction requested by the audit.
        classification = "OPTIMIZATION_FAILURE"
    else:
        classification = "TRAJECTORY_FEASIBLE"
    trajectory = {"format": "mujoco_qpos_multiple_shooting_v0", "dt_nominal_s": .04,
                  "rows": rows, "not_for_production": True}
    (output/"candidate_trajectory.json").write_text(json.dumps(trajectory, indent=2)+"\n", encoding="utf-8")
    (output/"convergence_diagnostics.json").write_text(json.dumps(diagnostics, indent=2)+"\n", encoding="utf-8")
    residuals = {"kinematic_feasible": kinematic_feasible, "dynamic_feasible": dynamic_feasible,
                 "classification": classification, "path_failure_count": len(path_failures), "path_failures": path_failures[:100],
                 "max_support_error_m": max(d["support_keypoint_error_max_m"] for d in diagnostics),
                 "max_swing_error_m": max(d["swing_target_error_m"] for d in diagnostics),
                 "max_base_dynamics_residual": max(d["base_dynamics_residual_norm"] for d in dynamics),
                 "max_effort_utilization": max(d["max_effort_utilization"] for d in dynamics),
                 "max_velocity_utilization": max(d["max_velocity_utilization"] for d in dynamics),
                 "max_friction_utilization": max(d["max_friction_cone_utilization"] for d in dynamics),
                 "min_normal_force_n": min(d["min_normal_force_n"] for d in dynamics)}
    (output/"constraint_residuals.json").write_text(json.dumps(residuals, indent=2)+"\n", encoding="utf-8")
    (output/"failure_examples.json").write_text(json.dumps({"path": path_failures[:20],
        "dynamics": sorted(dynamics, key=lambda d: max(d["max_effort_utilization"], d["max_friction_cone_utilization"]), reverse=True)[:20]}, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(residuals, indent=2))


if __name__ == "__main__": main()
