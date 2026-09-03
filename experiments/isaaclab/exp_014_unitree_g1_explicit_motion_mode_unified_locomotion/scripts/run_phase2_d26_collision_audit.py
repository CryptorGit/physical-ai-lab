"""Read-only USD collision geometry and numeric sole polygon audit."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve(); REPO = HERE.parents[4]
OUT = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26_wmove_reference_and_wbik"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


d3 = load("d3_collision_audit", HERE.parent / "run_phase2_d3.py")
import gymnasium as gym  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402


def hull(points):
    pts = sorted({(round(float(x), 9), round(float(y), 9)) for x, y in points})
    if len(pts) <= 1: return pts
    def cross(o, a, b): return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0: lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0: upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def area(poly):
    return abs(sum(poly[i][0]*poly[(i+1)%len(poly)][1] - poly[(i+1)%len(poly)][0]*poly[i][1] for i in range(len(poly))) * 0.5) if len(poly) >= 3 else 0.0


def matrix_np(m):
    return np.asarray([[float(m[i][j]) for j in range(4)] for i in range(4)], dtype=np.float64)


def main():
    p = argparse.ArgumentParser(); add_launcher_args(p); args, hydra = setup_preset_cli(p); sys.argv = [sys.argv[0], *hydra]
    cfg, agent = resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = 1; cfg.observations.policy.enable_corruption = False; cfg.events.base_external_force_torque = None; cfg.events.push_robot = None
    report = {"contract": "G1NumericFootSolePolygonV1", "source": "USD collision geometry", "feet": {}, "status": "FAIL"}
    with launch_simulation(cfg, args):
        # Import USD Python bindings only after SimulationApp has loaded.  An
        # early pxr import causes Isaac Sim's public pxr namespace collision.
        from pxr import Usd, UsdGeom, UsdPhysics
        wrapped = RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0", cfg=cfg), clip_actions=agent.clip_actions)
        stage = None
        # omni.usd is the authoritative stage context in Isaac Sim.
        try:
            import omni.usd
            stage = omni.usd.get_context().get_stage()
        except Exception as exc:
            report["error"] = f"stage_context: {type(exc).__name__}: {exc}"
        for side, token in (("left", "left_ankle_roll_link"), ("right", "right_ankle_roll_link")):
            foot = {"collision_prims": [], "mesh_points": 0, "fallback": None}
            if stage is not None:
                link = None
                for prim in stage.Traverse():
                    if token in str(prim.GetPath()) and str(prim.GetPath()).endswith(token): link = prim; break
                if link is not None:
                    link_inv = np.linalg.inv(matrix_np(UsdGeom.Xformable(link).ComputeLocalToWorldTransform(Usd.TimeCode.Default())))
                    points = []
                    for prim in stage.Traverse():
                        path = str(prim.GetPath())
                        if not path.startswith(str(link.GetPath())): continue
                        collision = prim.HasAPI(UsdPhysics.CollisionAPI) or "collision" in path.lower() or "collider" in path.lower()
                        if not collision: continue
                        foot["collision_prims"].append(path)
                        if prim.IsA(UsdGeom.Mesh):
                            mesh = UsdGeom.Mesh(prim); local = np.asarray(mesh.GetPointsAttr().Get() or [], dtype=np.float64)
                            if local.size:
                                world_m = matrix_np(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
                                link_m = link_inv @ world_m
                                hom = np.concatenate((local, np.ones((len(local), 1))), axis=1)
                                points.extend((hom @ link_m.T)[:, :3].tolist()); foot["mesh_points"] += len(local)
                    if points:
                        z = np.asarray(points)[:, 2]; tol = max(1.0e-4, 0.01 * max(np.ptp(np.asarray(points)[:, :2], axis=0).max(), 1.0e-3)); sole = np.asarray(points)[z <= z.min() + tol]
                        poly = hull(sole[:, :2].tolist()) if len(sole) >= 3 else hull(np.asarray(points)[:, :2].tolist())
                        foot.update({"polygon_vertices_xy": poly, "area_m2": area(poly), "centroid_xy": np.mean(np.asarray(poly), axis=0).tolist() if poly else [None, None], "sole_z_m": float(z.min()), "sole_plane_tolerance_m": float(tol), "geometry_method": "collision_mesh_convex_hull"})
                    else:
                        # No visual geometry is substituted.  A collision prim
                        # without vertices is a geometry extraction failure.
                        foot["fallback"] = "collision_prim_without_mesh_vertices"
                else:
                    foot["fallback"] = "foot_link_not_found"
            report["feet"][side] = foot
        report["status"] = "PASS" if all(len(report["feet"][s].get("polygon_vertices_xy", [])) >= 3 for s in ("left", "right")) else "FAIL"
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "foot_collision_geometry_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        polygons = {s: report["feet"][s] for s in ("left", "right")}
        (OUT / "numeric_foot_sole_polygon.json").write_text(json.dumps({"name": "G1NumericFootSolePolygonV1", "status": report["status"], "feet": polygons}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"status": report["status"], "left_prims": len(report["feet"]["left"].get("collision_prims", [])), "right_prims": len(report["feet"]["right"].get("collision_prims", []))}, indent=2), flush=True)
        wrapped.close()


if __name__ == "__main__": main()
