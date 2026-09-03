"""Direct RL environment for learning to keep Pathfinder standing."""
from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import euler_xyz_from_quat

import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics

from .pathfinder_stand_env_cfg import PathfinderStandEnvCfg


def hide_collision_visuals(root_path: str) -> int:
    """Hide collision geometry visually while keeping physics collisions active."""
    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(root_path)

    if not root_prim.IsValid():
        raise RuntimeError(f"Robot prim not found: {root_path}")

    hidden_count = 0

    for prim in Usd.PrimRange(root_prim):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if not prim.IsA(UsdGeom.Imageable):
            continue

        UsdGeom.Imageable(prim).MakeInvisible()
        hidden_count += 1

    return hidden_count


class PathfinderStandEnv(DirectRLEnv):
    cfg: PathfinderStandEnvCfg

    def __init__(self, cfg: PathfinderStandEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._joint_ids, self._joint_names = self.robot.find_joints(".*", preserve_order=True)
        if len(self._joint_ids) != self.cfg.action_space:
            raise RuntimeError(
                f"Expected {self.cfg.action_space} joints, found {len(self._joint_ids)}: {self._joint_names}"
            )

        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.previous_actions = torch.zeros_like(self.actions)
        self.joint_targets = self.robot.data.default_joint_pos.torch.clone()
        self.default_joint_pos = self.robot.data.default_joint_pos.torch.clone()

        self.target_root_height = self.robot.data.default_root_pose.torch[:, 2].clone()
        self.minimum_root_height = torch.clamp(
            self.target_root_height * self.cfg.min_height_ratio,
            min=0.04,
        )

        self._update_state()
        print(f"[PATHFINDER] joints={self._joint_names}")
        print(
            f"[PATHFINDER] initial_root_height={self.target_root_height[0].item():.4f} "
            f"minimum={self.minimum_root_height[0].item():.4f}"
        )

    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        self.scene.clone_environments(copy_from_source=False)

        hidden_count = 0

        for env_index in range(self.cfg.scene.num_envs):
            robot_path = f"/World/envs/env_{env_index}/Robot"
            hidden_count += hide_collision_visuals(robot_path)

        print(
            f"[PATHFINDER] hidden collision visuals={hidden_count} "
            "(physics collisions remain active)"
        )

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        self.scene.articulations["robot"] = self.robot

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.previous_actions.copy_(self.actions)
        self.actions = torch.clamp(actions, -1.0, 1.0)

        targets = self.default_joint_pos + self.cfg.action_scale * self.actions
        limits = self.robot.data.soft_joint_pos_limits.torch
        self.joint_targets = torch.clamp(targets, limits[..., 0], limits[..., 1])

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.joint_targets, joint_ids=self._joint_ids)

    def _update_state(self) -> None:
        self.root_pos = self.robot.data.root_pos_w.torch
        self.root_quat = self.robot.data.root_quat_w.torch
        self.root_lin_vel = self.robot.data.root_lin_vel_w.torch
        self.root_ang_vel = self.robot.data.root_ang_vel_w.torch
        self.joint_pos = self.robot.data.joint_pos.torch
        self.joint_vel = self.robot.data.joint_vel.torch
        self.roll, self.pitch, self.yaw = euler_xyz_from_quat(self.root_quat)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        self._update_state()
        height_error = (self.root_pos[:, 2] - self.target_root_height).unsqueeze(-1)
        tilt = torch.stack((self.roll, self.pitch), dim=-1)
        joint_pos_rel = self.joint_pos - self.default_joint_pos

        policy_obs = torch.cat(
            (
                height_error,
                tilt,
                self.root_lin_vel,
                self.root_ang_vel,
                joint_pos_rel,
                self.joint_vel * 0.1,
                self.actions,
            ),
            dim=-1,
        )
        return {"policy": policy_obs}

    def _get_rewards(self) -> torch.Tensor:
        height_error = self.root_pos[:, 2] - self.target_root_height
        tilt_sq = self.roll.square() + self.pitch.square()
        joint_error = self.joint_pos - self.default_joint_pos
        action_rate = self.actions - self.previous_actions

        reward = torch.full(
            (self.num_envs,),
            self.cfg.rew_alive,
            device=self.device,
        )

        reward += self.cfg.rew_upright * torch.exp(-6.0 * tilt_sq)
        reward += self.cfg.rew_height * torch.exp(-150.0 * height_error.square())

        reward += self.cfg.rew_joint_pose * torch.sum(
            joint_error.square(),
            dim=-1,
        )
        reward += self.cfg.rew_joint_vel * torch.sum(
            self.joint_vel.square(),
            dim=-1,
        )
        reward += self.cfg.rew_action * torch.sum(
            self.actions.square(),
            dim=-1,
        )
        reward += self.cfg.rew_action_rate * torch.sum(
            action_rate.square(),
            dim=-1,
        )
        reward += self.cfg.rew_termination * self.reset_terminated.float()

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._update_state()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        too_low = self.root_pos[:, 2] < self.minimum_root_height
        too_tilted = (torch.abs(self.roll) > self.cfg.max_tilt_rad) | (
            torch.abs(self.pitch) > self.cfg.max_tilt_rad
        )
        invalid = ~torch.isfinite(self.root_pos).all(dim=-1)
        terminated = too_low | too_tilted | invalid
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos.torch[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel.torch[env_ids].clone()
        joint_pos += self.cfg.joint_noise * (2.0 * torch.rand_like(joint_pos) - 1.0)

        limits = self.robot.data.soft_joint_pos_limits.torch[env_ids]
        joint_pos = torch.clamp(joint_pos, limits[..., 0], limits[..., 1])

        root_pose = self.robot.data.default_root_pose.torch[env_ids].clone()
        root_vel = self.robot.data.default_root_vel.torch[env_ids].clone()
        root_pose[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(root_pose, env_ids)
        self.robot.write_root_velocity_to_sim(root_vel, env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self.actions[env_ids] = 0.0
        self.previous_actions[env_ids] = 0.0
        self.joint_targets[env_ids] = joint_pos

