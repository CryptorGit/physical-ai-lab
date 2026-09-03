"""Observation/action contract extraction from the live manager configuration."""

from __future__ import annotations


def manager_terms(manager, group: str | None = None) -> list[dict]:
    names = manager.active_terms[group] if group else manager.active_terms
    if isinstance(names, dict):
        names = names.get(group, [])
    return [{"index": i, "name": name} for i, name in enumerate(names)]


def observation_action_contract(env, agent_cfg) -> dict:
    robot = env.scene["robot"]
    obs = env.observation_manager.compute()["policy"]
    action_term = env.action_manager.get_term("joint_pos")
    return {
        "observation": {
            "dimension": int(obs.shape[-1]),
            "field_order": manager_terms(env.observation_manager, "policy"),
            "base_linear_velocity": True,
            "base_angular_velocity": True,
            "projected_gravity": True,
            "velocity_command": True,
            "joint_positions": True,
            "joint_velocities": True,
            "previous_action": True,
            "height_scan": False,
            "history": False,
            "normalization": bool(getattr(agent_cfg.actor, "obs_normalization", False)),
            "privileged_observation": False,
        },
        "action": {
            "dimension": int(env.action_manager.total_action_dim),
            "joint_order": list(robot.joint_names),
            "type": type(action_term).__name__,
            "semantics": "default_joint_position + scale * policy_output",
            "scale": getattr(action_term.cfg, "scale", None),
            "default_joint_positions": robot.data.default_joint_pos[0].detach().cpu().tolist(),
            "policy_output_clip": getattr(agent_cfg, "clip_actions", None),
            "action_term_clip": getattr(action_term.cfg, "clip", None),
            "effort_limits": robot.data.joint_effort_limits[0].detach().cpu().tolist(),
            "velocity_limits": robot.data.joint_vel_limits[0].detach().cpu().tolist(),
            "previous_action_semantics": "last applied policy action; continuous within an episode",
        },
    }

