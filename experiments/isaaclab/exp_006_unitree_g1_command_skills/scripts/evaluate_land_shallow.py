"""Evaluate Stage-2 baseline and audit-only scripted LAND_SHALLOW primitives."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from importlib import metadata
from pathlib import Path

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

SCRIPT = Path(__file__).resolve(); EXP = SCRIPT.parent.parent; REPO = EXP.parents[2]
sys.path[:0] = [str(EXP / "src"), str(REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src")]
import isaaclab_tasks  # noqa: E402,F401
import g1_flat_run.tasks  # noqa: E402,F401
from g1_command_skills.scripted_land import LandPhase, landing_offset  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--mode", choices=("baseline", "grid", "pilot"), required=True)
parser.add_argument("--heights", default="0.02,0.04,0.06")
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--drop-height", type=float, default=0.06)
parser.add_argument("--preflex-depth", type=float, default=0.04)
parser.add_argument("--absorption-depth", type=float, default=0.04)
parser.add_argument("--absorption-duration", type=float, default=0.40)
parser.add_argument("--recovery-duration", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=20260722)
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser); sys.argv = [sys.argv[0]] + hydra

KEYPOINTS = {
    "toe": (0.06383880963290349, 0.0, -0.025807180037281774),
    "sole": (0.04321213238651294, 0.0, -0.025807180037281774),
    "heel": (0.022585455140122387, 0.0, -0.025807180037281774),
}


def mean(x): return sum(x)/len(x) if x else 0.0
def percentile(x, q):
    if not x: return 0.0
    x = sorted(x); return x[min(round((len(x)-1)*q/100), len(x)-1)]


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def conditions() -> list[dict]:
    if args.mode == "baseline":
        return [{"drop_height_m": h, "preflex_depth_m": 0.0, "absorption_depth_m": 0.0,
                 "absorption_duration_s": .20, "recovery_duration_s": .20, "replicate": i}
                for h in map(float, args.heights.split(",")) for i in range(args.episodes)]
    if args.mode == "pilot":
        return [{"drop_height_m": h, "preflex_depth_m": args.preflex_depth,
                 "absorption_depth_m": args.absorption_depth, "absorption_duration_s": args.absorption_duration,
                 "recovery_duration_s": args.recovery_duration, "replicate": i}
                for h in map(float, args.heights.split(",")) for i in range(args.episodes)]
    values = []
    for pre in (.02, .04, .06):
        for absorb in (.02, .04):
            if pre + absorb > .10: continue
            for duration in (.25, .40, .60):
                for recovery in (.8, 1.2):
                    for replicate in range(args.episodes):
                        values.append({"drop_height_m": args.drop_height, "preflex_depth_m": pre,
                                       "absorption_depth_m": absorb, "absorption_duration_s": duration,
                                       "recovery_duration_s": recovery, "replicate": replicate})
    return values


def main() -> None:
    checkpoint = Path(args.checkpoint).resolve(strict=True); output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=True)
    cfgs = conditions(); n = len(cfgs)
    env_cfg, agent_cfg = resolve_task_config("Isaac-Velocity-Flat-G1-Run-Eval-v0", "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = n; env_cfg.seed = args.seed; env_cfg.scene.contact_forces.history_length = 7
    if args.device is not None: env_cfg.sim.device = args.device
    with launch_simulation(env_cfg, args):
        raw = gym.make("Isaac-Velocity-Flat-G1-Run-Eval-v0", cfg=env_cfg)
        wrapped = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions); env = raw.unwrapped
        agent_cfg.device = env.device; agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=env.device)
        runner.load(str(checkpoint), load_cfg={"actor": True, "critic": False, "optimizer": False,
                                               "iteration": False, "rnd": False})
        policy = runner.get_inference_policy(device=env.device)
        robot = env.scene["robot"]; command = env.command_manager.get_term("base_velocity"); contact = env.scene.sensors["contact_forces"]
        foot_ids, foot_names = robot.find_bodies(".*_ankle_roll_link"); sensor_ids = [contact.body_names.index(x) for x in foot_names]
        torso_sensor_ids = [i for i, name in enumerate(contact.body_names) if any(key in name for key in ("pelvis", "torso"))]
        all_joint_ids, all_joint_names = robot.find_joints(".*"); ankle_ids, _ = robot.find_joints(".*ankle.*")
        knee_ids, _ = robot.find_joints(".*knee.*")
        landing_joint_local_ids = [
            j for j, name in enumerate(all_joint_names)
            if any(token in name for token in ("hip_", "knee", "ankle", "torso"))
        ]
        wrapped.reset(); dt = float(env.step_dt); physics_dt = float(env.cfg.sim.dt)
        # Obtain the same stable Stage-2 standing initial condition for all environments.
        base_action = None
        for _ in range(round(2.0/dt)):
            command.vel_command_b.zero_(); obs = wrapped.get_observations()
            with torch.inference_mode(): base_action = policy(obs); wrapped.step(base_action)
        standing_root_height = robot.data.root_pos_w.torch[:,2].clone()
        if args.mode != "baseline":
            pre = torch.tensor([c["preflex_depth_m"] for c in cfgs],device=env.device)
            absorb = torch.tensor([c["absorption_depth_m"] for c in cfgs],device=env.device)
            prepare_steps = round(.50/dt)
            prepare_phase = torch.full((n,),int(LandPhase.PREPARE),dtype=torch.long,device=env.device)
            for prepare_step in range(prepare_steps):
                command.vel_command_b.zero_(); obs=wrapped.get_observations()
                with torch.inference_mode(): base_action=policy(obs)
                prepare_progress=torch.full((n,),min((prepare_step+1)/prepare_steps,1.0),device=env.device)
                with torch.inference_mode(): wrapped.step(base_action+landing_offset(prepare_phase,prepare_progress,pre,absorb,action_dim=wrapped.num_actions))
        command.vel_command_b.zero_()
        settled_current_fz = contact.data.net_forces_w.torch[:,sensor_ids,2].abs().sum(dim=1).clone()
        pos = robot.data.body_pos_w.torch[:, foot_ids]; quat = robot.data.body_quat_w.torch[:, foot_ids]
        initial_points = []
        for local_value in KEYPOINTS.values():
            local = torch.tensor(local_value, device=env.device).expand(n, 2, 3)
            initial_points.append(pos + quat_apply(quat.reshape(-1,4), local.reshape(-1,3)).reshape(n,2,3))
        raw_keypoint_bottom = torch.stack(initial_points).select(-1, 2).amin(dim=(0,2))
        # The legacy STEP_OVER keypoint plane sits above the instantiated USD
        # collision contact plane (the loaded offset is pose-dependent).
        # Calibrate that offset
        # offset at the settled, loaded contact pose; otherwise requesting a
        # 20 mm drop would lower the robot and create immediate contact.
        settled_keypoint_to_collision_offset = raw_keypoint_bottom.clone()
        root_pose = robot.data.root_pose_w.torch.clone(); settled_root_height = standing_root_height
        requested = torch.tensor([c["drop_height_m"] for c in cfgs], device=env.device)
        lift = requested; root_pose[:,2] += lift
        root_velocity = torch.zeros((n,6), device=env.device); ids = torch.arange(n,device=env.device,dtype=torch.long)
        robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=ids)
        robot.write_root_velocity_to_sim_index(root_velocity=root_velocity, env_ids=ids)
        env.sim.forward()
        # Teleporting preserves the sensor's previous standing-contact sample
        # unless its buffers are reset explicitly.  That stale sample must not
        # be interpreted as a zero-time landing contact.
        contact.reset(ids)
        contact.update(0.0, force_recompute=True)
        actual_initial_clearance = raw_keypoint_bottom + lift - settled_keypoint_to_collision_offset
        initial_height = root_pose[:,2].clone(); initial_xy = root_pose[:,:2].clone()

        phase = torch.full((n,), int(LandPhase.AIRBORNE), dtype=torch.long, device=env.device)
        phase_steps = torch.zeros(n,dtype=torch.long,device=env.device); active=torch.ones(n,dtype=torch.bool,device=env.device)
        first_contact = torch.full((n,2),-1,dtype=torch.long,device=env.device); double_contact=torch.full((n,),-1,dtype=torch.long,device=env.device)
        precontact_vz=torch.zeros(n,device=env.device); previous_vz=torch.zeros(n,device=env.device)
        stable_streak=torch.zeros(n,dtype=torch.long,device=env.device); hold_streak=torch.zeros_like(stable_streak)
        fall=torch.zeros(n,dtype=torch.bool,device=env.device); torso_contact=torch.zeros_like(fall); premature=torch.zeros_like(fall)
        airborne_confirmed=torch.zeros_like(fall)
        force_peak=torch.zeros(n,device=env.device); force10=torch.zeros(n,device=env.device); force15=torch.zeros(n,device=env.device); force30=torch.zeros(n,device=env.device)
        impulse=torch.zeros(n,device=env.device); accel_peak=torch.zeros(n,device=env.device); pelvis_min=initial_height.clone(); rebound=torch.zeros(n,device=env.device)
        roll_max=torch.zeros(n,device=env.device); pitch_max=torch.zeros(n,device=env.device); angular_max=torch.zeros(n,device=env.device)
        drift_max=torch.zeros(n,device=env.device); slip_max=torch.zeros(n,device=env.device); effort_max=torch.zeros(n,device=env.device); velocity_max=torch.zeros(n,device=env.device)
        slip_run=torch.zeros(n,dtype=torch.long,device=env.device); slip_duration_max=torch.zeros(n,device=env.device)
        ankle_torque_steps=torch.zeros(n,device=env.device); knee_velocity_steps=torch.zeros(n,device=env.device); limit_max=torch.zeros(n,device=env.device); measured_steps=torch.zeros(n,device=env.device)
        recovery_time=torch.full((n,),math.inf,device=env.device); final_hspeed=torch.full((n,),math.inf,device=env.device); final_vspeed=torch.full((n,),math.inf,device=env.device)
        failure=["" for _ in range(n)]; step_rows=[]
        max_steps=round(6.0/dt)
        for step in range(max_steps):
            command.vel_command_b.zero_(); obs=wrapped.get_observations()
            with torch.inference_mode(): base_action=policy(obs)
            pre=torch.tensor([c["preflex_depth_m"] for c in cfgs],device=env.device)
            absorb=torch.tensor([c["absorption_depth_m"] for c in cfgs],device=env.device)
            progress=torch.zeros(n,device=env.device)
            for i,c in enumerate(cfgs):
                p=LandPhase(int(phase[i])); age=float(phase_steps[i])*dt
                duration=.25 if p==LandPhase.AIRBORNE else c["absorption_duration_s"] if p==LandPhase.IMPACT_ABSORPTION else c["recovery_duration_s"] if p==LandPhase.RETURN_TO_STAND else .1
                progress[i]=min(age/max(duration,1e-6),1.0)
            offset=landing_offset(phase,progress,pre,absorb,action_dim=wrapped.num_actions) if args.mode!="baseline" else torch.zeros_like(base_action)
            actions=base_action+offset
            with torch.inference_mode(): _,_,dones,_=wrapped.step(actions)
            command.vel_command_b.zero_()
            forces_hist=contact.data.net_forces_w_history.torch[:,:,sensor_ids,:]
            current_forces=contact.data.net_forces_w.torch[:,sensor_ids,:]
            # Current force, not history max: history still contains the
            # pre-lift standing contact for a few sensor samples.
            contacts=current_forces.norm(dim=-1)>5.0; contact_count=contacts.sum(dim=1)
            current_fz=current_forces[:,:,2].abs().sum(dim=1)
            hist_fz=forces_hist[...,2].abs().sum(dim=2)
            vz=robot.data.root_lin_vel_w.torch[:,2]; hspeed=robot.data.root_lin_vel_w.torch[:,:2].norm(dim=1)
            gravity=robot.data.projected_gravity_b.torch; roll=torch.atan2(gravity[:,1],-gravity[:,2]); pitch=torch.atan2(-gravity[:,0],torch.sqrt(gravity[:,1].square()+gravity[:,2].square()))
            angular=robot.data.root_ang_vel_b.torch.norm(dim=1); root_h=robot.data.root_pos_w.torch[:,2]
            effort=robot.data.applied_torque.torch[:,all_joint_ids].abs()/robot.data.joint_effort_limits.torch[:,all_joint_ids].abs().clamp_min(1e-6)
            velocity=robot.data.joint_vel.torch[:,all_joint_ids].abs()/robot.data.joint_vel_limits.torch[:,all_joint_ids].abs().clamp_min(1e-6)
            limits=robot.data.joint_pos_limits.torch[:,all_joint_ids]; q=robot.data.joint_pos.torch[:,all_joint_ids]
            center=(limits[...,0]+limits[...,1])/2; half=(limits[...,1]-limits[...,0]).clamp_min(1e-6)/2
            proximity_all=(q-center).abs()/half
            proximity=proximity_all[:,landing_joint_local_ids].amax(dim=1)
            foot_slip=robot.data.body_lin_vel_w.torch[:,foot_ids,:2].norm(dim=-1)*contacts
            torso_force=contact.data.net_forces_w.torch[:,torso_sensor_ids,:].norm(dim=-1).amax(dim=1) if torso_sensor_ids else torch.zeros(n,device=env.device)
            for i,c in enumerate(cfgs):
                if not active[i]: continue
                measured_steps[i]+=1; force_peak[i]=torch.maximum(force_peak[i],current_fz[i])
                first_seen = int(first_contact[i][first_contact[i]>=0].min()) if bool((first_contact[i]>=0).any()) else -1
                if first_seen >= 0 and (step-first_seen)*dt <= .20: impulse[i]+=current_fz[i]*dt
                accel_peak[i]=torch.maximum(accel_peak[i],abs(vz[i]-previous_vz[i])/dt); pelvis_min[i]=torch.minimum(pelvis_min[i],root_h[i])
                if bool((first_contact[i]>=0).any()): rebound[i]=torch.maximum(rebound[i],root_h[i]-pelvis_min[i])
                roll_max[i]=torch.maximum(roll_max[i],abs(roll[i])); pitch_max[i]=torch.maximum(pitch_max[i],abs(pitch[i])); angular_max[i]=torch.maximum(angular_max[i],angular[i])
                drift_max[i]=torch.maximum(drift_max[i],torch.linalg.vector_norm(robot.data.root_pos_w.torch[i,:2]-initial_xy[i])); slip_max[i]=torch.maximum(slip_max[i],foot_slip[i].max())
                effort_max[i]=torch.maximum(effort_max[i],effort[i].max()); velocity_max[i]=torch.maximum(velocity_max[i],velocity[i].max()); limit_max[i]=torch.maximum(limit_max[i],proximity[i])
                ankle_torque_steps[i]+=float((effort[i,ankle_ids]>=.95).any()); knee_velocity_steps[i]+=float((velocity[i,knee_ids]>=.95).any())
                torso_contact[i]|=torso_force[i]>20.0
                p=LandPhase(int(phase[i])); safe=(hspeed[i]<=.08 and abs(vz[i])<=.05 and abs(roll[i])<=.10 and abs(pitch[i])<=.10)
                sustained_slip=bool(contact_count[i]>0 and foot_slip[i].max()>.50 and p not in (LandPhase.AIRBORNE,LandPhase.FIRST_CONTACT))
                slip_run[i]=slip_run[i]+1 if sustained_slip else 0
                slip_duration_max[i]=torch.maximum(slip_duration_max[i],slip_run[i]*dt)
                valid_contact=True
                if p==LandPhase.AIRBORNE:
                    if contact_count[i]==0: airborne_confirmed[i]=True
                    ballistic_floor=.5*math.sqrt(2.0*c["drop_height_m"]/9.81)
                    force_changed=abs(float(current_fz[i]-settled_current_fz[i]))>max(5.0,.05*float(settled_current_fz[i]))
                    valid_contact=bool(contact_count[i]>0 and phase_steps[i]*dt>=ballistic_floor and (airborne_confirmed[i] or force_changed))
                for foot in range(2):
                    if first_contact[i,foot]<0 and contacts[i,foot] and valid_contact:
                        first_contact[i,foot]=step; precontact_vz[i]=previous_vz[i]
                        if first_contact[i].max()==step:
                            force10[i]=hist_fz[i,:min(2,hist_fz.shape[1])].mean(); force15[i]=hist_fz[i,:min(3,hist_fz.shape[1])].mean(); force30[i]=hist_fz[i,:min(6,hist_fz.shape[1])].mean()
                if double_contact[i]<0 and contact_count[i]==2 and valid_contact: double_contact[i]=step
                if p==LandPhase.AIRBORNE:
                    if valid_contact:
                        phase[i]=int(LandPhase.FIRST_CONTACT); phase_steps[i]=0
                    elif phase_steps[i]*dt>1.0: failure[i]="impact_failure"; active[i]=False
                elif p==LandPhase.FIRST_CONTACT:
                    phase[i]=int(LandPhase.IMPACT_ABSORPTION); phase_steps[i]=0
                elif p==LandPhase.IMPACT_ABSORPTION:
                    stable_streak[i]=stable_streak[i]+1 if contact_count[i]==2 and abs(vz[i])<.15 and abs(roll[i])<.15 and abs(pitch[i])<.15 else 0
                    if phase_steps[i]*dt>=c["absorption_duration_s"] and stable_streak[i]*dt>=.10:
                        phase[i]=int(LandPhase.DOUBLE_SUPPORT_RECOVERY); phase_steps[i]=0; stable_streak[i]=0
                    elif phase_steps[i]*dt>1.5: failure[i]="absorption_failure"; active[i]=False
                elif p==LandPhase.DOUBLE_SUPPORT_RECOVERY:
                    stable_streak[i]=stable_streak[i]+1 if contact_count[i]==2 and safe else 0
                    if stable_streak[i]*dt>=.20: phase[i]=int(LandPhase.RETURN_TO_STAND); phase_steps[i]=0
                    elif phase_steps[i]*dt>1.5: failure[i]="double_support_failure"; active[i]=False
                elif p==LandPhase.RETURN_TO_STAND:
                    if phase_steps[i]*dt>=c["recovery_duration_s"]:
                        if safe and contact_count[i]==2: phase[i]=int(LandPhase.STAND_HOLD); phase_steps[i]=0; hold_streak[i]=0
                        elif phase_steps[i]*dt>c["recovery_duration_s"]+.8: failure[i]="unstable_recovery"; active[i]=False
                elif p==LandPhase.STAND_HOLD:
                    hold_streak[i]=hold_streak[i]+1 if safe and contact_count[i]==2 else 0
                    if hold_streak[i]*dt>=.8:
                        recovery_time[i]=(step-int(first_contact[i].min()))*dt; final_hspeed[i]=hspeed[i]; final_vspeed[i]=abs(vz[i]); active[i]=False
                    elif phase_steps[i]*dt>1.5: failure[i]="stand_hold_failure"; active[i]=False
                if bool(dones[i]): fall[i]=True; failure[i]="fall"; active[i]=False
                step_rows.append({"episode":i,"time_s":step*dt,"phase":LandPhase(int(phase[i])).name,"pelvis_height_m":float(root_h[i]),"vertical_speed_mps":float(vz[i]),"force_n":float(current_fz[i]),"offset_norm":float(offset[i].norm()),"left_contact":bool(contacts[i,0]),"right_contact":bool(contacts[i,1])})
                phase_steps[i]+=1
            previous_vz.copy_(vz)
            if not bool(active.any()): break
        records=[]
        for i,c in enumerate(cfgs):
            times=[float(x)*dt if x>=0 else math.inf for x in first_contact[i].tolist()]; timing=abs(times[0]-times[1]) if all(math.isfinite(x) for x in times) else math.inf
            complete=hold_streak[i]*dt>=.8; height_error=abs(float(robot.data.root_pos_w.torch[i,2]-settled_root_height[i])) if complete else math.inf
            sat=bool(ankle_torque_steps[i]/measured_steps[i].clamp_min(1)>.05 or knee_velocity_steps[i]/measured_steps[i].clamp_min(1)>.05)
            clearance_error=abs(float(actual_initial_clearance[i])-c["drop_height_m"])
            dangerous_slip=bool(slip_duration_max[i]>.06)
            success=bool(complete and clearance_error<=.002 and not premature[i] and not fall[i] and not torso_contact[i] and timing<=.04 and force_peak[i]<=3500 and drift_max[i]<=.10 and final_hspeed[i]<=.08 and final_vspeed[i]<=.05 and height_error<=.05 and not sat and limit_max[i]<.95 and not dangerous_slip)
            derived="" if success else failure[i] or ("invalid_initial_clearance" if clearance_error>.002 else "premature_contact" if premature[i] else "excessive_contact_asymmetry" if timing>.04 else "impact_failure" if force_peak[i]>3500 else "saturation_failure" if sat else "joint_limit_failure" if limit_max[i]>=.95 else "foot_slip_failure" if dangerous_slip else "stand_hold_failure")
            records.append(dict(c,episode=i,controller=args.mode,actual_initial_clearance_m=float(actual_initial_clearance[i]),initial_clearance_error_m=clearance_error,keypoint_to_collision_plane_offset_m=float(settled_keypoint_to_collision_offset[i]),premature_contact=bool(premature[i]),initial_pelvis_height_m=float(initial_height[i]),precontact_vertical_velocity_mps=float(precontact_vz[i]),left_first_contact_s=times[0],right_first_contact_s=times[1],contact_timing_difference_s=timing,first_contact_foot="left" if times[0]<times[1] else "right" if times[1]<times[0] else "simultaneous",double_support_time_s=float(double_contact[i])*dt if double_contact[i]>=0 else math.inf,contact_force_peak_n=float(force_peak[i]),load_mean_10ms_n=float(force10[i]),load_mean_15ms_n=float(force15[i]),load_mean_30ms_n=float(force30[i]),impact_impulse_ns=float(impulse[i]),pelvis_vertical_acceleration_peak_mps2=float(accel_peak[i]),pelvis_minimum_height_m=float(pelvis_min[i]),pelvis_rebound_m=float(rebound[i]),roll_max_rad=float(roll_max[i]),pitch_max_rad=float(pitch_max[i]),angular_velocity_max_rps=float(angular_max[i]),horizontal_drift_m=float(drift_max[i]),foot_slip_max_mps=float(slip_max[i]),dangerous_foot_slip_duration_s=float(slip_duration_max[i]),effort_utilization_max=float(effort_max[i]),joint_velocity_utilization_max=float(velocity_max[i]),ankle_torque_saturation_fraction=float(ankle_torque_steps[i]/measured_steps[i].clamp_min(1)),knee_velocity_saturation_fraction=float(knee_velocity_steps[i]/measured_steps[i].clamp_min(1)),joint_limit_proximity_max=float(limit_max[i]),fall=bool(fall[i]),torso_contact=bool(torso_contact[i]),recovery_time_s=float(recovery_time[i]),final_horizontal_speed_mps=float(final_hspeed[i]),final_vertical_speed_mps=float(final_vspeed[i]),final_standing_height_error_m=height_error,stand_hold_success=bool(complete),landing_success=success,failure_class=derived))
        write_csv(output/"episodes.csv",records); write_csv(output/"steps.csv",step_rows)
        groups={}
        for height in sorted(set(r["drop_height_m"] for r in records)):
            rs=[r for r in records if r["drop_height_m"]==height]
            groups[str(height)]={"episodes":len(rs),"landing_success_rate":mean([float(r["landing_success"]) for r in rs]),"stand_hold_success_rate":mean([float(r["stand_hold_success"]) for r in rs]),"fall_rate":mean([float(r["fall"]) for r in rs]),"double_support_recovery_rate":mean([float(math.isfinite(r["double_support_time_s"])) for r in rs]),"premature_contact_rate":mean([float(r["premature_contact"]) for r in rs]),"contact_asymmetry_failure_rate":mean([float(r["contact_timing_difference_s"]>.04) for r in rs]),"force_peak_p95_n":percentile([r["contact_force_peak_n"] for r in rs],95),"force_over_3500_rate":mean([float(r["contact_force_peak_n"]>3500) for r in rs]),"impulse_p95_ns":percentile([r["impact_impulse_ns"] for r in rs],95),"rebound_p95_m":percentile([r["pelvis_rebound_m"] for r in rs],95),"saturation_failure_rate":mean([float(r["ankle_torque_saturation_fraction"]>.05 or r["knee_velocity_saturation_fraction"]>.05) for r in rs]),"failure_counts":dict(Counter(r["failure_class"] for r in rs if r["failure_class"]))}
        summary={"controller":args.mode,"checkpoint":str(checkpoint),"episodes":n,"seed":args.seed,"foot_keypoints_body_m":KEYPOINTS,"drop_definition":"minimum toe/sole/heel clearance relative to the settled USD collision contact plane","keypoint_collision_calibration":"per-episode settled contact offset; legacy keypoint plane is not assumed to be z=0","groups":groups,"gate_pass":all(g["landing_success_rate"]>=.9 and g["stand_hold_success_rate"]>=.9 and g["fall_rate"]<=.05 and g["force_over_3500_rate"]<=.05 and g["saturation_failure_rate"]<=.05 for g in groups.values())}
        (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8"); print(json.dumps(summary,indent=2)); raw.close()


if __name__=="__main__": main()
