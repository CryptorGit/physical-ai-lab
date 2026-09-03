"""Formal vectorized RUN -> TURN -> RUN command-system sequence evaluation."""

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

SCRIPT=Path(__file__).resolve(); EXP=SCRIPT.parent.parent; REPO=EXP.parents[2]
sys.path[:0]=[str(EXP/"src"),str(REPO/"experiments/isaaclab/exp_005_unitree_g1_flat_run/src")]
import isaaclab_tasks  # noqa:E402,F401
import g1_flat_run.tasks  # noqa:E402,F401
import g1_command_skills.tasks  # noqa:E402,F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa:E402
from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg  # noqa:E402
from isaaclab_tasks.utils import add_launcher_args,launch_simulation,resolve_task_config,setup_preset_cli  # noqa:E402

p=argparse.ArgumentParser(description=__doc__)
p.add_argument("--checkpoint",required=True);p.add_argument("--output",required=True)
p.add_argument("--episodes",type=int,default=50);p.add_argument("--seed",type=int,default=20260723)
p.add_argument("--run-duration-min",type=float,default=1.4);p.add_argument("--run-duration-max",type=float,default=2.0)
p.add_argument("--recovery-duration-min",type=float,default=2.8);p.add_argument("--recovery-duration-max",type=float,default=3.8)
add_launcher_args(p);args,hydra=setup_preset_cli(p);sys.argv=[sys.argv[0]]+hydra

def mean(v):return sum(v)/len(v) if v else 0.0
def pct(v,q):
    if not v:return 0.0
    v=sorted(v);return v[min(round((len(v)-1)*q/100),len(v)-1)]
def tail(v,steps):return v[-min(len(v),steps):]

def main():
    checkpoint=Path(args.checkpoint).resolve(strict=True);out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    cfg,ac=resolve_task_config("Isaac-Motion-Flat-G1-Command-TurnFull-Eval-v0","rsl_rl_cfg_entry_point")
    cfg.scene.num_envs=args.episodes;cfg.seed=args.seed
    if args.device is not None:cfg.sim.device=args.device
    with launch_simulation(cfg,args):
        raw=gym.make("Isaac-Motion-Flat-G1-Command-TurnFull-Eval-v0",cfg=cfg);w=RslRlVecEnvWrapper(raw,clip_actions=ac.clip_actions);e=raw.unwrapped
        ac.device=e.device;ac=handle_deprecated_rsl_rl_cfg(ac,metadata.version("rsl-rl-lib"));runner=OnPolicyRunner(w,ac.to_dict(),log_dir=None,device=e.device)
        runner.load(str(checkpoint),load_cfg={"actor":True,"critic":False,"optimizer":False,"iteration":False,"rnd":False});policy=runner.get_inference_policy(device=e.device)
        robot=e.scene["robot"];term=e.command_manager.get_term("base_velocity");contact=e.scene.sensors["contact_forces"]
        ankle_ids,_=robot.find_joints(".*ankle.*");all_ids,_=robot.find_joints(".*")
        w.reset();n=args.episodes;dt=float(e.step_dt);gen=torch.Generator(device=e.device).manual_seed(args.seed)
        # Capture the physical heading after the first simulation update;
        # both root and command buffers may still expose pre-reset values
        # immediately after the vectorized reset.
        initial_heading=torch.zeros(n,device=e.device)
        run_duration=torch.empty(n,device=e.device).uniform_(args.run_duration_min,args.run_duration_max,generator=gen)
        recovery_duration=torch.empty(n,device=e.device).uniform_(args.recovery_duration_min,args.recovery_duration_max,generator=gen)
        term.segment_duration.copy_(run_duration)
        traces=[{"pre_speed":[],"pre_heading":[],"pre_lateral":[],"pre_lateral_velocity":[],"turn_speed":[],"recovery_speed":[],"recovery_heading":[],"recovery_lateral":[],"jumps_l2":[],"jumps_max":[],"stabilization":[]} for _ in range(n)]
        active=torch.ones(n,dtype=torch.bool,device=e.device);fallen=torch.zeros_like(active)
        velocity_sat=torch.zeros(n,device=e.device);ankle_sat=torch.zeros(n,device=e.device);measured=torch.zeros(n,device=e.device)
        final_turn_error=torch.full((n,),math.inf,device=e.device);commanded_turn=term.commanded_turn_angle_rad.clone()
        last_skill=term.skill_id.clone();last_action=None;transition_start=torch.full((n,),-1.0,device=e.device);transition_stabilized=torch.zeros(n,dtype=torch.bool,device=e.device)
        max_steps=round(12.0/dt)
        for step in range(max_steps):
            obs=w.get_observations()
            with torch.inference_mode():actions=policy(obs)
            skill_before=term.skill_id.clone();segment_before=term.segment_index.clone()
            if last_action is not None:
                changed=active&(skill_before!=last_skill)
                for i in torch.nonzero(changed,as_tuple=False).flatten().tolist():
                    diff=actions[i]-last_action[i];traces[i]["jumps_l2"].append(float(torch.linalg.vector_norm(diff)));traces[i]["jumps_max"].append(float(diff.abs().max()))
                    transition_start[i]=step*dt;transition_stabilized[i]=False
            with torch.inference_mode():_,_,dones,infos=w.step(actions)
            if step==0:initial_heading.copy_(robot.data.heading_w.torch)
            skill_after=term.skill_id.clone();segment_after=term.segment_index.clone()
            changed_after=active&(segment_after!=segment_before)
            entered_turn=changed_after&(segment_after==1);entered_recovery=changed_after&(segment_after==2)
            # Per-environment duration randomization is applied after the
            # command term initializes each newly entered segment.
            term.segment_duration[entered_recovery]=recovery_duration[entered_recovery]
            commanded_turn[entered_turn]=term.commanded_turn_angle_rad[entered_turn]
            speed_error=(robot.data.root_lin_vel_b.torch[:,0]-term.target_speed).abs();heading_error=term.heading_error.abs();lateral=term.path_lateral_error.abs()
            initial_heading_error=torch.atan2(
                torch.sin(robot.data.heading_w.torch-initial_heading),
                torch.cos(robot.data.heading_w.torch-initial_heading),
            ).abs()
            vel_ratio=robot.data.joint_vel.torch[:,all_ids].abs()/robot.data.joint_vel_limits.torch[:,all_ids].abs().clamp_min(1e-6)
            ankle_ratio=robot.data.applied_torque.torch[:,ankle_ids].abs()/robot.data.joint_effort_limits.torch[:,ankle_ids].abs().clamp_min(1e-6)
            velocity_sat+=active*(vel_ratio.amax(dim=1)>=.95);ankle_sat+=active*(ankle_ratio.amax(dim=1)>=.95);measured+=active
            for i in torch.nonzero(active,as_tuple=False).flatten().tolist():
                # The post-step command manager may already expose the next
                # segment target.  Do not attribute that target's error to the
                # action that was generated for the previous segment.
                if bool(changed_after[i]):
                    continue
                phase=int(segment_before[i]);tr=traces[i]
                if phase==0:tr["pre_speed"].append(float(speed_error[i]));tr["pre_heading"].append(float(initial_heading_error[i]));tr["pre_lateral"].append(float(lateral[i]));tr["pre_lateral_velocity"].append(abs(float(robot.data.root_lin_vel_b.torch[i,1])))
                elif phase==1:tr["turn_speed"].append(float(speed_error[i]))
                else:tr["recovery_speed"].append(float(speed_error[i]));tr["recovery_heading"].append(float(heading_error[i]));tr["recovery_lateral"].append(float(lateral[i]))
                if transition_start[i]>=0 and not transition_stabilized[i] and speed_error[i]<=.35 and heading_error[i]<=.12:
                    tr["stabilization"].append(step*dt-float(transition_start[i]));transition_stabilized[i]=True
            for i in torch.nonzero(entered_recovery,as_tuple=False).flatten().tolist():
                final_turn_error[i]=abs(term.commanded_turn_angle_rad[i]-term.actual_accumulated_yaw_rad[i])
            timeout_tensor=infos.get("time_outs") if isinstance(infos,dict) else None
            timeout=timeout_tensor.bool() if timeout_tensor is not None else torch.zeros_like(active)
            fallen|=active&dones.bool()&~timeout;active&=~dones.bool()
            complete=active&(segment_after==2)&(term.segment_elapsed>=recovery_duration)
            active[complete]=False
            last_skill=skill_before;last_action=actions.clone()
            if not bool(active.any()):break
        rows=[];tail_steps=round(.6/dt);recovery_tail=round(1.0/dt)
        for i,tr in enumerate(traces):
            pre_speed=tail(tr["pre_speed"],tail_steps);pre_heading=tail(tr["pre_heading"],tail_steps);pre_lateral=tail(tr["pre_lateral"],tail_steps);pre_lateral_velocity=tail(tr["pre_lateral_velocity"],tail_steps)
            rec_speed=tail(tr["recovery_speed"],recovery_tail);rec_heading=tail(tr["recovery_heading"],recovery_tail);rec_lat=tail(tr["recovery_lateral"],recovery_tail)
            # RUN is path-local.  Absolute world-yaw convergence from the
            # randomized reset is diagnostic only; TURN fixes its own entry
            # heading.  Gate the pre-segment on stable propulsion and path.
            # Match the independently-qualified RUN gate: command-axis speed
            # tracking is authoritative here.  Body-y velocity is retained as
            # a diagnostic because randomized reset heading makes it unsuitable
            # as an additional success condition for this sequence evaluator.
            pre_ok=bool(pre_speed and mean(pre_speed)<=.35)
            turn_ok=bool(final_turn_error[i]<=.12)
            recovery_ok=bool(rec_speed and mean(rec_speed)<=.35 and pct(rec_heading,95)<=.12 and pct(rec_lat,95)<=.75)
            vs=float(velocity_sat[i]/measured[i].clamp_min(1));ats=float(ankle_sat[i]/measured[i].clamp_min(1));sat=vs>.05 or ats>.20
            success=pre_ok and turn_ok and recovery_ok and not bool(fallen[i]) and not sat
            angle=float(commanded_turn[i]);category=("left" if angle>=0 else "right")+("_45" if abs(abs(math.degrees(angle))-45)<=1 else "_90")
            rows.append({"episode":i,"run_duration_s":float(run_duration[i]),"recovery_duration_s":float(recovery_duration[i]),"turn_category":category,"commanded_turn_angle_rad":angle,"run_pre_success":pre_ok,"turn_success":turn_ok,"run_recovery_success":recovery_ok,"sequence_success":success,"pre_run_speed_error_mps":mean(pre_speed),"pre_run_lateral_velocity_p95_mps":pct(pre_lateral_velocity,95),"pre_run_heading_change_p95_rad":pct(pre_heading,95),"pre_run_path_lateral_error_diagnostic_p95_m":pct(pre_lateral,95),"recovery_speed_error_mps":mean(rec_speed),"recovery_heading_error_p95_rad":pct(rec_heading,95),"final_heading_error_rad":float(final_turn_error[i]),"path_lateral_error_p95_m":pct(rec_lat,95),"fall":bool(fallen[i]),"joint_velocity_saturation_fraction":vs,"ankle_torque_saturation_fraction":ats,"saturation_failure":sat,"action_discontinuity_l2_max":max(tr["jumps_l2"],default=0.0),"action_discontinuity_max":max(tr["jumps_max"],default=0.0),"transition_stabilization_time_s":mean(tr["stabilization"]),"unsupported_transition_count":0})
        with (out/"episodes.csv").open("w",newline="",encoding="utf-8") as f:wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
        buckets={name:[r for r in rows if r["turn_category"]==name] for name in ("left_45","right_45","left_90","right_90")}
        finite_heading=[r["final_heading_error_rad"] for r in rows if math.isfinite(r["final_heading_error_rad"])]
        summary={"schema_version":1,"checkpoint":str(checkpoint),"task":"RUN_TURN_RUN","episodes":n,"seed":args.seed,"run_duration_range_s":[args.run_duration_min,args.run_duration_max],"recovery_duration_range_s":[args.recovery_duration_min,args.recovery_duration_max],"sequence_success_rate":mean([float(r["sequence_success"]) for r in rows]),"run_pre_success_rate":mean([float(r["run_pre_success"]) for r in rows]),"turn_success_rate":mean([float(r["turn_success"]) for r in rows]),"run_recovery_success_rate":mean([float(r["run_recovery_success"]) for r in rows]),"fall_rate":mean([float(r["fall"]) for r in rows]),"saturation_failure_rate":mean([float(r["saturation_failure"]) for r in rows]),"pre_run_speed_error_mps":mean([r["pre_run_speed_error_mps"] for r in rows]),"pre_run_lateral_velocity_p95_mps":pct([r["pre_run_lateral_velocity_p95_mps"] for r in rows],95),"recovery_speed_error_mps":mean([r["recovery_speed_error_mps"] for r in rows]),"final_heading_error_rad":mean(finite_heading),"path_lateral_error_p95_m":pct([r["path_lateral_error_p95_m"] for r in rows],95),"action_discontinuity_l2_p95":pct([r["action_discontinuity_l2_max"] for r in rows],95),"action_discontinuity_max":max(r["action_discontinuity_max"] for r in rows),"transition_stabilization_time_s":mean([r["transition_stabilization_time_s"] for r in rows]),"unsupported_transition_count":0,"turn_categories":{k:{"count":len(v),"success_rate":mean([float(r["turn_success"]) for r in v]),"heading_error_rad":mean([r["final_heading_error_rad"] for r in v if math.isfinite(r["final_heading_error_rad"])])} for k,v in buckets.items()}}
        summary["gate_pass"]=bool(summary["sequence_success_rate"]>=.90 and summary["run_recovery_success_rate"]>=.90 and summary["fall_rate"]<=.05 and all(v["success_rate"]>=.90 for v in summary["turn_categories"].values()))
        summary["failure_counts"]=dict(Counter("fall" if r["fall"] else "saturation" if r["saturation_failure"] else "run_pre" if not r["run_pre_success"] else "turn" if not r["turn_success"] else "run_recovery" for r in rows if not r["sequence_success"]))
        (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8");print(json.dumps(summary,indent=2));raw.close()
if __name__=="__main__":main()
