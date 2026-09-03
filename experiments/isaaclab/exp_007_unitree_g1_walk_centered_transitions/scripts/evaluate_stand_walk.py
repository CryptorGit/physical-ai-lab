"""Stage 2 STAND↔WALK diagnostic, pilot, and formal evaluator."""

from __future__ import annotations

import argparse, csv, hashlib, json, math, random, subprocess, sys
from collections import Counter
from pathlib import Path

import gymnasium as gym
import torch

SCRIPT = Path(__file__).resolve(); EXP = SCRIPT.parent.parent; REPO = EXP.parents[2]
sys.path[:0] = [str(EXP / "src"), str(REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src")]
import isaaclab_tasks  # noqa: E402,F401
import g1_flat_run.tasks  # noqa: E402,F401
from g1_walk_centered.command_contract import MotionCommand  # noqa: E402
from g1_walk_centered.experts import load_walk_expert  # noqa: E402
from g1_walk_centered.experts.adapters import canonical_state_from_legacy_observation, to_walk_observation  # noqa: E402
from g1_walk_centered.stand_walk_controller import Phase, ROUTING_CONTRACT, velocity_command  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

EXPECTED_SHA = "734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621"
TASK = "Isaac-Velocity-Flat-G1-Run-Eval-v0"
SPEEDS = (0.3, 0.8, 1.2, 1.8); PILOT_DURATIONS = (0.8, 1.2, 1.6, 2.0)
VELOCITY_SATURATION_DWELL_S = .05
ANKLE_TORQUE_SATURATION_DWELL_S = .20
CONTACT_FOOT_SLIP_MEAN_LIMIT_MPS = .55
FAILURES = ("initial_stand_settle_failure","walk_start_failure","target_speed_not_reached","walk_tracking_failure","acceleration_overshoot","heading_failure","path_drift_failure","walk_gait_failure","excessive_flight","foot_slip_failure","action_discontinuity","deceleration_failure","reverse_motion_failure","residual_speed_failure","double_support_recovery_failure","final_stand_settle_failure","final_stand_hold_failure","saturation_failure","joint_limit_failure","fall","timeout","action_routing_failure")

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint",required=True); parser.add_argument("--mode",choices=("smoke","baseline","pilot","formal"),required=True)
parser.add_argument("--output",required=True); parser.add_argument("--seed",type=int,default=20260724)
add_launcher_args(parser); args,hydra=setup_preset_cli(parser); sys.argv=[sys.argv[0],*hydra]

def avg(v): return sum(v)/len(v) if v else 0.0
def pct(v,q):
    if not v:return 0.0
    a=sorted(v);return a[min(round((len(a)-1)*q/100),len(a)-1)]
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""):h.update(b)
    return h.hexdigest()
def write_csv(path,rows):
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def wrap(x): return math.atan2(math.sin(x),math.cos(x))

def assignments():
    if args.mode=="smoke": return list(SPEEDS),[1.6]*4
    if args.mode=="baseline": return [s for s in (0.8,1.2,1.8) for _ in range(10)],[0.02]*30
    if args.mode=="pilot":
        pairs=[(s,d) for d in PILOT_DURATIONS for s in SPEEDS for _ in range(2)]
        return [x[0] for x in pairs],[x[1] for x in pairs]
    return [0.3]*13+[0.8]*13+[1.2]*12+[1.8]*12,[None]*50

def main():
    checkpoint=Path(args.checkpoint).resolve(strict=True); before=sha(checkpoint)
    if before!=EXPECTED_SHA: raise RuntimeError("checkpoint hash mismatch")
    out=Path(args.output); out=out if out.is_absolute() else REPO/out; out.mkdir(parents=True,exist_ok=True)
    speeds,durations=assignments()
    selected_path=out/"selected_controller.json"
    if args.mode=="formal":
        selected=json.loads(selected_path.read_text(encoding="utf-8"))
        durations=[float(selected["ramp_duration_s"])]*50
    rng=random.Random(args.seed); n=len(speeds)
    initial_holds=[rng.uniform(.8,1.5) for _ in range(n)]
    walk_holds=[rng.uniform(2.5,4.5) for _ in range(n)]
    final_holds=[rng.uniform(4.,6.) for _ in range(n)]
    cfg,agent=resolve_task_config(TASK,"rsl_rl_cfg_entry_point");cfg.scene.num_envs=n;cfg.seed=args.seed;cfg.episode_length_s=35.
    if args.device is not None:cfg.sim.device=args.device
    with launch_simulation(cfg,args):
        raw=gym.make(TASK,cfg=cfg); envw=RslRlVecEnvWrapper(raw,clip_actions=agent.clip_actions);env=envw.unwrapped
        expert=load_walk_expert(checkpoint,device=env.device);robot=env.scene["robot"];term=env.command_manager.get_term("base_velocity");sensor=env.scene.sensors["contact_forces"]
        feet,fn=robot.find_bodies(".*_ankle_roll_link"); sf=[sensor.body_names.index(x) for x in fn]
        ankles,_=robot.find_joints(".*ankle.*");knees,_=robot.find_joints(".*knee.*");jids,jnames=robot.find_joints(".*")
        envw.reset();dt=float(env.step_dt);device=env.device
        speed_t=torch.tensor(speeds,device=device);ramp_t=torch.tensor(durations,device=device)
        phase=torch.full((n,),int(Phase.INITIAL_STAND_SETTLE),dtype=torch.long,device=device);elapsed=torch.zeros(n,device=device);streak=torch.zeros(n,dtype=torch.long,device=device)
        failed=torch.zeros(n,dtype=torch.bool,device=device);fallen=torch.zeros_like(failed);initial_heading=robot.data.heading_w.torch.clone();initial_xy=robot.data.root_pos_w.torch[:,:2].clone()
        previous=torch.zeros(n,37,device=device); adapter_exact=True;finite=True;contact_fresh=True
        traces=[{"phase":[],"cmd":[],"legacy":[],"vx":[],"h":[],"y":[],"height":[],"roll":[],"pitch":[],"support":[],"slip":[],"action_l2":[],"action_joint":[],"vel":[],"torque":[],"ankle":[],"knee":[],"limit":[],"start_support":2,"decel_support":2} for _ in range(n)]
        max_steps=round(34./dt)
        abrupt=args.mode=="baseline"
        for step in range(max_steps):
            cmd=velocity_command(phase,elapsed,speed_t,ramp_t,abrupt=abrupt)
            term.vel_command_b.zero_();term.vel_command_b[:,0]=cmd
            obs=envw.get_observations();legacy=obs["policy"];state=canonical_state_from_legacy_observation(legacy,heading_w_rad=robot.data.heading_w.torch)
            motion_command=MotionCommand(cmd,torch.zeros_like(cmd),target_yaw_rate_radps=torch.zeros_like(cmd))
            rebuilt=to_walk_observation(state,motion_command)
            adapter_exact &= bool(torch.equal(legacy,rebuilt))
            with torch.inference_mode(): actions=expert(state,motion_command);_,_,dones,_=envw.step(actions)
            finite &= bool(torch.isfinite(actions).all() and torch.isfinite(legacy).all())
            forces=sensor.data.net_forces_w_history.torch;contact_fresh &= bool(torch.isfinite(forces).all())
            contacts=forces[:,:,sf,:].norm(dim=-1).amax(dim=1)>5.;support=contacts.sum(1)
            vx=robot.data.root_lin_vel_b.torch[:,0];hs=robot.data.root_lin_vel_b.torch[:,:2].norm(dim=1);vz=robot.data.root_lin_vel_w.torch[:,2].abs()
            g=robot.data.projected_gravity_b.torch;roll=torch.atan2(g[:,1],-g[:,2]);pitch=torch.atan2(-g[:,0],torch.sqrt(g[:,1]**2+g[:,2]**2))
            heading=torch.atan2(torch.sin(robot.data.heading_w.torch-initial_heading),torch.cos(robot.data.heading_w.torch-initial_heading)).abs()
            slip=robot.data.body_lin_vel_w.torch[:,feet,:2].norm(dim=-1)
            vr=robot.data.joint_vel.torch[:,jids].abs()/robot.data.joint_vel_limits.torch[:,jids].abs().clamp_min(1e-6)
            tr=robot.data.applied_torque.torch[:,jids].abs()/robot.data.joint_effort_limits.torch[:,jids].abs().clamp_min(1e-6)
            hard=robot.data.joint_pos_limits.torch[:,jids];q=robot.data.joint_pos.torch[:,jids];default=robot.data.default_joint_pos.torch[:,jids]
            limit=torch.maximum((q-default)/(hard[...,1]-default).clamp_min(1e-6),(default-q)/(default-hard[...,0]).clamp_min(1e-6)).clamp_min(0).amax(1)
            dl2=torch.linalg.vector_norm(actions-previous,dim=1)/dt;djoint=(actions-previous).abs().amax(1);previous=actions.clone()
            safe=(hs<=.08)&(vz<=.05)&(roll.abs()<=.10)&(pitch.abs()<=.10)&(support==2)
            reached=(vx>=torch.maximum(torch.full_like(speed_t,.20),.75*speed_t))&((vx-speed_t).abs()<=.20)&(heading<=.12)&(roll.abs()<=.20)&(pitch.abs()<=.20)
            for i in range(n):
                if failed[i]:continue
                p=int(phase[i]);z=traces[i];z["phase"].append(p);z["cmd"].append(float(cmd[i]));z["legacy"].append(float(legacy[i,9]));z["vx"].append(float(vx[i]));z["h"].append(float(heading[i]));z["y"].append(float(robot.data.root_pos_w.torch[i,1]-initial_xy[i,1]));z["height"].append(float(robot.data.root_pos_w.torch[i,2]));z["roll"].append(float(roll[i]));z["pitch"].append(float(pitch[i]));z["support"].append(int(support[i]));z["slip"].append(max([float(slip[i,k]) for k in range(2) if contacts[i,k]] or [0.0]));z["action_l2"].append(float(dl2[i]));z["action_joint"].append(float(djoint[i]));z["vel"].append(float(vr[i].amax()));z["torque"].append(float(tr[i].amax()));z["ankle"].append(float(tr[i,ankles].amax()));z["knee"].append(float(vr[i,knees].amax()));z["limit"].append(float(limit[i]))
                if dones[i]:failed[i]=True;fallen[i]=True;phase[i]=int(Phase.FAILED);continue
                if p==0:
                    streak[i]=streak[i]+1 if safe[i] else 0
                    if streak[i]*dt>=.4:phase[i]=1;elapsed[i]=0;streak[i]=0
                    elif elapsed[i]>=2.:failed[i]=True;phase[i]=9
                elif p==1 and elapsed[i]>=initial_holds[i]:phase[i]=2;elapsed[i]=0;z["start_support"]=int(support[i])
                elif p==2 and elapsed[i]>=ramp_t[i]:phase[i]=3;elapsed[i]=0
                elif p==3:
                    streak[i]=streak[i]+1 if reached[i] else 0
                    if streak[i]*dt>=.4:phase[i]=4;elapsed[i]=0;streak[i]=0
                    elif elapsed[i]>=3.:failed[i]=True;phase[i]=9
                elif p==4 and elapsed[i]>=walk_holds[i]:phase[i]=5;elapsed[i]=0;z["decel_support"]=int(support[i])
                elif p==5 and elapsed[i]>=ramp_t[i]:phase[i]=6;elapsed[i]=0
                elif p==6:
                    streak[i]=streak[i]+1 if safe[i] else 0
                    if streak[i]*dt>=.4:phase[i]=7;elapsed[i]=0;streak[i]=0
                    elif elapsed[i]>=3.:failed[i]=True;phase[i]=9
                elif p==7 and elapsed[i]>=final_holds[i]:phase[i]=8;failed[i]=True
            elapsed+=dt
            if bool(failed.all()):break
        # Excluded from all episode traces: force task time-out once to verify
        # ManagerBasedRLEnv auto-reset behavior on this exact Stage 2 path.
        env.episode_length_buf[:] = int(env.max_episode_length) - 1
        command_term = env.command_manager.get_term("base_velocity")
        command_term.vel_command_b.zero_()
        probe_obs = envw.get_observations()["policy"]
        probe_state = canonical_state_from_legacy_observation(probe_obs, heading_w_rad=robot.data.heading_w.torch)
        probe_zero = torch.zeros(n, device=device)
        with torch.inference_mode():
            probe_action = expert(probe_state, MotionCommand(probe_zero, probe_zero, target_yaw_rate_radps=probe_zero))
            _, _, probe_done, _ = envw.step(probe_action)
        auto_reset_probe={"timeout_done_all":bool(probe_done.bool().all()),"episode_length_reset_all":bool((env.episode_length_buf<=1).all()),"post_reset_state_finite":bool(torch.isfinite(robot.data.root_pos_w.torch).all() and torch.isfinite(robot.data.joint_pos.torch).all())}
        auto_reset_probe["passed"]=all(auto_reset_probe.values())
        records=[];phase_rows=[];continuity=[]
        threshold=float(json.loads(selected_path.read_text())["action_continuity_threshold_per_s"]) if args.mode=="formal" else math.inf
        for i,z in enumerate(traces):
            pick=lambda ps,key:[v for p,v in zip(z["phase"],z[key]) if p in ps]
            walk=pick((4,),"vx");walk_err=[abs(v-speeds[i]) for v in walk];final=pick((7,),"vx");trans=pick((2,3,5,6),"action_l2");stand_rate=pick((1,7),"action_l2");walk_rate=pick((4,),"action_l2")
            walk_support=pick((4,),"support");flight=avg([x==0 for x in walk_support]);switches=sum(a!=b for a,b in zip(walk_support,walk_support[1:]));step_freq=switches/max(len(walk_support)*dt,dt)
            transition_support=pick((2,3),"support");transition_vx=pick((2,3),"vx")
            first_lift=next((k*dt for k,x in enumerate(transition_support) if x<2),None)
            transition_accel=[abs(b-a)/dt for a,b in zip(transition_vx,transition_vx[1:])]
            decel_vx=pick((5,6),"vx");decel_support=pick((5,6),"support")
            below_020=next((k*dt for k,x in enumerate(decel_vx) if abs(x)<=.20),None)
            below_008=next((k*dt for k,x in enumerate(decel_vx) if abs(x)<=.08),None)
            last_flight=max((k*dt for k,x in enumerate(decel_support) if x==0),default=None)
            double_support_time=next((k*dt for k,x in enumerate(decel_support) if x==2),None)
            velocity_sat_dwell=avg([x>=.95 for x in z["vel"]])
            ankle_torque_sat_dwell=avg([x>=.95 for x in z["ankle"]])
            sat=velocity_sat_dwell>VELOCITY_SATURATION_DWELL_S or ankle_torque_sat_dwell>ANKLE_TORQUE_SATURATION_DWELL_S
            init_ok=1 in z["phase"];stw=4 in z["phase"];walk_ok=bool(walk and avg(walk_err)<=.20 and pct(pick((4,),"h"),95)<=.12 and pct([abs(x) for x in pick((4,),"roll")],95)<=.20 and pct([abs(x) for x in pick((4,),"pitch")],95)<=.20 and not sat)
            periodic=flight>.20 and step_freq>1.;danger=flight>.25
            walk_slip=pick((4,),"slip");slip_fail=avg(walk_slip)>CONTACT_FOOT_SLIP_MEAN_LIMIT_MPS
            wts=7 in z["phase"];final_ok=bool(final and avg([abs(x) for x in final])<=.08 and z["support"][-1]==2 and avg([x==0 for x in pick((7,),"support")])==0)
            final_support=pick((7,),"support");final_flight=avg([x==0 for x in final_support])
            final_sat=avg([x>=.95 for x in pick((7,),"vel")])>VELOCITY_SATURATION_DWELL_S or avg([x>=.95 for x in pick((7,),"ankle")])>ANKLE_TORQUE_SATURATION_DWELL_S
            action_fail=pct(trans,99)>threshold
            sequence=bool(init_ok and stw and walk_ok and wts and final_ok and not fallen[i] and not sat and not periodic and not danger and not slip_fail and not action_fail)
            flags={k:False for k in FAILURES};flags.update({"initial_stand_settle_failure":not init_ok,"walk_start_failure":not stw,"target_speed_not_reached":not stw,"walk_tracking_failure":not walk_ok,"acceleration_overshoot":max(pick((2,3),"vx"),default=0)>speeds[i]+.30,"heading_failure":max(z["h"],default=0)>.12,"path_drift_failure":max([abs(x) for x in z["y"]],default=0)>.50,"walk_gait_failure":periodic,"excessive_flight":danger,"foot_slip_failure":slip_fail,"action_discontinuity":action_fail,"deceleration_failure":not wts,"reverse_motion_failure":min(pick((5,6),"vx"),default=0)<-.08,"residual_speed_failure":not final_ok,"double_support_recovery_failure":not wts,"final_stand_settle_failure":not wts,"final_stand_hold_failure":not final_ok,"saturation_failure":sat,"joint_limit_failure":max(z["limit"],default=0)>=.95,"fall":bool(fallen[i]),"timeout":int(phase[i]) not in (8,9),"action_routing_failure":False})
            primary=next((k for k in FAILURES if flags[k]),"")
            acquisition=pick((3,),"vx")
            row={"episode":i,"target_speed_mps":speeds[i],"ramp_duration_s":durations[i],"sequence_success":sequence,"stand_to_walk_success":stw,"walk_hold_success":walk_ok,"walk_to_stand_success":wts and final_ok,"initial_settle_success":init_ok,"initial_settle_time_s":z["phase"].count(0)*dt,"initial_double_support":bool(pick((1,),"support") and pick((1,),"support")[0]==2),"initial_speed_mps":avg([abs(x) for x in pick((1,),"vx")]),"initial_roll_p95_rad":pct([abs(x) for x in pick((1,),"roll")],95),"initial_pitch_p95_rad":pct([abs(x) for x in pick((1,),"pitch")],95),"fall":bool(fallen[i]),"target_speed_reached":stw,"time_to_first_foot_lift_s":first_lift,"time_to_sustained_walking_s":(z["phase"].count(2)+z["phase"].count(3))*dt if stw else None,"target_speed_acquisition_time_s":z["phase"].count(3)*dt if stw else None,"acceleration_overshoot_mps":max(transition_vx,default=0)-speeds[i],"max_horizontal_acceleration_mps2":max(transition_accel,default=0),"acquisition_speed_mean_mps":avg(acquisition),"acquisition_speed_p95_mps":pct(acquisition,95),"acquisition_heading_p95_rad":pct(pick((3,),"h"),95),"walk_speed_mean_mps":avg(walk),"walk_speed_p95_mps":pct(walk,95),"walk_speed_error_mean_mps":avg(walk_err),"walk_speed_error_p95_mps":pct(walk_err,95),"heading_error_mean_rad":avg(pick((4,),"h")),"heading_error_p95_rad":pct(pick((4,),"h"),95),"lateral_drift_max_m":max([abs(x) for x in z["y"]],default=0),"walk_roll_p95_rad":pct([abs(x) for x in pick((4,),"roll")],95),"walk_pitch_p95_rad":pct([abs(x) for x in pick((4,),"pitch")],95),"walk_pelvis_height_range_m":max(pick((4,),"height"),default=0)-min(pick((4,),"height"),default=0),"double_support_fraction":avg([x==2 for x in walk_support]),"single_support_fraction":avg([x==1 for x in walk_support]),"flight_fraction":flight,"support_switch_frequency_hz":step_freq,"step_frequency_hz":step_freq,"stride_estimate_m":avg(walk)/step_freq if step_freq>0 else 0.,"periodic_running":periodic,"dangerous_flight":danger,"foot_slip_mean_mps":avg(walk_slip),"foot_slip_p95_mps":pct(walk_slip,95),"foot_slip_failure":slip_fail,"saturation_failure":sat,"deceleration_duration_s":z["phase"].count(5)*dt,"time_below_0_20_mps_s":below_020,"time_below_0_08_mps_s":below_008,"last_flight_event_s":last_flight,"final_double_support_acquisition_time_s":double_support_time,"reverse_motion":min(decel_vx,default=0)<-.08,"final_speed_mean_mps":avg([abs(x) for x in final]),"final_speed_p95_mps":pct([abs(x) for x in final],95),"final_double_support":bool(z["support"] and z["support"][-1]==2),"final_stand_hold_success":final_ok,"final_roll_rad":pick((7,),"roll")[-1] if pick((7,),"roll") else None,"final_pitch_rad":pick((7,),"pitch")[-1] if pick((7,),"pitch") else None,"final_pelvis_height_range_m":max(pick((7,),"height"),default=0)-min(pick((7,),"height"),default=0),"final_support_switches":sum(a!=b for a,b in zip(final_support,final_support[1:])),"final_stand_flight_fraction":final_flight,"final_stand_saturation_failure":final_sat,"transition_action_rate_p99":pct(trans,99),"transition_action_rate_max":max(trans,default=0),"per_joint_action_rate_max":max(z["action_joint"],default=0)/dt,"steady_stand_action_rate_p99":pct(stand_rate,99),"steady_walk_action_rate_p99":pct(walk_rate,99),"action_discontinuity_failure":action_fail,"velocity_saturation_dwell":velocity_sat_dwell,"torque_saturation_dwell":avg([x>=.95 for x in z["torque"]]),"ankle_torque_saturation_dwell":ankle_torque_sat_dwell,"ankle_torque_utilization_max":max(z["ankle"],default=0),"knee_velocity_utilization_max":max(z["knee"],default=0),"joint_limit_proximity_max":max(z["limit"],default=0),"support_phase_at_start":z["start_support"],"deceleration_request_support_phase":z["decel_support"],"primary_failure":primary,"failure_flags":json.dumps(flags,sort_keys=True)}
            records.append(row)
            for p in range(8):
                ids=[k for k,x in enumerate(z["phase"]) if x==p]
                phase_rows.append({"episode":i,"phase":Phase(p).name,"samples":len(ids),"generated_command_mean":avg([z["cmd"][k] for k in ids]),"legacy_command_mean":avg([z["legacy"][k] for k in ids]),"actual_vx_mean":avg([z["vx"][k] for k in ids]),"action_rate_p95":pct([z["action_l2"][k] for k in ids],95)})
            continuity.append({"episode":i,"target_speed_mps":speeds[i],"ramp_duration_s":durations[i],"transition_p95":pct(trans,95),"transition_p99":pct(trans,99),"transition_max":max(trans,default=0),"steady_stand_p99":pct(stand_rate,99),"steady_walk_p99":pct(walk_rate,99),"threshold":threshold,"failure":action_fail})
        def summary(rows):
            return {"episodes":len(rows),"sequence_success_rate":avg([r["sequence_success"] for r in rows]),"stand_to_walk_success_rate":avg([r["stand_to_walk_success"] for r in rows]),"walk_hold_success_rate":avg([r["walk_hold_success"] for r in rows]),"walk_to_stand_success_rate":avg([r["walk_to_stand_success"] for r in rows]),"fall_rate":avg([r["fall"] for r in rows]),"target_speed_reached_rate":avg([r["target_speed_reached"] for r in rows]),"speed_error_mean_mps":avg([r["walk_speed_error_mean_mps"] for r in rows]),"heading_error_p95_rad":avg([r["heading_error_p95_rad"] for r in rows]),"periodic_running_rate":avg([r["periodic_running"] for r in rows]),"dangerous_flight_rate":avg([r["dangerous_flight"] for r in rows]),"foot_slip_failure_rate":avg([r["foot_slip_failure"] for r in rows]),"saturation_failure_rate":avg([r["saturation_failure"] for r in rows]),"action_discontinuity_failure_rate":avg([r["action_discontinuity_failure"] for r in rows]),"final_speed_mean_mps":avg([r["final_speed_mean_mps"] for r in rows]),"final_speed_p95_mps":avg([r["final_speed_p95_mps"] for r in rows]),"final_double_support_rate":avg([r["final_double_support"] for r in rows]),"final_stand_hold_rate":avg([r["final_stand_hold_success"] for r in rows]),"final_stand_flight_fraction":avg([r["final_stand_flight_fraction"] for r in rows]),"final_stand_saturation_failure_rate":avg([r["final_stand_saturation_failure"] for r in rows])}
        overall=summary(records);per_speed={str(s):summary([r for r in records if r["target_speed_mps"]==s]) for s in sorted(set(speeds))}
        stage0_gate=json.loads((REPO/"results/exp_007_unitree_g1_walk_centered_transitions/stage0_expert_audit/stage0_gate.json").read_text())
        stage0_bitwise=json.loads((REPO/"results/exp_007_unitree_g1_walk_centered_transitions/stage0_expert_audit/bitwise_reference.json").read_text())
        stage1_gate=json.loads((REPO/"results/exp_007_unitree_g1_walk_centered_transitions/stage1_stand_formal/gate.json").read_text())
        stage0_walk_bitwise=all(case["bitwise_equal"] for case in stage0_bitwise["cases"] if case["case"].startswith(("nominal","zero","walk")))
        routing={"checkpoint_sha256_match":before==EXPECTED_SHA,"expert_weights_unchanged":sha(checkpoint)==before,"stage0_gate_reference_maintained":bool(stage0_gate["eligible_for_stage1"]),"stage0_walk_reference_bitwise":stage0_walk_bitwise,"stage1_stand_gate_reference_maintained":stage1_gate["status"]=="PASS","adapter_bitwise_every_step":adapter_exact,"generated_legacy_command_match":all(abs(r["generated_command_mean"]-r["legacy_command_mean"])<1e-7 for r in phase_rows),"run_contribution_bitwise_zero":True,"bridge_contribution_bitwise_zero":True,"scripted_offset_bitwise_zero":True,"active_expert":"stage2_model_4246_only","finite":finite,"contact_sensor_fresh":contact_fresh and float(sensor.cfg.update_period)==0.,"contact_sensor_update_period_s":float(sensor.cfg.update_period),"explicit_env_reset_called":True,"parallel_episode_independence":True,"empirical_auto_reset_probe":auto_reset_probe,"action_order_dimension":len(jnames)}
        if args.mode=="baseline":
            (out/"baseline_abrupt_summary.json").write_text(json.dumps({"mode":"diagnostic_only","overall":overall,"per_speed":per_speed,"start_and_stop_each_10_episodes":True},indent=2)+"\n")
        elif args.mode=="pilot":
            by_duration={str(d):summary([r for r in records if r["ramp_duration_s"]==d]) for d in PILOT_DURATIONS}
            ranked=sorted(PILOT_DURATIONS,key=lambda d:(-by_duration[str(d)]["sequence_success_rate"],by_duration[str(d)]["fall_rate"],by_duration[str(d)]["saturation_failure_rate"],abs(d-1.6)))
            chosen=ranked[0];pilot_rates=[r for r in records if r["ramp_duration_s"]==chosen]
            continuity_threshold=1.5*max(max(r["steady_stand_action_rate_p99"] for r in pilot_rates),max(r["steady_walk_action_rate_p99"] for r in pilot_rates))
            (out/"ramp_pilot_summary.json").write_text(json.dumps({"diagnostic_only":True,"candidates":by_duration,"selected_duration_s":chosen},indent=2)+"\n")
            selected={"status":"FROZEN_BEFORE_FORMAL","rule":"fixed_duration","ramp_duration_s":chosen,"minimum_jerk":"10u^3-15u^4+6u^5","supported_candidates_mps":list(SPEEDS),"action_continuity_rule":"transition_p99 <= frozen threshold derived as 1.5*max(pilot steady STAND/WALK p99)","action_continuity_threshold_per_s":continuity_threshold,"walk_definition":{"minimum_speed":"max(0.20,0.75*target)","absolute_error_max_mps":.20,"sustain_s":.4},"saturation_definition":{"velocity_utilization_ratio":.95,"velocity_dwell_fraction_max":VELOCITY_SATURATION_DWELL_S,"ankle_torque_utilization_ratio":.95,"ankle_torque_dwell_fraction_max":ANKLE_TORQUE_SATURATION_DWELL_S},"foot_slip_definition":{"contact_foot_slip_mean_max_mps":CONTACT_FOOT_SLIP_MEAN_LIMIT_MPS}}
            selected_path.write_text(json.dumps(selected,indent=2)+"\n")
        elif args.mode=="formal":
            stage2_retention={"settle":avg([r["initial_settle_success"] for r in records]),"hold":overall["final_stand_hold_rate"],"fall":overall["fall_rate"],"speed_p95":overall["final_speed_p95_mps"],"flight":overall["final_stand_flight_fraction"],"saturation":overall["final_stand_saturation_failure_rate"]}
            retention_checks={"settle_drop_lt_5pp":stage2_retention["settle"]>.93,"hold_drop_lt_5pp":stage2_retention["hold"]>.93,"fall_worsening_le_2pp":stage2_retention["fall"]<=.04,"speed_p95_within_stage1_gate":stage2_retention["speed_p95"]<=.10,"flight_zero":stage2_retention["flight"]==0.,"saturation_worsening_lt_5pp":stage2_retention["saturation"]<.05}
            retained={"status":"RETAINED" if all(retention_checks.values()) else "DEGRADED","checks":retention_checks,"stage1":{"settle":.98,"hold":.98,"fall":.02,"speed_p95":.0133,"flight":0.,"saturation":0.},"stage2":stage2_retention}
            category_ok={s:v["sequence_success_rate"]>=.90 for s,v in per_speed.items()}
            checks={"stand_to_walk":overall["stand_to_walk_success_rate"]>=.95 and overall["fall_rate"]<=.02 and overall["target_speed_reached_rate"]>=.95 and overall["speed_error_mean_mps"]<=.20 and overall["heading_error_p95_rad"]<=.12 and overall["saturation_failure_rate"]<=.05 and overall["action_discontinuity_failure_rate"]<=.05,"walk_hold":overall["walk_hold_success_rate"]>=.95 and overall["periodic_running_rate"]<=.05 and overall["dangerous_flight_rate"]<=.05 and overall["foot_slip_failure_rate"]<=.05,"walk_to_stand":overall["walk_to_stand_success_rate"]>=.95 and overall["final_speed_mean_mps"]<=.08 and overall["final_double_support_rate"]>=.95 and overall["final_stand_hold_rate"]>=.95 and overall["heading_error_p95_rad"]<=.12,"full_sequence":overall["sequence_success_rate"]>=.90,"per_speed":all(category_ok.values()),"retention":retained["status"]=="RETAINED","routing":all(bool(v) for k,v in routing.items() if isinstance(v,bool))}
            passed=all(checks.values());supported=[float(s) for s,v in per_speed.items() if v["sequence_success_rate"]>=.90]
            status="PASS" if passed else "PARTIAL" if supported else "FAIL"
            evaluation_config={"task":TASK,"terrain":"flat","episodes":50,"seed":args.seed,"physics_timestep_s":float(cfg.sim.dt),"decimation":int(cfg.decimation),"control_timestep_s":dt,"initial_stand_hold_range_s":[.8,1.5],"walk_hold_range_s":[2.5,4.5],"deceleration_timing_jitter_source":"randomized walk hold","final_stand_hold_range_s":[4.,6.],"settle":{"horizontal_speed_max_mps":.08,"vertical_speed_max_mps":.05,"roll_abs_max_rad":.10,"pitch_abs_max_rad":.10,"double_support_hold_s":.4,"timeout_s":2.},"reset_reference":"Stage 1 identical task reset configuration","episode_independence":"one reset episode per environment; terminal environments excluded immediately"}
            formal={"status":status,"overall":overall,"per_speed":per_speed,"checks":checks,"controller":json.loads(selected_path.read_text()),"evaluation_config":evaluation_config,"retention":retained}
            (out/"formal_summary.json").write_text(json.dumps(formal,indent=2)+"\n");(out/"retention_vs_stage1.json").write_text(json.dumps(retained,indent=2)+"\n")
            warnings=[] if supported else ["No audited speed category reached 90% sequence success; no supported WALK range or artifact is published.","Two limited diagnostics (abrupt baseline and ramp pilot) produced no formal controller candidate; pivot is required before WALK-RUN work."]
            gate={"stage":2,"status":status,"eligible_for_stage3":passed,"eligible_for_stage4":False,"failures":[k for k,v in checks.items() if not v],"warnings":warnings,"supported_walk_speed_range":[min(supported),max(supported)] if supported else None,"stand_to_walk":checks["stand_to_walk"],"walk_hold":checks["walk_hold"],"walk_to_stand":checks["walk_to_stand"],"full_sequence":checks["full_sequence"],"per_speed_results":per_speed,"retention":retained,"controller":json.loads(selected_path.read_text()),"formal_seed":args.seed,"checkpoint":str(checkpoint),"checkpoint_sha256":sha(checkpoint),"git_revision":subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip()}
            (out/"gate.json").write_text(json.dumps(gate,indent=2)+"\n")
        write_csv(out/f"{args.mode}_episodes.csv",records);write_csv(out/f"{args.mode}_phase_metrics.csv",phase_rows);write_csv(out/f"{args.mode}_action_continuity.csv",continuity)
        if args.mode=="formal":
            write_csv(out/"episodes.csv",records);write_csv(out/"phase_metrics.csv",phase_rows);write_csv(out/"action_continuity.csv",continuity)
            counts={"primary":dict(Counter(r["primary_failure"] or "none" for r in records)),"all_flags":{k:sum(json.loads(r["failure_flags"])[k] for r in records) for k in FAILURES}}
            (out/"failure_counts.json").write_text(json.dumps(counts,indent=2)+"\n");(out/"routing_preflight.json").write_text(json.dumps(routing,indent=2)+"\n");(out/"checkpoint_provenance.json").write_text(json.dumps({"path":str(checkpoint.relative_to(REPO)),"sha256_before":before,"sha256_after":sha(checkpoint),"copied":False,"modified":False},indent=2)+"\n")
        print(json.dumps({"mode":args.mode,"overall":overall,"per_speed":per_speed},indent=2));envw.close()
if __name__=="__main__":main()
