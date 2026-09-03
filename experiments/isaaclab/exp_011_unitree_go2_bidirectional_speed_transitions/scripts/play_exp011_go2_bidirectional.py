"""GUI diagnostic playback for the frozen exp_011 single Go2 policy."""

from __future__ import annotations
import argparse, hashlib, json, math, sys
from pathlib import Path
import gymnasium as gym
import torch
import warp as wp

SCRIPT=Path(__file__).resolve();EXP=SCRIPT.parent.parent;REPO=EXP.parents[2];sys.path.insert(0,str(EXP/"src"))
import isaaclab_tasks  # noqa
from isaaclab_tasks.utils import add_launcher_args,launch_simulation,resolve_task_config,setup_preset_cli
from go2_bidirectional.command_profiles import FULL_SEQUENCE,LIMITED_SEQUENCE,sequence_command,transition_command
from go2_bidirectional.contact_analysis import resolve_foot_mapping
from go2_bidirectional.evaluation import build_runner
from go2_bidirectional.gait_classifier import classify
from go2_bidirectional.phase_gated_heading import PhaseGatedFixedHeadingController
from go2_bidirectional.stage6_endpoint_protocol import (
    classify_go2_gait_v1,
    heading_error,
    quat_xyzw_to_gravity_tilt,
    quat_xyzw_to_roll_pitch_yaw,
)
from isaaclab_physx.sensors import ContactSensorCfg

p=argparse.ArgumentParser(description=__doc__)
p.add_argument("--mode",choices=("Stand","Steady","SteadyState","Transition","ReducedSequence","AnchorSequence","FullSequence","Showcase"),default="AnchorSequence")
p.add_argument("--heading-controller",choices=("OpenLoop","AlwaysOn","PhaseGated"),default="PhaseGated")
p.add_argument("--checkpoint",required=True,type=Path);p.add_argument("--target-speed",type=float,default=1.2);p.add_argument("--source-speed",type=float,default=0.0);p.add_argument("--target-yaw-rate",type=float,default=0.0);p.add_argument("--ramp-duration",type=float,default=1.5);p.add_argument("--seed",type=int,default=20260901);p.add_argument("--record-video",action="store_true");p.add_argument("--output-path",default="");p.add_argument("--show-floor-guides",action=argparse.BooleanOptionalAction,default=True);p.add_argument("--width",type=int,default=1920);p.add_argument("--height",type=int,default=1080)
add_launcher_args(p);a,h=setup_preset_cli(p);sys.argv=[sys.argv[0],*h]
if a.record_video:a.enable_cameras=True
# Preserve the formally validated anchor route while presenting its five
# semantic phases as STOP / WALK / RUN / WALK / STOP.
SHOWCASE_SEQUENCE=(0.0,0.6,1.2,2.0,1.2,0.6,0.0)


class FloorGuides:
    """Visual-only debug lines, adapted from the exp_007 showcase lineage."""
    def __init__(self, origin, heading):
        self.line_count=0
        if not a.show_floor_guides:return
        from isaacsim.core.experimental.utils.app import enable_extension
        enable_extension("isaacsim.util.debug_draw")
        from isaacsim.util.debug_draw import _debug_draw
        interface=_debug_draw.acquire_debug_draw_interface();interface.clear_lines()
        origin=origin.detach().cpu()[:2];f=torch.tensor((math.cos(heading),math.sin(heading)));l=torch.tensor((-math.sin(heading),math.cos(heading)))
        def world(x,y,z):
            q=origin+x*f+y*l;return(float(q[0]),float(q[1]),z)
        starts=[];ends=[];colors=[];widths=[]
        for x in range(-5,71):
            major=x%5==0;starts.append(world(x,-1.5,.012));ends.append(world(x,1.5,.012));colors.append((.25,.72,.90,1) if major else (.55,.60,.64,1));widths.append(5.0 if major else 2.0)
        for y in (-1.5,1.5):
            starts.append(world(-5,y,.012));ends.append(world(70,y,.012));colors.append((.88,.90,.92,1));widths.append(4.0)
        interface.draw_lines(starts,ends,colors,widths);self.line_count=len(starts)


class TrackingCamera:
    def __init__(self,env,heading):
        self.env=env;f=torch.tensor((math.cos(heading),math.sin(heading),0.));l=torch.tensor((-math.sin(heading),math.cos(heading),0.));self.f=f;self.offset=-3.8*f-3.2*l+torch.tensor((0.,0.,1.9));self.filtered=None
    def update(self,root):
        point=root.detach().cpu().float();self.filtered=point if self.filtered is None else .65*self.filtered+.35*point
        eye=self.filtered+self.offset;look=self.filtered+.5*self.f+torch.tensor((0.,0.,.45))
        eye_tuple=tuple(map(float,eye));target_tuple=tuple(map(float,look))
        self.env.sim.set_camera_view(eye=eye_tuple,target=target_tuple)
        recorder=getattr(self.env,"video_recorder",None);capture=getattr(recorder,"_capture",None);capture_cfg=getattr(capture,"cfg",None)
        if capture_cfg is not None:
            from isaacsim.core.rendering_manager import ViewportManager
            ViewportManager.set_camera_view(capture_cfg.camera_prim_path,eye=list(eye_tuple),target=list(target_tuple))


class Overlay:
    def __init__(self):
        self.label=None;self.window=None;self.updates=0
        try:
            import omni.ui as ui
            self.window=ui.Window("EXP_011 GO2 PHASE-GATED FIXED HEADING",width=640,height=440)
            with self.window.frame:
                with ui.VStack():
                    ui.Label("EXP_011 GO2 PHASE-GATED FIXED HEADING",height=28)
                    self.label=ui.Label("",word_wrap=True)
        except Exception as exc:
            print(f"overlay_backend=console reason={type(exc).__name__}:{exc}")
    def update(self,text):
        self.updates+=1
        if self.label is not None:self.label.text=text
        elif self.updates==1 or self.updates%50==0:print(text.replace("\n"," | "))


def prepare_recording():
    if not a.record_video:return None
    import cv2
    destination=Path(a.output_path).resolve()
    if destination.suffix.lower()!=".mp4":destination=destination/"exp011_go2_stop_walk_run_walk_stop.mp4"
    destination.parent.mkdir(parents=True,exist_ok=True)
    raw_folder=destination.parent/"capture_raw";raw_folder.mkdir(parents=True,exist_ok=True)
    raw_path=raw_folder/f"{destination.stem}_raw.mp4"
    writer=cv2.VideoWriter(str(raw_path),cv2.VideoWriter_fourcc(*"mp4v"),50.0,(a.width,a.height))
    if not writer.isOpened():raise RuntimeError(f"unable to open recorder: {raw_path}")
    return destination,raw_path,writer


def burn_overlay(recording,lines_by_frame):
    if recording is None:return
    import cv2
    destination,source,source_writer=recording;source_writer.release()
    capture=cv2.VideoCapture(str(source));fps=capture.get(cv2.CAP_PROP_FPS) or 50.0
    width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    temporary=destination.with_name(destination.stem+"_burnin_tmp.mp4")
    writer=cv2.VideoWriter(str(temporary),cv2.VideoWriter_fourcc(*"mp4v"),fps,(width,height));frame_index=0
    while True:
        ok,image=capture.read()
        if not ok:break
        lines=lines_by_frame[min(frame_index,len(lines_by_frame)-1)] if lines_by_frame else []
        panel=image.copy();cv2.rectangle(panel,(18,18),(850,58+30*len(lines)),(18,22,28),-1);cv2.addWeighted(panel,.78,image,.22,0,image)
        for index,line in enumerate(lines):cv2.putText(image,line,(36,52+30*index),cv2.FONT_HERSHEY_SIMPLEX,.68,(245,248,252),2,cv2.LINE_AA)
        writer.write(image);frame_index+=1
    capture.release();writer.release();source.unlink(missing_ok=True)
    if destination.exists():destination.unlink()
    temporary.replace(destination)
    print(f"record_result=PASS path={destination} frames={frame_index} fps={fps}")


def main():
    if a.mode=="Stand":duration=8.;command_fn=lambda t:(0.,"HOLD")
    elif a.mode in ("Steady","SteadyState"):duration=8.;command_fn=lambda t:(a.target_speed,"HOLD")
    elif a.mode=="Transition":duration=3+a.ramp_duration+5;command_fn=lambda t:transition_command(t,a.source_speed,a.target_speed,a.ramp_duration)
    else:
        speeds=SHOWCASE_SEQUENCE if a.mode=="Showcase" else LIMITED_SEQUENCE if a.mode in ("ReducedSequence","AnchorSequence") else FULL_SEQUENCE
        duration=3+(len(speeds)-1)*(3+a.ramp_duration);command_fn=lambda t:(lambda v,i,s:(v,f"{speeds[max(0,i-1)]}->{speeds[i]} {s.upper()}"))(*sequence_command(t,speeds,a.ramp_duration))
    c,agent=resolve_task_config("Isaac-Velocity-Flat-Unitree-Go2-v0","rsl_rl_cfg_entry_point");c.scene.num_envs=1;c.seed=a.seed;c.episode_length_s=duration+2;c.observations.policy.enable_corruption=False;c.events.base_external_force_torque=None;c.events.push_robot=None;c.viewer.origin_type="world";c.video_recorder.window_width=a.width;c.video_recorder.window_height=a.height
    if a.mode=="Showcase":
        # The policy is yaw-equivariant on the flat task. Fix only the normal
        # reset yaw so the world-grid texture and course-relative visual guides
        # share an axis in the public-facing video.
        c.events.reset_base.params["pose_range"]["yaw"]=(0.0,0.0)
        c.commands.base_velocity.debug_vis=False
    for label,foot in zip(("fl","fr","rl","rr"),("FL_foot","FR_foot","RL_foot","RR_foot")):
        setattr(c.scene,f"stage6_{label}_contact",ContactSensorCfg(prim_path=f"{{ENV_REGEX_NS}}/Robot/{foot}",update_period=0.0,track_pose=True,track_contact_points=True,track_friction_forces=True,max_contact_data_count_per_prim=8,filter_prim_paths_expr=["/World/ground/terrain/GroundPlane/CollisionPlane"]))
    if a.device:c.sim.device=a.device
    with launch_simulation(c,a):
        raw=gym.make("Isaac-Velocity-Flat-Unitree-Go2-v0",cfg=c,render_mode="rgb_array" if a.record_video else None)
        if a.record_video:
            if not a.output_path:raise ValueError("--record-video requires --output-path")
        recording=prepare_recording();burn_lines=[]
        w,_,policy=build_runner(raw,agent,a.checkpoint.resolve(strict=True));env=w.unwrapped;robot=env.scene["robot"];term=env.command_manager.get_term("base_velocity");sensor=env.scene.sensors["contact_forces"];mapping=resolve_foot_mapping(robot,sensor);sensor_ids=[x["contact_sensor_index"] for x in mapping];body_ids=[x["robot_body_index"] for x in mapping]
        point_sensors=[env.scene.sensors[f"stage6_{label}_contact"] for label in ("fl","fr","rl","rr")]
        w.reset();heading=float(robot.data.heading_w.torch[0]);FloorGuides(robot.data.root_pos_w.torch[0],heading);camera=TrackingCamera(env,heading);overlay=Overlay();contacts_trace=[];previous_points=[None]*4;heading_history=[];yaw_history_abs=[];contact_age=[0]*4
        calibration_path=REPO/"results/exp_011_unitree_go2_bidirectional_speed_transitions/stage11_tangential_slip_reduction/slip_reward_calibration.json"
        slip_lambda=float(__import__("json").loads(calibration_path.read_text())["lambda_slip"]) if calibration_path.exists() else 0.0
        controller_mode={"OpenLoop":"OPEN_LOOP","AlwaysOn":"ALWAYS_ON_FIXED_HEADING","PhaseGated":"PHASE_GATED_FIXED_HEADING"}[a.heading_controller]
        controller_kind="transition" if a.mode=="Transition" else "steady"
        heading_controller=PhaseGatedFixedHeadingController(controller_mode,controller_kind,a.target_speed,float(env.step_dt));sequence_segment=0
        dt=float(env.step_dt)
        for step in range(round(duration/dt)):
            now=step*dt;target,direction=command_fn(now)
            quaternion_now=[float(value) for value in robot.data.root_quat_w.torch[0]];_,_,yaw_now=quat_xyzw_to_roll_pitch_yaw(quaternion_now);actual_now=float(robot.data.root_lin_vel_b.torch[0,0])
            if a.mode=="Transition":schedule_phase={"source_hold":"source","ramp":"ramp","target_hold":"target"}.get(direction,"target");controller_time=now
            elif a.mode in ("ReducedSequence","AnchorSequence","FullSequence","Showcase"):
                speeds=SHOWCASE_SEQUENCE if a.mode=="Showcase" else LIMITED_SEQUENCE if a.mode in ("ReducedSequence","AnchorSequence") else FULL_SEQUENCE;_,segment,sequence_phase=sequence_command(now,speeds,a.ramp_duration)
                if segment!=sequence_segment:
                    sequence_segment=segment;heading_controller=PhaseGatedFixedHeadingController(controller_mode,"transition",speeds[segment],dt);heading_controller.reference_samples=list(yaw_history_abs[-max(1,round(.5/dt)):])
                schedule_phase="ramp" if sequence_phase=="ramp" else ("steady" if segment==0 else "target");controller_time=now if segment==0 else 3.0+((now-3.0)-(segment-1)*(3.0+a.ramp_duration))
            else:schedule_phase="steady";controller_time=now
            control=heading_controller.update(controller_time,yaw_now,actual_now,schedule_phase)
            term.vel_command_b[:,0]=target;term.vel_command_b[:,1]=0;term.vel_command_b[:,2]=control.command
            with torch.inference_mode():action=policy(w.get_observations());_,_,done,_=w.step(action)
            forces=sensor.data.net_forces_w_history.torch[0,:,sensor_ids,:].norm(dim=-1).amax(dim=0);contacts=[bool(x) for x in (forces>5).cpu()];contacts_trace.append(contacts);contacts_trace=contacts_trace[-round(2/dt):]
            actual=float(robot.data.root_lin_vel_b.torch[0,0]);gait,evidence=classify_go2_gait_v1(contacts_trace,abs(actual),bool(done[0]));camera.update(robot.data.root_pos_w.torch[0])
            if recording is not None:
                import cv2
                frame=raw.render();recording[2].write(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
            point_speeds=[];foot_point_speeds=[0.0]*4;stage9_tangent=[];stage9_util=[];stage9_moment=[];stage9_points=[];stage9_load=[]
            for foot,point_sensor in enumerate(point_sensors):
                point=point_sensor.data.contact_pos_w.torch[0,0,0,:2]
                if contacts[foot] and bool(torch.isfinite(point).all()) and previous_points[foot] is not None:
                    foot_point_speeds[foot]=float(torch.linalg.vector_norm(point-previous_points[foot])/dt);point_speeds.append(foot_point_speeds[foot])
                if contacts[foot] and bool(torch.isfinite(point).all()):previous_points[foot]=point.clone()
                else:previous_points[foot]=None
                normal_force,positions,normals,_,counts,starts=point_sensor.contact_view.get_contact_data(dt=point_sensor._sim_physics_dt)
                friction_forces,friction_points,friction_counts,friction_starts=point_sensor.contact_view.get_friction_data(dt=point_sensor._sim_physics_dt)
                normal_force=wp.to_torch(normal_force).reshape(-1);positions=wp.to_torch(positions).reshape(-1,3);normals=wp.to_torch(normals).reshape(-1,3);count=int(wp.to_torch(counts).reshape(-1)[0]);start=int(wp.to_torch(starts).reshape(-1)[0]);fcount=int(wp.to_torch(friction_counts).reshape(-1)[0]);fstart=int(wp.to_torch(friction_starts).reshape(-1)[0]);friction_forces=wp.to_torch(friction_forces).reshape(-1,3);friction_points=wp.to_torch(friction_points).reshape(-1,3)
                if count:
                    p3=positions[start:start+count];n3=normals[start:start+count];fn=normal_force[start:start+count];n3=n3/n3.norm(dim=-1,keepdim=True).clamp_min(1e-12);body=body_ids[foot];radius=p3-robot.data.body_pos_w.torch[0,body];surface=robot.data.body_lin_vel_w.torch[0,body]+torch.linalg.cross(robot.data.body_ang_vel_w.torch[0,body].expand_as(radius),radius);tangent=surface-(surface*n3).sum(-1,keepdim=True)*n3;stage9_tangent.append(float((tangent.norm(dim=-1)*fn).sum()/fn.sum().clamp_min(1e-12)));centroid=(p3*fn[:,None]).sum(0)/fn.sum().clamp_min(1e-12);stage9_points.append([float(x) for x in centroid])
                    load=float(fn.sum());stage9_load.append(load);ft=friction_forces[fstart:fstart+fcount].sum(0) if fcount else torch.zeros(3,device=fn.device);stage9_util.append(float(ft.norm()/(.6*fn.sum())) if load>5.0 else 0.0);normal_vector=(fn[:,None]*n3);root=robot.data.root_com_pos_w.torch[0];moment=torch.linalg.cross(p3-root,normal_vector)[:,2].sum()
                    if fcount:moment+=torch.linalg.cross(friction_points[fstart:fstart+fcount]-root,friction_forces[fstart:fstart+fcount])[:,2].sum()
                    stage9_moment.append(float(moment))
                else:stage9_tangent.append(0.0);stage9_util.append(0.0);stage9_moment.append(0.0);stage9_points.append([float("nan")]*3);stage9_load.append(0.0)
            contact_age=[age+1 if load>5.0 else 0 for age,load in zip(contact_age,stage9_load)]
            weighted_score=0.0;weight_sum=0.0
            for slip,load,age in zip(stage9_tangent,stage9_load,contact_age):
                if age<3:continue
                weight=min(load/100.0,1.0);x=max(0.0,(slip-.20)/.30);rho=.5*x*x if x<=1.0 else x-.5;weighted_score+=weight*min(rho,5.0);weight_sum+=weight
            slip_score=weighted_score/max(weight_sum,1e-12);slip_reward=-slip_lambda*slip_score
            contact_point_slip=max(point_speeds or [0.0])
            quaternion=[float(value) for value in robot.data.root_quat_w.torch[0]];roll,pitch,yaw=quat_xyzw_to_roll_pitch_yaw(quaternion);tilt=quat_xyzw_to_gravity_tilt(quaternion);heading_err=heading_error(yaw,heading);yaw_history_abs.append(yaw);heading_history.append(heading_err);heading_history=heading_history[-max(2,round(1.0/dt)):];drift_slope=(heading_history[-1]-heading_history[0])/(dt*max(1,len(heading_history)-1));actual_yaw_rate=float(robot.data.root_ang_vel_w.torch[0,2]);left_slip=max(foot_point_speeds[0],foot_point_speeds[2]);right_slip=max(foot_point_speeds[1],foot_point_speeds[3])
            left_tangent=max(stage9_tangent[0],stage9_tangent[2]);right_tangent=max(stage9_tangent[1],stage9_tangent[3]);net_yaw_moment=sum(stage9_moment)
            showcase_phase="STOP" if target<=.05 else "WALK" if target<1.0 else "RUN"
            lines=["EXP_011 GO2 SINGLE-POLICY SHOWCASE",f"SEQUENCE: STOP > WALK > RUN > WALK > STOP",f"PHASE: {showcase_phase}   TRANSITION: {direction}",f"TARGET SPEED: {target:.2f} m/s   ACTUAL SPEED: {actual:.2f} m/s",f"HEADING ERROR: {control.error:.3f} rad   PHASE GATE: {control.gate:.2f}",f"GAIT: {gait}   FALL: {bool(done[0])}"]
            burn_lines.append(lines)
            overlay.update(f"TARGET SPEED              {target:6.2f} m/s\nACTUAL SPEED              {actual:6.2f} m/s\nHEADING ERROR             {control.error:6.3f} rad\nPHASE GATE                {control.gate:6.3f}\nYAW COMMAND               {control.command:6.3f} rad/s\nFOOT CONTACTS             {contacts}\nTANGENTIAL SLIP PER FOOT  {[round(x,3) for x in stage9_tangent]} m/s\nSLIP REWARD               {slip_reward:8.5f}\nFRICTION UTILIZATION      {[round(x,3) for x in stage9_util]}\nFALL                      {bool(done[0])}")
            if bool(done[0]):break
        w.close()
        burn_overlay(recording,burn_lines)
        if recording is not None:
            destination=recording[0];telemetry={"mode":a.mode,"seed":a.seed,"checkpoint":str(a.checkpoint.resolve()),"checkpoint_sha256":hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),"sequence":list(SHOWCASE_SEQUENCE) if a.mode=="Showcase" else None,"showcase_reset_yaw_range_rad":[0.0,0.0] if a.mode=="Showcase" else None,"duration_s":duration,"frames":len(burn_lines),"tracking_camera":True,"floor_guides":a.show_floor_guides,"floor_guides_physics":False,"heading_controller":a.heading_controller,"policy_switches":0,"falls":sum("FALL: True" in " ".join(lines) for lines in burn_lines)}
            destination.with_suffix(".json").write_text(json.dumps(telemetry,indent=2)+"\n",encoding="utf-8")
if __name__=="__main__":main()
