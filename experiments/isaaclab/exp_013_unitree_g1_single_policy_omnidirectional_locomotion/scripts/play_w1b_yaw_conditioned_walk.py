"""W1BYawConditionedWalk playback; actor action is never externally corrected."""
import argparse,hashlib,math,msvcrt,sys
from pathlib import Path
import gymnasium as gym,torch
EXP=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(EXP/"src"))
import isaaclab_tasks,g1_omnidirectional.tasks
from g1_omnidirectional.policy import FrozenGaitActor
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import add_launcher_args,launch_simulation,resolve_task_config,setup_preset_cli
p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);add_launcher_args(p);a,h=setup_preset_cli(p);sys.argv=[sys.argv[0],*h]
def main():
 cfg,ac=resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0","rsl_rl_cfg_entry_point");cfg.scene.num_envs=1;cfg.episode_length_s=3600
 with launch_simulation(cfg,a):
  w=RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0",cfg=cfg),clip_actions=ac.clip_actions);e=w.unwrapped;actor=FrozenGaitActor(a.checkpoint).to(e.device).eval();robot=e.scene["robot"];sensor=e.scene.sensors["contact_forces"];feet=[i for i,n in enumerate(sensor.body_names) if "ankle_roll" in n];rf=[robot.body_names.index(sensor.body_names[i]) for i in feet];term=e.command_manager.get_term("base_velocity");term.external_override_enabled=True
  angle,speed,yaw,step,slipn,satn=0.,.3,0.,0,0,0;obs,_=w.reset();obs=obs.to(e.device);checksum=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest();print("W1BYawConditionedWalk: A/D direction, W/S speed, Q/E yaw, X stop, ESC quit")
  run=True
  while run:
   while msvcrt.kbhit():
    k=msvcrt.getwch().lower()
    if k=="\x1b":run=False
    elif k=="a":angle=(angle+5)%360
    elif k=="d":angle=(angle-5)%360
    elif k=="w":speed=min(1.2,speed+.05)
    elif k=="s":speed=max(0,speed-.05)
    elif k=="q":yaw=min(.6,yaw+.05)
    elif k=="e":yaw=max(-.6,yaw-.05)
    elif k=="x":speed=yaw=0
   r=math.radians(angle);target=torch.tensor([speed*math.cos(r),speed*math.sin(r),yaw],device=e.device);term.external_override[0]=target
   with torch.inference_mode():act=actor(obs["policy"],torch.zeros(1,device=e.device))
   obs,_,done,_=w.step(act);obs=obs.to(e.device);actual=robot.data.root_lin_vel_b[0,:2];ay=float(robot.data.root_ang_vel_b[0,2]);aspeed=float(torch.linalg.vector_norm(actual));adir=math.degrees(math.atan2(float(actual[1]),float(actual[0])))%360;derr=abs((adir-angle+180)%360-180) if speed>.05 else float("nan");mae=float(torch.linalg.vector_norm(actual-target[:2]));ymae=abs(ay-yaw);forces=sensor.data.net_forces_w_history[0,-1,feet,:].norm(dim=-1);contacts=forces>5;fs=torch.linalg.vector_norm(robot.data.body_lin_vel_w[0,rf,:2],dim=-1);slip=bool(((fs>.55)&contacts).any());slipn=slipn+1 if slip else 0;lim=robot.data.joint_vel_limits[0];lim=lim[:,1].abs() if lim.ndim==2 else lim;sat=bool((robot.data.joint_vel[0].abs()/lim.clamp_min(1e-6)>.95).any());satn=satn+1 if sat else 0;g=robot.data.projected_gravity_b[0];roll=float(torch.atan2(g[1].abs(),g[2].abs().clamp_min(1e-6)));pitch=float(torch.atan2(g[0].abs(),g[2].abs().clamp_min(1e-6)));gait="FALL" if bool(done[0]) else ("TURN_IN_PLACE" if speed<.05 and abs(yaw)>.05 else ("TURNING_WALK" if abs(yaw)>.05 else "TRANSLATION_ONLY"))
   if step%10==0:print(f"TARGET VX/VY {target[0]:+.2f}/{target[1]:+.2f} ACTUAL {actual[0]:+.2f}/{actual[1]:+.2f} | TARGET/ACTUAL DIR {angle:.1f}/{adir:.1f} | TARGET/ACTUAL YAW {yaw:+.2f}/{ay:+.2f} | VECTOR MAE {mae:.3f} DIR ERROR {derr:.1f} YAW MAE {ymae:.3f} | GAIT {gait} | LEFT/RIGHT CONTACT {bool(contacts[0])}/{bool(contacts[1])} FLIGHT {not bool(contacts.any())} | ROLL/PITCH {roll:.3f}/{pitch:.3f} SLIP {slipn>=5} IMPACT {float(forces.max())>3500} SAT {satn>=5} FALL {bool(done[0])} | SHA {checksum[:16]}")
   step+=1
  w.close()
if __name__=="__main__":main()
