"""W1A2DirectionSpeedEnvelope interactive playback; actor actions only."""
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
  w=RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0",cfg=cfg),clip_actions=ac.clip_actions);e=w.unwrapped;actor=FrozenGaitActor(a.checkpoint).to(e.device).eval();robot=e.scene["robot"];sensor=e.scene.sensors["contact_forces"];feet=[i for i,n in enumerate(sensor.body_names) if "ankle_roll" in n];rfeet=[robot.body_names.index(sensor.body_names[i]) for i in feet];term=e.command_manager.get_term("base_velocity");term.external_override_enabled=True
  angle,speed,step,slipn,satn=0.,.3,0,0,0;obs,_=w.reset();obs=obs.to(e.device);checksum=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest();print("W1A2DirectionSpeedEnvelope: A/D direction, W/S speed, X stop, ESC quit")
  run=True
  while run:
   while msvcrt.kbhit():
    k=msvcrt.getwch().lower()
    if k=="\x1b":run=False
    elif k=="a":angle=(angle+5)%360
    elif k=="d":angle=(angle-5)%360
    elif k=="w":speed=min(1.2,speed+.05)
    elif k=="s":speed=max(0.,speed-.05)
    elif k=="x":speed=0
   r=math.radians(angle);cmd=torch.tensor([speed*math.cos(r),speed*math.sin(r),0.],device=e.device);term.external_override[0]=cmd
   with torch.inference_mode():act=actor(obs["policy"],torch.zeros(1,device=e.device))
   obs,_,done,_=w.step(act);obs=obs.to(e.device);actual=robot.data.root_lin_vel_b[0,:2];aspeed=float(torch.linalg.vector_norm(actual));adir=math.degrees(math.atan2(float(actual[1]),float(actual[0])))%360;derr=abs((adir-angle+180)%360-180) if speed>.05 else float("nan");mae=float(torch.linalg.vector_norm(actual-cmd[:2]));forces=sensor.data.net_forces_w_history[0,-1,feet,:].norm(dim=-1);contacts=forces>5;fs=torch.linalg.vector_norm(robot.data.body_lin_vel_w[0,rfeet,:2],dim=-1);slip=bool(((fs>.55)&contacts).any());slipn=slipn+1 if slip else 0;limits=robot.data.joint_vel_limits[0];limits=limits[:,1].abs() if limits.ndim==2 else limits;sat=bool((robot.data.joint_vel[0].abs()/limits.clamp_min(1e-6)>.95).any());satn=satn+1 if sat else 0;g=robot.data.projected_gravity_b[0];roll=float(torch.atan2(g[1].abs(),g[2].abs().clamp_min(1e-6)));pitch=float(torch.atan2(g[0].abs(),g[2].abs().clamp_min(1e-6)));gait="FALL" if bool(done[0]) else ("WALK_LIKE" if contacts.any() else "ISOLATED_FLIGHT")
   if step%10==0:print(f"TARGET VX {cmd[0]:+.2f} | TARGET VY {cmd[1]:+.2f} | ACTUAL VX {actual[0]:+.2f} | ACTUAL VY {actual[1]:+.2f} | TARGET SPEED {speed:.2f} | ACTUAL SPEED {aspeed:.2f} | TARGET DIRECTION {angle:.1f} | ACTUAL DIRECTION {adir:.1f} | VECTOR MAE {mae:.3f} | DIRECTION ERROR {derr:.1f} | DETECTED GAIT {gait} | ROLL {roll:.3f} | PITCH {pitch:.3f} | SLIP {slipn>=5} | IMPACT {float(forces.max())>3500} | SATURATION {satn>=5} | FALL {bool(done[0])} | CHECKPOINT SHA {checksum[:16]}")
   step+=1
  w.close()
if __name__=="__main__":main()
