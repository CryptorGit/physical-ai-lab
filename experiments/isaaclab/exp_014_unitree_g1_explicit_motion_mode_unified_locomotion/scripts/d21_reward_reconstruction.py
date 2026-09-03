"""Pure NumPy reconstruction for the D21 Reward V2R1 capture."""
from __future__ import annotations
import numpy as np

def minimum_jerk(u):
 u=np.clip(np.asarray(u,dtype=np.float32),0,1);return 10*u**3-15*u**4+6*u**5

def reconstruct(a,scales,weights):
 t=np.asarray(a["time_since_start"],dtype=np.float32);target=np.where(t<.35,.7*minimum_jerk(t/.35),np.where(t<.75,.7,0)).astype(np.float32);env=np.where(t<.50,1,np.where(t<.75,1-minimum_jerk((t-.50)/.25),0)).astype(np.float32)
 early=np.where(t<=.50,1,np.where(t<.75,1-minimum_jerk((t-.50)/.25),0)).astype(np.float32);velw=np.where(t<=.20,.15,np.where(t<.50,.15+.45*minimum_jerk((t-.20)/.30),np.where(t<.75,.60+.40*minimum_jerk((t-.50)/.25),1))).astype(np.float32);yaww=np.where(t<=.20,.25,np.where(t<.50,.25+.50*minimum_jerk((t-.20)/.30),1)).astype(np.float32);unload=np.where((t>=.20)&(t<=.60),1,0).astype(np.float32)
 valid=np.asarray(a["support_valid"],dtype=np.float32);load=valid*np.exp(-((a["unsigned_load_balance"]-target)/scales["sigma_load"])**2);total=np.exp(-((a["support_ratio"]-1)/scales["sigma_support"])**2);unloadr=np.exp(-((a["low_load_ratio"]-scales["unload_target"])/scales["sigma_unload"])**2)
 preventive=early*(np.exp(-(a["Lz"]/scales["sigma_Lz"])**2)+np.exp(-(a["dLz_dt"]/scales["sigma_dLz"])**2)+np.exp(-(a["contact_yaw_moment"]/scales["sigma_Mz"])**2));velocity=6*velw*np.exp(-(a["velocity_error"]**2)/.25);yaw=8*yaww*np.exp(-(a["yaw_error"]**2)/.25)
 upright=2*np.exp(-((1-a["upright_scalar"])**2)/.1);termination=-200*np.asarray(a["fall"],dtype=np.float32);safetyrest=-.2*a["pelvis_vertical_velocity"]**2-a["dangerous_slip"].astype(np.float32)-a["impact"].astype(np.float32)-a["velocity_saturation"].astype(np.float32)-a["torque_saturation"].astype(np.float32);regular=-2e-6*a["torque_sq"]-1e-7*a["joint_acc_sq"]-.005*a["action_rate_sq"]-.02*a["residual_mag_sq"]
 terms={"load_reward":env*load,"total_support_reward":env*total,"support_slip_reward":env*(-.2*a["support_foot_slip"]),"swing_unload_reward":unload*unloadr,"preventive_yaw_reward":preventive,"velocity_reward":velocity,"yaw_reward":yaw,"upright_reward":upright,"termination_reward":termination,"safety_rest_reward":safetyrest,"regularization_reward":regular}
 terms["online_reward"]=weights["preventive"]*preventive+weights["support"]*(terms["load_reward"]+terms["total_support_reward"]+terms["support_slip_reward"]+terms["swing_unload_reward"])+weights["tracking"]*(velocity+yaw)+weights["safety"]*(upright+termination+safetyrest+regular)
 terms["target_load"]=target;terms["support_envelope"]=env;return {k:np.asarray(v,dtype=np.float32) for k,v in terms.items()}
