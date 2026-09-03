"""D19 offline support-objective implementation and symmetry audit.

No simulator, actor inference, optimizer, or checkpoint is used. The registered
synthetic-test fail-closed rule prevents policy probes when the D18 reward
implementation disagrees with its frozen contract.
"""
from __future__ import annotations
import hashlib, json, math, re
from pathlib import Path

HERE=Path(__file__).resolve(); REPO=HERE.parents[4]
OUT=REPO/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d19_support_objective_symmetry_audit"; RAW=OUT/"raw"
D18SRC=HERE.parent/"run_phase2_d18_precursor.py"
MIRROR=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1b_d1_yaw_translation_interference_diagnosis/robot_mirror_contract.json"
D15=REPO/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d15_stand_to_omniwalk_start_audit"
A5=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w2_p1_a5_versioned_four_step_start_trajectory_overlay_preflight/four_step_runtime_positive_control.json"

def dump(path,x): path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def mj(x): return x*x*x*(x*(x*6-15)+10)
def implemented_schedule(t,peak=.7):
 target=peak*mj(min(1,max(0,t/.35))); target=target if t<=.50 else 0.
 weight=1. if t<=.50 else (1-mj(min(1,max(0,(t-.50)/.25))) if t<.75 else 0.)
 return target,weight
def contract_schedule(t,peak=.7):
 if t>=.75:return None,0.
 target=peak*mj(min(1,max(0,t/.35)))
 weight=1. if t<=.50 else 1-mj((t-.50)/.25)
 return target,weight
def rewards(fl,fr,target,sigma=.3,sigma_support=1.):
 total=fl+fr; signed=(fl-fr)/(total+1e-6); unsigned=abs(signed)
 load=math.exp(-((unsigned-target)/sigma)**2); support=math.exp(-(((total)-1)/sigma_support)**2)
 left=math.exp(-((signed-target)/sigma)**2); right=math.exp(-((signed+target)/sigma)**2)
 return {"signed_balance":signed,"unsigned_balance":unsigned,"unsigned_reward":load,"signed_left_reward":left,"signed_right_reward":right,"total_support_reward":support,"combined_load_plus_support":load+support}

def locations(text):
 needles={"contact_force":"force=world.sensor.data.net_forces_w_history","F_L_F_R":"fl,fr=fz[:,0],fz[:,1]","load_imbalance":"imbalance=(fl-fr).abs()","low_load_ratio":"low=torch.minimum(fl,fr)","total_support_ratio":"support_ratio=total/","support_foot":"support=(fz.argmax(1))","support_slip":"support_slip=slip.gather","target_schedule":"target=self.target_peak*minimum_jerk","target_zero_after_0p50":"target=torch.where(t<=.50,target,torch.zeros_like(t))","reward_envelope":"support_env=torch.where(t<=.50","physics_action":"self.wrapped.step(action)","privileged_after_action":"p=privileged(self","reward_computation":"support=support_env*"}
 lines=text.splitlines(); out={}
 for key,needle in needles.items():
  idx=next((i+1 for i,x in enumerate(lines) if needle in x),None); out[key]={"line":idx,"text":None if idx is None else lines[idx-1].strip()}
 return out

def main():
 text=D18SRC.read_text(encoding="utf-8"); loc=locations(text)
 force_cases=[("50_50",.5,.5),("85_15",.85,.15),("15_85",.15,.85),("100_0",1.,0.),("0_100",0.,1.),("0_0",0.,0.)]
 rows=[]
 for name,fl,fr in force_cases: rows.append({"case":name,"F_L":fl,"F_R":fr,**rewards(fl,fr,.7)})
 tests=[]
 def test(name,passed,actual,expected): tests.append({"name":name,"pass":bool(passed),"actual":actual,"expected":expected})
 a={r["case"]:r for r in rows}
 test("unsigned_85_15_equals_15_85",abs(a["85_15"]["unsigned_reward"]-a["15_85"]["unsigned_reward"])<1e-12,[a["85_15"]["unsigned_reward"],a["15_85"]["unsigned_reward"]],"equal")
 test("signed_left_selects_85_15",a["85_15"]["signed_left_reward"]>a["15_85"]["signed_left_reward"],[a["85_15"]["signed_left_reward"],a["15_85"]["signed_left_reward"]],"first greater")
 test("signed_right_selects_15_85",a["15_85"]["signed_right_reward"]>a["85_15"]["signed_right_reward"],[a["15_85"]["signed_right_reward"],a["85_15"]["signed_right_reward"]],"first greater")
 # A load reward must not score flight as a maximum merely because signed balance is zero.
 zero_at_zero_target=rewards(0,0,0)["unsigned_reward"]
 test("zero_total_support_not_high_load_reward",zero_at_zero_target<.5,zero_at_zero_target,"<0.5 or explicit support-valid mask")
 times=[0,.10,.20,.35,.50,.60,.75]; schedule=[]
 for t in times:
  impl=implemented_schedule(t); expected=contract_schedule(t); ok=(expected[0] is None and impl[1]==0) or (abs(impl[0]-expected[0])<1e-12 and abs(impl[1]-expected[1])<1e-12)
  schedule.append({"time_s":t,"implemented_target":impl[0],"implemented_weight":impl[1],"contract_target":expected[0],"contract_weight":expected[1],"pass":ok})
  test(f"schedule_t_{t:.2f}",ok,impl,expected)
 unit_pass=all(x["pass"] for x in tests)
 mirror=json.loads(MIRROR.read_text(encoding="utf-8")); perm=mirror["mirror_indices"]; signs=mirror["mirror_signs"]
 involution=all(perm[perm[i]]==i for i in range(len(perm))); sign_involution=all(signs[i]*signs[perm[i]]==1 for i in range(len(perm)))
 # Reward mirror algebra for 32 deterministic synthetic force pairs.
 pairs=[]
 for i in range(32):
  fl=.1+.8*(i/31); fr=1-fl; original=rewards(fl,fr,.7); mirrored=rewards(fr,fl,.7)
  pairs.append({"pair":i,"unsigned_error":abs(original["unsigned_reward"]-mirrored["unsigned_reward"]),"signed_error":abs(original["signed_left_reward"]-mirrored["signed_right_reward"])})
 mirror_pass=involution and sign_involution and max(x["unsigned_error"] for x in pairs)<1e-12 and max(x["signed_error"] for x in pairs)<1e-12
 # Positive artifacts contain aggregate gates only, not force/momentum trajectories.
 d15_json=json.loads((D15/"formal_start_matrix.json").read_text(encoding="utf-8")); a5=json.loads(A5.read_text(encoding="utf-8"))
 positive={"D15":{"artifact":str((D15/"formal_start_matrix.json").relative_to(REPO)).replace("\\","/"),"sha256":sha(D15/"formal_start_matrix.json"),"walk_acquisition_count":d15_json["aggregate"]["walk_acquisition_count"],"raw_trajectory_available":False},"A5":{"artifact":str(A5.relative_to(REPO)).replace("\\","/"),"sha256":sha(A5),"profiles":len(a5.get("profiles",[])),"raw_support_dynamics_available":False},"conclusion":"SUPPORT_POSITIVE_REFERENCE_INSUFFICIENT"}
 result={"source_locations":loc,"timing":{"action_then_physics_line":loc["physics_action"]["line"],"sensor_and_reward_after_physics":True,"time_index":"pre-increment self.age * 0.02","mismatch":"target is reset to zero immediately after 0.50 s while its registered weight decays until 0.75 s"},"force_fixture":rows,"schedule_fixture":schedule,"unit_tests":tests,"unit_test_status":"PASS" if unit_pass else "FAIL","failed_tests":[x["name"] for x in tests if not x["pass"]],"mirror_contract":mirror,"mirror_pairs":pairs,"mirror_status":"PASS" if mirror_pass else "FAIL","positive":positive,"policy_probe_status":"NOT_EXECUTED" if not unit_pass else "AUTHORIZED","policy_probe_reason":None if unit_pass else "registered fail-closed rule: synthetic reward unit test failed before temporary policy probes","persistent_updates":0,"new_checkpoints":0}
 dump(RAW/"audit.json",result); print(json.dumps({"unit_tests":result["unit_test_status"],"failed":result["failed_tests"],"mirror":result["mirror_status"],"policy_probes":result["policy_probe_status"]},indent=2))
if __name__=="__main__":main()
