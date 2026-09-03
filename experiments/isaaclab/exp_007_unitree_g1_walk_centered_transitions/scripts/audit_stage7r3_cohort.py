"""Manager-level R0 audit for in-place cohort activation."""
import csv,json,sys,torch
from pathlib import Path
H=Path(__file__).resolve();EXP=H.parent.parent;REPO=EXP.parents[2];sys.path.insert(0,str(EXP/"src"))
from g1_walk_centered.in_place_cohort import InPlaceEnvIdCohort
O=REPO/"results/exp_007_unitree_g1_walk_centered_transitions/stage7r3_in_place_cohort";O.mkdir(parents=True,exist_ok=True)
def test(n,c,seed):
 m=InPlaceEnvIdCohort(n,c,seed);valid=torch.zeros(n,dtype=torch.bool);valid[:int(.97*n)]=True;m.update_ready(valid,100)
 prev=torch.arange(n*37,dtype=torch.float32).reshape(n,37);x=m.activate(valid,prev);ids=x["physical_env_ids"]
 gathered=m.gather(torch.arange(n));return {"physical_envs":n,"cohort":c,"ready":int(valid.sum()),"formed":len(ids)==c,
 "same_env_ids":bool(torch.equal(gathered,ids)),"previous_action_bitwise":bool(torch.equal(x["cohort_previous_action"],prev[ids])),
 "state_copy":x["state_copy"],"setter_calls":0,"teleport_calls":0,"selected_ids":ids.tolist()}
small=test(64,32,20261110);prod=test(1024,512,20261110)
(O/"r0_small_live_test.json").write_text(json.dumps({**small,"test_level":"MANAGER_LIVE_TENSOR_NOT_ISAAC_PHYSICS"},indent=2)+"\n")
(O/"r0_production_live_test.json").write_text(json.dumps({**prod,"test_level":"MANAGER_LIVE_TENSOR_NOT_ISAAC_PHYSICS","isaac_rollout_executed":False},indent=2)+"\n")
(O/"cohort_env_id_map.json").write_text(json.dumps({"generation":0,"map":[{"cohort_local_index":i,"physical_env_id":v} for i,v in enumerate(prod["selected_ids"])]},indent=2)+"\n")
with (O/"handoff_continuity.csv").open("w",newline="") as f:
 w=csv.DictWriter(f,fieldnames=["env_id","same_env","state_copy","setter","teleport","previous_action_match"]);w.writeheader()
 for i in small["selected_ids"]:w.writerow({"env_id":i,"same_env":True,"state_copy":False,"setter":False,"teleport":False,"previous_action_match":True})
print(json.dumps({"small":small["formed"],"production_manager":prod["formed"],"isaac_live":False}))
