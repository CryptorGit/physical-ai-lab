"""Run mapped-teacher-to-A4 safe-horizon sweep, three conditions per ReplayV2 batch."""
from __future__ import annotations
import argparse,json,subprocess
import audit_w2_p1_a8_validation as audit
p=argparse.ArgumentParser();p.add_argument("--shard",type=int,required=True);p.add_argument("--shards",type=int,required=True);a=p.parse_args()
mapping=json.loads((audit.OUT/"offline_start_teacher_condition_map_v1.json").read_text())["condition_map"];horizons=(2,4,6,8,12,16,24,32);tasks=[]
for horizon in horizons:
 for update in sorted({x["selected_checkpoint_update"] for x in mapping}):
  cs=[int(x["physical_command"]["direction_deg"]//45)*3+(-.3,0.,.3).index(x["physical_command"]["yaw_radps"]) for x in mapping if x["selected_checkpoint_update"]==update]
  for i in range(0,len(cs),3):tasks.append((horizon,update,cs[i:i+3]))
raw=audit.OUT/"raw/safe_horizon";raw.mkdir(parents=True,exist_ok=True);evaluator=audit.HERE.parent/"evaluate_w2_p1_a8_multi_condition.py"
for ti,(horizon,update,conditions) in enumerate(tasks):
 if ti%a.shards!=a.shard:continue
 temp=raw/f"h{horizon:02d}_u{update:03d}_{conditions[0]:02d}_{conditions[-1]:02d}";temp.mkdir(parents=True,exist_ok=True)
 command=[str(audit.ISAAC),"-p",str(evaluator),"--policy",str(audit.checkpoint(update)),"--batch","5","--split","heldout","--conditions",','.join(map(str,conditions)),"--episodes","200","--output-dir",str(temp),"--candidate-takeover-horizon",str(horizon),"--headless","--device","cuda:0"]
 with temp.with_suffix(".log").open("w",encoding="utf-8") as log:subprocess.run(command,cwd=audit.REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
 print(json.dumps({"horizon":horizon,"update":update,"conditions":conditions,"status":"COMPLETE"}),flush=True)
