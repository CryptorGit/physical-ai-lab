"""Evaluate the frozen A8 condition map on held-out recipes in batched ReplayV2 runs."""
from __future__ import annotations
import argparse,json,subprocess
from pathlib import Path
import audit_w2_p1_a8_validation as audit

p=argparse.ArgumentParser();p.add_argument("--shard",type=int,required=True);p.add_argument("--shards",type=int,required=True);a=p.parse_args()
mapping=json.loads((audit.OUT/"offline_start_teacher_condition_map_v1.json").read_text())["condition_map"]
tasks=[]
for update in sorted({x["selected_checkpoint_update"] for x in mapping}):
 conditions=[int(x["physical_command"]["direction_deg"]//45)*3+(-.3,0.,.3).index(x["physical_command"]["yaw_radps"]) for x in mapping if x["selected_checkpoint_update"]==update]
 for i in range(0,len(conditions),3):tasks.append((update,conditions[i:i+3]))
raw=audit.OUT/"raw/heldout_oracle";raw.mkdir(parents=True,exist_ok=True);evaluator=audit.HERE.parent/"evaluate_w2_p1_a8_multi_condition.py"
for ti,(update,conditions) in enumerate(tasks):
 if ti%a.shards!=a.shard:continue
 temp=raw/f"multi_u{update:03d}_{conditions[0]:02d}_{conditions[-1]:02d}";temp.mkdir(parents=True,exist_ok=True)
 command=[str(audit.ISAAC),"-p",str(evaluator),"--policy",str(audit.checkpoint(update)),"--batch","5","--split","heldout","--conditions",','.join(map(str,conditions)),"--episodes","300","--output-dir",str(temp),"--headless","--device","cuda:0"]
 with temp.with_suffix(".log").open("w",encoding="utf-8") as log:subprocess.run(command,cwd=audit.REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
 for c in conditions:
  for suffix in (".csv",".json"):(raw/f"condition_{c:02d}{suffix}").write_bytes((temp/f"condition_{c:02d}{suffix}").read_bytes())
 print(json.dumps({"update":update,"conditions":conditions,"status":"COMPLETE"}),flush=True)
