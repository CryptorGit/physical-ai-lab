"""Evaluate missing A8 validation conditions three at a time in one ReplayV2 batch."""
from __future__ import annotations
import argparse,json,shutil,subprocess
from pathlib import Path
import audit_w2_p1_a8_validation as audit

p=argparse.ArgumentParser();p.add_argument("--updates",required=True);p.add_argument("--shard",type=int,required=True);p.add_argument("--shards",type=int,required=True);a=p.parse_args()
evaluator=audit.HERE.parent/"evaluate_w2_p1_a8_multi_condition.py"
tasks=[]
for update in [int(x) for x in a.updates.split(",")]:
 missing=[c for c in range(24) if not (audit.RAW/f"update_{update:03d}_condition_{c:02d}.json").exists()]
 for i in range(0,len(missing),3):tasks.append((update,missing[i:i+3]))
for task_index,(update,conditions) in enumerate(tasks):
 if task_index%a.shards!=a.shard:continue
 temp=audit.RAW/f"multi_u{update:03d}_{conditions[0]:02d}_{conditions[-1]:02d}";temp.mkdir(parents=True,exist_ok=True)
 command=[str(audit.ISAAC),"-p",str(evaluator),"--policy",str(audit.checkpoint(update)),"--batch","4","--split","validation","--conditions",','.join(map(str,conditions)),"--episodes","300","--output-dir",str(temp),"--headless","--device","cuda:0"]
 with temp.with_suffix(".log").open("w",encoding="utf-8") as log:subprocess.run(command,cwd=audit.REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
 for c in conditions:
  for suffix in (".csv",".json"):
   source=temp/f"condition_{c:02d}{suffix}";destination=audit.RAW/f"update_{update:03d}_condition_{c:02d}{suffix}";shutil.copy2(source,destination)
 print(json.dumps({"update":update,"conditions":conditions,"status":"COMPLETE"}),flush=True)
