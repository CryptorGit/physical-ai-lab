"""Evaluate 3x3 command neighborhoods at every frozen oracle checkpoint boundary."""
from __future__ import annotations
import argparse,json,subprocess
from pathlib import Path
import audit_w2_p1_a8_validation as audit
p=argparse.ArgumentParser();p.add_argument("--shard",type=int,required=True);p.add_argument("--shards",type=int,required=True);a=p.parse_args()
mapping=json.loads((audit.OUT/"offline_start_teacher_condition_map_v1.json").read_text())["condition_map"];lookup={(int(x["physical_command"]["direction_deg"]),float(x["physical_command"]["yaw_radps"])):x["selected_checkpoint_update"] for x in mapping};boundary=set()
for d in range(0,360,45):
 for y in (-.3,0.,.3):
  if lookup[d,y]!=lookup[(d+45)%360,y]:boundary|={(d,y),((d+45)%360,y)}
for d in range(0,360,45):
 for y1,y2 in ((-.3,0.),(0.,.3)):
  if lookup[d,y1]!=lookup[d,y2]:boundary|={(d,y1),(d,y2)}
tasks=sorted(boundary);raw=audit.OUT/"raw/local_neighborhood";raw.mkdir(parents=True,exist_ok=True);evaluator=audit.HERE.parent/"evaluate_w2_p1_a8_multi_condition.py"
for ti,(direction,yaw) in enumerate(tasks):
 if ti%a.shards!=a.shard:continue
 update=lookup[direction,yaw];name=f"d{direction:03d}_y{yaw:+.1f}".replace('+','p').replace('-','m').replace('.','p');temp=raw/name;temp.mkdir(parents=True,exist_ok=True);specs=[]
 for di,dd in enumerate((-5,0,5)):
  for yi,dy in enumerate((-.03,0.,.03)):specs.append({"id":f"d{di}_y{yi}","direction":float((direction+dd)%360),"speed":.3,"yaw":float(yaw+dy),"direction_delta":dd,"yaw_delta":dy})
 spec_path=temp/"specs.json";spec_path.write_text(json.dumps(specs,indent=2)+"\n")
 command=[str(audit.ISAAC),"-p",str(evaluator),"--policy",str(audit.checkpoint(update)),"--batch","4","--split","validation","--condition-specs-json",str(spec_path),"--episodes","100","--output-dir",str(temp),"--headless","--device","cuda:0"]
 with temp.with_suffix(".log").open("w",encoding="utf-8") as log:subprocess.run(command,cwd=audit.REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
 print(json.dumps({"formal_condition":[direction,yaw],"update":update,"status":"COMPLETE"}),flush=True)
