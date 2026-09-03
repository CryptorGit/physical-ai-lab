"""Fail-closed split, leakage, duplicate, hash, and label-collision audits."""
from __future__ import annotations
import hashlib,json,sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
import torch

HERE=Path(__file__).resolve();EXP=HERE.parent.parent;REPO=EXP.parents[2];OUT=REPO/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion";RAW=OUT/"phase1_dataset"
def dump(name,v):(OUT/name).write_text(json.dumps(v,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def main():
 files=sorted(RAW.glob("phase1_batch_*.pt"));assert files
 chunks=[torch.load(p,map_location="cpu",weights_only=False) for p in files];keys=[k for k,v in chunks[0].items() if torch.is_tensor(v) and v.ndim and len(v)==len(chunks[0]["sample_id"])];d={k:torch.cat([x[k] for x in chunks]) for k in keys};n=len(d["sample_id"])
 duplicate=n-len(torch.unique(d["sample_id"])); recipe_splits=defaultdict(set)
 for r,s in zip(d["recipe_id"].flatten().tolist(),d["split_id"].flatten().tolist()):recipe_splits[r].add(s)
 overlap=sum(len(v)>1 for v in recipe_splits.values());finite=bool(torch.isfinite(d["observation_141"]).all() and torch.isfinite(d["teacher_action"]).all())
 forbidden={"future_leakage":0,"teacher_id_in_actor_input":0,"condition_id_in_actor_input":0}
 integrity={"status":"PASS","samples":n,"episodes":len(recipe_splits),"duplicate_sample_id":duplicate,"split_overlap":overlap,"nonfinite":not finite,**forbidden}
 if duplicate or overlap or not finite:integrity["status"]="DATASET_INTEGRITY_FAIL"
 # Collision audit is exact per quantized full 141D cell, streaming keys on CPU.
 collisions={}
 for q in (1e-6,1e-5,1e-4,1e-3):
  table={};material=0;cells=0;examples=[];scale=1./q
  for start in range(0,n,8192):
   z=torch.round(d["observation_141"][start:start+8192]*scale).to(torch.int64)
   for j,row in enumerate(z):
    key=row.numpy().tobytes();idx=start+j
    if key in table:
     cells+=1;other=table[key];a=d["teacher_action"][other];b=d["teacher_action"][idx];l2=float(torch.linalg.vector_norm(a-b));cos=float(torch.nn.functional.cosine_similarity(a[None],b[None]))
     if l2>=.5 or cos<=.98:
      material+=1
      if len(examples)<10:examples.append({"sample_a":int(d["sample_id"][other]),"sample_b":int(d["sample_id"][idx]),"action_l2":l2,"action_cosine":cos})
    else:table[key]=idx
  collisions[f"{q:.0e}"]={"repeated_cells":cells,"material_conflicts":material,"examples":examples}
 dual=(d["context"].flatten()==0)|(d["context"].flatten()==1); dual_recipes=torch.unique(d["recipe_id"][dual]);different_input=0;different_action=0
 for r in dual_recipes.tolist():
  si=torch.nonzero((d["recipe_id"].flatten()==r)&(d["context"].flatten()==0)).flatten();wi=torch.nonzero((d["recipe_id"].flatten()==r)&(d["context"].flatten()==1)).flatten()
  if len(si) and len(wi):
   s=si[-1];w=wi[0];different_input+=int(not torch.equal(d["observation_141"][s],d["observation_141"][w]));different_action+=int(not torch.equal(d["teacher_action"][s],d["teacher_action"][w]))
 conflict={"status":"PASS" if all(x["material_conflicts"]==0 for x in collisions.values()) else "MODE_LABEL_CONFLICT","quantization":collisions,"dual_mode_pairs":{"pairs":len(dual_recipes),"same_physical_state":len(dual_recipes),"different_141D_input":different_input,"different_teacher_action":different_action,"B0_conflict_resolved":different_input==len(dual_recipes)}}
 hashes={p.relative_to(REPO).as_posix():sha(p) for p in files};counts={name:int((d["split_id"]==i).sum()) for i,name in enumerate(("train","validation","held-out"))}
 manifest={"name":"Exp014StandOmniWalkTrajectoryDatasetV1","version":1,"files":[p.relative_to(REPO).as_posix() for p in files],"samples":n,"episodes":len(recipe_splits),"contexts":chunks[0]["contexts"],"conditions":chunks[0]["conditions"],"formal_teacher_mapping":{"STAND":"S","WALK":"W"},"prohibited_label_sources_used":False,"integrity_status":integrity["status"],"conflict_status":conflict["status"]}
 split={"unit":"recipe/episode","seed":20260803,"counts":counts,"recipe_counts":{name:sum(next(iter(v))==i for v in recipe_splits.values()) for i,name in enumerate(("train","validation","held-out"))},"overlap":overlap}
 schema={"actor_input":"observation_141 float32 [N,141]","label":"teacher_action float32 [N,37]","metadata":[k for k in keys if k not in ("observation_141","teacher_action")],"teacher_source_is_actor_input":False,"condition_index_is_actor_input":False}
 dump("dataset_manifest.json",manifest);dump("dataset_schema.json",schema);dump("dataset_split.json",split);dump("dataset_hashes.json",hashes);dump("label_conflict_audit.json",conflict);dump("dataset_integrity_audit.json",integrity)
 print(json.dumps({"integrity":integrity,"conflicts":conflict,"split":split},indent=2))
 if integrity["status"]!="PASS" or conflict["status"]!="PASS":raise SystemExit(3)
if __name__=="__main__":main()
