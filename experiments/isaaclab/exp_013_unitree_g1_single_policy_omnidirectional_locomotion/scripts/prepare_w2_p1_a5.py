"""A5 identity and A4-candidate deterministic reproduction preflight."""
from __future__ import annotations
import hashlib,json,subprocess,sys
from pathlib import Path
import torch
import w2_p1_a5_common as c
OUT=c.A5;REPO=c.REPO;RES=c.BASE/"phase_w2_p1_r1_d2_dataset_provenance_reconciliation";A4=c.A4
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def dump(n,x):OUT.mkdir(parents=True,exist_ok=True);(OUT/n).write_text(json.dumps(x,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def main():
 OUT.mkdir(parents=True,exist_ok=True);device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu");m1,r1,datasets,splits,lookup=c.reproduce_a4(device);m2,r2,_,_,_=c.reproduce_a4(device);stored=json.loads((A4/"selected_v2_probe_candidate.json").read_text())["selected"]
 # Re-evaluate exact stored validation metrics with A4's evaluator.
 import probe_w2_p1_a4_b0_contract as a4
 ev=a4.evaluate(m1,datasets,splits,"validation",device,lookup);diff=max(abs(ev[g][k]-stored["metrics"][g][k]) for g in ev for k in ("mse","cosine"));parity=r1==r2 and diff<=1e-12
 resolved=json.loads((RES/"w2_p1_dataset_hashes_resolved_v2.json").read_text());actual={p:sha(REPO/p) for p in resolved["hashes"]};v2sha=sha(A4/"start_boundary_b0_label_overlay_v2.pt")
 dump("base_dataset_identity_audit.json",{"resolved_manifest":str((RES/"w2_p1_dataset_hashes_resolved_v2.json").relative_to(REPO)).replace("\\","/"),"all_base_hashes_match":actual==resolved["hashes"],"base_hashes":actual,"v2_overlay_sha256":v2sha,"v2_overlay_matches_A4_manifest":v2sha==json.loads((A4/"w2_p1_dataset_overlay_manifest_v2.json").read_text())["overlay_sha256"],"base_changes":0,"label_changes":0,"split_changes":0})
 dump("v2_candidate_reproduction.json",{"initialization":"W2-P1-R2 step37000","contract_source":str((A4/"resolved_v2_probe_training_config.yaml").relative_to(REPO)).replace("\\","/"),"selected_step":500,"same_process_runs":[r1,r2],"tensor_hash":r1["tensor_hash"],"trace_hash":r1["trace_hash"],"stored_A4_tensor_hash_available":False,"stored_metric_max_abs_difference":diff,"same_process_exact":r1==r2,"reproduction_pass":parity,"persistent_checkpoint_written":False,"note":"A4 did not serialize the in-memory tensor hash; exact independent reconstruction plus stored metric parity is the strongest available fingerprint."})
 dump("stage_reference.json",{"stage":"W2-P1-A5","starting_head":"d2315c079df36e1e782910049726cf947efec3c8","integration_base":"W2-P1-R2 step37000","A4_reproduced_tensor_hash":r1["tensor_hash"],"persistent_checkpoint":0,"remote_push":False})
 dump("protocol.json",{"positive_control_first":True,"proposed":"B0 stop + B1-B4 W1B + A4 candidate","visited_collection":"candidate runtime only; W1B label queries only","overlay":"StartBoundaryTrajectoryOverlayV3","formal_DAgger":False,"persistent_policy":False})
 print(json.dumps({"reproduction_pass":parity,"tensor_hash":r1["tensor_hash"],"metric_diff":diff}))
if __name__=="__main__":main()
