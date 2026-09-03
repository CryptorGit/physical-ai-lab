"""Freeze A9 local command labelability before diagnostic dataset collection."""
from __future__ import annotations
import hashlib,json
from pathlib import Path

HERE=Path(__file__).resolve(); REPO=HERE.parents[4]
BASE=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion"
A8=BASE/"phase_w2_p1_a8_offline_start_teacher_oracle"
OUT=BASE/"phase_w2_p1_a9_observation_history_contract_preflight"
old=json.loads((A8/"offline_teacher_local_neighborhood.json").read_text())
raw=OUT/"raw/local_counterfactual"
counter={}
for p in raw.glob("update*/condition_*.json"):
    x=json.loads(p.read_text()); counter[x["multi_condition_specs"][0].get("id",p.stem.replace("condition_",""))]=x
# Specs carry the stable cf index; collect results by file stem.
counter={p.stem.replace("condition_",""):json.loads(p.read_text()) for p in raw.glob("update*/condition_*.json")}
rows=[]
for i,r in enumerate(old["rows"]):
    chosen=dict(r); chosen["point_id"]=f"local_{i:03d}"; chosen["original_diagnostic_pass"]=bool(r["diagnostic_pass"])
    if r["diagnostic_pass"]:
        chosen.update(labelable=True,selection="A8_FROZEN_SELECTED_TEACHER",alternative_evaluated=False)
    else:
        x=counter[f"cf{i:03d}"]; alt=x["row"]; ok=alt["endpoint_success"]>=.90 and alt["acquisition_0p20"]>=.80 and alt["fall_rate"]<=.02
        if ok:
            chosen.update(alt); chosen.update(labelable=True,selection="ALTERNATIVE_A7_R2_CHECKPOINT",alternative_evaluated=True,diagnostic_pass=True)
        else:
            chosen.update(labelable=False,selection="UNLABELABLE_LOCAL_COMMAND",alternative_evaluated=True,alternative_metrics=alt)
    rows.append(chosen)
payload={"contract":"validation-only update10 versus update150","total":len(rows),"labelable":sum(r["labelable"] for r in rows),"unlabelable":sum(not r["labelable"] for r in rows),"rows":rows}
canonical=json.dumps(payload,sort_keys=True,separators=(",",":"),allow_nan=False).encode();payload["semantic_sha256"]=hashlib.sha256(canonical).hexdigest()
OUT.mkdir(parents=True,exist_ok=True);(OUT/"local_command_labelability.json").write_text(json.dumps(payload,indent=2)+"\n")
print(json.dumps({k:payload[k] for k in ("total","labelable","unlabelable","semantic_sha256")}))
