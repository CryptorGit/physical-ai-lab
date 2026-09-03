"""Prepare the preregistered opposite-teacher check for A8 local failures."""
from __future__ import annotations
import json
from pathlib import Path

HERE=Path(__file__).resolve(); REPO=HERE.parents[4]
BASE=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion"
A8=BASE/"phase_w2_p1_a8_offline_start_teacher_oracle"
OUT=BASE/"phase_w2_p1_a9_observation_history_contract_preflight/raw/local_counterfactual"
local=json.loads((A8/"offline_teacher_local_neighborhood.json").read_text())
groups={"update010":[],"update150":[]}
for i,row in enumerate(local["rows"]):
    if row["diagnostic_pass"]: continue
    source=next((A8/"raw/local_neighborhood"/row["formal_boundary_condition"]).glob("condition_*.json"))
    selected=Path(json.loads(source.read_text())["policy"]).stem
    alternative="update150" if selected=="model_010" else "update010"
    groups[alternative].append({"id":f"cf{i:03d}","direction":row["direction"],"speed":row["speed"],"yaw":row["yaw"],"formal_boundary_condition":row["formal_boundary_condition"],"selected_checkpoint":selected})
OUT.mkdir(parents=True,exist_ok=True)
for key,rows in groups.items():
    for j in range(0,len(rows),9):
        (OUT/f"{key}_specs_{j//9}.json").write_text(json.dumps(rows[j:j+9],indent=2)+"\n")
print(json.dumps({k:len(v) for k,v in groups.items()}))
