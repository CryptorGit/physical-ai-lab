"""Verify selected A7-R3 checkpoint load and metric parity in fresh processes."""
from __future__ import annotations
import hashlib, json, subprocess
from pathlib import Path

HERE=Path(__file__).resolve(); REPO=HERE.parents[4]
OUT=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w2_p1_a7_r3_start_retention_recovery"
EVAL=HERE.parent/"evaluate_w2_p1_a7_r3.py"; ISAAC=Path.home()/"workspace/IsaacLab/isaaclab.bat"
selected=json.loads((OUT/"selected_checkpoint.json").read_text(encoding="utf-8")); policy=OUT/selected["path"] if not Path(selected["path"]).is_absolute() else Path(selected["path"])
rows=[]
for run in range(2):
 output=OUT/f"raw/selected_parity_run_{run}.csv"
 command=[str(ISAAC),"-p",str(EVAL),"--policy",str(policy),"--batch","4","--split","validation","--direction","315","--speed","0.3","--yaw","0.3","--episodes","200","--group","selected_parity","--output",str(output),"--headless","--device","cuda:0"]
 with output.with_suffix(".log").open("w",encoding="utf-8") as log: subprocess.run(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
 rows.append(json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))["row"])
result={"checkpoint":str(policy.relative_to(REPO)),"sha256":hashlib.sha256(policy.read_bytes()).hexdigest(),"fresh_process_runs":2,"metric_parity":rows[0]==rows[1],"rows":rows,"next_collection_assignment_parity":"NOT_APPLICABLE_EXISTING_RESCUE" if selected.get("source")=="existing_A7_R2_checkpoint" else selected.get("next_collection_assignment_parity","PASS"),"status":"PASS" if rows[0]==rows[1] else "FAIL"}
(OUT/"selected_checkpoint_process_parity.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
