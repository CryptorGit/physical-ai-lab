"""Aggregate the preregistered A7-R3 teacher-attractor horizon sweep."""
from __future__ import annotations
import csv, json, subprocess
from pathlib import Path

HERE=Path(__file__).resolve(); REPO=HERE.parents[4]
OUT=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w2_p1_a7_r3_start_retention_recovery"
RAW=OUT/"raw/safe_horizon"; EVAL=HERE.parent/"evaluate_w2_p1_a7_r3_safe_horizon.py"; ISAAC=Path.home()/"workspace/IsaacLab/isaaclab.bat"
selected=json.loads((OUT/"selected_checkpoint.json").read_text(encoding="utf-8")); policy=Path(selected["path"]); policy=policy if policy.is_absolute() else OUT/policy
RAW.mkdir(parents=True,exist_ok=True); rows=[]
for horizon in (2,4,6,8,12,16,24):
 for direction in range(0,360,45):
  for yi,yaw in enumerate((-.3,0.,.3)):
   output=RAW/f"h{horizon:02d}_d{direction:03d}_y{yi}.json"
   if not output.exists():
    command=[str(ISAAC),"-p",str(EVAL),"--teacher-policy",str(policy),"--output",str(output),"--batch","5","--episodes","200","--horizon",str(horizon),"--direction",str(direction),"--yaw",str(yaw),"--headless","--device","cuda:0"]
    with output.with_suffix(".log").open("w",encoding="utf-8") as log: subprocess.run(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
   rows.append(json.loads(output.read_text(encoding="utf-8"))["row"]); print(json.dumps({"horizon":horizon,"direction":direction,"yaw":yaw}),flush=True)
summary=[]
for horizon in (2,4,6,8,12,16,24):
 subset=[row for row in rows if int(row["horizon"])==horizon]
 aggregate_endpoint=sum(float(row["endpoint_success"]) for row in subset)/24; aggregate_acquisition=sum(float(row["acquisition_0p20"]) for row in subset)/24; aggregate_fall=sum(float(row["fall_rate"]) for row in subset)/24
 passed=aggregate_endpoint>=.95 and aggregate_acquisition>=.85 and aggregate_fall<=.02 and min(float(row["acquisition_0p20"]) for row in subset)>=.85
 summary.append({"horizon":horizon,"aggregate_endpoint":aggregate_endpoint,"aggregate_acquisition":aggregate_acquisition,"aggregate_fall":aggregate_fall,"minimum_condition_acquisition":min(float(row["acquisition_0p20"]) for row in subset),"pass":passed})
with (OUT/"teacher_safe_horizon.csv").open("w",newline="",encoding="utf-8") as stream:
 writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
(OUT/"teacher_safe_horizon.json").write_text(json.dumps({"teacher":str(policy.relative_to(REPO)),"candidate":"A4 V2 exact in-memory reconstruction","B0":"A4 V2 stop label","conditions":24,"episodes_per_condition":200,"rows":rows,"summary":summary,"authorized_safe_teacher_horizon":next((row["horizon"] for row in summary if row["pass"]),None),"runtime_teacher_authorized":False},indent=2)+"\n",encoding="utf-8")
