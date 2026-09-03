"""Aggregate per-condition and global A8 safe horizons."""
from __future__ import annotations
import csv,json
import audit_w2_p1_a8_validation as audit
def main():
 mapping=json.loads((audit.OUT/"offline_start_teacher_condition_map_v1.json").read_text())["condition_map"];raw=audit.OUT/"raw/safe_horizon";rows=[]
 for h in (2,4,6,8,12,16,24,32):
  for x in mapping:
   c=int(x["physical_command"]["direction_deg"]//45)*3+(-.3,0.,.3).index(x["physical_command"]["yaw_radps"]);update=x["selected_checkpoint_update"]
   matches=list(raw.glob(f"h{h:02d}_u{update:03d}_*/condition_{c:02d}.json"));assert len(matches)==1,(h,c,matches)
   r=json.loads(matches[0].read_text())["row"];r["horizon"]=h;r["condition_id"]=x["condition_id"];r["selected_checkpoint_update"]=update;rear=r["direction"]==180 and abs(r["yaw"])==.3;r["condition_pass"]=r["endpoint_success"]>=.90 and r["acquisition_0p20"]>=(.90 if rear else .85) and r["fall_rate"]<=.05;rows.append(r)
 summaries=[]
 for h in (2,4,6,8,12,16,24,32):
  rs=[r for r in rows if r["horizon"]==h];s={"horizon":h,"aggregate_endpoint":sum(r["endpoint_success"] for r in rs)/24,"aggregate_acquisition":sum(r["acquisition_0p20"] for r in rs)/24,"aggregate_fall":sum(r["fall_rate"] for r in rs)/24,"minimum_endpoint":min(r["endpoint_success"] for r in rs),"minimum_acquisition":min(r["acquisition_0p20"] for r in rs),"conditions_passed":sum(r["condition_pass"] for r in rs)};s["pass"]=s["aggregate_endpoint"]>=.95 and s["aggregate_acquisition"]>=.85 and s["aggregate_fall"]<=.02 and s["conditions_passed"]==24;summaries.append(s)
 per={}
 for x in mapping:
  rs=[r for r in rows if r["condition_id"]==x["condition_id"]];per[x["condition_id"]]=next((r["horizon"] for r in rs if r["condition_pass"]),None)
 global_horizon=max(per.values()) if all(v is not None for v in per.values()) else None
 with (audit.OUT/"oracle_safe_horizon.csv").open("w",newline="",encoding="utf-8") as s:w=csv.DictWriter(s,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (audit.OUT/"oracle_safe_horizon.json").write_text(json.dumps({"status":"PASS" if global_horizon is not None else "FAIL","rows":rows,"summary":summaries,"per_condition_shortest_horizon":per,"global_safe_teacher_horizon":global_horizon,"runtime_teacher_authorized":False},indent=2)+"\n")
 print(json.dumps({"global_safe_teacher_horizon":global_horizon,"summary":summaries}))
if __name__=="__main__":main()
