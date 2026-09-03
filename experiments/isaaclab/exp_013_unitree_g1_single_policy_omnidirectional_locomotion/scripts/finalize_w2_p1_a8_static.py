"""Aggregate held-out pure-yaw and static retention for mapped checkpoints."""
from __future__ import annotations
import json
import audit_w2_p1_a8_validation as audit
def main():
 pure=[];static=[]
 for update in (10,150):
  for yi,yaw in enumerate((-.3,.3)):
   row=json.loads((audit.OUT/f"raw/pure_yaw/u{update}_y{yi}.json").read_text())["row"];row["update"]=update;pure.append(row)
  rows=json.loads((audit.OUT/f"raw/static/u{update}.json").read_text())["rows"]
  groups={g:[r for r in rows if r["group"]==g] for g in {r["group"] for r in rows}}
  static.append({"update":update,"zero_yaw_pass_count":sum(r["endpoint_success"]>=.90 for r in groups["zero_yaw"]),"forward_0p6_endpoint":next(r["endpoint_success"] for r in groups["forward_anchor"] if r["speed"]==.6),"forward_1p2_endpoint":next(r["endpoint_success"] for r in groups["forward_anchor"] if r["speed"]==1.2),"moving_turn_pass_count":sum(r["endpoint_success"]>=.90 for r in groups["moving_turn"]),"pure_yaw_endpoint_pass_count":sum(r["endpoint_success"]>=.90 for r in groups["pure_yaw"]),"rows":rows})
 pure_status=all(next(r for r in pure if r["update"]==u and r["yaw"]<0)["acquisition_0p20"]>=.90 and next(r for r in pure if r["update"]==u and r["yaw"]>0)["acquisition_0p20"]>=.85 and max(r["fall_rate"] for r in pure if r["update"]==u)<=.05 for u in (10,150))
 static_status=all(r["zero_yaw_pass_count"]==16 and r["forward_0p6_endpoint"]>=.95 and r["forward_1p2_endpoint"]>=.95 and r["moving_turn_pass_count"]==24 and r["pure_yaw_endpoint_pass_count"]==2 for r in static)
 (audit.OUT/"heldout_oracle_pure_yaw.json").write_text(json.dumps({"status":"PASS" if pure_status else "FAIL","rows":pure},indent=2)+"\n")
 (audit.OUT/"heldout_oracle_static_retention.json").write_text(json.dumps({"status":"PASS" if static_status else "FAIL","checkpoints":static},indent=2)+"\n")
 print(json.dumps({"pure_yaw":pure_status,"static":static_status}))
if __name__=="__main__":main()
