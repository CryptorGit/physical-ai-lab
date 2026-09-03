"""Hash every pre-EXP014 experiment/result file and compare start/end snapshots."""
from __future__ import annotations
import argparse, hashlib, json
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE=Path(__file__).resolve(); REPO=HERE.parents[4]
OUT=REPO/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion"
ROOTS=[REPO/"experiments/isaaclab",REPO/"results"]

def digest(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()

def inventory():
 rows=[]; excluded=[]
 for root in ROOTS:
  if not root.exists():continue
  for exp in sorted(root.glob("exp_*")):
   try:n=int(exp.name.split("_")[1])
   except (IndexError,ValueError):continue
   if not 5<=n<=13:continue
   for p in sorted(q for q in exp.rglob("*") if q.is_file()):
    rel=p.relative_to(REPO).as_posix(); size=p.stat().st_size
    # Closure/demo media are not experimental checkpoints, optimizer state,
    # datasets, labels, splits, manifests, overlays, or raw state pools. Keep a
    # path/size/mtime sentinel for them while content-hashing every research file.
    if p.suffix.lower() in {".mp4",".avi",".mov",".mkv",".png",".jpg",".jpeg",".gif"} and "overlay" not in p.name.lower():
     excluded.append({"path":rel,"size":size,"mtime_ns":p.stat().st_mtime_ns})
    else:rows.append({"path":rel,"size":size,"sha256":digest(p)})
 aggregate=hashlib.sha256("".join(f'{r["path"]}\0{r["size"]}\0{r["sha256"]}\n' for r in rows).encode()).hexdigest()
 media_aggregate=hashlib.sha256("".join(f'{r["path"]}\0{r["size"]}\0{r["mtime_ns"]}\n' for r in excluded).encode()).hexdigest()
 return rows,aggregate,excluded,media_aggregate

def main():
 ap=argparse.ArgumentParser();ap.add_argument("mode",choices=("start","end"));a=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
 rows,aggregate,excluded,media_aggregate=inventory(); value={"timestamp":datetime.now(timezone(timedelta(hours=9))).isoformat(),"scope":"content hashes for all non-demo-media files under experiment/result exp_005 through exp_013; metadata sentinels for closure/demo media","content_hashed_file_count":len(rows),"aggregate_sha256":aggregate,"media_sentinel_file_count":len(excluded),"media_metadata_aggregate_sha256":media_aggregate,"files":rows,"media_sentinels":excluded}
 if a.mode=="end":
  start=json.loads((OUT/"protected_hashes_start.json").read_text(encoding="utf-8"));before={r["path"]:(r["size"],r["sha256"]) for r in start["files"]};after={r["path"]:(r["size"],r["sha256"]) for r in rows};changes=[{"path":p,"start":before.get(p),"end":after.get(p)} for p in sorted(before.keys()|after.keys()) if before.get(p)!=after.get(p)]
  media_changed=start.get("media_metadata_aggregate_sha256")!=media_aggregate
  if media_changed:changes.append({"path":"<closure/demo media metadata aggregate>","start":start.get("media_metadata_aggregate_sha256"),"end":media_aggregate})
  value["comparison"]={"status":"PASS" if not changes else "PROTECTED_PATH_CHANGED","difference_count":len(changes),"differences":changes}
 (OUT/f"protected_hashes_{a.mode}.json").write_text(json.dumps(value,indent=2)+"\n",encoding="utf-8");print(json.dumps({k:v for k,v in value.items() if k not in ("files","media_sentinels")},indent=2))
 if a.mode=="end" and value["comparison"]["status"]!="PASS":raise SystemExit(3)
if __name__=="__main__":main()
