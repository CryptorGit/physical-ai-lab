"""Parent-owned durable coordinator for D15 formal validation."""
from __future__ import annotations
import hashlib,importlib.util,json,os,subprocess
from datetime import datetime,timezone
from pathlib import Path

HERE=Path(__file__).resolve();REPO=HERE.parents[4];OUT=REPO/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d15_stand_to_omniwalk_start_audit";DB=OUT/"durable_evaluation.sqlite"
WORKER=HERE.parent/"run_phase2_d15_worker.py";STORE=HERE.parent/"durable_evaluation_store.py";ISAAC=Path(r"C:\Users\user\workspace\IsaacLab\isaaclab.bat");RUN="exp014-d15-formal-v1";START="d9ab9326f29f2723d6d8156d5d3091771c9bf5c6"
W_SHA="61dd37f9f1fb7036357d54975c1120c66c30650ed7d5f2e1d4de91174fdca27d";H_SHA="734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621"
def canonical(x):return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def dump(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+".tmp")
 with t.open("wb") as f:f.write(json.dumps(x,indent=2,sort_keys=True,allow_nan=False).encode()+b"\n");f.flush();os.fsync(f.fileno())
 os.replace(t,p)
def load_store():
 s=importlib.util.spec_from_file_location("d15store",STORE);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def expected():return [{"episode_id":f"D15-F-{c:02d}-{j:03d}","condition_id":c} for c in range(34) for j in range(102)]
def main():
 OUT.mkdir(parents=True,exist_ok=True);items=expected();mod=load_store();store=mod.DurableEvaluationStore(DB)
 if not store.db.execute("SELECT 1 FROM run_manifest WHERE run_id=?",(RUN,)).fetchone():store.create_run(RUN,W_SHA,H_SHA,"Exp014OmnidirectionalStartTransitionContractV1",items)
 else:store.validate_and_repair(RUN)
 pending=[r[0] for r in store.db.execute("SELECT episode_id FROM episodes WHERE run_id=? AND status!='COMPLETED' ORDER BY episode_id",(RUN,))]
 if not pending:
  store.close();return
 env=os.environ.copy();env["D15_EPISODE_IDS_JSON"]=json.dumps(pending,separators=(",",":"))
 cmd=subprocess.list2cmdline([str(ISAAC),"-p",str(WORKER),"--headless","--device","cuda:0"]);p=subprocess.Popen(["cmd.exe","/d","/s","/c",cmd],cwd=REPO,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding="utf-8",errors="replace",bufsize=1)
 assert p.stdin and p.stdout;finished=None
 with (OUT/"simulation_worker.log").open("w",encoding="utf-8") as log:
  for line in p.stdout:
   log.write(line);log.flush()
   if not line.startswith("D15_IPC:"):continue
   msg=json.loads(line[len("D15_IPC:"):]);kind=msg["kind"];value=msg["value"]
   if kind=="SNAPSHOTS":dump(OUT/"stand_start_snapshot_manifest.json",{"created_at":datetime.now(timezone.utc).isoformat(),"count":102,**value});p.stdin.write("D15_ACK\n");p.stdin.flush()
   elif kind=="START_REQUEST":
    for eid in value["episode_ids"]:store.start_episode(RUN,eid,"d15-isaac-worker")
    p.stdin.write("D15_ACK\n");p.stdin.flush()
   elif kind=="RESULT":
    store.commit_result(RUN,value["episode_id"],value,{"candidate_sha":W_SHA,"sealed_sha":H_SHA,"contract_version":"Exp014OmnidirectionalStartTransitionContractV1","code_version":START});p.stdin.write("D15_ACK\n");p.stdin.flush()
   elif kind=="WORKER_FINISHED":finished=value
 code=p.wait();inv=store.invariants(RUN);completed=store.db.execute("SELECT COUNT(*) FROM episodes WHERE run_id=? AND status='COMPLETED'",(RUN,)).fetchone()[0]
 dump(OUT/"durable_transaction_audit.json",{"journal_mode":store.db.execute("PRAGMA journal_mode").fetchone()[0],"synchronous":store.db.execute("PRAGMA synchronous").fetchone()[0],"persistence_owner":"parent process","expected":3468,"completed":completed,"worker_exit_code":code,"worker_finished":finished,"invariants":inv,"status":"PASS" if code==0 and completed==3468 and not any(inv.values()) else "FAIL"})
 store.close()
 if code or completed!=3468 or any(inv.values()):raise RuntimeError("EXP014_D15_DURABLE_EVALUATION_FAIL")
if __name__=="__main__":main()
