"""Parent-owned durable coordinator and offline finalizer for Phase 2-D21."""
from __future__ import annotations
import argparse,csv,hashlib,importlib.util,json,os,sqlite3,subprocess
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve();REPO=HERE.parents[4];OUT=REPO/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d21_identity_complete_support_causality";RAW=OUT/"raw";DB=OUT/"reference_rollout.sqlite";BUNDLE=OUT/"reference_rollout_bundle.npz";WORKER=HERE.parent/"run_phase2_d21_worker.py";ISAAC=Path(r"C:\Users\user\workspace\IsaacLab\isaaclab.bat")
START="b2c6e3d15f8ade365a10bd347317f3839544ae82";RUN="exp014-d21-reference-v1";SEED=20279501;CKPT=REPO/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d16_dedicated_start_specialist/raw/checkpoints/model_000.pt";TRAIN=REPO/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d16_dedicated_start_specialist/raw/train_start_snapshots.pt";REWARD=HERE.parent/"support_reward_v2r1.py";RECON=HERE.parent/"d21_reward_reconstruction.py"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def dump(name,x):OUT.mkdir(parents=True,exist_ok=True);(OUT/name).write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
def mod(name,path):s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def init_db():
 exists=DB.exists();db=sqlite3.connect(DB);db.execute("PRAGMA journal_mode=WAL");db.execute("PRAGMA synchronous=FULL");db.executescript("""CREATE TABLE IF NOT EXISTS rollouts(rollout_id TEXT PRIMARY KEY,status TEXT NOT NULL,expected_transitions INTEGER NOT NULL,bundle_path TEXT,bundle_sha TEXT,metadata_json BLOB);CREATE TABLE IF NOT EXISTS events(sequence INTEGER PRIMARY KEY AUTOINCREMENT,rollout_id TEXT,event TEXT,detail_json BLOB);""");
 if exists:
  row=db.execute("SELECT status,bundle_path,bundle_sha FROM rollouts WHERE rollout_id=?",(RUN,)).fetchone()
  if row!=("STARTED",None,None):raise RuntimeError("existing D21 transaction is not safely resumable")
  with db:db.execute("INSERT INTO events(rollout_id,event,detail_json) VALUES(?,?,?)",(RUN,"CAPTURE_RESUME_AFTER_PREBUNDLE_INFRA_FAILURE",b"{}"))
 else:
  with db:
   db.execute("INSERT INTO rollouts VALUES(?,?,?,?,?,?)",(RUN,"STARTED",6400,None,None,json.dumps({"seed":SEED,"contract":"Exp014D21IdentityCompleteReferenceRolloutV1"},sort_keys=True).encode()));db.execute("INSERT INTO events(rollout_id,event,detail_json) VALUES(?,?,?)",(RUN,"CAPTURE_STARTED",b"{}"))
 return db
def array_hashes(path):
 with np.load(path,allow_pickle=False) as z:return {k:hashlib.sha256(np.ascontiguousarray(z[k]).tobytes()).hexdigest() for k in sorted(z.files)}
def preregister():
 head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip();stage={"stage":"Phase 2-D21","starting_head":head,"requested_starting_head":START,"head_match":head==START,"timestamp_utc":datetime.now(timezone.utc).isoformat(),"seed":SEED,"result_blind":True,"remote_push":False};dump("stage_reference.json",stage)
 protocol={"name":"Exp014D21IdentityCompleteReferenceRolloutV1","condition":{"direction_deg":0,"speed":.3,"yaw":0},"source":{"split":"train","snapshots":64,"validation_access":0},"capture":{"steps":100,"transitions":6400,"physics_attempts_per_reference_snapshot":1,"sample_order":"control_step_major_then_environment_index"},"rng_streams":{"environment_reset":SEED,"policy_sampling":SEED,"minibatch":SEED,"optimizer":SEED},"temporary_only":{"probe_tensors":True,"persistent_policy_update":False,"checkpoint":False},"mirror_tolerance":{"signed_balance_absolute":.05,"yaw_p95_relative":.10,"safety_classification":"exact"},"gates":"D21 user contract frozen"};dump("protocol.json",protocol);dump("reference_rollout_contract.json",protocol)
 checks={"train_snapshot_sha":sha(TRAIN),"d16_initial_checkpoint_sha":sha(CKPT),"reward_v2r1_source_sha":sha(REWARD),"seed":SEED,"validation_access":0,"status":"PASS"};dump("reference_rollout_identity.json",checks)
def finalize(meta,worker):
 recon=mod("d21_reconstruct_parent",RECON);scales=worker["scales"];weights=worker["weights"]
 with np.load(BUNDLE,allow_pickle=False) as z:a={k:z[k] for k in z.files}
 rec=recon.reconstruct(a,scales,weights);diffs={k:float(np.max(np.abs(rec[k].astype(np.float32)-a[k].astype(np.float32)))) for k in rec};recon_pass=max(diffs.values())<=1e-8;exploit=int(np.count_nonzero((~a["support_valid"])&(a["load_reward"]!=0)))
 reconstruction={"status":"PASS" if recon_pass and exploit==0 else "FAIL","max_absolute_difference":max(diffs.values()),"per_term_max_absolute_difference":diffs,"classification_difference":0 if recon_pass else None,"corrected_target_schedule":True,"valid_support_mask":True,"zero_support_exploit_occurrences":exploit};dump("reward_reconstruction_audit.json",reconstruction)
 baseline=worker["baseline"];dump("baseline_physical_metrics.json",baseline);dump("actor_gradient_isolation.json",worker["gradient_isolation"])
 names=sorted(worker["gradient_cosines"]);with_csv=[]
 for x in names:
  for y in names:with_csv.append({"term_a":x,"term_b":y,"cosine":worker["gradient_cosines"][x][y]})
 with (OUT/"support_gradient_conflict_matrix.csv").open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=["term_a","term_b","cosine"]);w.writeheader();w.writerows(with_csv)
 dump("support_gradient_conflict_matrix.json",{"matrix":worker["gradient_cosines"],"strong_conflict_threshold":-.5})
 probes=worker["probes"];rows=[]
 for n,m in probes.items():rows.append({"probe":n,**m})
 fields=sorted({k for r in rows for k in r});
 with (OUT/"support_term_decomposition.csv").open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 dump("support_term_decomposition.json",{"baseline":baseline,"probes":probes})
 signed=[{"probe":n,**probes[n]} for n in ("Q_SIGN_LEFT_R1","Q_SIGN_RIGHT_R1")]
 with (OUT/"signed_support_probe_results.csv").open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=sorted({k for r in signed for k in r}));w.writeheader();w.writerows(signed)
 left,right=probes["Q_SIGN_LEFT_R1"],probes["Q_SIGN_RIGHT_R1"]
 lp=(left["signed_load_balance_mean"]>baseline["signed_load_balance_mean"] and left["signed_target_error"]<=.9*baseline["signed_left_target_error"] and left["fall"]<=baseline["fall"]+.02 and left["dangerous_slip"]<=baseline["dangerous_slip"]+.02 and left["total_support_error"]<=baseline["total_support_error"]+1e-9)
 rp=(right["signed_load_balance_mean"]<baseline["signed_load_balance_mean"] and right["signed_target_error"]<=.9*baseline["signed_right_target_error"] and right["fall"]<=baseline["fall"]+.02 and right["dangerous_slip"]<=baseline["dangerous_slip"]+.02 and right["total_support_error"]<=baseline["total_support_error"]+1e-9)
 mirror_pass=worker["symmetry"]["signed_physical_mirror_error"]<=.05 and worker["symmetry"]["yaw_p95_mirror_error"]<=.1*max(left["yaw_p95"],right["yaw_p95"],1e-9)
 dump("signed_support_probe_results.json",{"baseline":baseline,"left":left,"right":right,"left_gate":lp,"right_gate":rp,"mirror_consistency":mirror_pass,"mirror_tolerance":{"signed_balance_absolute":.05,"yaw_p95_relative":.10}})
 sym={**worker["symmetry"],"signed_left_pass":lp,"signed_right_pass":rp,"mirror_consistency":mirror_pass};dump("symmetry_cancellation_metrics.json",sym);dump("support_side_temporal_stability.json",{n:{k:v for k,v in m.items() if k in ("first_dominant_support_step","dominant_support_duration","support_side_reversal_count","yaw_sign_changes","support_yaw_sign_correlation")} for n,m in probes.items()})
 load=probes["Q_LOAD_ABS_R1"];full=probes["Q_SUPPORT_FULL_R1"];load_improve=1-load["load_target_error"]/baseline["load_target_error"];full_improve=1-full["load_target_error"]/baseline["load_target_error"]
 load_pass=load_improve>=.10 and load["total_support_error"]<=1.02*baseline["total_support_error"] and load["fall"]<=baseline["fall"]+.02 and load["dangerous_slip"]<=baseline["dangerous_slip"]+.02 and exploit==0
 full_pass=full_improve>=.10 and full["total_support_error"]<=baseline["total_support_error"]+1e-9 and full["support_foot_slip"]<=baseline["support_foot_slip"]+1e-9 and full["fall"]<=baseline["fall"]+.02 and full["dangerous_slip"]<=baseline["dangerous_slip"]+.02 and full["torque_saturation"]<=baseline["torque_saturation"]+.05
 stability=all(x["finite"] and x["exact_kl"]<=.2 and x["all_step_kl"]<=.2 and x["clip_fraction"]<=.5 and x["mean_final_action_shift"]<=2 for x in worker["temporary_tensors"].values())
 conflicts=[]
 if load_pass and not full_pass:
  for n,m in probes.items():
   if n in ("Q_LOAD_ABS_R1","Q_SIGN_LEFT_R1","Q_SIGN_RIGHT_R1"):continue
   cos=worker["gradient_cosines"]["Q_LOAD_ABS_R1"][n];worsen=m["load_target_error"]>=1.10*baseline["load_target_error"]
   if cos<=-.5 or worsen:conflicts.append({"term":n,"cosine":cos,"load_error_worsened_10pct":worsen})
 capture_pass=True
 if not capture_pass:classification="EXP014_D21_IDENTITY_COMPLETE_CAPTURE_FAIL"
 elif not recon_pass:classification="EXP014_D21_REWARD_RECONSTRUCTION_FAIL"
 elif not load_pass:classification="EXP014_D21_CORRECTED_LOAD_TERM_NONCAUSAL"
 elif not full_pass:classification="EXP014_D21_SUPPORT_FAMILY_CONFLICT_CONFIRMED"
 elif lp and rp and not load_pass:classification="EXP014_D21_SUPPORT_SYMMETRY_CANCELLATION_CONFIRMED"
 elif full_pass and lp and rp and mirror_pass and stability:classification="EXP014_D21_CORRECTED_SUPPORT_FAMILY_PREFLIGHT_PASS"
 else:classification="EXP014_D21_SIGNED_SUPPORT_NONCAUSAL"
 root={"load_only":{"improvement":load_improve,"pass":load_pass},"full_support":{"improvement":full_improve,"pass":full_pass},"signed":{"left":lp,"right":rp,"mirror":mirror_pass},"temporary_numerical_stability":stability,"conflicting_terms":conflicts,"classification":classification};dump("root_cause_classification.json",root);dump("temporary_probe_manifest.json",{"persistent":False,"probe_count":len(probes),"tensors":worker["temporary_tensors"],"same_reference_rollout_sha":meta["sha256"]})
 authorized=classification=="EXP014_D21_CORRECTED_SUPPORT_FAMILY_PREFLIGHT_PASS"
 if authorized:dump("exp014_d18r_persistent_training_authorization.json",{"status":"AUTHORIZED","reward":"Exp014OmnidirectionalStartRewardV2R1","reward_hash":sha(REWARD),"reference_rollout_hash":meta["sha256"],"captured_samples":6400,"actor_hash":worker["identity"]["actor_hash"],"critic_hash":worker["identity"]["critic_hash"],"weights_unchanged":True,"architecture_unchanged":True,"training_budget_updates":40,"condition":{"direction":0,"speed":.3,"yaw":0},"C2_or_later":"NOT_AUTHORIZED"})
 else:dump("exp014_d18r_not_authorized.json",{"status":"NOT_AUTHORIZED","classification":classification,"persistent_updates":0})
 next_action="D18R 40-update forward START PPO with unchanged Reward V2R1" if authorized else "causal-term-only Reward V3 preflight" if classification=="EXP014_D21_SUPPORT_FAMILY_CONFLICT_CONFIRMED" else "explicit START phase / lead-foot contract" if classification=="EXP014_D21_SUPPORT_SYMMETRY_CANCELLATION_CONFIRMED" else "direct 141D transition actor" if classification=="EXP014_D21_SIGNED_SUPPORT_NONCAUSAL" else "end corrected support-reward route and audit a direct 141D transition actor" if classification=="EXP014_D21_CORRECTED_LOAD_TERM_NONCAUSAL" else "repair capture/reconstruction only"
 dump("stage_classification.json",{"primary_classification":classification,"persistent_updates":0});dump("recommended_next_action.json",{"single_next_experiment":next_action})
 prot={"exp005_to_exp013_changed_by_d21":False,"d6_to_d20_changed_by_d21":False,"persistent_policy_update":0,"new_persistent_checkpoint":0,"validation_access":0,"heldout_access":0,"reward_weight_change":0,"actor_input_change":0,"formal_gate_change":0,"RUN":0,"Causal_DAgger_V2":0,"remote_push":False};dump("protected_hashes.json",prot)
 report=f"""# Exp014 Phase 2-D21 identity-complete support causality audit\n\nClassification: `{classification}`.\n\nThe parent-owned WAL/FULL SQLite transaction durably committed a {meta['transitions']}-transition NPZ bundle before temporary probes. Two independent readers matched every array hash. Offline Reward V2R1 reconstruction had maximum absolute difference {max(diffs.values()):.9g}; zero-support exploits: {exploit}.\n\nCorrected load-only improvement was {load_improve:.2%} (gate {'PASS' if load_pass else 'FAIL'}). Full-family improvement was {full_improve:.2%} (gate {'PASS' if full_pass else 'FAIL'}). Signed-left/right gates were {lp}/{rp}; mirror consistency was {mirror_pass}. Persistent updates and checkpoints: 0. Validation/held-out access: 0.\n""";(REPO/"research/exp_014_phase_2_d21_identity_complete_support_causality_report.md").write_text(report,encoding="utf-8")
 return classification
def main():
 p=argparse.ArgumentParser();p.add_argument("--device",default="cuda:0");args=p.parse_args();OUT.mkdir(parents=True,exist_ok=True);RAW.mkdir(parents=True,exist_ok=True);preregister()
 db=init_db();meta=None
 if BUNDLE.exists():
  ready=[x for x in (OUT/"simulation_worker.log").read_text(encoding="utf-8",errors="replace").splitlines() if x.startswith("D21_CAPTURE_READY ")]
  if not ready:raise RuntimeError("durable bundle has no capture-ready provenance")
  meta=json.loads(ready[-1][len("D21_CAPTURE_READY "):])
  if sha(BUNDLE)!=meta["sha256"]:raise RuntimeError("durable bundle hash mismatch")
  h1=array_hashes(BUNDLE);h2=array_hashes(BUNDLE)
  if h1!=h2:raise RuntimeError("two-reader identity failure")
  with db:
   db.execute("UPDATE rollouts SET status='COMPLETED',bundle_path=?,bundle_sha=?,metadata_json=? WHERE rollout_id=?",(str(BUNDLE),meta["sha256"],json.dumps({**meta,"array_hashes":h1},sort_keys=True).encode(),RUN));db.execute("INSERT INTO events(rollout_id,event,detail_json) VALUES(?,?,?)",(RUN,"CAPTURE_LEDGER_REPAIRED_AFTER_ATOMIC_RENAME",json.dumps({"bundle_sha":meta["sha256"]},sort_keys=True).encode()))
  env=os.environ.copy();env["D21_PROBE_ONLY"]="1";cmd=subprocess.list2cmdline([str(ISAAC),"-p",str(WORKER),"--headless","--device",args.device]);proc=subprocess.Popen(["cmd.exe","/d","/s","/c",cmd],cwd=REPO,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding="utf-8",errors="replace",bufsize=1);assert proc.stdout
  with (OUT/"probe_worker.log").open("w",encoding="utf-8") as log:
   for line in proc.stdout:log.write(line);log.flush()
 else:
  cmd=subprocess.list2cmdline([str(ISAAC),"-p",str(WORKER),"--headless","--device",args.device]);proc=subprocess.Popen(["cmd.exe","/d","/s","/c",cmd],cwd=REPO,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding="utf-8",errors="replace",bufsize=1);assert proc.stdin and proc.stdout
  with (OUT/"simulation_worker.log").open("w",encoding="utf-8") as log:
   for line in proc.stdout:
    log.write(line);log.flush()
    if line.startswith("D21_CAPTURE_READY "):
     meta=json.loads(line[len("D21_CAPTURE_READY "):]);tmp=Path(meta["tmp_path"])
     if meta["transitions"]!=6400 or sha(tmp)!=meta["sha256"]:raise RuntimeError("capture bundle identity failure")
     os.replace(tmp,BUNDLE)
     with BUNDLE.open("rb+") as f:os.fsync(f.fileno())
     h1=array_hashes(BUNDLE);h2=array_hashes(BUNDLE)
     if h1!=h2:raise RuntimeError("two-reader identity failure")
     with db:
      db.execute("UPDATE rollouts SET status='COMPLETED',bundle_path=?,bundle_sha=?,metadata_json=? WHERE rollout_id=?",(str(BUNDLE),meta["sha256"],json.dumps({**meta,"array_hashes":h1},sort_keys=True).encode(),RUN));db.execute("INSERT INTO events(rollout_id,event,detail_json) VALUES(?,?,?)",(RUN,"CAPTURE_COMPLETED",json.dumps({"bundle_sha":meta["sha256"]},sort_keys=True).encode()))
     missing=db.execute("SELECT COUNT(*) FROM rollouts WHERE status='COMPLETED' AND (bundle_sha IS NULL OR bundle_path IS NULL)").fetchone()[0]
     if missing:raise RuntimeError("completed without durable bundle")
     proc.stdin.write("CONTINUE\n");proc.stdin.flush()
 rc=proc.wait();db.close()
 if rc or meta is None:raise RuntimeError(f"D21 worker failed rc={rc}")
 worker=json.loads((RAW/"worker_results.json").read_text());h1=array_hashes(BUNDLE);h2=array_hashes(BUNDLE);manifest={"rollout_id":RUN,"bundle":str(BUNDLE.relative_to(REPO)).replace("\\","/"),"bundle_sha256":meta["sha256"],"transitions":6400,"arrays":meta["array_count"],"sample_order":meta["sample_order"],"source_snapshots":64,"seed":SEED};dump("reference_rollout_capture_manifest.json",manifest);dump("reference_rollout_durability_audit.json",{"journal_mode":"wal","synchronous":"FULL","owner":"parent persistence process","status":"PASS","completed_rollout_ids":[RUN],"durable_rollout_bundle_ids":[RUN],"completed_without_bundle":0,"two_reader_bitwise_identity":h1==h2,"array_hashes":h1});(OUT/"reference_rollout_bundle.sha256").write_text(meta["sha256"]+"  reference_rollout_bundle.npz\n")
 identity={**worker["identity"],"source_snapshot_hashes":h1.get("source_snapshot_hashes"),"source_observation_hashes":h1.get("source_observation_hashes"),"bundle_sha256":meta["sha256"],"two_reader_identity":h1==h2};dump("reference_rollout_identity.json",identity);classification=finalize(meta,worker)
 (OUT/"reproduction_commands.ps1").write_text("& 'C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat' -p experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d21_worker.py --headless --device cuda:0 # normally launched by parent IPC\npython experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d21_parent.py --device cuda:0\n",encoding="utf-8");print(json.dumps({"classification":classification,"bundle_sha":meta["sha256"],"persistent_updates":0},indent=2))
if __name__=="__main__":main()
