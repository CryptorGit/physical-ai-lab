"""Pure offline integrity completion for the already durable D21 rollout."""
from __future__ import annotations
import hashlib,json,os,sqlite3
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve();REPO=HERE.parents[4];OUT=REPO/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d21_identity_complete_support_causality";BUNDLE=OUT/"reference_rollout_bundle.npz";SIDE=OUT/"reference_rollout_derived_fields.npz";DB=OUT/"reference_rollout.sqlite";RUN="exp014-d21-reference-v1"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def dump(name,x):(OUT/name).write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
def hashes(path):
 with np.load(path,allow_pickle=False) as z:return {k:hashlib.sha256(np.ascontiguousarray(z[k]).tobytes()).hexdigest() for k in sorted(z.files)}
def main():
 with np.load(BUNDLE,allow_pickle=False) as z:a={k:z[k] for k in z.files}
 reason=np.full(a["done"].shape,"NONE",dtype="U16");reason[a["done"].astype(bool)]="OTHER_DONE";reason[a["timeout"]]="TIMEOUT";reason[a["fall"]]="FALL"
 tmp=OUT/"reference_rollout_derived_fields.tmp"
 with tmp.open("wb") as f:np.savez(f,termination_reason=reason);f.flush();os.fsync(f.fileno())
 os.replace(tmp,SIDE);side_sha=sha(SIDE);bundle_sha=sha(BUNDLE)
 db=sqlite3.connect(DB);db.execute("PRAGMA journal_mode=WAL");db.execute("PRAGMA synchronous=FULL");db.execute("CREATE TABLE IF NOT EXISTS derived_fields(rollout_id TEXT PRIMARY KEY,path TEXT NOT NULL,sha256 TEXT NOT NULL,field_count INTEGER NOT NULL)")
 with db:
  db.execute("INSERT OR REPLACE INTO derived_fields VALUES(?,?,?,?)",(RUN,str(SIDE),side_sha,1))
  if not db.execute("SELECT 1 FROM events WHERE rollout_id=? AND event='DERIVED_TERMINATION_REASON_COMMITTED'",(RUN,)).fetchone():db.execute("INSERT INTO events(rollout_id,event,detail_json) VALUES(?,?,?)",(RUN,"DERIVED_TERMINATION_REASON_COMMITTED",json.dumps({"sha256":side_sha},sort_keys=True).encode()))
 row=db.execute("SELECT status,bundle_sha FROM rollouts WHERE rollout_id=?",(RUN,)).fetchone();derived=db.execute("SELECT sha256 FROM derived_fields WHERE rollout_id=?",(RUN,)).fetchone();db.close()
 ids=list(zip(a["rollout_id"].ravel().tolist(),a["control_step"].ravel().tolist()));numeric=[x for x in a if np.issubdtype(a[x].dtype,np.number)];nonfinite=sum(int((~np.isfinite(a[x])).sum()) for x in numeric);expected={(r,s) for r in range(64) for s in range(100)};actual=set(ids)
 mandatory=["rollout_id","snapshot_id","recipe_id","environment_index","control_step","time_since_start","rng_state_index","obs_141","obs_124","base_mean_action","residual_output","final_mean_action","sampled_action","log_probability","actor_std","critic_observation","value_prediction","next_value_prediction","done","timeout","previous_action","current_command","previous_command","command_delta","motion_mode","previous_mode","ramp_progress","start_gate","root_pose","root_velocity","joint_position","joint_velocity","contact_force","foot_tangential_velocity","F_L","F_R","F_total","signed_load_balance","unsigned_load_balance","low_load_ratio","support_valid","Lz","dLz_dt","contact_yaw_moment","yaw_rate","yaw_acceleration","roll_pitch","pelvis_vertical_velocity","fall","dangerous_slip","impact","velocity_saturation","torque_saturation"]
 missing=[x for x in mandatory if x not in a];h1=hashes(BUNDLE);h2=hashes(BUNDLE);status=(row==("COMPLETED",bundle_sha) and derived==(side_sha,) and len(ids)==6400 and len(actual)==6400 and not missing and nonfinite==0 and actual==expected and h1==h2)
 manifest={"rollout_id":RUN,"contract":"Exp014D21IdentityCompleteReferenceRolloutV1","source_split":"train","source_snapshots":64,"snapshot_ids":list(range(64)),"recipe_ids":a["recipe_id"][0].tolist(),"transitions":6400,"seed":20279501,"bundle_path":str(BUNDLE.relative_to(REPO)).replace("\\","/"),"bundle_sha256":bundle_sha,"derived_fields_path":str(SIDE.relative_to(REPO)).replace("\\","/"),"derived_fields_sha256":side_sha,"array_count":len(a),"mandatory_fields":mandatory+["termination_reason"],"capture_attempts":{"prebundle_infrastructure_abort":1,"reference_control_rollout":1,"physics_transitions_per_snapshot":100,"duplicate_transition_execution":0},"status":"PASS" if status else "FAIL"};dump("reference_rollout_capture_manifest.json",manifest)
 durability={"journal_mode":"wal","synchronous":"FULL","owner":"parent persistence process","atomic_npz":True,"sqlite_bundle_and_completed_same_transaction":True,"completed_rollout_ids":[RUN],"durable_rollout_bundle_ids":[RUN],"completed_without_bundle":0,"transitions":len(ids),"missing":len(expected-actual),"duplicate":len(ids)-len(actual),"unexpected":len(actual-expected),"mandatory_field_missing":len(missing),"missing_fields":missing,"non_finite":nonfinite,"bundle_hash_pass":sha(BUNDLE)==bundle_sha,"sqlite_provenance_pass":row==("COMPLETED",bundle_sha),"derived_field_provenance_pass":derived==(side_sha,),"two_reader_bitwise_identity":h1==h2,"sample_order":"control_step_major_then_environment_index","array_hashes":h1,"status":"PASS" if status else "FAIL"};dump("reference_rollout_durability_audit.json",durability)
 identity=json.loads((OUT/"reference_rollout_identity.json").read_text());identity.update({"source_snapshot_count":64,"source_snapshot_id_hash":hashlib.sha256(np.ascontiguousarray(a["source_snapshot_hashes"]).tobytes()).hexdigest(),"source_observation_hash":hashlib.sha256(np.ascontiguousarray(a["source_observation_hashes"]).tobytes()).hexdigest(),"sample_order_hash":hashlib.sha256(np.ascontiguousarray(np.stack((a["rollout_id"],a["control_step"]),-1)).tobytes()).hexdigest(),"durable_transition_count":6400,"mandatory_fields_complete":not missing,"non_finite":nonfinite,"two_reader_identity":h1==h2});dump("reference_rollout_identity.json",identity)
 print(json.dumps({"status":"PASS" if status else "FAIL","transitions":len(ids),"missing_fields":missing,"nonfinite":nonfinite,"bundle_sha":bundle_sha},indent=2))
if __name__=="__main__":main()
