"""Write W2-P1-D3 metadata/report after read-only probes complete."""
from __future__ import annotations
import hashlib, json, subprocess
from pathlib import Path

HERE=Path(__file__).resolve(); REPO=HERE.parents[4]
BASE=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion"
OUT=BASE/"phase_w2_p1_d3_initialization_gap_diagnosis"
REPORT=REPO/"research/exp_013_g1_phase_w2_p1_d3_initialization_gap_diagnosis_report.md"

def dump(name,value): (OUT/name).write_text(json.dumps(value,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""): h.update(b)
 return h.hexdigest()
def load(name): return json.loads((OUT/name).read_text())

start_head=(OUT/"_starting_head.txt").read_text().strip(); actual_head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip()
status=(OUT/"_starting_status.txt").read_text(encoding="utf-8").splitlines()
dump("stage_reference.json",{"stage":"Phase W2-P1-D3","experiment":"exp_013_unitree_g1_single_policy_omnidirectional_locomotion","starting_head":start_head,"reported_starting_head":"bbe454235b7e2597b448339e66c4577936cf33b1","start_head_match":start_head=="bbe454235b7e2597b448339e66c4577936cf33b1","pre_commit_head":actual_head,"starting_unrelated_dirty_count":len(status),"persistent_training_runs":0,"closed_loop_rollouts":0})
dump("protocol.json",{"purpose":"canonical-parent versus pretrained-student initialization gap diagnosis","dataset_manifest":"w2_p1_dataset_hashes_resolved_v2.json","dataset_identity":"PASS","P3":{"optimizer":"Adam","learning_rate":2e-4,"seed":20277717,"pool_seed":20276049,"steps":2000,"gradient_clip":10.0,"weights":[.25,.25,.25,.25]},"balanced_horizons":[2000,5000,10000,15000,20000,25000,40000],"two_stage_paths":["PATH_A_BALANCED_ONLY","PATH_B_ORIGINAL_THEN_BALANCED","PATH_C_ORIGINAL_25K_THEN_BALANCED","PATH_D_BALANCED_THEN_ORIGINAL","PATH_E_LINEAR_WEIGHT_SCHEDULE"],"interpolation_lambdas":[0,.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,1],"persistent_checkpoint_writes":0,"closed_loop":False,"DAgger":False,"PPO":False,"promotion":False})

before=json.loads((OUT/"_protected_before.json").read_text(encoding="utf-8-sig")); after=[]; changed=[]
for row in before:
 p=(REPO/row["path"].replace(".\\","")).resolve(); current=sha(p); item={**row,"after_sha256":current,"unchanged":current==row["sha256"]}; after.append(item)
 if not item["unchanged"]: changed.append(item["path"])
dump("protected_hashes.json",{"algorithm":"SHA-256","protected_file_count":len(after),"changed":changed,"all_protected_unchanged":not changed,"files":after,"dataset_unchanged":not any("chunk" in x for x in changed),"checkpoint_unchanged":not any(x.endswith(".pt") for x in changed),"manifest_unchanged":not any("manifest" in x or "hashes" in x for x in changed),"new_persistent_policy_checkpoint":0})

inv=load("initialization_checkpoint_inventory.json")["checkpoints"]
p3=load("initialization_p3_replay_matrix.json")["runs"]
long=load("canonical_balanced_long_horizon.json")["runs"]
paths=load("two_stage_optimization_path_comparison.json")["paths"]
interp=load("canonical_to_pretrained_interpolation.json")
layers=load("layerwise_initialization_ablation.json")["runs"]
opts=load("initialization_optimizer_state_ablation.json")["runs"]
latent=load("initialization_gap_latent_analysis.json")["metrics"]
warm=load("old_w2_p1_student_warm_start_validity.json")
def f(x): return f"{x:.8f}"
def group(run,g): return run["final_metrics"][g]["mean_mse"]
p3lines="\n".join(f"- {x['initialization']}: {'PASS' if x['joint_pass'] else 'FAIL'}; start {f(group(x,'START_RETENTION'))}; stop-recovery {f(group(x,'STOP_RECOVERY'))}; exact-zero {f(x['exact_zero_mse'])}" for x in p3)
longlines="\n".join(f"- {x['horizon']:,}: {'PASS' if x['joint_pass'] else 'FAIL'}; start {f(x['metrics']['START_RETENTION']['mean_mse'])}; stop-recovery {f(x['metrics']['STOP_RECOVERY']['mean_mse'])}" for x in long)
pathlines="\n".join(f"- {x['path']}: {'PASS' if x['joint_pass'] else 'FAIL'}; start {f(x['metrics']['START_RETENTION']['mean_mse'])}; stop-recovery {f(x['metrics']['STOP_RECOVERY']['mean_mse'])}" for x in paths)
layerlines="\n".join(f"- {x['initialization']}: {'PASS' if x['joint_pass'] else 'FAIL'}; start {f(x['metrics']['START_RETENTION']['mean_mse'])}; stop-recovery {f(x['metrics']['STOP_RECOVERY']['mean_mse'])}" for x in layers)
optlines="\n".join(f"- {x['case']}: {x['availability']}; joint={x.get('joint_pass','not_evaluable')}" for x in opts)
first_interp=interp["first_success_lambda"]
lat3={x['checkpoint']:x for x in latent if x['layer']==3}
report=f"""# exp_013 Phase W2-P1-D3 initialization-gap diagnosis

## Outcome

Primary classification: `CANONICAL_BALANCED_TRAINING_TOO_SHORT`.

The canonical W1B-R2 actor did reach the preregistered joint static gate under the unchanged balanced objective, first at the 10,000-step checkpoint and again at 40,000 steps. Old-objective pretraining made the 2,000-step consolidation reliable from old step 10,000 onward, but was not necessary for reachability. No persistent policy checkpoint, closed-loop rollout, DAgger round, PPO update, or promotion was performed.

## Initializations

- Canonical parent: W1B-R2 iteration 200, SHA-256 `61dd37f9f1fb7036357d54975c1120c66c30650ed7d5f2e1d4de91174fdca27d`.
- Old supervised checkpoints 0/500/1k/2k/5k/10k/15k/20k/25k were all AVAILABLE.
- Parameter L2 from parent increased from 2.366 at step 500 to 6.525 at step 20,000 and 7.098 at step 25,000.
- Old step 0 is actor-tensor identical to the canonical parent.

## P3 replay matrix

{p3lines}

The success boundary across existing initializations is old step 10,000. Old step-20,000 reproduced tensor hash `975f2cb165e48853f87d79cb93de83ed50954627b5b3a37f38c3b2bd6d4a159b` and trace hash `50d15a131577d64015c5793af01da4db20c1d811cbcb6b105af362517f0b724c` exactly.

## Balanced-only horizon

{longlines}

The gate is narrow and non-monotonic at preregistered checkpoints: 10k and 40k pass, while 15k/20k/25k straddle the 0.001 boundary. This establishes reachability without old pretraining but not a stable selection plateau.

## Two-stage paths

{pathlines}

Original-then-balanced is the most reproducible path among tested paths. Nevertheless, because canonical balanced-only reaches the gate, pretraining is an optimizer-path stabilizer rather than a required representation stage under the decision rules.

## Parameter and layer analysis

- Interpolation first passes after P3 at lambda {first_interp}; improvement is continuous, with stop-recovery MSE falling from 0.00103521 at lambda 0 to 0.00099824 at lambda 0.7.
- There is no intervening loss barrier on canonical-to-old20 linear interpolation; worst-group loss decreases from the canonical endpoint.
{layerlines}

The old trunk with canonical head passes, while canonical trunk with old head fails. Old first layer alone also passes. The useful warm-start effect is therefore primarily feature/trunk initialization, not a special output head.

## Latent and gradient

- Layer-3 start/stop AUROC: canonical {lat3['canonical_parent']['linear_probe_auroc']:.6f}, formal R1-1750 {lat3['formal_r1_step1750']['linear_probe_auroc']:.6f}, old20 {lat3['old_step20000']['linear_probe_auroc']:.6f}, old20+P3 {lat3['old_step20000_p3_final']['linear_probe_auroc']:.6f}.
- Layer-3 centroid distance grows from {lat3['canonical_parent']['centroid_distance']:.4f} at canonical to {lat3['old_step20000']['centroid_distance']:.4f} at old20.
- Exact-zero start and steady-stop gradients are antagonistic; cosine is -0.706 at canonical, -0.977 at R1-1750, -0.863 at old20, and -0.989 after P3. Training succeeds by better feature separation and balancing, not by eliminating this local gradient conflict.

## Optimizer state

{optlines}

Fresh Adam, old Adam state, and zero-moment/old-step-counter variants all pass from old20. The canonical PPO optimizer cannot be strictly loaded because its parameter-group structure differs. Optimizer state is not primary.

## Warm-start validity

`{warm['status']}`. The step-20,000 actor is traceable to the canonical parent, uses the resolved immutable dataset and current label contract, has saved optimizer/objective/seed information, and reproduces P3 exactly. The preferred next action remains a formal long-horizon balanced run from canonical because it has stronger end-to-end provenance.

## Protection and current artifact

- Dataset, labels, split, manifests, existing checkpoints, and optimizers: unchanged.
- New persistent policy checkpoint: 0.
- Closed-loop evaluation / DAgger / promotion: 0 / 0 / 0.
- Canonical parent remains W1B-R2 iteration 200. Candidate students remain diagnostic only.
"""
REPORT.write_text(report,encoding="utf-8")

ps1='''$python = "C:\\Users\\user\\workspace\\IsaacLab\\env_isaaclab\\Scripts\\python.exe"\n& $python experiments\\isaaclab\\exp_013_unitree_g1_single_policy_omnidirectional_locomotion\\scripts\\diagnose_w2_p1_d3_initialization_gap.py\n# If interruption occurs after interpolation, resume analysis-only tail:\n& $python experiments\\isaaclab\\exp_013_unitree_g1_single_policy_omnidirectional_locomotion\\scripts\\finish_w2_p1_d3_initialization_gap.py\n& $python experiments\\isaaclab\\exp_013_unitree_g1_single_policy_omnidirectional_locomotion\\scripts\\finalize_w2_p1_d3_initialization_gap.py\n'''
(OUT/"reproduction_commands.ps1").write_text(ps1,encoding="utf-8")
print(json.dumps({"protected_unchanged":not changed,"report":str(REPORT.relative_to(REPO)),"classification":"CANONICAL_BALANCED_TRAINING_TOO_SHORT"}))
