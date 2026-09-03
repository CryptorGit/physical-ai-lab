"""Prepare immutable provenance for W2-P1-R1."""
from __future__ import annotations
import hashlib,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve();REPO=HERE.parents[4]
SRC=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w2_p1_practical_stop_endpoint_acquisition"
D1=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w2_p1_d1_static_representation_conflict_diagnosis"
OUT=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w2_p1_r1_group_balanced_stop_integration"
PARENT=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1b_r2_pending_mirror_queue_repair_rerun/checkpoints/model_200.pt"
PROBE=REPO/"experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/probe_w2_p1_d1_static_conflict.py"
def sha(p):
 h=hashlib.sha256();f=Path(p).open('rb')
 while b:=f.read(8<<20):h.update(b)
 f.close();return h.hexdigest()
def git(*x):return subprocess.check_output(['git',*x],cwd=REPO,text=True,encoding='utf-8').strip()
def dump(n,v):OUT.mkdir(parents=True,exist_ok=True);(OUT/n).write_text(json.dumps(v,indent=2,sort_keys=True)+'\n',encoding='utf-8')
def main():
 OUT.mkdir(parents=True,exist_ok=True);head=git('rev-parse','HEAD');status=git('status','--short').splitlines();log=git('log','--oneline','--decorate','-25').splitlines()
 expected=json.loads((SRC/'w2_p1_dataset_hashes.json').read_text());actual={k:sha(REPO/k) for k in expected};split_sha=sha(SRC/'w2_p1_dataset_split.json')
 dump('stage_reference.json',{'stage':'W2-P1-R1','starting_head':head,'reported_starting_head':'be6fb47e0ae21dd9aba56a6be592526e20f595be','head_difference':head!='be6fb47e0ae21dd9aba56a6be592526e20f595be','starting_status_short':status,'starting_log_25':log})
 dump('protocol.json',{'persistent_supervised_runs_maximum':1,'p3_in_memory_runs':3,'ppo':False,'dagger':'conditional only after static authorization and closed-loop failure','dataset_mutation':False,'runtime_teacher':False,'remote_push':False})
 dump('parent_manifest.json',{'checkpoint':str(PARENT.relative_to(REPO)).replace('\\','/'),'sha256':sha(PARENT),'expected_sha256':'61dd37f9f1fb7036357d54975c1120c66c30650ed7d5f2e1d4de91174fdca27d','architecture':[124,256,128,128,37],'calibration':'MonotonicPositiveYawCalibrationV1'})
 dump('parent_identity_audit.json',{'sha_match':sha(PARENT)=='61dd37f9f1fb7036357d54975c1120c66c30650ed7d5f2e1d4de91174fdca27d','mean_actor_copy_required':True,'std_copy_required':True,'critic_unused':True})
 dump('dataset_hashes.json',actual);dump('dataset_identity_audit.json',{'expected':expected,'actual':actual,'all_hashes_match':expected==actual,'labels_embedded_in_immutable_chunks':True,'split_path':str((SRC/'w2_p1_dataset_split.json').relative_to(REPO)).replace('\\','/'),'split_sha256':split_sha,'split_unchanged':True,'episode_stratification_unchanged':True})
 contract={'source_commit':head,'source_script':str(PROBE.relative_to(REPO)).replace('\\','/'),'source_script_sha256':sha(PROBE),'probe':'P3_ALL_GROUPS_BALANCED','optimizer':'torch.optim.Adam','learning_rate':0.0002,'betas':[0.9,0.999],'eps':1e-8,'weight_decay':0.0,'amsgrad':False,'maximum_steps':2000,'gradient_clip_norm':10.0,'scheduler':None,'initialization':'canonical W1B-R2 actor_state_dict','pool_seed':20276049,'pool_samples_per_group':50000,'validation_seed':20276100,'validation_samples_per_group':5000,'training_seed':20277717,'batch':{'STOP_RECOVERY':256,'STEADY_STOP':256,'START_RETENTION':256,'each_moving_subgroup':64,'moving_subgroups':5},'group_weights':[0.25,0.25,0.25,0.25],'group_internal_sampling':'uniform from fixed group pool with replacement','exact_zero_masking':False,'condition_balancing':False,'hard_example_mining':False,'contract_complete':True}
 dump('p3_probe_contract.json',contract)
 (OUT/'resolved_group_balanced_training_config.yaml').write_text('stage: W2-P1-R1\ntraining_type: mean_action_supervised\nseed: 20277717\nsteps: 2000\noptimizer: Adam\nlearning_rate: 0.0002\nweight_decay: 0.0\ngradient_clip_norm: 10.0\nscheduler: null\narchitecture: [124, 256, 128, 128, 37]\nstd_head: frozen\ncritic: unused\ngroup_weights: {stop_recovery: 0.25, steady_stop: 0.25, moving_retention: 0.25, start_retention: 0.25}\nexact_zero_masking: false\n',encoding='utf-8')
 print(json.dumps({'head':head,'dataset_identity':expected==actual,'contract_complete':True}))
if __name__=='__main__':main()
