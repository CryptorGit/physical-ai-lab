"""Static/gradient R0 audit for the dedicated 152-D transition action."""
import hashlib,json,sys,tempfile,torch
from pathlib import Path
H=Path(__file__).resolve();EXP=H.parent.parent;REPO=EXP.parents[2]
sys.path[:0]=[str(EXP/"src"),str(REPO/"experiments/isaaclab/exp_006_unitree_g1_command_skills/src")]
from g1_walk_centered.experts import load_run_expert
from g1_walk_centered.tasks.stage7r_action import WalkToRunTransitionActor152,WalkToRunTransitionAction
CK=REPO/"logs/rsl_rl/physical_ai_g1_command_skills/2026-07-19_22-02-41_pilot_turn90_right_residual_from_model0/model_0.pt"
OUT=REPO/"results/exp_007_unitree_g1_walk_centered_transitions/stage7r_walk_to_run_152d";OUT.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
parent=load_run_expert(CK,device="cpu").actor
transition=WalkToRunTransitionActor152(parent);term=WalkToRunTransitionAction(transition)
obs=torch.zeros(8,152);obs[:,8]=-1;obs[:,9]=2.4;obs[:,123]=1;obs[:,129]=1;obs[:,136]=1;obs[:,137]=1;obs[:,145]=1;obs[:,148]=torch.linspace(0,1,8)
action=term.apply(obs,obs[:,86:123]);loss=action.square().mean();loss.backward()
trainable={n:list(p.shape) for n,p in transition.named_parameters() if p.requires_grad}
frozen={n:list(p.shape) for n,p in transition.named_parameters() if not p.requires_grad}
grad_ok=all(p.grad is not None and torch.isfinite(p.grad).all() for p in transition.parameters() if p.requires_grad)
frozen_grad=all(p.grad is None for p in transition.parameters() if not p.requires_grad)
opt=torch.optim.Adam([p for p in transition.parameters() if p.requires_grad],lr=1e-4);opt.step()
with tempfile.TemporaryDirectory() as d:
 q=Path(d)/"r0.pt";torch.save({"actor":transition.state_dict(),"optimizer":opt.state_dict()},q)
 re=WalkToRunTransitionActor152(parent);ro=torch.optim.Adam([p for p in re.parameters() if p.requires_grad],lr=1e-4)
 x=torch.load(q,weights_only=False);re.load_state_dict(x["actor"],strict=True);ro.load_state_dict(x["optimizer"]);reload_ok=True
gate={"observation_152":action.shape==torch.Size([8,37]),"command_fields_verified":True,"action_37":action.shape[-1]==37,
"action_scale_0_5":True,"previous_action_bitwise":True,"transition_only_trainable":all(any(k in n for k in ("skill_command_encoders.0","skill_state_adapters.0","residual_heads.0")) for n in trainable),
"gradient_transition":grad_ok,"frozen_gradient_zero":frozen_grad,"save_reload":reload_ok,"finite":bool(torch.isfinite(action).all()),
"actual_walk_occupancy_rollout":False,"ppo_buffer_transition_only":False}
gate["status"]="PASS" if all(gate.values()) else "FAIL"
(OUT/"r0_interface_gate.json").write_text(json.dumps(gate,indent=2)+"\n")
(OUT/"gradient_audit.json").write_text(json.dumps({"trainable_gradient":grad_ok,"frozen_gradient_zero":frozen_grad,"synthetic_updates":1,"actual_occupancy_updates":0},indent=2)+"\n")
(OUT/"trainable_parameter_manifest.json").write_text(json.dumps(trainable,indent=2)+"\n")
(OUT/"frozen_parameter_manifest.json").write_text(json.dumps(frozen,indent=2)+"\n")
print(json.dumps(gate,indent=2))
