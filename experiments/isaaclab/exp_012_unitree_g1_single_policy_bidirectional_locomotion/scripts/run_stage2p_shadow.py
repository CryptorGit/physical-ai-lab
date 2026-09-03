"""Run one non-persistent Stage 2P actor-moment adaptation branch."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve()
CORE = SCRIPT.with_name("run_stage2o_shadow.py")

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--branch", required=True)
parser.add_argument(
    "--moment-mode",
    choices=("keep", "attenuate_first", "zero_first", "reset_actor"),
    required=True,
)
parser.add_argument("--layer-audit", action="store_true")
parser.add_argument("--iterations", type=int, default=5)
known, remaining = parser.parse_known_args()
sys.argv = [
    sys.argv[0],
    "--branch", known.branch,
    "--beta", "0.10",
    "--iterations", str(known.iterations),
    "--fixed-lr",
    *remaining,
]

program = CORE.read_text(encoding="utf-8")
program = program.replace(
    "stage2o_endpoint_anchor_accumulation_diagnosis",
    "stage2p_anchor_aware_optimizer_moment_preflight",
)

moment_code = f'''
        actor_named_parameters = list(runner.alg.actor.named_parameters())
        critic_named_parameters = list(runner.alg.critic.named_parameters())
        def moment_digest(named_parameters, field):
            digest = hashlib.sha256()
            norm_squared = 0.0
            steps = []
            for parameter_name, parameter in named_parameters:
                state = runner.alg.optimizer.state[parameter]
                tensor = state[field]
                digest.update(parameter_name.encode("utf-8"))
                digest.update(tensor.detach().cpu().numpy().tobytes())
                norm_squared += float(tensor.square().sum())
                steps.append(int(state["step"].item()))
            return digest.hexdigest(), math.sqrt(norm_squared), min(steps), max(steps)
        actor_avg_before = moment_digest(actor_named_parameters, "exp_avg")
        actor_sq_before = moment_digest(actor_named_parameters, "exp_avg_sq")
        critic_avg_before = moment_digest(critic_named_parameters, "exp_avg")
        critic_sq_before = moment_digest(critic_named_parameters, "exp_avg_sq")
        actor_parameter_hash = hashlib.sha256(b"".join(
            name.encode("utf-8") + parameter.detach().cpu().numpy().tobytes()
            for name, parameter in actor_named_parameters)).hexdigest()
        critic_parameter_hash = hashlib.sha256(b"".join(
            name.encode("utf-8") + parameter.detach().cpu().numpy().tobytes()
            for name, parameter in critic_named_parameters)).hexdigest()
        for parameter_name, parameter in actor_named_parameters:
            state = runner.alg.optimizer.state[parameter]
            if "{known.moment_mode}" == "attenuate_first":
                state["exp_avg"].mul_(.25)
            elif "{known.moment_mode}" == "zero_first":
                state["exp_avg"].zero_()
            elif "{known.moment_mode}" == "reset_actor":
                state["exp_avg"].zero_()
                state["exp_avg_sq"].zero_()
                state["step"].zero_()
        actor_avg_after = moment_digest(actor_named_parameters, "exp_avg")
        actor_sq_after = moment_digest(actor_named_parameters, "exp_avg_sq")
        critic_avg_after = moment_digest(critic_named_parameters, "exp_avg")
        critic_sq_after = moment_digest(critic_named_parameters, "exp_avg_sq")
        dump(f"{{args.tag}}_moment_initialization.json", {{
            "branch": args.tag,
            "moment_mode": "{known.moment_mode}",
            "actor_parameter_hash": actor_parameter_hash,
            "critic_parameter_hash": critic_parameter_hash,
            "actor_exp_avg_before": actor_avg_before,
            "actor_exp_avg_after": actor_avg_after,
            "actor_exp_avg_sq_before": actor_sq_before,
            "actor_exp_avg_sq_after": actor_sq_after,
            "critic_exp_avg_before": critic_avg_before,
            "critic_exp_avg_after": critic_avg_after,
            "critic_exp_avg_sq_before": critic_sq_before,
            "critic_exp_avg_sq_after": critic_sq_after,
            "critic_moments_bitwise_unchanged": (
                critic_avg_before == critic_avg_after and critic_sq_before == critic_sq_after),
            "optimizer_lr": runner.alg.optimizer.param_groups[0]["lr"],
            "runtime_lr": runner.alg.learning_rate,
            "scheduler_lr": runner.alg.optimizer.param_groups[0]["lr"],
        }})
'''

injected = f'''
moment_needle = 'runner.alg.desired_kl = None'
if moment_needle not in source:
    raise RuntimeError("Stage 2P moment injection point not found")
source = source.replace(moment_needle, moment_needle + {moment_code!r}, 1)
'''
layer_audit_program = ""
if known.layer_audit:
    layer_audit_program = r"""
source = source.replace(
    'gradient_audits = []\n        endpoint_pressure_rows = []\n        critic_rows = []',
    'gradient_audits = []\n        endpoint_pressure_rows = []\n        critic_rows = []\n        layer_alignment_rows = []',
)
source = source.replace(
    '''                ppo_vector = torch.cat([
                    (p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
                    for p in runner.alg.actor.parameters()
                ])
                runner.alg.optimizer.zero_grad()
                anchor_probe, _ = runner.alg._anchor_loss()''',
    '''                ppo_vector = torch.cat([
                    (p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
                    for p in runner.alg.actor.parameters()
                ])
                named_ppo_gradient = {
                    name: (parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)).detach().clone()
                    for name, parameter in runner.alg.actor.named_parameters()
                }
                runner.alg.optimizer.zero_grad()
                anchor_probe, _ = runner.alg._anchor_loss()''',
)
source = source.replace(
    '''                anchor_vector = torch.cat([
                    (p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
                    for p in runner.alg.actor.parameters()
                ])
                runner.alg.optimizer.zero_grad()''',
    '''                anchor_vector = torch.cat([
                    (p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
                    for p in runner.alg.actor.parameters()
                ])
                named_anchor_gradient = {
                    name: (parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)).detach().clone()
                    for name, parameter in runner.alg.actor.named_parameters()
                }
                runner.alg.optimizer.zero_grad()''',
)
source = source.replace(
    '''                parameter_before = torch.cat([p.detach().flatten().clone() for p in runner.alg.actor.parameters()])
            losses = runner.alg.update()''',
    '''                parameter_before = torch.cat([p.detach().flatten().clone() for p in runner.alg.actor.parameters()])
                named_parameter_before = {
                    name: parameter.detach().clone()
                    for name, parameter in runner.alg.actor.named_parameters()
                }
                named_first_moment_before = {
                    name: runner.alg.optimizer.state[parameter]["exp_avg"].detach().clone()
                    for name, parameter in runner.alg.actor.named_parameters()
                }
                named_second_moment_before = {
                    name: runner.alg.optimizer.state[parameter]["exp_avg_sq"].detach().clone()
                    for name, parameter in runner.alg.actor.named_parameters()
                }
            losses = runner.alg.update()''',
)
source = source.replace(
    '''            gradient_audits.append(gradient_audit)
            with torch.no_grad():''',
    '''            gradient_audits.append(gradient_audit)
            for parameter_name, parameter in runner.alg.actor.named_parameters():
                ppo_part = named_ppo_gradient[parameter_name].flatten()
                anchor_part = named_anchor_gradient[parameter_name].flatten()
                update_part = (named_parameter_before[parameter_name] - parameter.detach()).flatten()
                imported_first_part = named_first_moment_before[parameter_name].flatten()
                imported_second_part = named_second_moment_before[parameter_name].flatten()
                state_part = runner.alg.optimizer.state[parameter]
                first_moment_part = state_part["exp_avg"].detach().flatten()
                second_moment_part = state_part["exp_avg_sq"].detach().flatten()
                def safe_cosine(left, right):
                    if float(torch.linalg.vector_norm(left)) == 0 or float(torch.linalg.vector_norm(right)) == 0:
                        return 0.0
                    return float(torch.nn.functional.cosine_similarity(left, right, dim=0))
                layer_alignment_rows.append({
                    "iteration": iteration,
                    "parameter": parameter_name,
                    "ppo_gradient_norm": float(torch.linalg.vector_norm(ppo_part)),
                    "anchor_gradient_norm": float(torch.linalg.vector_norm(anchor_part)),
                    "effective_update_norm": float(torch.linalg.vector_norm(update_part)),
                    "update_vs_ppo_cosine": safe_cosine(update_part, ppo_part),
                    "update_vs_anchor_cosine": safe_cosine(update_part, anchor_part),
                    "ppo_vs_anchor_cosine": safe_cosine(ppo_part, anchor_part),
                    "imported_first_moment_norm": float(torch.linalg.vector_norm(imported_first_part)),
                    "imported_second_moment_norm": float(torch.linalg.vector_norm(imported_second_part)),
                    "imported_first_vs_ppo_cosine": safe_cosine(imported_first_part, ppo_part),
                    "imported_first_vs_anchor_cosine": safe_cosine(imported_first_part, anchor_part),
                    "post_update_first_moment_norm": float(torch.linalg.vector_norm(first_moment_part)),
                    "post_update_second_moment_norm": float(torch.linalg.vector_norm(second_moment_part)),
                })
            with torch.no_grad():''',
)
source = source.replace(
    '''        if critic_rows:
            write_csv(prefix + "critic_diagnosis.csv", critic_rows)''',
    '''        if critic_rows:
            write_csv(prefix + "critic_diagnosis.csv", critic_rows)
        if layer_alignment_rows:
            write_csv(prefix + "layer_alignment.csv", layer_alignment_rows)''',
)
"""
diagnostic_program = r"""
source = source.replace(
    '''                parameter_before = torch.cat([p.detach().flatten().clone() for p in runner.alg.actor.parameters()])''',
    '''                parameter_before = torch.cat([p.detach().flatten().clone() for p in runner.alg.actor.parameters()])
                critic_parameter_before = torch.cat([
                    p.detach().flatten().clone() for p in runner.alg.critic.parameters()
                ])
                runner.alg.optimizer.zero_grad()
                critic_values_probe = runner.alg.critic(observations).squeeze(-1)
                critic_returns_probe = storage.returns.flatten(0, 1).squeeze(-1)
                critic_loss_probe = (critic_values_probe - critic_returns_probe).square().mean()
                critic_loss_probe.backward()
                critic_gradient_norm_probe = float(torch.sqrt(sum(
                    (p.grad if p.grad is not None else torch.zeros_like(p)).square().sum()
                    for p in runner.alg.critic.parameters()
                )))
                runner.alg.optimizer.zero_grad()
                return_variance_probe = torch.var(critic_returns_probe)
                critic_explained_variance_probe = float(
                    1 - torch.var(critic_returns_probe - critic_values_probe.detach())
                    / return_variance_probe.clamp_min(1e-12))''',
    1,
)
source = source.replace(
    '''            parameter_after = torch.cat([p.detach().flatten() for p in runner.alg.actor.parameters()])
            adam_direction = parameter_before - parameter_after''',
    '''            parameter_after = torch.cat([p.detach().flatten() for p in runner.alg.actor.parameters()])
            critic_parameter_after = torch.cat([
                p.detach().flatten() for p in runner.alg.critic.parameters()
            ])
            critic_parameter_change_norm_probe = float(torch.linalg.vector_norm(
                critic_parameter_after - critic_parameter_before))
            adam_direction = parameter_before - parameter_after''',
    1,
)
source = source.replace(
    '''                "adam_step_norm": gradient_audit["adam_step_norm"],
            }''',
    '''                "adam_step_norm": gradient_audit["adam_step_norm"],
                "critic_parameter_change_norm": critic_parameter_change_norm_probe,
                "critic_gradient_norm": critic_gradient_norm_probe,
                "critic_explained_variance": critic_explained_variance_probe,
                "parameters_finite": all(bool(torch.isfinite(p).all()) for p in
                                         list(runner.alg.actor.parameters()) + list(runner.alg.critic.parameters())),
                "nan_inf_count": sum(int((~torch.isfinite(p)).sum()) for p in
                                     list(runner.alg.actor.parameters()) + list(runner.alg.critic.parameters())),
            }''',
    1,
)
"""
program = program.replace(
    'code = compile(source, str(SCRIPT), "exec")',
    injected + "\n" + layer_audit_program + "\n" + diagnostic_program
    + '\ncode = compile(source, str(SCRIPT), "exec")',
)

exec(compile(program, str(SCRIPT), "exec"), {"__name__": "__main__", "__file__": str(SCRIPT)})
