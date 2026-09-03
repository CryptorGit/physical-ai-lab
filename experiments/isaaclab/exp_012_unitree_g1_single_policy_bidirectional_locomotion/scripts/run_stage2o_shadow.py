"""Run one non-persistent Stage 2O five-update shadow branch.

This launcher reuses the audited Stage 2N harness by compiling a diagnostic-only
copy in memory. It never saves model or optimizer checkpoints.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve()
SOURCE = SCRIPT.with_name("run_stage2n_retention.py")

outer = argparse.ArgumentParser(add_help=False)
outer.add_argument("--branch", required=True)
outer.add_argument("--beta", type=float, required=True)
outer.add_argument("--fixed-lr", action="store_true")
outer.add_argument("--iterations", type=int, default=5)
outer.add_argument("--pressure-audit", action="store_true")
outer.add_argument("--current-state-anchor", action="store_true")
known, remaining = outer.parse_known_args()
sys.argv = [
    sys.argv[0], "--mode", "train", "--beta", str(known.beta),
    "--iterations", str(known.iterations), "--tag", known.branch, *remaining,
]

source = SOURCE.read_text(encoding="utf-8")
source = source.replace(
    'OUT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2n_gait_conditioned_ppo_retention_preflight"',
    'OUT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2o_endpoint_anchor_accumulation_diagnosis"',
)
source = source.replace(
    'anchor_payload = torch.load(OUT / "raw/endpoint_anchor.pt", map_location=runner.device, weights_only=False)',
    'anchor_payload = torch.load(REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2n_gait_conditioned_ppo_retention_preflight/raw/endpoint_anchor.pt", map_location=runner.device, weights_only=False)',
)
source = source.replace(
    'save_model(runner, parent_dir / "model_initial.pt", 0, args.beta)',
    '# Stage 2O: persistent model save prohibited',
)
source = source.replace(
    'if iteration in checkpoints and not args.tag:',
    'if False:',
)
source = source.replace(
    'if iteration <= 5 and (',
    'if False and (',
)
source = source.replace(
    'gradient_audit = None',
    'gradient_audits = []\n        endpoint_pressure_rows = []\n        critic_rows = []',
)
source = source.replace(
    'if iteration == 1:\n                rng = torch.random.get_rng_state()',
    'if True:\n                rng = torch.random.get_rng_state()',
)
source = source.replace(
    '                    "cap_pass": args.beta * anchor_norm / max(ppo_norm, 1e-30) <= .25,\n                }\n            losses = runner.alg.update()',
    '''                    "cap_pass": args.beta * anchor_norm / max(ppo_norm, 1e-30) <= .50,
                    "iteration": iteration,
                }
                parameter_before = torch.cat([p.detach().flatten().clone() for p in runner.alg.actor.parameters()])
            losses = runner.alg.update()
            parameter_after = torch.cat([p.detach().flatten() for p in runner.alg.actor.parameters()])
            adam_direction = parameter_before - parameter_after
            combined_vector = ppo_vector + args.beta * anchor_vector
            gradient_audit.update({
                "adam_vs_ppo_cosine": float(torch.nn.functional.cosine_similarity(adam_direction, ppo_vector, dim=0)),
                "adam_vs_anchor_cosine": float(torch.nn.functional.cosine_similarity(adam_direction, anchor_vector, dim=0))
                    if anchor_norm > 0 else 0.0,
                "adam_vs_combined_cosine": float(torch.nn.functional.cosine_similarity(adam_direction, combined_vector, dim=0)),
                "adam_step_norm": float(torch.linalg.vector_norm(adam_direction)),
                "actor_parameter_change_norm": float(torch.linalg.vector_norm(parameter_after - parameter_before)),
            })
            gradient_audits.append(gradient_audit)''',
)
source = source.replace(
    'anchor_kl = exact_anchor_kl(runner.alg.actor, reference, anchor_payload)\n            row = {',
    '''anchor_kl = exact_anchor_kl(runner.alg.actor, reference, anchor_payload)
            current_kl = {}
            current_reverse = {}
            current_mean = {}
            current_std = {}
            with torch.no_grad():
                reference(observations, stochastic_output=True)
                ref_mean, ref_std = (value.clone() for value in reference.output_distribution_params)
                runner.alg.actor(observations, stochastic_output=True)
                cur_mean, cur_std = runner.alg.actor.output_distribution_params
                policy_obs = observations["policy"]
                speed = policy_obs[:, 9]
                gait = policy_obs[:, -1]
                masks = {
                    "walk_1p2": (gait < .05) & ((speed - 1.2).abs() < .08),
                    "run_1p2": (gait > .95) & ((speed - 1.2).abs() < .08),
                    "run_2p4": (gait > .95) & ((speed - 2.4).abs() < .08),
                    "run_2p6": (gait > .95) & ((speed - 2.6).abs() < .08),
                }
                forward_mean = .5 * (((ref_mean - cur_mean) / cur_std).square()).sum(-1)
                forward_std = (torch.log(cur_std / ref_std) + .5 * (ref_std / cur_std).square() - .5).sum(-1)
                reverse_mean = .5 * (((cur_mean - ref_mean) / ref_std).square()).sum(-1)
                reverse_std = (torch.log(ref_std / cur_std) + .5 * (cur_std / ref_std).square() - .5).sum(-1)
                raw_dir = OUT / "raw"
                raw_dir.mkdir(parents=True, exist_ok=True)
                state_payload = {}
                for name, mask in masks.items():
                    if mask.any():
                        current_mean[name] = float(forward_mean[mask].mean())
                        current_std[name] = float(forward_std[mask].mean())
                        current_kl[name] = current_mean[name] + current_std[name]
                        current_reverse[name] = float((reverse_mean[mask] + reverse_std[mask]).mean())
                        ids = torch.nonzero(mask, as_tuple=False).flatten()[::max(1, int(mask.sum()) // 2048)][:2048]
                        state_payload[name] = policy_obs[ids].detach().cpu()
                torch.save(state_payload, raw_dir / f"{args.tag}_iter{iteration}_states.pt")
            row = {''',
)
source = source.replace(
    '**{f"anchor_kl_{key}": value for key, value in anchor_kl.items()},\n            }',
    '''**{f"anchor_kl_{key}": value for key, value in anchor_kl.items()},
                **{f"current_kl_{key}": value for key, value in current_kl.items()},
                **{f"current_reverse_kl_{key}": value for key, value in current_reverse.items()},
                **{f"current_mean_kl_{key}": value for key, value in current_mean.items()},
                **{f"current_std_kl_{key}": value for key, value in current_std.items()},
                "ppo_gradient_norm": gradient_audit["ppo_gradient_norm"],
                "anchor_gradient_norm": gradient_audit["anchor_gradient_norm"],
                "effective_anchor_ppo_ratio": gradient_audit["effective_anchor_ppo_ratio"],
                "ppo_anchor_cosine": gradient_audit["gradient_cosine"],
                "adam_vs_ppo_cosine": gradient_audit["adam_vs_ppo_cosine"],
                "adam_vs_anchor_cosine": gradient_audit["adam_vs_anchor_cosine"],
                "adam_vs_combined_cosine": gradient_audit["adam_vs_combined_cosine"],
                "adam_step_norm": gradient_audit["adam_step_norm"],
            }''',
)
source = source.replace(
    'dump(prefix + "gradient_audit.json", gradient_audit)',
    '''dump(prefix + "gradient_audit.json", gradient_audits)
        if endpoint_pressure_rows:
            write_csv(prefix + "endpoint_pressure.csv", endpoint_pressure_rows)
        if critic_rows:
            write_csv(prefix + "critic_diagnosis.csv", critic_rows)''',
)
source = source.replace(
    'if args.tag:\n            save_model(runner, OUT / f"shadow_{args.tag}.pt", len(curves), args.beta)',
    'if False:  # Stage 2O persistent checkpoint prohibited\n            pass',
)
if known.fixed_lr:
    source = source.replace(
        'runner.alg.configure_anchor(anchor, reference, args.beta)',
        'runner.alg.configure_anchor(anchor, reference, args.beta)\n        runner.alg.schedule = "fixed"\n        runner.alg.desired_kl = None',
    )
if known.current_state_anchor:
    source = source.replace(
        '''            actions = storage.actions.flatten(0, 1)
            old_log = storage.actions_log_prob.flatten(0, 1).squeeze(-1)''',
        '''            actions = storage.actions.flatten(0, 1)
            old_log = storage.actions_log_prob.flatten(0, 1).squeeze(-1)
            current_policy_anchor = observations["policy"].detach()
            current_speed_anchor = current_policy_anchor[:, 9]
            current_gait_anchor = current_policy_anchor[:, -1]
            current_anchor_masks = (
                (current_gait_anchor < .05) & ((current_speed_anchor - 1.2).abs() < .08),
                (current_gait_anchor > .95) & ((current_speed_anchor - 1.2).abs() < .08),
                (current_gait_anchor > .95) & ((current_speed_anchor - 2.4).abs() < .08),
                (current_gait_anchor > .95) & ((current_speed_anchor - 2.6).abs() < .08),
            )
            current_anchor_obs_parts = []
            current_anchor_id_parts = []
            for current_anchor_id, current_anchor_mask in enumerate(current_anchor_masks):
                current_anchor_ids = torch.nonzero(current_anchor_mask, as_tuple=False).flatten()
                if current_anchor_ids.numel() < 512:
                    raise RuntimeError(f"current-state anchor endpoint {current_anchor_id} has only {current_anchor_ids.numel()} samples")
                current_anchor_obs_parts.append(current_policy_anchor[current_anchor_ids])
                current_anchor_id_parts.append(torch.full(
                    (current_anchor_ids.numel(),), current_anchor_id,
                    device=runner.device, dtype=torch.long))
            runner.alg.anchor_observations = TensorDict(
                {
                    "policy": torch.cat(current_anchor_obs_parts),
                    "endpoint_id": torch.cat(current_anchor_id_parts),
                },
                batch_size=[sum(part.shape[0] for part in current_anchor_obs_parts)],
                device=runner.device,
            )''',
    )
if known.pressure_audit:
    source = source.replace(
        '''                runner.alg.optimizer.zero_grad()
                torch.random.set_rng_state(rng)
                ppo_norm = float(torch.linalg.vector_norm(ppo_vector))''',
        '''                runner.alg.optimizer.zero_grad()
                torch.random.set_rng_state(rng)
                policy_obs_probe = observations["policy"]
                speed_probe = policy_obs_probe[:, 9]
                gait_probe = policy_obs_probe[:, -1]
                endpoint_masks_probe = {
                    "walk_1p2": (gait_probe < .05) & ((speed_probe - 1.2).abs() < .08),
                    "run_1p2": (gait_probe > .95) & ((speed_probe - 1.2).abs() < .08),
                    "run_2p4": (gait_probe > .95) & ((speed_probe - 2.4).abs() < .08),
                    "run_2p6": (gait_probe > .95) & ((speed_probe - 2.6).abs() < .08),
                }
                endpoint_vectors_probe = {}
                anchor_norm_probe = float(torch.linalg.vector_norm(anchor_vector))
                for endpoint_name_probe, endpoint_mask_probe in endpoint_masks_probe.items():
                    if not endpoint_mask_probe.any():
                        continue
                    runner.alg.optimizer.zero_grad()
                    runner.alg.actor(observations, stochastic_output=True)
                    endpoint_log_probe = runner.alg.actor.get_output_log_prob(actions)
                    endpoint_loss_probe = -(storage.advantages.flatten(0, 1).squeeze(-1)[endpoint_mask_probe]
                                            * torch.exp(endpoint_log_probe[endpoint_mask_probe]
                                                        - old_log[endpoint_mask_probe])).mean()
                    endpoint_loss_probe.backward()
                    endpoint_vector_probe = torch.cat([
                        (p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
                        for p in runner.alg.actor.parameters()
                    ]).detach().clone()
                    endpoint_vectors_probe[endpoint_name_probe] = endpoint_vector_probe
                    endpoint_norm_probe = float(torch.linalg.vector_norm(endpoint_vector_probe))
                    endpoint_pressure_rows.append({
                        "iteration": iteration,
                        "endpoint": endpoint_name_probe,
                        "sample_count": int(endpoint_mask_probe.sum()),
                        "gradient_norm": endpoint_norm_probe,
                        "cosine_to_combined_ppo": float(torch.nn.functional.cosine_similarity(
                            endpoint_vector_probe, ppo_vector, dim=0)),
                        "cosine_to_static_anchor": float(torch.nn.functional.cosine_similarity(
                            endpoint_vector_probe, anchor_vector, dim=0)) if anchor_norm_probe > 0 else 0.0,
                        "projection_on_static_anchor": float(torch.dot(
                            endpoint_vector_probe, anchor_vector)) if anchor_norm_probe > 0 else 0.0,
                    })
                    endpoint_values_probe = storage.values.flatten(0, 1).squeeze(-1)[endpoint_mask_probe]
                    endpoint_returns_probe = storage.returns.flatten(0, 1).squeeze(-1)[endpoint_mask_probe]
                    endpoint_adv_probe = storage.advantages.flatten(0, 1).squeeze(-1)[endpoint_mask_probe]
                    critic_rows.append({
                        "iteration": iteration,
                        "endpoint": endpoint_name_probe,
                        "sample_count": int(endpoint_mask_probe.sum()),
                        "value_mean": float(endpoint_values_probe.mean()),
                        "return_mean": float(endpoint_returns_probe.mean()),
                        "value_bias": float((endpoint_values_probe - endpoint_returns_probe).mean()),
                        "advantage_mean": float(endpoint_adv_probe.mean()),
                        "advantage_std": float(endpoint_adv_probe.std()),
                        "advantage_positive_fraction": float((endpoint_adv_probe > 0).float().mean()),
                    })
                if "run_1p2" in endpoint_vectors_probe and "walk_1p2" in endpoint_vectors_probe:
                    endpoint_pressure_rows.append({
                        "iteration": iteration,
                        "endpoint": "run_1p2_vs_walk_1p2",
                        "sample_count": 0,
                        "gradient_norm": 0.0,
                        "cosine_to_combined_ppo": float(torch.nn.functional.cosine_similarity(
                            endpoint_vectors_probe["run_1p2"], endpoint_vectors_probe["walk_1p2"], dim=0)),
                        "cosine_to_static_anchor": 0.0,
                        "projection_on_static_anchor": 0.0,
                    })
                runner.alg.optimizer.zero_grad()
                torch.random.set_rng_state(rng)
                ppo_norm = float(torch.linalg.vector_norm(ppo_vector))''',
    )

code = compile(source, str(SCRIPT), "exec")
exec(code, {"__name__": "__main__", "__file__": str(SCRIPT)})
