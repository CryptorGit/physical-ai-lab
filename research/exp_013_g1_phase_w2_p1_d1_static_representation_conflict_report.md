# exp_013 Phase W2-P1-D1 static representation conflict diagnosis

## Outcome

Main classification: `OPTIMIZATION_PATH_NOT_REPRESENTATION_LIMIT`.

The W2-P1 published held-out result was reproduced exactly: mean action MSE `0.00129128795`, p95 `0.00010496832`, cosine `0.99969745`. This is not a unit or weighting mismatch. The per-sample MSE is the mean over 37 action dimensions, and the original group mean is an unweighted mean over 10,000 episode-uniform/timestep-uniform samples drawn with replacement from accepted held-out episodes.

## Metric contract and heavy tail

The distribution is a two-component heavy tail. The top 1% contributes `53.85%` of total loss and the top 5% contributes `97.95%`. The exact-zero sample at the actor-switch/ramp boundary occurs once per held-out episode (`240` samples, `2.40%` of the full held-out episode population) and contributes `97.60%` of loss. Excluding only that boundary sample is diagnostic-only and changes mean MSE from `0.00128926` to `0.00003147`; no formal gate or dataset was changed.

## Checkpoint timeline

The parent step 0 reproduces W1B labels essentially bitwise (`1.62e-14` start MSE) but has no stop integration. The boundary tail appears by step 500 and remains through step 25,000. The lowest existing trained start mean is `0.00000000` at step `0`, still above `0.001`; no existing checkpoint passes every group. Better stop fitting and the exact-zero start tail therefore trade off throughout the original optimization path.

## Label routing and boundary timing

Routing matches the preregistered contract: stop recovery uses W1B before SW3 and exp_012 at/after SW3, steady stop uses exp_012, moving/start retention use W1B. At `t=3.0 s`, command scheduling, observation, and label calculation are aligned; there is no one-step buffer offset. The issue is semantic: runtime switches to W1B exactly when the minimum-jerk command is still bitwise zero and the observation's previous action is the stop teacher action. Thus the first start label asks for a W1B action at a stay-stopped input boundary.

## Exact and near conflicts

Across `1,773,566` unique input hashes there were `0` bitwise-identical 124D cross-group material conflicts, so `EXACT_ZERO_COMMAND_LABEL_CONFLICT` is not supported literally. In the normalized input neighborhood, however, every exact-zero start sample's nearest comparison is materially conflicting (`100.0%`), with mean label MSE `0.06996`; neighbors are predominantly steady-stop with the remainder stop-recovery. This is a near-zero manifold competition, not corrupted episodes: the top 100 reconstructed episodes are all `ZERO_COMMAND_BOUNDARY`, span all 24 direction/yaw conditions, and show no reset/padding/label-source mismatch.

## Conditions and joints

All 24 start conditions have nearly the same mean error (`0.00127881`–`0.00130336`), and their exact-zero samples account for 96.38–98.09% of each condition's loss. The tail is not rear-, direction-, or calibrated-positive-yaw-specific. Broad joint contribution at the boundary is dominated by elbow/hand/shoulder categories (`85.75%`), while hip/knee/ankle contribute `5.31%`. Joint names were not inferred beyond stored action index and broad diagnostic categories.

## State sufficiency, latent, and gradients

Over complete start trajectories the full observation separates start from steady stop well, but at the exact-zero boundary the linear full-124D probe is weak (AUROC `0.583`); a small nonlinear probe reaches AUROC `0.876`. Previous action alone is insufficient at that boundary (nonlinear AUROC `0.657`). Hidden layers nevertheless linearly separate ordinary steady-stop and start samples, so the trunk is not collapsed.

The selected checkpoint gradients confirm a localized conflict: steady-stop vs start-outlier cosine is `-0.8320`, stop-recovery vs start-outlier is `-0.7634`, while moving-retention vs normal-start is `0.9913`. The whole start group conflict is therefore driven by the exact-zero outlier component, not normal moving acquisition samples.

## Fixed temporary probes

All probes used in-memory clones and wrote no checkpoint. P1 start-only fits start but destroys stop groups. P2 start+steady leaves stop-recovery just above gate. P4 exact-zero exclusion leaves the formal original start population above gate, and P5 last-layer-only narrowly misses start (`0.0010057`). P6 original full-network also narrowly misses (`0.0010224`). Crucially, P3 all-groups-balanced reaches simultaneous static PASS without changing the architecture: stop-recovery `0.0009873`, steady-stop `0.0000384`, start `0.0009829`, and worst moving subgroup `0.0000280`. The margin is narrow and does not authorize a production student, but it establishes static representational feasibility.

## Interpretation and next action

The primary classification is optimization-path, not a hard 124D representational limit. Near-zero labels compete strongly, and the original top-level start weight (`10%`) plus checkpoint selection leaves the one-sample-per-episode tail underfit. A fixed group-balanced objective can fit all static groups with the unchanged network.

Next, and only next: **group-balanced supervised integration rerun from the canonical W1B-R2 parent**. This diagnosis does not itself authorize that run.

## Protection

All raw dataset/checkpoint hashes match their starting values. No dataset or label bytes changed; no persistent checkpoint, closed-loop rollout, DAgger round, PPO update, or canonical promotion occurred. Existing W2-P1 classification remains unchanged. Remote push was not performed.
