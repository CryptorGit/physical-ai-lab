# exp_013 Phase W2-P1-R1 resolved-manifest rerun

## Outcome

Classification: `EXP013_W2_P1_R1_STATIC_REPRODUCTION_FAIL`. Dataset identity and exact P3 reproduction passed. The one authorized persistent run completed 2,000 optimizer steps, but no saved checkpoint passed every held-out static group. Closed-loop evaluation and DAgger were therefore not authorized.

## Dataset and P3

The immutable v2 manifest was the sole dataset authority. Dataset, labels, split, order, conditions, and exact-zero samples were unchanged. Same-process and fresh-process P3 reproduced the D1 gate metrics, trace, and tensor hash `975f2cb165e48853f87d79cb93de83ed50954627b5b3a37f38c3b2bd6d4a159b` exactly. The executable D1 source initializes P3 from the prior selected W2-P1 student, despite the earlier R1 metadata describing the canonical parent. The formal persistent run correctly initialized from canonical W1B-R2 as required.

## Static result

All post-update checkpoints passed stop recovery, steady stop, and every moving subgroup after step 250. Start retention remained above the `0.001` MSE gate. The best post-update saved start result was `0.001176059828` at step `1750`. Step 2000 was `0.001183362911`. Consequently, no checkpoint was selected or promoted.

## Protection

The run created only its own student checkpoints. Existing chunks, labels, split, stale manifest, resolved manifest, checkpoints, optimizer artifacts, sampler, reward, physics, Isaac Lab/RSL-RL core, and canonical parent were not changed. Runtime teacher use, closed-loop rollout, DAgger, action blending, and remote push were zero.

## Next

One method only: diagnose the canonical-parent versus D1-selected-student initialization gap before any further integration training.
