# EXP-012 Stage 2P — Anchor-aware actor optimizer-moment preflight

## Outcome

Classification: `ACTOR_MOMENT_ADAPTATION_NO_EFFECT`.

The Stage 2K integration materially changed every inherited mean-network layer,
and the gait column plus two conditioned std endpoints have new semantics. The
imported RUN moments are therefore shape-compatible but not semantically current.
Nevertheless, attenuating or resetting those actor moments did not solve five-update
endpoint retention.

Across the inherited mean network, 99.40% of coordinates changed, with relative
L2 difference 0.496 and parameter cosine 0.893. The imported first moment was
nearly orthogonal to the initial PPO gradient (cosine 0.006) and opposed the
iteration-2 anchor gradient (cosine -0.114), confirming the semantic-age concern
without confirming it as the causal retention bottleneck.

## Branch comparison

M100 final KL was WALK 0.03838, RUN-1.2
0.02524, RUN-2.4 0.02834, and RUN-2.6
0.02879. First-moment zero gave the best WALK result,
0.03626, only a 5.5%
reduction, while RUN-2.4 and RUN-2.6 rose above 0.03.

No branch increased effective Adam-to-anchor alignment materially. Full actor
moment reset did not overshoot—the first update norm remained within the M100
bound—but it also did not retain all endpoints. All branches were finite and stayed
inside mean rollout-KL, clip, critic-gradient, value-loss, and fixed-LR gates.
M000 produced a conservative maximum per-sample KL of 0.3168 and is therefore
marked numerically ineligible; the other branches remained below 0.20.

Closed-loop endpoint/toggle evaluation was fail-closed after the analytic endpoint
KL gate failed in every branch. No temporary branch was promoted and no
fresh-process reproduction was run.

## Interpretation

The Stage 2O correlation between imported Adam history and anchor suppression does
not survive the causal intervention: removing first moments or all actor moments
does not restore semantic retention. The soft-anchor plus Adam-adaptation route
should be closed. The current best artifact remains the untouched Stage 2N initial
checkpoint.

## Protection

All branches were temporary, at most five updates, and discarded. No formal
checkpoint, optimizer, reward, curriculum, network, core package, or production
policy was modified. No remote push was performed.
