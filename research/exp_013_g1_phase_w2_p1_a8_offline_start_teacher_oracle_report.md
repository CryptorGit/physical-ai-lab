# Phase W2-P1-A8 Offline Start-Teacher Oracle Authorization

## Scope and outcome

This stage performed read-only evaluation of the eleven saved A7-R2 checkpoints. It performed no PPO update, supervised student update, checkpoint creation, overlay generation, reward or physics change, or canonical promotion.

The validation-only optimization found a two-checkpoint cover (`update 10`, `update 150`) and froze `Exp013OfflineStartTeacherOracleV1` before held-out evaluation. The frozen map passed all 24 held-out start conditions and its full-episode positive control. Authorization nevertheless failed closed because the mapped checkpoints did not both retain the required pure-yaw/static capabilities, the command-neighborhood audit was partial, and neither rear-yaw condition admitted a passing candidate-takeover horizon through 32 control steps.

Main classification:

```text
EXP013_W2_P1_A8_MULTIPLE_FAILURES
```

## Candidate coverage and compact cover

Each saved A7-R2 checkpoint (`initial`, updates `1`, `10`, `20`, `45`, `75`, `100`, `120`, `130`, `140`, and `150`) was evaluated on 24 conditions with 300 deterministic validation episodes per condition under ReplayRecipeV2. Covered-condition counts were respectively:

```text
22, 21, 23, 20, 20, 22, 19, 19, 21, 20, 21
```

No single checkpoint covered all conditions. Eleven two-checkpoint covers existed. Registered tie-breaking selected updates `10` and `150`: the resulting per-condition validation minimum acquisition was 99.0%, aggregate acquisition 99.889%, and rear-pair minimum 99.0%.

The frozen map assigns 13 conditions to update 10 and 11 conditions to update 150. Its canonical semantic SHA-256 is:

```text
817b904cb0f52db345b42420d84378987190047aecb9d406bd2f45bf53c79f29
```

Checkpoint byte SHA-256 values:

```text
update 10:
3d6c54c7afcabdbf47e84a62dfc1d47190881f6be5140ea16f6d514a34adcf55

update 150:
2964a9eeebbadea8918340623d3b30695dd08f3ef1ee2184878904f9c27e2135
```

## Held-out oracle and positive control

The condition map was fixed before held-out evaluation and was not changed afterward. Across 24 conditions with 300 held-out episodes each:

```text
conditions passing: 24 / 24
minimum endpoint: 100.0%
minimum acquisition: 99.33%
rear -yaw acquisition: 100.0%
rear +yaw acquisition: 100.0%
aggregate fall/slip/impact/saturation: 0 / 0 / 0 / 0
held-out fallback: 0
```

The identical full-episode oracle positive control passed: aggregate endpoint 100.0%, aggregate acquisition 99.958%, minimum condition acquisition 99.33%, and fall/slip 0/0.

## Continuity and local neighborhood

There were 16 adjacent formal-condition checkpoint boundaries. On matched B0-B8 counterfactual observations, the maximum update10/update150 whole-body action L2 was 1.6595 and the minimum cosine was 0.99426; no configured continuity warning fired. The largest joint-group contribution was torso/arms (1.4577), followed by hands (0.5917), waist (0.4291), and legs (0.4041).

The local-neighborhood audit evaluated 21 boundary conditions × 9 perturbations (`direction ±5°`, `yaw ±0.03`) × 100 validation episodes. Twenty-seven of 189 points failed the diagnostic target. Minimum endpoint and acquisition were both 0%; maximum fall was 1%. This is evidence of formal-point overfit despite good action-space continuity.

## Pure-yaw and static retention

Update 10 retained all zero-yaw directions, both forward anchors, and 24/24 static moving-turn endpoints, but positive pure-yaw acquisition was only 52.67% (required 85%). Update 150 passed both pure-yaw starts, all zero-yaw directions, and 24/24 moving turns, but failed the forward 1.2 m/s endpoint (0%, required 95%). Therefore the cover does not satisfy the required teacher retention contract.

## Safe horizon

All requested horizons (`2, 4, 6, 8, 12, 16, 24, 32`) were evaluated with 200 held-out episodes per condition. Twenty-two of 24 conditions already passed at two steps, but rear `180°/-0.3` and `180°/+0.3` failed at every tested horizon after takeover by the A4 V2 candidate. Across horizons, aggregate endpoint was 99.73-100%, aggregate acquisition 92.71-93.27%, and aggregate fall 0-0.229%. The minimum condition acquisition remained 15.5-21.0%. No global safe teacher horizon exists.

## Interpretation and next decision

The existing checkpoints are complementary enough to form a strong full-episode offline oracle, but not an authorized overlay-label oracle under the complete contract. In particular, capability cannot be handed back safely to A4 within 32 steps for rear yaw, and the mapped teachers are not jointly valid on required pure-yaw/static anchors.

Per the terminal pivot rule, no A7-R4 or additional teacher PPO/search is authorized. The recommended next decision is an observation/history-contract change preflight; otherwise close single-actor stop/restart as unresolved.

## Protection

Existing experiments, datasets, labels, splits, manifests, overlays, checkpoints, optimizers, ReplayRecipeV1/V2, MaskedPPOV1, reward, and physics were unchanged. New policy checkpoint count, PPO updates, V3 overlays, student training, canonical promotion, and remote pushes were all zero.
