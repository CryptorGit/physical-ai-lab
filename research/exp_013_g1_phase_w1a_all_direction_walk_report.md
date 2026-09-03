# exp_013 Phase W1A 360-degree translation-only WALK report

## Decision

正式分類は `EXP013_W1A_ALL_DIRECTION_WALK_PASS_LOW_SPEED_ONLY`。selected checkpointはiteration 120、SHA-256 `b128f6b164d151b411eeaf2caf22edc1ea2a69e68fca9534e7d6a965ae4dbba9`。
次に実施する方式は **Phase W1A2: all-direction WALK speed-envelope expansion** の一つだけとする。

## Parent and training contract

Stage 2Q actor `66ca45753aa6175109bfea90b5a8d751c49888f04a69e9ce0a8ec963ac750698`（124→256→128→128→37）をbitwiseコピーした。criticはStage 2Nの互換124D criticを使用。
actor/critic optimizerはfresh Adam step 0、固定LR 1.5e-5。WALK stdは校正済み0.30倍でlog-stdを全iteration凍結した。
rewardは既存body-frame vector trackingを使用し、vx/vyの単位・frame・正負対称性監査に合格した。

Phase A 40、B 60、C 60、D 40の計200 iteration、1024 env × 24 step、合計4,915,200 interaction。
one-update preflightはexact KL 0.01354、all-step max KL 0.01571、clip 0.2000でPASS。iteration 1〜10 early guardもPASS。

## Formal low-speed matrix

0.3 m/sは **16/16 PASS**、0.6 m/sは **4/16 PASS**。
bestはS0.3_D000.0（MAE 0.056 m/s）、worstはS0.6_D180.0（MAE 0.271 m/s）。
forward 0.6は100%、forward 1.2は100%で保持した。

## Directional envelope

前進1.2 m/sはPASS。lateral 0.8、rear-diagonal 0.6、backward 0.6はformal gate未達。
したがって、低速360度の成立は確認できたが、0.6 m/s以上の全方向envelopeは未完成である。

## Continuous direction diagnostic

30秒×30 episodeでfall 0.0%、vector MAE 0.116 m/s、
direction error 12.2°、dangerous slip 10.0%。
dominant failureは4秒ごとのdirection切替直後の速度ベクトル遅れとslipである。この診断はformal W1A gate外。

## Safety and symmetry

正式low-speed 1,600 episodeでfall 0.00%、tilt 2.31%、
dangerous slip 0.88%、impact 0.00%、
long-dwell saturation 0.00%。
mirror MAE差平均は0.012 m/sでsymmetry gate PASS。

## RUN diagnostic

RUN 1.2/2.4およびWALK↔RUN各20 episodeを診断し、gait分類内訳は `{"RUN_1P2": {"PERIODIC_RUNNING": 20}, "RUN_2P4": {"PERIODIC_RUNNING": 20}, "RUN_TO_WALK": {"PERIODIC_RUNNING": 20}, "WALK_TO_RUN": {"PERIODIC_RUNNING": 20}}`。
これはRUNを学習・選択gate化したものではなく、W1A checkpointはfinal integrated policyではない。

## Protection

exp_005〜exp_012、exp_012 closure、exp_013 Stage 0、既存checkpoint/optimizer、robot asset、
physics、Isaac Lab/RSL-RL coreは変更していない。新規checkpointはW1A lineageのみ。remote pushは行っていない。
