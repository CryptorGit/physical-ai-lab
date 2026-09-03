# exp_012 closure summary

## Final status

```text
Status:
CLOSED

Project-level classification:
EXP_012_CLOSED_WITH_SINGLE_POLICY_LOCOMOTION_SUCCESS_AND_STRICT_STAND_LIMITATION
```

このproject-level分類は既存stage classificationを変更しない。中心成果は、Isaac Lab上のUnitree G1で、一つのcheckpoint、一つのmean actor、一つのgait-conditioned Gaussian headにWALKとRUNを統合し、同一速度で歩容を選択し、双方向に遷移できたことである。未解決点はstrict static-contact STANDとcontinued PPO semantic retentionである。

## Final artifact

最終sequence artifactはStage 2Q selected checkpoint:

```text
SHA-256:
66ca45753aa6175109bfea90b5a8d751c49888f04a69e9ce0a8ec963ac750698

architecture:
124 → 256 → 128 → 128 → 37

runtime:
1 checkpoint
1 actor
1 gait-conditioned Gaussian head
0 teacher/expert/router/switch/blend
```

これはWALK 0.6/0.8/1.0/1.2、RUN 1.2/2.4/2.6、STAND→WALK、WALK→RUN、RUN acceleration/deceleration、RUN→WALKを各100%達成した。integrated sequenceではWALK/RUN/return-WALKを維持し、平均最終速度0.0547 m/sまで減速した。

## Best gait-core artifact

Stage 2N initial checkpoint:

```text
SHA-256:
04b43e5497bc35e2d00fa4476f9120f9439e0953283c69cf8ca1e9635dedd121

alpha_walk:
0.30

alpha_run:
0.65
```

決定論的/校正stochasticでWALK 1.2は100/100%、RUN 1.2は100/99%、RUN 2.4/2.6は100/100%、WALK→RUNとRUN→WALKは100/100%だった。追加PPO前の統合歩容artifactとして再利用価値が最も高い。

## Successful capabilities

- 同じ1.2 m/sで`gait_cmd=0`のWALKと`gait_cmd=1`のRUNを選択
- WALK 0.6–1.2 m/s
- RUN 1.2–2.6 m/s
- WALK→RUN、RUN→WALK
- RUN 1.2→2.4→1.2
- 走行後のWALK復帰と実用的停止
- single-weight runtime contract

## Unresolved capabilities

- strict flight-zero STAND
- strict final double support
- final sequence formal completion
- soft KL anchor下のcontinued PPO semantic retention
- scratch RLのみでの全endpoint統合

Stage 2Rのpositive controlではSTAND専門方策もformal success 3%、WALK_TO_STAND専門方策も3%だった。両者は平均速度0.00551/0.00122 m/s、fall 1/2%と実質的には静止していたが、接触gateを満たさなかった。このためStage 2Qの0%をstudentだけの学習失敗とは解釈しない。

## Scientific conclusions

速度commandだけでは同一速度の歩容を指定できない。WALKとRUNは1.2 m/sでもAUROC約0.999986で分離し、局所action perturbationからWALKへ到達できなかった。独立scalar gait commandを加えると、一つのmean actorが両attractorを明示選択できた。従って障害はnetwork capacityよりoptimization pathとdynamical basinにあった。

teacherのfull stdは安全なruntime探索量ではなかった。WALK/RUN別の温度校正により非zeroの安全探索窓を得た。一方、rewardが直接保護しないgait semanticsはsoft KL anchorだけでは連続PPOでdriftした。Adam moment不整合は存在したが、moment adaptationで解決せず主因ではなかった。

## Why the project is closed

中心問いには肯定的な実証が得られた。walk-parentからRUNを獲得する経路、completion reuse、reverse continuation、soft-anchor PPO、moment adaptationは十分に診断され、追加の係数調整よりも問題を別研究として切り出す方が妥当である。strict STAND gateは専門positive controlにも通らず、評価定義を含む独立課題である。

## Recommended reuse

- gait-core checkpoint: same-speed gait disambiguation、gait toggle、calibrated explorationの基準
- final-sequence checkpoint: deterministic WALK/RUN sequence、可視化、下流のSTOP/contact評価
- scalar gait command: speedだけでは曖昧な接触様式を一つのpolicyへ統合する入力設計
- Stage 2J/2M/2O/2P diagnostics: manifold、exploration、semantic retentionを分離する手順
