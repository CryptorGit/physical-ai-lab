# exp_013 Phase W1A2 WALK speed-envelope report

Selected checkpointはiteration 160、SHA `3cb25a32c24b0ca9d8e70e1f418942d28c46da22dd542a4830e33bf54101ef2b`。strict resumeはactor/critic bitwise、optimizer Adam step 2400、Identity normalizer、固定LR 1.5e-5でPASSした。160 iterations、3,932,160 interactionsを一回だけ実行した。

Boundary preflightではlateral 0.30〜0.40、rear 0.30〜0.55、backwardは非単調だが最大0.55 m/s。固定E1〜E4 curriculumを使用した。

Formal結果は0.3 m/s **14/16**、0.6 m/s **9/16**。envelopeは5/16。fall 0.00%、tilt 0.00%、slip 0.69%、impact 0.00%、saturation 0.00%、mirror MAE差 0.024 m/s。

0.6 m/sはW1Aの4/16から改善したが、0.3 m/sの225°と247.5°を失った。正式分類は `EXP013_W1A2_LOW_SPEED_RETENTION_FAIL`。次は **0.3m/s rear-left retention boundary diagnosis (225° and 247.5°)** のみ。

continuous 30秒診断およびRUN retention診断はformal gate外。W1A2はfinal integrated policyではない。保護対象と既存dirty stateは変更せず、remote pushは行っていない。
