# exp_006 command_system_v1 目視確認チェックリスト

このチェックリストはIsaac Sim GUIでの人手確認用であり、正式な自動gate結果を置き換えない。確認日、担当者、所見を記入する。

- 確認日: [ ]
- 確認者: [ ]
- 使用commit: [ ]
- Isaac Sim / Isaac Lab環境: [ ]

## RUN_TURN_RUN

- [ ] 直進走行が安定している
- [ ] 左45°旋回を確認した
- [ ] 右45°旋回を確認した
- [ ] 左90°旋回を確認した
- [ ] 右90°旋回を確認した
- [ ] 旋回後に直進へ復帰する
- [ ] 転倒がない
- [ ] 大きな横逸脱がない
- [ ] consoleのcontroller family、base option、sequence resultが期待どおり
- 所見: [ ]

## STAND_CROUCH_STAND

- [ ] 初期STANDが安定している
- [ ] 0.08–0.10 m範囲の浅いしゃがみを確認した
- [ ] HOLD中に高さと姿勢を維持する
- [ ] RETURNが滑らかである
- [ ] 最終STANDが安定している
- [ ] 転倒、危険なcontact loss、saturationがない
- [ ] Stage 2 baseと`scripted_shallow_v1`が表示される
- 所見: [ ]

## Unsupported requests

`UNSUPPORTED_RUN_TO_CROUCH`、`UNSUPPORTED_STEP_OVER`、`UNSUPPORTED_LAND`を個別に確認する。

- [ ] request後も動作を開始しない
- [ ] 現在controllerを維持する
- [ ] baseを突然切り替えない
- [ ] unsafe offsetがzeroと表示される
- [ ] 正しいrejection reasonを表示する
- [ ] unsupported requestを成功技能として表示しない
- 所見: [ ]

## 判定

- [ ] 自動評価結果と目視結果を別々に保存した
- [ ] GUI目視PASS
- [ ] 要再確認（理由: [ ]）
