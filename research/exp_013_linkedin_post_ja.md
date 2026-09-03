Unitree G1の単一方策による全方向移動の研究（exp_013）をクローズしました。

Isaac Lab上で、body-frameのvx / vy / yaw-rateを受け取る一つのactorを使い、前後左右と斜めを含む360°の歩行、その場旋回、移動しながらの旋回を実現しました。全方向歩行は22.5°刻みの16方向で正式評価しています。

動画にはexp_013の16方向移動、その場旋回、前進・後退しながらの旋回だけをまとめています。前段のexp_012ではWALK / RUNの双方向遷移、加減速、実用的な停止を別のStage 2Q actorで実現しましたが、今回の動画には含めていません。exp_013 actorが歩行・走行遷移まで単独で行ったものではありません。

研究上の重要な結果は、全方向歩行とyaw制御を一つのmemoryless actorで保持できたことです。一方、停止能力そのものは別方策で成立したものの、停止維持、全方向への安全な再発進、moving yaw、stop recoveryを一つのactorへ完全統合する課題は未解決でした。command history、contact phase、短期GRU、教師軌道統合、限定PPOも検証しましたが、全能力を同時に保持する正式gateには届きませんでした。

成功と失敗の両方を記録し、exp_013はここで終了します。なお、すべてシミュレーション結果であり、G1実機の性能を示すものではありません。

#PhysicalAI #ReinforcementLearning #Robotics #IsaacLab #UnitreeG1
