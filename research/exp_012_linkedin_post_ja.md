Unitree G1を、一つのニューラルネットワークだけで「歩行→走行→歩行→停止」まで動かす研究をクローズしました。

Isaac Lab上のシミュレーション研究です。実行時に使うのは一つのcheckpoint、一つのactorだけ。expert router、checkpoint切替、action blendingは使っていません。

一番大きな発見は、速度commandだけでは歩容を指定できなかったことです。同じ1.2m/sでも、WALKはflight約3.5%・stride約1.4Hz、RUNはflight約48%・stride約6.2Hz。状態分布もほぼ完全に分離し、小さなaction perturbationでは片方からもう片方へ移れませんでした。

そこで速度とは独立したscalar gait commandを追加しました。

```text
gait=0 → WALK
gait=1 → RUN
```

その結果、同じ1.2m/sでWALKとRUNを一つのmean actorから明示的に出し分けられました。

主な決定論的評価結果:

- WALK 0.6〜1.2m/s: 100%
- RUN 1.2〜2.6m/s: 100%
- WALK→RUN: 100%
- RUN→WALK: 100%

走行後は再びWALKへ戻り、最終的に平均約0.055m/sまで減速しました。一方、厳格なSTAND gateは未解決です。転倒はほぼなく実用的には止まっていますが、微小な足踏み・接触振動が残り、flight-zero / final double-support条件を満たしませんでした。この限界も映像で隠していません。

研究を通じて、主問題はnetwork capacityよりもoptimization pathと歩容のdynamical basinにあると分かりました。また、teacher policyのfull exploration stdは完成歩容の安全なruntime分布ではなく、歩容ごとの温度校正が必要でした。

この結果をもってexp_012をクローズします。

[GitHub repository link]

#PhysicalAI #Robotics #ReinforcementLearning #IsaacLab #HumanoidRobot #MachineLearning
