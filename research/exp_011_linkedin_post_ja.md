# LinkedIn post — exp_011 Unitree Go2

Unitree Go2で、ひとつの連続速度条件付き方策だけを使い、

**停止 → 歩行 → 2.0 m/sの高速走行領域 → 歩行 → 停止**

を一つのepisode内で接続しました。

動画の最初から最後まで、actorの重みは同じです。

- expert switchingなし
- checkpoint switchingなし
- action blendingなし
- transition専用policyなし
- residual controllerなし

方策へ与える前進速度commandを連続的に変えることで、同じ12関節のGo2 policyが停止、低速移動、高速移動、減速、再停止を処理しています。

今回、特に重要だったのは「単に最高速度を出すこと」ではなく、速度領域を双方向につなぐことでした。

公式Isaac Lab Go2 policyを出発点に、停止、低速定常、加速、減速、高速能力保持を同じcommand curriculumへ入れました。また、低速域にあった歩容の不安定帯を重点的に再学習しました。

もう一つ分かったのは、48次元のpolicy observationには絶対heading errorが含まれていないことです。そこでpolicy自体は凍結したまま、速度獲得後だけ小さなyaw-rate補正を有効化する固定のphase-gated heading command controllerを使用しています。これは別expertへの切替ではなく、同じ方策へ渡すcommandの生成層です。

G1で取り組んだときは、停止、歩行、走行を複数の専門方策とroutingで接続する構成になりました。一方、自由度と接触構造がより単純なGo2では、単一の共有重みでこの往復を成立させることができました。

Go2から得た学びは明確です。

1. 速度ごとのpolicyを分ける前に、双方向command distributionを一つの方策へ十分に与える
2. 停止と低速歩容の境界を独立した学習対象として扱う
3. 加速だけでなく減速sampleを対称に入れる
4. warm-start時はmodelだけでなくoptimizerの履歴も整合させる
5. 観測できない状態は、policyの失敗と決めつける前にinterface contractとして切り分ける

次は、この知見をG1へ戻します。

目標は、G1でも複数expertの切替に頼らず、**共通の一組の重みで「停止 → 歩行 → 走行 → 歩行 → 停止」**を学習することです。

Go2で成立した設計が、そのまま37自由度のhumanoidへ移るとは考えていません。ただ、どこまでを一つの連続policyで共有でき、どこからが身体構造由来の本当の限界なのかを、より明確に検証できる段階に来ました。

Simulation: NVIDIA Isaac Lab  
Robot: Unitree Go2  
Policy: single continuous speed-conditioned actor  
Sequence command: 0 → 0.6 → 1.2 → 2.0 → 1.2 → 0.6 → 0 m/s

#PhysicalAI #ReinforcementLearning #Robotics #IsaacLab #NVIDIAIsaac #UnitreeGo2 #Quadruped #HumanoidRobotics #SimToReal
