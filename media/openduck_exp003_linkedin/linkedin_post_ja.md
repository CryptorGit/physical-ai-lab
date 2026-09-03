# LinkedIn投稿文案

ロボットに歩かせることより、「この研究方向はここで閉じる」と再現可能な形で判断する方が難しかったかもしれません。

OpenDuckMiniのexp_003では、単一のPPO方策で停止・前後・左右・旋回・複合指令を扱うomnidirectional locomotionに取り組みました。

途中で分かったのは、見えている失敗が必ずしも方策だけの失敗ではないということでした。

・legacy evaluatorのsceneとteacher routingが学習時と不一致  
・yaw報酬が指令値より約3.5倍の旋回を有利にするobjective conflict  
・GPU batched MJXがscatter-addから微小に分岐  
・production-size学習ではper-update device_getが100k付近のCUDA crashを予測  
・固定batchで正しく見えた局所gradient介入が、closed-loopでは状態分布を変えて別の失敗を生む

評価経路を直し、checkpoint/RNG/env stateを保存し、batch-1のexact paired evaluationまで行った上で、最終判断は`CLOSED_NO_GO`です。

これは「歩行学習に失敗した」というだけの話ではありません。どの失敗がevaluation-induced、objective-induced、policy-induced、infrastructure-inducedなのかを分け、改善しない介入を止められたことが成果です。

v52 hybrid controllerはsimulation parentとして残します。v59/v60以降のpilotは採用せず、実機投入もしません。

映像はMuJoCo simulationです。v59のシーンは診断方策であり、hardware-qualified policyではありません。

#ReinforcementLearning #Robotics #MuJoCo #JAX #PPO #Sim2Real #RobotLearning #OpenDuckMini #ResearchEngineering #NegativeResults

