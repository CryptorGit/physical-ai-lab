# LinkedIn投稿案

Unitree G1を使ったモジュール型運動制御の研究を、Isaac Sim上でひと区切りにしました。

独立したexpertをつなぐ構成で、`STAND ↔ WALK`と、WALK 1.2 m/sからRUN 2.6 / 2.8 m/sへの遷移までは成立しました。一方、逆方向の`RUN → WALK`は成立しませんでした。

exp_008では、この失敗を「将来のcontact breakを観測から予測できるか」「安全なbounded correctionでWALK状態を20 step維持できるか」に分けて診断しました。

- 2,048 episodes / 201,882 steps
- breakの接近は高いAUROCで順位付け可能
- ただしtime-to-break MAEは約5.2 stepで、事前基準1.5 stepを未達
- 20-step成功は0件
- bounded correctionは全候補0/512

高い分類精度だけを見て「予測できた」とはせず、補正制御に必要な時刻精度と制御可能性までgateに含めました。結果はNo-Goです。失敗も含めて成果として残し、G1の局所phase-aware correction路線をcloseします。

添付動画はexp_008が新しく獲得した能力ではなく、exp_007で正式に成立した`STAND ↔ WALK`と`WALK → RUN`の再生です。2.8 m/s sceneは固定showcase seedでRUN_LOWへ到達したものの、7秒hold判定は失敗した事実もmanifestに残しています。

次は、自由度と接触構造がより単純なUnitree Go2で、双方向歩容遷移を基礎から再検証します。Go2なら成功すると証明したわけではなく、問題をより切り分けやすい身体からやり直す、というプロジェクト判断です。

使用動画: `exp008_g1_state_graph_closeout_showcase.mp4`

#PhysicalAI #ReinforcementLearning #Robotics #IsaacSim #UnitreeG1
