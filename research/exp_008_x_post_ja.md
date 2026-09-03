# A. 単独投稿

Isaac Sim上のUnitree G1運動制御を一区切り。STAND↔WALK、WALK→RUN 2.6/2.8 m/sは成立。一方RUN→WALKはNo-Goでした。exp_008では2,048 episodesを診断し、break接近は検出できても正確な時刻予測は未達、bounded correctionも0/512。失敗を含めてcloseし、次はGo2で再検証します。#PhysicalAI #IsaacSim

# B. Thread

## 1/5

Isaac Sim上で進めてきたUnitree G1のモジュール型運動制御を一区切りにしました。成功だけでなく、接続できなかった遷移も含めてcloseout動画と最終reportに残しています。

## 2/5

できたこと：STAND↔WALK、WALK 1.2 m/s→RUN 2.6/2.8 m/s。動画はこのexp_007正式能力の再生です。exp_008が新能力を獲得した、という意味ではありません。

## 3/5

できなかったこと：RUN→WALK。RUN周期は終えられてもWALK contractを維持できず、full bidirectional graphにはなりませんでした。

## 4/5

exp_008では2,048 episodes / 201,882 stepsを解析。break接近の順位付けは高AUROCでしたが、時刻MAEは約5.2 stepで基準1.5 stepを未達。安全なbounded correctionも全候補0/512でした。

## 5/5

局所phase-aware correctionはNo-Goとしてclose。次はUnitree Go2で、より単純な自由度・接触構造から双方向歩容遷移を再検証します。Go2での成功を主張するものではなく、次の検証方針です。#PhysicalAI #Robotics #IsaacSim
