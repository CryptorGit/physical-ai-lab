# FBXモデルをIsaac Sim / Isaac Labへ載せ、Articulation化して強化学習するまで

## 1. 目的

本ドキュメントは、FBX形式のキャラクターモデルを出発点として、以下を完了するまでの手順をまとめたものです。

1. FBXモデルをIsaac Simへ取り込む
2. VisualとCollisionを整理する
3. テクスチャ付きUSD / USDCとして保存する
4. 関節・剛体を持つArticulation USDを生成する
5. Isaac Labで単体表示する
6. DirectRLEnvへ登録する
7. RL-GamesでGPU並列学習する
8. GUIで学習中の挙動を確認する
9. Collision形状を物理的には残したまま、画面上だけ非表示にする
10. モデル更新時に学習用USDを再生成する

対象モデルはApex LegendsのPathfinderを元にした個人学習用モデルです。

> 注意  
> キャラクターやモデルの権利は権利者に帰属します。公開・配布・商用利用時は、取得元ライセンスと権利関係を必ず確認してください。

---

## 2. 今回使用した環境

```text
OS              : Windows
Isaac Sim       : 6.0.0系
Isaac Lab       : 3.0
Python          : Isaac Lab専用仮想環境
RLライブラリ     : RL-Games
GPU device      : cuda:0
プロジェクト     : C:\Users\user\workspace\physical-ai-lab
Isaac Lab       : C:\Users\user\workspace\IsaacLab
Python          : C:\Users\user\workspace\IsaacLab\env_isaaclab\Scripts\python.exe
```

---

## 3. 最終的なディレクトリ構成

```text
physical-ai-lab/
├─ shared/
│  └─ models/
│     └─ pathfinder/
│        ├─ source/
│        │  ├─ pathfinder.fbx
│        │  └─ pathfinder_visual_collision.usdc
│        ├─ textures/
│        │  ├─ basecolor.png
│        │  ├─ normal.png
│        │  ├─ roughness.png
│        │  └─ metallic.png
│        └─ usd/
│           └─ pathfinder_articulation.usd
│
├─ experiments/
│  └─ isaaclab/
│     └─ exp_003_pathfinder_articulation/
│        ├─ scripts/
│        │  ├─ spawn_pathfinder.py
│        │  ├─ test_joint_motion.py
│        │  └─ train_stand.py
│        └─ src/
│           └─ pathfinder_stand/
│              └─ tasks/
│                 └─ stand/
│                    ├─ __init__.py
│                    ├─ pathfinder_stand_env_cfg.py
│                    ├─ pathfinder_stand_env.py
│                    └─ agents/
│                       └─ rl_games_ppo_cfg.yaml
│
└─ logs/
   └─ rl_games/
      └─ PathfinderStand/
```

---

# Part A: FBXからUSD / USDCを作る

## 4. FBXをIsaac Simへインポートする

Isaac Simを起動し、FBXをStageへインポートします。

GUI上では概ね以下の操作です。

```text
File / Import
→ FBXファイルを選択
→ Import
```

FBXを読み込んだ直後に確認する項目:

- スケールが適切か
- 上方向がZ軸になっているか
- 原点位置が極端にずれていないか
- メッシュがバラバラになっていないか
- スケルトンやボーン階層が保持されているか
- テクスチャ参照が切れていないか

---

## 5. スケールと座標系を確認する

Isaac Sim / Isaac Labでは、基本的にメートル単位を前提に扱います。

確認事項:

```text
1 unit = 1 m
Z-up
正面方向を統一
ルート位置は原点付近
足裏が地面付近
```

モデルが巨大または極端に小さい場合は、FBX側またはIsaac Sim側でスケールを調整します。

今回の初期Root高さは次でした。

```text
z = 0.15 m
```

これは小型モデルとして扱ったためです。一般的な人型サイズなら、モデル寸法に応じて修正してください。

---

## 6. テクスチャを設定する

Isaac Simでは画像テクスチャを表示できます。

利用できる代表的なマップ:

- Base Color / Albedo
- Normal
- Roughness
- Metallic
- Emissive
- Opacity

きれいに表示するには、メッシュ側にUV展開が必要です。

推奨フロー:

```text
Blender
→ UV展開
→ マテリアル設定
→ PNGなどのテクスチャを割り当て
→ FBXまたはUSDで出力
→ Isaac Simで確認
```

推奨配置:

```text
shared/models/pathfinder/
├─ source/
├─ textures/
└─ usd/
```

テクスチャの絶対パス依存は避け、可能ならUSDから相対参照にします。

---

## 7. VisualとCollisionを分離する

重要なのは、見た目用メッシュと物理衝突用メッシュを分けることです。

理想構造:

```text
Robot
└─ Links
   └─ torso
      ├─ Visuals
      │  └─ textured_mesh
      └─ Collisions
         └─ simple_box
```

役割:

```text
Visuals
  見た目用
  高ポリゴンでもよい
  テクスチャを持つ

Collisions
  当たり判定用
  低ポリゴン
  Box / Capsule / Convex Hull推奨
  画面上は非表示でよい
```

Collision用メッシュをVisualにも残すと、学習GUIで白い箱や簡易形状が見えます。

---

## 8. USDCとして保存する

VisualとCollisionを整理したStageを、次へ保存します。

```text
shared/models/pathfinder/source/pathfinder_visual_collision.usdc
```

今回の実ファイル:

```text
C:\Users\user\workspace\physical-ai-lab\
shared\models\pathfinder\source\pathfinder_visual_collision.usdc
```

USDCはバイナリUSDです。

特徴:

- 容量が比較的小さい
- 読み込みが速い
- `Select-String`では内容検索できない
- テキスト確認したい場合はUSDAへ変換する

---

# Part B: Articulation USDを作る

## 9. Articulationとは

Articulationは、複数の剛体と関節から成るロボットモデルです。

今回のPathfinderでは10関節を使用しました。

```text
shoulder_L
shoulder_R
elbow_L
elbow_R
hip_L
hip_R
knee_L
knee_R
ankle_L
ankle_R
```

確認されたBody数は15です。

```text
torso
head
arms
pelvis
legs
feet
```

---

## 10. 関節構造を作る

各可動部にRevolute Joint等を設定します。

必要項目:

- 親Body
- 子Body
- 回転軸
- 関節原点
- Position Limits
- Velocity Limits
- Effort Limits
- Stiffness
- Damping

今回のIsaac Lab側の設定値:

```python
ImplicitActuatorCfg(
    joint_names_expr=[".*"],
    effort_limit_sim=150.0,
    velocity_limit_sim=8.0,
    stiffness=40.0,
    damping=4.0,
)
```

関節制限例:

```text
shoulder : [-1.571, 1.571]
elbow    : [-0.175, 2.356]
hip      : [-1.047, 0.873]
knee     : [-0.175, 2.356]
ankle    : [-0.611, 0.611]
```

---

## 11. Articulation Rootを設定する

ロボット全体のルートPrimにArticulation Root APIを設定します。

重要項目:

- Root prim
- Fixed baseかFree baseか
- Self collision
- Solver iteration
- 各BodyへのRigid Body API
- 各Collision PrimへのCollision API

立位学習では、RootをFree baseにして重力下で倒れる状態にします。

---

## 12. 学習用Articulation USDを生成する

最終出力先:

```text
shared/models/pathfinder/usd/pathfinder_articulation.usd
```

今回の実ファイル:

```text
C:\Users\user\workspace\physical-ai-lab\
shared\models\pathfinder\usd\pathfinder_articulation.usd
```

重要:

```text
source/pathfinder_visual_collision.usdc
```

を更新しても、

```text
usd/pathfinder_articulation.usd
```

は自動更新されません。

Visualやテクスチャを変更したら、Articulation生成処理を再実行する必要があります。

更新確認:

```powershell
$root = "$HOME\workspace\physical-ai-lab"

Get-ChildItem `
  "$root\shared\models\pathfinder" `
  -Recurse `
  -Include *.usd,*.usda,*.usdc |
  Select-Object FullName, Length, LastWriteTime
```

今回、次の状態になっていました。

```text
pathfinder_visual_collision.usdc : 新しい
pathfinder_articulation.usd      : 古い
```

この場合、学習側には古い外観が表示されます。

---

# Part C: Isaac Labで単体表示する

## 13. 単体表示スクリプト

表示スクリプト:

```text
experiments/isaaclab/exp_003_pathfinder_articulation/
└─ scripts/
   └─ spawn_pathfinder.py
```

基本構造:

```python
cfg = ArticulationCfg(
    prim_path="/World/Pathfinder",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(usd_path.resolve())
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.15)
    ),
    actuators={
        "debug_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim=150.0,
            velocity_limit_sim=8.0,
            stiffness=40.0,
            damping=4.0,
        )
    },
)
```

単体表示スクリプトは、関節数、関節名、Body数、Body名を確認できるようにしておきます。

---

## 14. 単体GUI表示

```powershell
cd "$HOME\workspace\physical-ai-lab"

& "$HOME\workspace\IsaacLab\env_isaaclab\Scripts\python.exe" `
  ".\experiments\isaaclab\exp_003_pathfinder_articulation\scripts\spawn_pathfinder.py" `
  --viz kit
```

確認内容:

```text
Joint count (DoF): 10
Joint names: [...]
Body count: 15
Body names: [...]
```

---

## 15. Headlessスモークテスト

```powershell
cd "$HOME\workspace\physical-ai-lab"

& "$HOME\workspace\IsaacLab\env_isaaclab\Scripts\python.exe" `
  ".\experiments\isaaclab\exp_003_pathfinder_articulation\scripts\spawn_pathfinder.py" `
  --max-steps 300
```

300 step完走してPowerShellへ戻れば、最低限の読み込みと物理ステップは成功です。

---

# Part D: Collisionを画面上だけ非表示にする

## 16. なぜ白い箱が見えるのか

Collision用のBoxやCapsuleが、通常のImageable Primとして表示されているためです。

重要:

```text
Collisionを非表示
≠
Collisionを削除
```

Visibilityを`invisible`にしても、物理衝突は残ります。

---

## 17. Collision APIを使って一括非表示にする

Prim名に`Collisions`が含まれるかどうかだけで判定すると、別名のCollisionを見逃します。

正しくは、`UsdPhysics.CollisionAPI`が付いているPrimを判定します。

```python
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics


def hide_collision_visuals(root_path: str) -> int:
    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(root_path)

    if not root_prim.IsValid():
        raise RuntimeError(f"Robot prim not found: {root_path}")

    hidden_count = 0

    for prim in Usd.PrimRange(root_prim):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue

        if not prim.IsA(UsdGeom.Imageable):
            continue

        UsdGeom.Imageable(prim).MakeInvisible()
        hidden_count += 1

    return hidden_count
```

単体表示時:

```python
hide_collision_visuals("/World/Pathfinder")
```

---

## 18. 複数環境ですべて非表示にする

学習では次のようなPrimパスになります。

```text
/World/envs/env_0/Robot
/World/envs/env_1/Robot
...
```

1体分だけ消しても、他環境には表示が残ります。

```python
hidden_count = 0

for env_index in range(self.cfg.scene.num_envs):
    robot_path = f"/World/envs/env_{env_index}/Robot"
    hidden_count += hide_collision_visuals(robot_path)

print(
    f"[PATHFINDER] hidden collision visuals={hidden_count} "
    "(physics collisions remain active)"
)
```

---

# Part E: DirectRLEnvを作る

## 19. タスク構造

```text
src/pathfinder_stand/tasks/stand/
├─ __init__.py
├─ pathfinder_stand_env_cfg.py
├─ pathfinder_stand_env.py
└─ agents/
   └─ rl_games_ppo_cfg.yaml
```

---

## 20. Gymタスク登録

```python
gym.register(
    id="Isaac-Pathfinder-Stand-Direct-v0",
    entry_point=(
        "pathfinder_stand.tasks.stand."
        "pathfinder_stand_env:PathfinderStandEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "pathfinder_stand.tasks.stand."
            "pathfinder_stand_env_cfg:PathfinderStandEnvCfg"
        ),
        "rl_games_cfg_entry_point": (
            "pathfinder_stand.tasks.stand.agents:"
            "rl_games_ppo_cfg.yaml"
        ),
    },
)
```

---

## 21. Configと実環境を分離する

Isaac Sim起動前に`pxr`や`omni`系を読み込むと、次のような問題が起きます。

```text
pxr modules were loaded before SimulationApp
TfNotice wrapper error
UsdAPISchemaBase error
```

対策:

```text
pathfinder_stand_env_cfg.py
  軽量Configだけ
  SimulationApp前に読まれる

pathfinder_stand_env.py
  omni.usd
  pxr
  物理処理
  SimulationApp起動後に読まれる
```

この分離は必須です。

---

## 22. 環境設定

今回の主要設定:

```python
decimation = 4
episode_length_s = 5.0

action_scale = 0.35
action_space = 10
observation_space = 39
state_space = 0

sim = SimulationCfg(
    dt=1.0 / 120.0,
    render_interval=decimation,
)
```

制御周期:

```text
Physics : 120 Hz
Policy  : 30 Hz
```

---

## 23. 観測

39次元の観測:

```text
Root高さ誤差        : 1
Roll / Pitch       : 2
Root linear vel    : 3
Root angular vel   : 3
Joint position     : 10
Joint velocity     : 10
Previous action    : 10
合計                : 39
```

実装:

```python
policy_obs = torch.cat(
    (
        height_error,
        tilt,
        self.root_lin_vel,
        self.root_ang_vel,
        joint_pos_rel,
        self.joint_vel * 0.1,
        self.actions,
    ),
    dim=-1,
)
```

---

## 24. 行動

行動は10関節の位置目標です。

```python
targets = (
    self.default_joint_pos
    + self.cfg.action_scale * self.actions
)
```

Soft joint limitsでclampします。

```python
limits = self.robot.data.soft_joint_pos_limits.torch
self.joint_targets = torch.clamp(
    targets,
    limits[..., 0],
    limits[..., 1],
)
```

---

## 25. 報酬

今回の初期報酬:

```text
生存報酬
直立報酬
高さ維持報酬
関節姿勢ペナルティ
関節速度ペナルティ
行動量ペナルティ
行動変化ペナルティ
転倒ペナルティ
```

実装例:

```python
reward = torch.full(
    (self.num_envs,),
    self.cfg.rew_alive,
    device=self.device,
)

reward += self.cfg.rew_upright * torch.exp(
    -4.0 * tilt_sq
)

reward += self.cfg.rew_height * torch.exp(
    -80.0 * height_error.square()
)

reward += self.cfg.rew_joint_pose * torch.sum(
    joint_error.square(),
    dim=-1,
)

reward += self.cfg.rew_joint_vel * torch.sum(
    self.joint_vel.square(),
    dim=-1,
)

reward += self.cfg.rew_action * torch.sum(
    self.actions.square(),
    dim=-1,
)

reward += self.cfg.rew_action_rate * torch.sum(
    action_rate.square(),
    dim=-1,
)
```

---

## 26. 終了条件

```python
too_low = (
    self.root_pos[:, 2]
    < self.minimum_root_height
)

too_tilted = (
    torch.abs(self.roll) > self.cfg.max_tilt_rad
) | (
    torch.abs(self.pitch) > self.cfg.max_tilt_rad
)

invalid = ~torch.isfinite(
    self.root_pos
).all(dim=-1)

terminated = (
    too_low
    | too_tilted
    | invalid
)
```

---

# Part F: Isaac Lab 3.0のProxyArray対応

## 27. ProxyArrayエラー

Isaac Lab 3.0では、一部データが`ProxyArray`として返ります。

そのままTorch関数へ渡すと失敗します。

エラー例:

```text
Expected Tensor but found ProxyArray
```

修正前:

```python
self.robot.data.root_quat_w
```

修正後:

```python
self.robot.data.root_quat_w.torch
```

代表例:

```python
self.robot.data.root_pos_w.torch
self.robot.data.root_quat_w.torch
self.robot.data.root_lin_vel_w.torch
self.robot.data.root_ang_vel_w.torch
self.robot.data.joint_pos.torch
self.robot.data.joint_vel.torch
self.robot.data.soft_joint_pos_limits.torch
self.robot.data.default_joint_pos.torch
self.robot.data.default_joint_vel.torch
self.robot.data.default_root_pose.torch
self.robot.data.default_root_vel.torch
```

注意:

置換を複数回実行すると、

```text
.torch.torch
```

になることがあります。

確認:

```powershell
Select-String `
  -Path $file `
  -Pattern "torch.torch"
```

修正:

```powershell
$text = Get-Content $file -Raw
$text = $text.Replace(".torch.torch", ".torch")
Set-Content $file $text -Encoding UTF8
```

---

# Part G: RL-Games設定

## 28. YAMLの注意点

次の書き方は失敗しました。

```yaml
mu_activation: null
sigma_activation: null
```

RL-Gamesは活性化関数名として`None`文字列を期待します。

修正後:

```yaml
mu_activation: None
sigma_activation: None
```

Regularizerも同様です。

```yaml
regularizer:
  name: None
```

---

## 29. Minibatchサイズ

RL-Gamesでは次を満たす必要があります。

```text
batch_size % minibatch_size == 0
```

今回:

```text
batch_size = num_envs × horizon_length
```

例1: 1環境

```text
1 × 32 = 32
minibatch_size = 32
```

例2: 32環境

```text
32 × 32 = 1024
minibatch_size = 256
```

例3: 64環境

```text
64 × 32 = 2048
minibatch_size = 512
```

GUI用と本学習用でYAMLを分けると安全です。

---

# Part H: 学習実行

## 30. 5 epochスモークテスト

```powershell
cd "$HOME\workspace\physical-ai-lab"

& "$HOME\workspace\IsaacLab\env_isaaclab\Scripts\python.exe" `
  ".\experiments\isaaclab\exp_003_pathfinder_articulation\scripts\train_stand.py" `
  --task Isaac-Pathfinder-Stand-Direct-v0 `
  --num_envs 64 `
  --max_iterations 5 `
  --seed 42
```

実績:

```text
64 environments
8192 frames
5 epochs
Training time: 6.35 sec
```

保存先例:

```text
logs/rl_games/PathfinderStand/
└─ 2026-07-16_00-00-57/
   └─ nn/
      └─ last_PathfinderStand_ep_5_....pth
```

---

## 31. GUI付き学習

Isaac Lab 3.0では、GUI visualizerは`kit`です。

```powershell
cd "$HOME\workspace\physical-ai-lab"

& "$HOME\workspace\IsaacLab\env_isaaclab\Scripts\python.exe" `
  ".\experiments\isaaclab\exp_003_pathfinder_articulation\scripts\train_stand.py" `
  --task Isaac-Pathfinder-Stand-Direct-v0 `
  --viz kit `
  --num_envs 32 `
  --max_iterations 1000 `
  --seed 42
```

誤り:

```text
--viz native
```

正しい指定:

```text
--viz kit
```

---

## 32. 本学習の推奨段階

いきなり1万epochを回すより、段階的に確認します。

```text
5 epoch
  起動確認

100 epoch
  発散や即死の確認

1000 epoch
  報酬設計が機能するか確認

3000 epoch
  立位性能の改善確認

10000 epoch
  報酬が伸び続ける場合のみ
```

32環境・1000epochの場合:

```text
32 × horizon 32 × 1000
= 約102万frames
```

まず1000epochで報酬曲線を見るのが妥当です。

---

# Part I: トラブルシューティング

## 33. `configclass`がmodule objectになる

エラー:

```text
TypeError: 'module' object is not callable
```

修正:

```python
from isaaclab.utils.configclass import configclass
```

---

## 34. SimulationApp起動前にpxrを読んでしまう

症状:

```text
TfNotice wrapper error
UsdAPISchemaBase error
omni.usd import failure
```

原因:

```text
環境本体をSimulationApp起動前にimportした
```

対策:

- Configを軽量モジュールへ分離
- `omni.usd`と`pxr`は起動後にimport
- 公式train.pyを`runpy`で起動
- タスク登録だけ先に行う

---

## 35. GUIでCollisionが見える

原因候補:

1. CollisionAPI付きPrimがVisible
2. VisualとCollisionが同じメッシュ
3. env_0だけ非表示にして、他環境が残っている
4. Articulation USDが古い
5. source USDCだけ更新し、学習用USDを再生成していない

確認順:

```text
単体表示
→ CollisionAPIを一括非表示
→ 複数環境すべてに適用
→ Articulation USD更新日時確認
→ 再生成
→ Isaac Sim完全終了後に再起動
```

---

## 36. USDCを更新したのに見た目が変わらない

確認:

```powershell
Get-ChildItem `
  "$HOME\workspace\physical-ai-lab\shared\models\pathfinder" `
  -Recurse `
  -Include *.usd,*.usda,*.usdc |
  Select-Object FullName, Length, LastWriteTime
```

重要:

```text
source/pathfinder_visual_collision.usdc
```

と

```text
usd/pathfinder_articulation.usd
```

は別ファイルです。

後者を再生成しない限り、学習側の見た目は更新されません。

---

## 37. `Select-String`でUSD内容が出ない

`.usd`拡張子でも、実体がバイナリUSDCの場合があります。

その場合:

```powershell
Select-String ...
```

では検索できません。

対策:

- USDAへ変換して確認
- usdcatを使う
- Isaac SimのStageで参照を確認
- ファイルサイズと更新日時を確認

---

## 38. 1環境GUIでAssertionError

エラー:

```text
assert(batch_size % minibatch_size == 0)
```

対策:

```yaml
horizon_length: 32
minibatch_size: 32
```

32環境なら:

```yaml
horizon_length: 32
minibatch_size: 256
```

---

# Part J: 最終チェックリスト

## 39. モデル側

- [ ] FBXが正しいスケール
- [ ] Z-up
- [ ] 原点位置が適切
- [ ] UV展開済み
- [ ] テクスチャ参照が切れていない
- [ ] VisualとCollisionを分離
- [ ] Collisionは簡易形状
- [ ] BodyごとにRigid Body API
- [ ] 関節軸が正しい
- [ ] Joint limitsが正しい
- [ ] Articulation Root設定済み
- [ ] source USDCを保存
- [ ] articulation USDを再生成

## 40. Isaac Lab側

- [ ] 単体spawn成功
- [ ] 10 DoF確認
- [ ] 15 Body確認
- [ ] 300 stepスモークテスト成功
- [ ] ProxyArrayを`.torch`へ変換
- [ ] ConfigとEnvを分離
- [ ] Gym登録成功
- [ ] 39次元観測確認
- [ ] 10次元行動確認
- [ ] Collision表示を一括非表示
- [ ] 5 epoch完走
- [ ] checkpoint保存
- [ ] GUIで32環境表示
- [ ] 1000 epoch試験へ移行

---

# 41. 今回の到達点

今回、次を実現しました。

```text
FBXモデル
→ テクスチャ付きVisual
→ Collision構造
→ USDC保存
→ Articulation USD生成
→ Isaac Labで単体spawn
→ 10関節認識
→ DirectRLEnv実装
→ 39次元観測
→ 10次元行動
→ RL-Games接続
→ GPU並列学習
→ 32環境GUI表示
→ Collision形状の非表示
```

この時点で、モデル取り込み作業は完了です。

次の主作業は環境構築ではなく、以下になります。

```text
報酬設計
初期姿勢設計
関節剛性・減衰調整
転倒判定
観測アブレーション
立位安定化
歩行タスクへの拡張
```

---

# 42. 代表コマンド一覧

## 単体GUI

```powershell
& "$HOME\workspace\IsaacLab\env_isaaclab\Scripts\python.exe" `
  ".\experiments\isaaclab\exp_003_pathfinder_articulation\scripts\spawn_pathfinder.py" `
  --viz kit
```

## Headlessスモークテスト

```powershell
& "$HOME\workspace\IsaacLab\env_isaaclab\Scripts\python.exe" `
  ".\experiments\isaaclab\exp_003_pathfinder_articulation\scripts\spawn_pathfinder.py" `
  --max-steps 300
```

## 学習スモークテスト

```powershell
& "$HOME\workspace\IsaacLab\env_isaaclab\Scripts\python.exe" `
  ".\experiments\isaaclab\exp_003_pathfinder_articulation\scripts\train_stand.py" `
  --task Isaac-Pathfinder-Stand-Direct-v0 `
  --num_envs 64 `
  --max_iterations 5 `
  --seed 42
```

## GUI付き32環境学習

```powershell
& "$HOME\workspace\IsaacLab\env_isaaclab\Scripts\python.exe" `
  ".\experiments\isaaclab\exp_003_pathfinder_articulation\scripts\train_stand.py" `
  --task Isaac-Pathfinder-Stand-Direct-v0 `
  --viz kit `
  --num_envs 32 `
  --max_iterations 1000 `
  --seed 42
```

## ファイル更新確認

```powershell
Get-ChildItem `
  "$HOME\workspace\physical-ai-lab\shared\models\pathfinder" `
  -Recurse `
  -Include *.usd,*.usda,*.usdc |
  Select-Object FullName, Length, LastWriteTime
```
