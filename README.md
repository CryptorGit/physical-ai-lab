# Physical AI Lab

物理世界をモデル化し、シミュレーションで改善し、
現実または異なる環境条件で検証するための研究リポジトリ。

## Current Focus
- MuJoCoによる環境パラメータ実験
- Isaac Labによる深層強化学習
- 重力・摩擦・地形条件への汎化

## Core Questions
1. どの物理条件で方策が破綻するか
2. どのランダム化が汎化性能に寄与するか
3. 異なる重力・地形間で何を共通化できるか

## Pathfinder Articulation

Pathfinder素材USDから、デバッグ用の簡略Isaac Sim Articulationを生成する。
元の \`shared/models/pathfinder/source/pathfinder_visual_collision.usdc\` は変更しない。

通常のWindows Pythonではなく、Isaac LabのPython起動スクリプトを使う。
以下はPowerShellでIsaac Labルートにいる場合の実行例。

\`\`\`powershell
$physicalAiLabRoot = Resolve-Path "C:\path\to\physical-ai-lab"
.\isaaclab.bat -p "$physicalAiLabRoot\shared\models\pathfinder\scripts\build_pathfinder_articulation.py"
\`\`\`

生成物は \`shared/models/pathfinder/usd/pathfinder_articulation.usd\`。
続いてGUIでspawn確認する。

\`\`\`powershell
.\isaaclab.bat -p "$physicalAiLabRoot\experiments\isaaclab\exp_003_pathfinder_articulation\scripts\spawn_pathfinder.py" --viz native
\`\`\`

膝、次に肘へ小さなsin波目標を与える単関節テストは次の通り。

\`\`\`powershell
.\isaaclab.bat -p "$physicalAiLabRoot\experiments\isaaclab\exp_003_pathfinder_articulation\scripts\test_joint_motion.py" --viz native
\`\`\`

片側だけ確認する場合は、例えば \`--sequence left_knee left_elbow\` を追加する。
\`--amplitude-deg\`、\`--frequency-hz\`、\`--segment-seconds\` も変更できる。
Isaac Lab 3.0でGUI確認するため \`--viz native\` を付ける。

## Repository Data Policy

生成されたrun出力、checkpoint、raw trace、ログ、ローカル機器情報、
エージェント固有の作業状態はGit管理しない。再現に必要なソース、設定、テスト、
研究文書、および明示的に選別した小さなサマリーだけを共有対象とする。
