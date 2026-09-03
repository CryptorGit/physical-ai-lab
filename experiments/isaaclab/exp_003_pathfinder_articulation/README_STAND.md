# Pathfinder standing smoke training

This bundle adds a local DirectRLEnv task without modifying Isaac Lab itself.

Task ID: `Isaac-Pathfinder-Stand-Direct-v0`

## Smoke run

```powershell
cd "$HOME\workspace\physical-ai-lab"

& "$HOME\workspace\IsaacLab\env_isaaclab\Scripts\python.exe" `
  ".\experiments\isaaclab\exp_003_pathfinder_articulation\scripts\train_stand.py" `
  --task Isaac-Pathfinder-Stand-Direct-v0 `
  --headless `
  --num_envs 64 `
  --max_iterations 5 `
  --seed 42
```

Expected log root:

`logs/rl_games/PathfinderStand/`
