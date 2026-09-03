cd "$HOME\workspace\physical-ai-lab"

# Startup diagnostic: 4 seeds x 50 candidates
1..4 | ForEach-Object {
  $seed = 20260920 + $_
  .\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\evaluate_state_conditioned.ps1 -Mode startup -Seed $seed -Label "startup_seed_$seed"
}

# Main conditioned formal: 3 seeds x 60 selected (72 candidates per seed)
1..3 | ForEach-Object {
  $seed = 20260930 + $_
  .\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\evaluate_state_conditioned.ps1 -Mode main -Seed $seed -Label "main_seed_$seed"
}

# 0.6 m/s confirmation: 3 seeds x 50 selected (60 candidates per seed)
1..3 | ForEach-Object {
  $seed = 20260940 + $_
  .\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\evaluate_state_conditioned.ps1 -Mode zero_point_six -Seed $seed -Label "zero_point_six_seed_$seed"
}

python .\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\finalize_stage5e.py
