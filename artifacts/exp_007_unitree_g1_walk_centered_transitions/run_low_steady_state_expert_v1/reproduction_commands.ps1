cd "$HOME\workspace\physical-ai-lab"
.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\evaluate_run_low.ps1 -Mode preflight -Seed 20261001 -Label preflight
.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\evaluate_run_low.ps1 -Mode audit -Seed 20261011 -Label operating_point_audit -EpisodesPerSpeed 10
.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\evaluate_run_low.ps1 -Mode formal -Seed 20261021 -Label formal_seed_20261021 -EpisodesPerSpeed 20
.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\evaluate_run_low.ps1 -Mode formal -Seed 20261022 -Label formal_seed_20261022 -EpisodesPerSpeed 20
.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\evaluate_run_low.ps1 -Mode formal -Seed 20261023 -Label formal_seed_20261023 -EpisodesPerSpeed 20
.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\evaluate_run_low.ps1 -Mode formal -Seed 20261024 -Label formal_seed_20261024 -EpisodesPerSpeed 20
