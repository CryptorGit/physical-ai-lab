# Observation Inventory

## Baseline policy observation

| Name | Dimension |
|---|---:|
| fingertip_pos_rel_fixed | 3 |
| fingertip_quat | 4 |
| ee_linvel | 3 |
| ee_angvel | 3 |
| prev_actions | 6 |
| Total | 19 |

## Ablation 001

Removed:

- ee_angvel

Policy observation dimension:

- 16

Critic state:

- unchanged
- 43 dimensions