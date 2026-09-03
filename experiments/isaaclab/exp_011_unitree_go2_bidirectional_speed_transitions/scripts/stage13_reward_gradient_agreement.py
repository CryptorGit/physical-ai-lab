"""Compare Stage 12 reward-derived action preference with fresh finite differences."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage13_fresh_process_counterfactual_replay"
STAGE12_RAW = REPO / (
    "results/exp_011_unitree_go2_bidirectional_speed_transitions/"
    "stage12_tangential_slip_reward_directionality/raw/gradient_batch.pt"
)
CHECKPOINT = REPO / (
    "results/exp_011_unitree_go2_bidirectional_speed_transitions/"
    "stage11_tangential_slip_reduction/checkpoints/model_initial.pt"
)
JOINTS = (
    "FL_hip", "FR_hip", "RL_hip", "RR_hip",
    "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh",
    "FL_calf", "FR_calf", "RL_calf", "RR_calf",
)

batch = torch.load(STAGE12_RAW, map_location="cpu", weights_only=False)
checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
state = checkpoint["actor_state_dict"]
actor = nn.Sequential(
    nn.Linear(48, 128), nn.ELU(),
    nn.Linear(128, 128), nn.ELU(),
    nn.Linear(128, 128), nn.ELU(),
    nn.Linear(128, 12),
)
actor.load_state_dict({
    key.removeprefix("mlp."): value
    for key, value in state.items() if key.startswith("mlp.")
})
actor.eval()
with torch.inference_mode():
    mean = actor(batch["observation"])
std = state["distribution.std_param"]
# Direction that policy-gradient ascent applies to the deterministic mean.
sample_preference = (
    (batch["action"] - mean) / std.square()[None]
    * batch["A_slip"][:, None]
)
reward_by_speed = {}
for target_speed in (0.2, 0.4, 0.6, 1.2, 2.0):
    mask = (batch["target_speed"] - target_speed).abs() < 1.0e-4
    reward_by_speed[str(target_speed)] = sample_preference[mask].mean(0).tolist()

empirical = json.loads(
    (OUT / "empirical_local_slip_gradient.json").read_text(encoding="utf-8")
)["rows"]
rows = []
cosines = []
signs = []
for speed in (0.2, 0.4, 0.6, 1.2, 2.0):
    subset = [item for item in empirical if abs(item["speed"] - speed) < 1.0e-6]
    derivative = np.array([
        np.mean([
            item["empirical_slip_derivative"]
            for item in subset if item["dimension"] == dimension
        ])
        for dimension in range(12)
    ])
    empirical_preference = -derivative
    reward_preference = np.asarray(reward_by_speed[str(speed)])
    denominator = np.linalg.norm(empirical_preference) * np.linalg.norm(reward_preference)
    cosine = float(np.dot(empirical_preference, reward_preference) / max(denominator, 1.0e-12))
    sign_agreement = float(np.mean(
        np.sign(empirical_preference) == np.sign(reward_preference)
    ))
    cosines.append(cosine)
    signs.append(sign_agreement)
    for dimension in range(12):
        rows.append({
            "speed": speed, "dimension": dimension, "joint": JOINTS[dimension],
            "reward_preference": float(reward_preference[dimension]),
            "empirical_preference": float(empirical_preference[dimension]),
            "sign_agreement": bool(
                np.sign(empirical_preference[dimension])
                == np.sign(reward_preference[dimension])
            ),
        })
agreement = {
    "definition": (
        "reward preference = mean[(sampled action - policy mean)/std^2 * "
        "diagnostic A_slip]; empirical preference = negative central finite "
        "difference of 8-step cumulative raw slip"
    ),
    "stage12_gradient_ratio": 0.001453556353226304,
    "stage12_base_slip_cosine": -0.3287697434425354,
    "speed_wise_cosine": {
        str(speed): value for speed, value in zip((0.2, 0.4, 0.6, 1.2, 2.0), cosines)
    },
    "speed_wise_sign_agreement": {
        str(speed): value for speed, value in zip((0.2, 0.4, 0.6, 1.2, 2.0), signs)
    },
    "median_cosine": float(np.median(cosines)),
    "mean_sign_agreement": float(np.mean(signs)),
    "gate": "PASS" if np.median(cosines) >= 0.20 and np.mean(signs) >= 0.60 else "FAIL",
    "joint_rows": rows,
}
(OUT / "reward_counterfactual_gradient_agreement.json").write_text(
    json.dumps(agreement, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
