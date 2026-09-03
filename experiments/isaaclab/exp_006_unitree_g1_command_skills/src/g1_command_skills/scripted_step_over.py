"""Safety-gated STEP_OVER primitive contract.

The first reachability audit found no complete safe pose chain, so v0 exposes
the phase/controller API but deliberately has no supported obstacle range.
"""
from __future__ import annotations
from enum import IntEnum
import torch

class StepOverPhase(IntEnum):
    SETTLE=0; WEIGHT_SHIFT_TO_SUPPORT=1; LEAD_FOOT_LIFT=2; LEAD_FOOT_SWING=3
    LEAD_FOOT_PLACE=4; WEIGHT_TRANSFER=5; TRAIL_FOOT_LIFT=6; TRAIL_FOOT_SWING=7
    TRAIL_FOOT_PLACE=8; STAND_RECOVERY=9; STAND_HOLD=10

CONTROLLER_ID="scripted_step_over_v0_guarded"
SUPPORTED_OBSTACLE_HEIGHT_RANGE_M: tuple[float,float] | None = None
UNSUPPORTED_REASON="KINEMATIC_POSE_CHAIN_UNRESOLVED"
FOOT_KEYPOINTS_BODY_M={"toe_bottom":(0.0638388096,0.0,-0.0258071800),"sole_center":(0.0432121324,0.0,-0.0258071800),"heel_bottom":(0.0225854551,0.0,-0.0258071800)}

def minimum_jerk(progress: torch.Tensor) -> torch.Tensor:
    value=progress.clamp(0.0,1.0);return value**3*(10.0-15.0*value+6.0*value**2)

def command_supported(obstacle_height_m: torch.Tensor) -> torch.Tensor:
    # Fail closed until both lead directions have a collision-free, return-safe chain.
    return torch.zeros_like(obstacle_height_m, dtype=torch.bool)

def guarded_offset(reference: torch.Tensor) -> torch.Tensor:
    """Return bitwise zero while the calibrated supported range is empty."""
    return torch.zeros_like(reference)


def classify_obstacle_region(
    point_x: torch.Tensor, point_z: torch.Tensor, *, front_x: float, rear_x: float,
    top_z: float, tolerance_m: float = 0.005,
) -> torch.Tensor:
    """Classify a collision keypoint: 1 front, 2 top, 3 rear edge, 0 unknown."""
    result = torch.zeros_like(point_x, dtype=torch.long)
    top = (point_x >= front_x - tolerance_m) & (point_x <= rear_x + tolerance_m) & (
        point_z >= top_z - tolerance_m
    )
    front = (point_x <= front_x + tolerance_m) & (point_z < top_z + tolerance_m)
    rear = (point_x >= rear_x - tolerance_m) & (point_z < top_z + tolerance_m)
    result = torch.where(front, torch.ones_like(result), result)
    result = torch.where(top, torch.full_like(result, 2), result)
    result = torch.where(rear & ~top, torch.full_like(result, 3), result)
    return result
