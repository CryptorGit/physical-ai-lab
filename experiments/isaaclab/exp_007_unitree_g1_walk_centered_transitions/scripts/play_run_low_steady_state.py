"""Stage 6 RUN_LOW GUI entrypoint (straight RUN only)."""
from pathlib import Path
import runpy

print("SYSTEM STATE: UNINITIALIZED_FOR_RUN -> RUN_LOW")
print("ACTIVE MODEL: run_low_steady_state_expert_v1")
print("TURN: DISABLED")
runpy.run_path(str(Path(__file__).with_name("play_run_expert.py")), run_name="__main__")
