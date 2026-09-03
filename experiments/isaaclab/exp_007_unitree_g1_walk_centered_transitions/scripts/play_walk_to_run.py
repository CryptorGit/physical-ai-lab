"""Stage 7 diagnostic-only GUI notice."""
import argparse
p=argparse.ArgumentParser();p.add_argument("--run-speed",type=float,required=True);a=p.parse_args()
if a.run_speed not in (2.4,2.6,2.8):raise ValueError("unsupported RUN target")
print("STATE: WALK / WALK_TO_RUN / RUN_LOW")
print("STAGE 7: FORMAL FAIL - DIAGNOSTIC PLAYBACK ONLY")
print(f"SOURCE WALK SPEED=1.2 TARGET RUN SPEED={a.run_speed}")
print("ACTIVE CONTROLLER: no production WALK_TO_RUN controller")
print("RUN TAKEOVER RESULT: NOT AUTHORIZED")
