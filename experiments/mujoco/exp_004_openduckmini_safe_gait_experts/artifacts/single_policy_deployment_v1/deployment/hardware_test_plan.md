# Human-controlled hardware verification plan

Status: NOT RUN. This plan is a safety gate, not an authorization to energize
the robot from the current workstation.

Before every stage, keep the robot supported/secured, keep an emergency-stop
handy, and record the package ID and SHA-256 values from `manifest.json`.

1. Offline: run `offline_target_sanity.py`; require `passed=true`, no serial
   device opened, all head targets zero, all raw targets finite/in range, and
   zero hash mismatches.
2. Communication/read-only: on the Raspberry Pi run the existing motor
   connection check. Require all ten leg IDs 10–14 and 20–24 to respond and
   no unexpected ID. Do not enable torque.
3. IMU/feet: verify BNO055 axis orientation with the pinned
   `imu_upside_down` setting, then verify active-low D22/D27 foot switches.
   PASS requires stationary upright readings and correct left/right response.
4. Torque-off target audit: generate the zero/stand target list from the
   package and compare each target to the measured zero/offset/sign mapping.
   PASS requires no missing ID, sign inversion, boundary crossing, or target
   outside the physical safe limit.
5. Supported stand: enable low torque only while the robot is supported. PASS
   requires stable stand, no runaway target, no boundary guard event, and an
   immediate successful torque-off.
6. Unsupported stand: brief test with a person ready to catch the robot. PASS
   requires upright body, no violent oscillation, and no joint-limit/saturation
   event.
7. Very-low-speed forward then stop: use a command no larger than
   `[+0.03, 0, 0]` for at most 2 s, then `[0,0,0]`. PASS requires forward
   motion, no slant growth, no fall, and stable stop.
8. Reverse then yaw/lateral: only after stages 1–7 pass. Increase one axis at a
   time inside the manifest envelope. PASS requires the commanded sign, no
   foot-slip runaway, no violent oscillation, and no guard/boundary event.
9. Combined/transition: test the manifest's forward/backward arc commands and
   arbitrary-to-zero transitions, one at a time. PASS requires no fall at the
   command change and return to stable stand.

Record for every stage: command, duration, package hash, raw present positions,
raw goal positions, IMU orientation, foot switch states, emergency-stop result,
and PASS/FAIL. A single FAIL stops escalation and reverts to torque-off.
