# exp_014 Phase 2-D10 frozen S1 stop closed-loop report

## Result

Classification: **EXP014_D10_FROZEN_S1_CLOSED_LOOP_PASS**. The immutable D7 S1 step 30,000 actor ran alone from STOP request through acquisition confirmation; no Teacher, internal route switch, action blend, phase classifier, or policy update was used.

Formal validation evaluated 3,400 fixed snapshots across all 34 conditions. Moving-start validity was 99.12% (3370/3,400). Given the fixed valid snapshots, STOP acquisition, conditional S_HOLD, joint success, and minimum-condition joint success were all 100%. End-to-end success including moving start was 99.12%. Fall, slip, impact, velocity saturation, torque saturation, and NaN/Inf were zero. The fixed 204 local-neighborhood episodes also achieved 100% joint success with zero safety failures.

S1-to-S_HOLD action L2 p95 was 0.043684, cosine p05 0.999983, and joint-target jump p95 0.021842 rad L2. Root discontinuity, contact change, and new handoff safety failures were zero.

Frozen S1 passed directly, so first-divergence, Student-visited labelability, DAgger collection, and DAgger training were not executed. Two independently reconstructed scenes in one process and two fresh processes matched snapshot/observation/action hashes, classifications, and aggregate metrics exactly. Held-out remains sealed and unopened. The next single experiment is one-time sealed held-out evaluation with no fallback.
