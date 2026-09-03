# command_system_v1

Parameter-free external command router for the qualified `RUN`, `TURN`,
`STAND`, and `CROUCH_SHALLOW` controllers. It keeps the 152-D observation and
the original six-way policy skill encoding unchanged. Cross-family requests
are rejected atomically; no implicit STOP or command rewriting is performed.

`STOP` remains a prototype. `CROUCH_DEEP`, `STEP_OVER`, and `LAND` remain
unsupported. The measured 0.02 m standing drop tolerance is passive robustness,
not a LAND capability.
