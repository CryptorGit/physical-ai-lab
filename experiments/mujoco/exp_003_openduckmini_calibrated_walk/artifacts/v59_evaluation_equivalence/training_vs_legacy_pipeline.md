# v59 Training vs Legacy Evaluation Pipeline

## Processing-stage comparison

| Stage | MJX v59 training path | Legacy formal evaluation path | Evidence / consequence |
| --- | --- | --- | --- |
| Raw external command | 7-vector: body `vx, vy, yaw` plus four head commands | CLI 3-vector; four head values inserted as zero | `joystick.py:_get_obs`; `evaluate_official_policy.py:run_episode` |
| Command adapter | No ramp; command held for 500 control steps | No ramp; optional positive-yaw `vy -= 0.06` (default) | `joystick.py:step`; evaluator CLI default |
| Command head | Same 7-vector observed by actor | Synthesized 7-vector | observation slots 6:13 |
| Teacher command | Reverse branch when `vx < -0.02` | Reverse feedforward guarded by calibrated-scene flag | `joystick.py:647-710`; evaluator `calibrated_hardware` |
| Teacher action | Optimized periodic reverse frames; left/right yaw profile blend | Disabled in the formal non-calibrated scene | formal JSON `scene` |
| Policy observation | 101 float32 values; backlash-added joint position | 101 values, but non-backlash scene/state | `joystick.py:_get_obs`; evaluator `_policy_observation` |
| Normalization | Brax checkpoint Welford mean/std | ONNX embeds same mean/std | checkpoint tree; `export_onnx.py` |
| Actor | 101→512→256→128→28, swish; tanh of first 14 | Same ONNX architecture | checkpoint tree; ONNX graph |
| Residual composition | Reverse: teacher target + scaled 14-joint residual | Formal scene: actor used as standalone action | scene-name teacher gate |
| Direct action scaling | Calibrated nonlinear directional span | Non-calibrated linear `default + 0.25*action` | `joystick.py:647-681`; evaluator |
| Action delay | Training samples history index 0..2 | No matching history sampler | config vs evaluator |
| Backlash | Deterministic backlash joints in scene and observation | Absent | scene XML paths |
| Motor clamp | speed limit, coupled-head envelope, measured joint limits | Legacy model/control clamp | implementations differ before physics |
| Simulator | MJX, calibrated/backlash XML, 0.002 s ×10 | CPU MuJoCo, flat XML, 0.002 s ×10 | training launch and formal JSON |

## Structured paths

### Training

`raw command(7,float32,body/head)` → fixed command storage → reverse teacher
selection → optimized periodic reference and phase → 101-D observation
(backlash-aware) → checkpoint normalization → actor residual → reverse
teacher+residual or calibrated nonlinear direct target → sampled action delay →
speed/head/joint clamps → calibrated motor target → MJX step.

### Legacy formal evaluation

`raw command(3,float64,body)` → optional positive-yaw lateral compensation →
zero head extension → no reverse teacher because the selected scene stem does
not contain `calibrated` → 101-D legacy observation → ONNX normalization and
actor → standalone linear target → legacy clamps → non-backlash CPU MuJoCo
step.

The first structural divergence is **scene selection**, before observation
construction.  The first command-dependent controller divergence for reverse
commands is **teacher routing**.  Numerical trace results are recorded in
`parity_report.json` and the CSV tables.
