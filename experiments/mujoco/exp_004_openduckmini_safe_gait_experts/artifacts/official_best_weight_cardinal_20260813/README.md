# Public weight: cardinal-command comparison

`official_best_walk_onnx_2_cardinal_comparison_fixed_camera.mp4` compares
four simultaneous 6-second replays of the public `BEST_WALK_ONNX_2.onnx`
policy.  The video layout is:

| Upper left | Upper right | Lower left | Lower right |
| --- | --- | --- | --- |
| Forward `vx=+0.15` | Backward `vx=-0.15` | Left `vy=+0.20` | Right `vy=-0.20` |

Every panel starts independently from the same official `home` state and uses
the unmodified public Open Duck Playground replay contract at commit
`1842c8f46a67cb5d6b74e5aaf08c8702cde6e74f`.  The comparison encoder only adds
labels and a 2×2 layout; it does not change simulation, observations, actions,
or targets.

| Command | World displacement after 6 s (m) | Minimum upright cosine | Frames |
| --- | ---: | ---: | ---: |
| Forward | `[+0.553932, +0.081175]` | `0.996489` | 150 |
| Backward | `[-0.181198, -0.073346]` | `0.996616` | 150 |
| Left | `[+0.056709, +0.469607]` | `0.997662` | 150 |
| Right | `[+0.004954, -0.373539]` | `0.996816` | 150 |

All four deterministic 6-second replays remain upright; this is not a
multi-seed robustness claim and not a real-robot result.  In particular,
backward motion is substantially slower than forward in this published weight.

## Evidence

- The comparison MP4 is H.264, 960×720, 25 fps, 150 frames, 6.000 seconds.
  SHA-256: `53e700318798f025d19d4c4e05a8870c4bb6673a6a653fe3389d0a5b48336a1d`.
- `forward/`, `backward/`, `left/`, and `right/` each retain the raw fixed-camera
  MP4 and a `manifest.json` with command, hashes, first policy action, source
  commit, and rollout data.
- All raw manifests identify the public policy SHA-256
  `3c606f9381a1710cc8fecdb7442787dcbfce3ee9bc02a6f1224774ab2b3a1067` and the
  same renderer-script hash.

## Scope

This is a public-simulation replay only.  It deliberately does **not** use the
local hardware servo offsets or send a real actuator command.  It demonstrates
that the earlier falling video came from an incompatible local evaluator, not
from this official replay contract.
