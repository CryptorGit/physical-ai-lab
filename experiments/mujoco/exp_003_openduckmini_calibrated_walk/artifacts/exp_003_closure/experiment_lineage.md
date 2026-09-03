# exp_003 Experiment Lineage

| Stage | Parent / package | Main change | Decision |
| --- | --- | --- | --- |
| calibrated v22 | official actor/reference | measured range constrained scene and runtime | simulation baseline |
| v45 | learned actor | stronger locomotion parent | actor parent for v52 and later branches |
| v49/v50 | v45 + reverse profile | reverse straight periodic feedforward | retained in v52 package |
| v51 | v45 + teacher/residual | learned residual around reverse reference | diagnostic |
| v52 | v45 + v50/v52 profiles | reverse straight/turn hybrid package | adopted simulation parent |
| v53 | v52-related | robust turn branch | not adopted |
| v54 | v22 checkpoint | separate omnidirectional branch | not continuation of v53/v55 |
| v55 | v45 checkpoint | new omnidirectional branch | stopped early |
| v56 | v45 checkpoint | teacher/turn diagnostic branch | not parent of v57 |
| v57 | v45 checkpoint | separate diagnostic branch | not continuation of v56 |
| v58 | v45 checkpoint | long omnidirectional run | diagnostic only |
| v59 | v58 policy/value/normalizer | fresh Adam, half LR, shared progress change | `diagnostic_not_qualified` |
| v60 C | v52/v45 package | matched old objective, fresh Adam | pilot failed |
| v60 T | same parent | bounded yaw objective only | `STOP_AT_1M` |
| reduced-LR | v52 package | initial LR 3e-4 -> 1.5e-4 | inconclusive |
| group normalization | v52 fixed batches | command-regime mean/std normalization | offline Gate FAIL; not trained |
| bounded scaling | v52 fixed batches | bounded regime RMS scaling | offline Gate FAIL; not trained |
| C04 negative cap | v52 package | C04 negative actor contribution x0.65 | closed-loop NOT SUPPORTED |

Important: version labels are research labels, not a guaranteed parent-child chain. Always use the parent/checkpoint/source/config manifests rather than inferring lineage from the version number.

