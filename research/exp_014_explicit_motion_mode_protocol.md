# EXP 014 explicit motion-mode protocol

The pre-registered question is whether an explicit target motion mode resolves the zero-command ambiguity that prevented EXP 013 from retaining practical stop and start acquisition simultaneously. The actor sees current physical command, target and previous target one-hot modes, previous command, command delta, time since mode change, and ramp progress after the unchanged legacy 124D input. A WALK request therefore becomes observable at B0 even while physical velocity remains zero.

Formal labels are restricted to EXP 012 Stage 2Q for practical STAND/stop and forward WALK/RUN, and EXP 013 W1B-R2 iteration 200 for omnidirectional WALK. A7/A8/A9 policies and probes are diagnosis-only and cannot label the formal dataset.
