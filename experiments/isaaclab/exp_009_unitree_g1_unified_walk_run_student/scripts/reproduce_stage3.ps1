$ErrorActionPreference = "Stop"
$python = "C:\Users\user\workspace\IsaacLab\env_isaaclab\Scripts\python.exe"
$base = $PSScriptRoot
& $python "$base\collect_stage3_student_rollouts.py" --viz none
& $python "$base\build_stage3_surrogate_dataset.py" --force
& $python "$base\train_stage3_surrogate.py"
& $python "$base\evaluate_stage3_surrogate.py"
& $python "$base\finalize_stage3.py"
