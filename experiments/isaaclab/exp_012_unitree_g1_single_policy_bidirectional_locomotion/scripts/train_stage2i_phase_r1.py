"""Stage 2I launcher for the audited Stage-2E strict-resume training harness."""

from pathlib import Path

source_path = Path(__file__).with_name("train_stage2e_phase_a.py")
source = source_path.read_text(encoding="utf-8")
replacements = {
    "Stage 2E Phase A: one focused continuation from the iteration-100 checkpoint.":
        "Stage 2I reverse continuation Phase R1 from the exp_005 Stage-4 RUN parent.",
    "stage2e_phase_a_run_acquisition_preflight": "stage2i_reverse_continuation_phase_r1",
    'PARENT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2_pilot1_retry1/checkpoints/model_100.pt"':
        'PARENT = REPO / "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-18_00-44-32_stage4_1024_500/model_5244.pt"',
    "8d8afac60cafbd4adf0b98469fab01f711f32771a40899653d962cc08a5d8143":
        "90d1a360587142d7e312db00a281505a027ecb221733eea6451a885868f6ccc9",
    "Isaac-Exp012-G1-PhaseA-RunAcquisition-v0": "Isaac-Exp012-G1-Reverse-PhaseR1-v0",
    "20265021": "20266021",
    "PHASE_A_": "REVERSE_PHASE_R1_",
    "PhaseA": "ReverseR1",
    "[87000]": "[105000]",
    "100 + local_iteration": "5244 + local_iteration",
    '"source_iteration": 100': '"source_iteration": 5244',
    '"phase_a_iteration": local_iteration': '"phase_r1_iteration": local_iteration',
    "runner.current_learning_iteration = 100": "runner.current_learning_iteration = 5244",
    '"all_environments_phase_a": True': '"phase_r1_curriculum": True',
    "phase_a_run_event_timeline.csv": "phase_r1_run_event_timeline.csv",
}
for old, new in replacements.items():
    if old not in source:
        raise RuntimeError(f"Stage 2I harness replacement target missing: {old}")
    source = source.replace(old, new)

code = compile(source, str(Path(__file__).resolve()), "exec")
exec(code, {"__name__": "__main__", "__file__": str(Path(__file__).resolve())})
