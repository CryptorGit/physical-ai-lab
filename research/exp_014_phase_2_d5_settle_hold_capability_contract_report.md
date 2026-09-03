# EXP014 Phase 2-D5 Settle/Hold Capability Contract

## Outcome

Classification: `EXP014_D5_SETTLE_HOLD_CONTRACT_PASS`. Selected S_HOLD: `C1_EXP007_STAND` (`734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621`). Selection used validation only; held-out had no fallback.

## Contract

RESET_TO_STAND requires simultaneous XY speed and absolute yaw <=0.08 by 1.0s and 50 continuous in-threshold steps, with no fall, dangerous slip, impact, or long-dwell saturation. STAND_HOLD starts at the first state after that policy-generated 50-step hold and evaluates an additional 100 steps using mean <=0.08 and p95 <=0.12 for speed and yaw. Legacy metric measures immediate reset quietness and conflates transient settling with steady-state holding. It is retained for historical comparability but is not the primary capability gate in Exp014StandCapabilityContractV2.

## Validation

| Candidate | Reset | Conditional hold | Joint | Acquisition p95 | Legacy |
|---|---:|---:|---:|---:|---:|
| Stage 2Q | 96.08% | 98.98% | 95.10% | 0.940s | 58.82% |
| exp007 STAND | 96.08% | 98.98% | 95.10% | 0.903s | 63.73% |

Both were eligible. exp007 won the acquisition-p95 tiebreak before the final C0 preference.

## Held-out and parity

Frozen held-out: reset 96.08%, conditional hold 98.98%, joint 95.10%, legacy 57.84%. Same-process independent-scene runs and two fresh-process runs matched exactly, including actions, acquisition times, classifications, and aggregate metrics. An aborted preflight accessed held-out before a post-evaluation dtype bug; it produced no adopted result and is disclosed in protocol.

## Boundary and post-stop

Boundary labels: 2720 samples, continuation 97.79%, pre-authorization PASS. Post-stop diagnostic: S_STOP practical stop 48.25%; among 193 confirmed stop states, selected S_HOLD held at 100.00%. Aggregate fall/slip (51.50%/25.25%) is caused by missing 90/180-degree S_STOP coverage. Action L2 was 0.0347. `STAND_AFTER_STOP` may therefore use S_HOLD only after an explicit S_STOP practical-stop gate.

No policy update, PPO, checkpoint creation, reward/physics change, DAgger Dataset V2, Student training, or RUN integration occurred. The old D3 result remains unchanged and is not retroactively passed.
