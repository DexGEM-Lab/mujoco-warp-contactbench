# All59 normal contact repairs: final model

## Goal and delivered behavior
All59source-v5 trajectories have normal-motion contact repairs. Registry:
outputs/full_repair/normal_selected/registry.json,59accepted/65000normal frames.
NAS full bundle: dexgem_vla_demo_guangxue_astra_repair_all59_20260910. Source and
first5bundle unchanged. Full package normal videos/data exclude diagnostic tails;
each stability_tests/rowXX holds a separate200framecontinuation.

## Timing correction
Earlier all59long-test runs were physically successful but copied the donor's
10svalidationhold into moving-bowl task motion. The user caught this. Final
normal streams remove only long constant-command plateaus, preserve moving
commands, retain0.3ssettle allowance, and were physically rerun. Typical moving
bowls25→13.34s, held row49 13.85→5.30s. Movement is still conservative/retimed;
exact original-speed fidelity is not claimed. Each normal trajectory is the
bit-equal prefix of its NEW physical audit run, never just an oldvideo cut.
Normal terminal contacts/support are checked independently before the audit tail.

## Stronger grip and repeatability
User permits tighter targets. Uniform+.04/.08radclosure did not generalize to
pitcher handles: it can change exit geometry. Final row32uses modest .025–.04rad
closing preload. Row39uses measured middle-finger support: MCP+.27rad and
PIP/DIP-.185rad, establishing a missing side contact; two independent full
replays pass. Row38uses a previously verified short donor37contact primitive
conditioned on receiver scene, and repeated runs pass. Other57rows retain already
sufficient targets.

Two pitcher templates were marginal: identical controls/initial coordinates
amplified micrometre solver differences at support removal into centimetres.
Theorist isolated overload/servo lag, not clock or Euler-command mismatch.
Local2xpickupdilation reduced wrist lag but did not resolve the full task; generic
normal-force offsets and raising wrist under arch also failed. These negatives
justify replacing the weak primitive with short verified donor37and adding the
missing middle contact, rather than treating one lucky run as a universal fix.

## Physical/data invariants
Free-body objects after reset, actuator targets only; no object wrench/weld/pose
forcing or learned policy. Explicit local elliptic/impratio100, unchanged
mass/mu/gravity/gains. Some source prefixes/timing retained; hard examples use
retimed task primitives conditioned on receiver initial/bowl/goal placements.
Output Lance is measured generated motion, not a drop-in source-capture format.
No fluid simulation/volume-delivery claim; outlet-over-bowl geometry is validated.
Small contact variation remains; these are saved task repairs, not robust policy
coverage for arbitrary perturbed initial states.

## Evidence and finalization
Parent inspected all59normal terminal mosaics and critical39/38/33/26keyframes.
Critic8797289e corroborated all long-run task outcomes; criticec43a41a verified all
normal/audit prefixes, exact65000frameaccounting, no moving-target loss, separate
200frameextensions and valid normal terminal states. Final uniform per-row
validation exists, including first5carryovers, without modifying original bundle.
First critic failed by accidentally decoding source video columns; corrected
read-only runs used localNPZonly. Empty case submodules are not asset drift; use
primary materialized source for imports. Final package command checks, source
UUIDroundtrip and user replay entrypoint are completing. Audit trail: OPS.md.
