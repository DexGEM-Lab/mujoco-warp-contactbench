# Operations and evidence

## 2026-09-09T20:51:36+08:00 — pre-change dataset inspection

Prediction: the runtime can train one active object per environment, but the selector will reject composite `index.scene` strings before MJX initialization.

Observed dataset version 5 with 59 rows. Every row contains two `objects`, one right hand, and exactly one `object_move` entry. Active-object counts are bowl=39, mayonnaisebottle=12, pitcherbase=8. The active object is the first comma-separated scene object in all 59 rows. Current `load_assigned_trajectory_batch` rejects every row with `no eligible non-generated Lance object/action pairs` because it requires `index.scene` to equal one object name.

The current asset registry and grasp mapping accept all five intended pairs: bowl:03/07/09, mayonnaisebottle:05, pitcherbase:06.

## 2026-09-09T21:00:00+08:00 — adapted dataset validation

Prediction: all 59 source rows will become eligible after deriving the active object from `object_move` and mapping it through the ordered comma-separated scene list.

Observed candidate counts: bowl:03=19, bowl:07=10, bowl:09=10, mayonnaisebottle:05=12, pitcherbase:06=8, total=59. A 95-environment balanced assignment window covered all 59 unique source rows. A 59-environment window covered 52 unique rows because the established selector balances slots across pairs before cycling within each pair; this is sampling behavior, not dropped candidates. The focused trajectory suite passed 22 tests with 7 integration skips.

## 2026-09-09T21:10:32+08:00 — policy-free GPU replay

Prediction: bowl:03 row 0 will run for its full 586-step reference horizon with zero residual actions and publish a Rerun artifact.

The first attempt exposed an existing recorder clock bug: a no-checkpoint 100 Hz replay inherited the default 120 Hz control clock and failed the coupled public-clock contract. `record_manorl_rerun.py` was corrected so policy-free control follows the resolved reference FPS while checkpoint replay retains checkpoint ABI timing.

The next attempt reached asset validation but found the new worktree's DexStream submodule uninitialized. After checking out the pinned `f98da997f316c8a6b4bc2931cabed19e831ef163` assets, GPU replay completed. Rerun published `outputs/replays/dexgem_vla_bowl03_reference.rrd` (18 MiB); `rerun rrd verify` loaded it without error, and its final recorded trajectory step is 584, covering the 586-state source reference after reset semantics. A web viewer serves the recording on ports 9090/9876 from tmux session `manorl-composite-replay:web`.

## 2026-09-09T21:25:00+08:00 — DISPLAY=:1 interactive replay

`DISPLAY=:1` and `/run/user/1000/gdm/Xauthority` accepted X11 connections. An interactive GPU viewer is running in tmux window `manorl-composite-replay:display1` with bowl:03, version 5, 100 Hz, residual actions disabled, terminal deviation disabled, and loop enabled. The live log reached calls 0 and 100 without reset or runtime error; the process remains active for user inspection.
