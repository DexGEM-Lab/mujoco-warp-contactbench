# Epistemic Model

## Phenomenon
The source RL trajectories are internally coherent, but their physical transfer to a fixed `cheyingtong` hand under MJX-Warp depends on where the task-motion window begins and how much residual authority the policy receives. The current intervention asks whether reference-derived onset, a resolved 0.5s pre-window, 1.0s early phase, 0.15m terminal bound and 2× finger residual range produce learnable shared Mayo policies.

## Supported mechanism
The binary/schema boundary is solved: pylance7 reads encodings21; the refined-RL importer preserves source provenance and emits Lance-free MTPs. Timestamp spacing, not unreliable `data_fps`, determines physical duration and is resampled to120Hz.

Reference movement onset is now explicit for all12,200 rows. The detector separates initial gravity settling from sustained translation/rotation by anchoring a 100ms quiescent pose, then requiring a persistent2mm/2° departure with5mm/5° confirmation. It records high/low confidence rather than silently dropping ambiguity. Server1-local annotation validation proves12,200 original rows/columns remain exactly equal;8514 rows are high confidence, with duplicate source UUID multiplicity preserved by row index.

For Mayo,397/600 rows have high-confidence onset across all six actions. Resolving every row to120Hz and requesting pre60 gives movement_start_step60 universally. Only132 rows contain the full captured0.5s margin;265 need a frame-zero hold, up to48 control steps. This is intentional and audit-visible—not equivalent to claiming captured pre60 for every row.

The fresh training contract is executable: pre60, early120, max deviation0.15m, joint increment/cap multipliers2.0, 120Hz control/480Hz physics, Cheyingtong assets, package digest `fc279...`. Three independent Server1 preflights completed without memory/contact-capacity faults and produced25–28 stochastic successes in update5. Three production seeds crossed checkpoint100. The earlier pre0/early30/0.10m/joint1x run is stopped at checkpoint7600 and is not a resume source because its MDP/action scale differ.

## Ruled out
The original failure is not simply a clock bug: zero-residual success was similarly low for observed120Hz and200Hz cohorts. It is not initial coordinate mismatch: failed rows begin near the target but make weak/brief contact while the target moves away. A first-numerical-change onset detector is invalid because it labels gravity settling as task motion.

Four GPUs on Server1 are not all available: GPU1 belongs to user `fyr`. No foreign process was stopped. One policy cannot be synchronously spread across current trainer GPUs; independent seeds are the valid parallel unit. Server2 cannot yet be considered equivalent production capacity: it has recurrent kernel-level Python/libcuda/Vulkan segfault history through Sep13, even after MTP isolation.

## Anomalies
Mayo high-confidence onset is uneven by action (01:59,02:43,03:51,04:80,05:100,08:64). Action05 has no full captured0.5s pre-window; all its examples require edge holds, so future per-action comparisons must separate edge-hold exposure from policy difficulty. The full annotated Lance is now published to NAS with byte-identical source columns and adjacent diagnostics; its duplicate source UUIDs remain intentional row-level identities.

## Current claim
Fresh training is active under the requested contract. Server1 runs seeds42/43/44 on GPUs0/2/3, each N600/U8000 and checkpoint100 durable; W&B IDs are `v6rnhhzv`, `g3wpu866`, and `zt6azhn6`. Server2 seed45 is a bounded U200 canary on GPU0; it crossed checkpoints25/50/75 and remained healthy at update93, W&B `g864xyml`. Every inspected sidecar binds pre60, early120,0.15m, joint2×,120/480Hz, package `fc279...`, and Cheyingtong asset commit778614d. Long-lived trainers have no Lance/PyArrow mappings. The annotated12,200-row Lance and audit sidecars are published under `/mnt/nas-222-projects/sunjieqiang/new_vla/`.

## Most informative next observation
For Server1, compare deterministic six-action evaluation and rolling per-action success at checkpoint200/500 across the three seeds, especially action05 versus actions with captured pre context. For Server2, completion of update200 with reloadable checkpoint200 and no new kernel segfault is the minimum evidence to promote that host beyond canary status.
