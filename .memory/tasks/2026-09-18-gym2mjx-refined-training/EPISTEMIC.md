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
The previous Mayo-only campaign is preserved and released. Server1 seeds42/43/44 stopped at complete checkpoint1100; Server2 seed45 completed its200-update canary with EXIT0. No task-owned trainer remains. Server1 GPUs0/2/3 and both Server2 GPUs are available; Server1 GPU1 remains owned by `fyr`.

The all-object request contains21 objects/122pairs/12,200 rows. Four balanced multi-object policy shards are defined in `configs/manorl/all_objects_four_gpu_shards.json`:3100/3000/3100/3000 rows and high-confidence loads2129/2116/2089/2180. N620 for31-pair shards and N600 for30-pair shards allocates exactly20 trajectories per pair per stage; five pair-assignment cycles cover every trajectory exactly while strict resume preserves each shard's policy and optimizer. This creates four independently deployable policies over disjoint object sets—not one universal checkpoint.

Twenty objects are runtime-ready under the pinned778 asset profile. `scissor` is the sole blocker to honest all-object delivery. An authoritative candidate exists at DexStream120c (0.075kg,9 convex pieces), but that repository history also changes other object physics and its manifest layout differs. Scissor needs an explicit mixed-provenance/runtime asset boundary; substituting another geometry or silently omitting its500 rows would invalidate the objective.

## Most informative next observation
Pin and validate the scissor runtime without changing the other20 objects'778 physics, then compile/preflight the four shard MTPs. The first discriminating preflight is shardB (largeclamp27 pieces) versus shardA (bowl25 pieces) at N600/620; their peak memory determines whether the five-cycle plan fits24GiB or needs a smaller per-cycle world count.
