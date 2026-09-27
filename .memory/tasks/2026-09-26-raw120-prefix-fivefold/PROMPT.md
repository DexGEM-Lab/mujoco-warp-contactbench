## Objective
For every completed object–action policy pair, produce a FIXED number of accepted rows equal to five times that pair's ORIGINAL training-source count. The25% cutoff changes which sources may generate rows; it NEVER reduces that pair's output target. Reallocate excluded sources' unfinished output quota to eligible sources in the SAME object–action pair, allowing each eligible source to produce more than five distinct variants. Example:100 original sources -> target500; excluding50 sources still leaves target500, supplied by the remaining50 sources. Existing verified rows continue to count for their own pair. Gradually use genuinely idle GPUs without interrupting other jobs or overwriting saved data. The previously reported overall completion was an assistant interpretation/implementation error.

## Workbench
1. Align pre80 raw packages, checkpoint control semantics, prefix perturbations, and offline retreat.
2. Implement resumable fixed per-object–action output budgets, independently selected eligible sources, unique extra-variant lineage and compact publication.
3. Compare real-asset GPU batch throughput and contact/capacity diagnostics.
4. Keep production supervised; expand onto idle GPUs and admit new final checkpoints with append-only host-local quotas and disjoint cross-host object ownership.

## Context
Feature branch: feat/raw120-prefix-fivefold; base dev 70153d8. Existing code: sim/manorl/approach_prefix.py, tools/run_manorl_prefix_only_coverage_plan_vectorized.py, tools/export_manorl_synthetic_lance.py, tools/build_manorl_full_retreat.py. Historical Far/Near contracts and old Lance readers must remain supported.
Current completed-model inventory: 16 objects, 113 object/action pairs, 13,540 raw sources; goal 67,700 accepted rows. Inventory is a dated observation, not a permanently fixed discovery rule.

## Task specifications
- The1:5 ratio fixes each object–action pair's accepted-row target from its original source count, NOT its surviving source count and NOT attempts. Good sources within that pair may supply additional distinct variants; another pair's surplus cannot satisfy its budget. Remove the per-source five-output ceiling without colliding with existing seeds, UUIDs or acceptance credit.
- Apply25% eligibility per raw source, independently of output budget. The previously declared20-settled-attempt floor and exclusion of unfinished/infrastructure-abandoned attempts remain unless explicitly revised; exactly25% is eligible. Source removal must not cancel budget slots. Preserve all already-published valid rows and credit them only to their original pair. A pair with no eligible sources is blocked, not complete.
- Reference base is pre80; prefix executes in MJX-Warp physics with zero residual during its policy-free phase. Position displacement is 5–50 cm relative to the source reference start; stratify distance/direction/initial hand posture across sources and slots.
- Preserve the original manipulation reference after the prefix. Retreat is offline full-tail replacement anchored at last real contact +15 frames, not an in-simulation retreat.
- Use Cheyingtong right 28-DoF and 120/480 Hz clocks. Preserve provenance, raw source identity, physical hand, checkpoint, sampled perturbation, actual capacities, acceptance, and contact origin.
- Output the compact VLA schema without persisted rollout observations/actions/rewards. Final quota counts require successful postprocessing and verified durable row storage.
- Training CCD remains 8; requested inference CCD is 16. Measure overflow; capacity setting alone is not evidence of non-truncation. Do not silently alter trained controller/reward semantics to load a checkpoint.
- Benchmark comparable workloads with unchanged acceptance criteria; separately report warmup, simulation, contact extraction, postprocessing and disk costs.
- Workers on each host share a local recoverable task queue, use homogeneous-object batches and cache reusable runtime state. Never place SQLite on CIFS/NAS. A durable NAS object-owner registry prevents the same source quotas from being independently generated on both hosts. Persist success, failed attempts, retry position, worker heartbeats and process ownership.
- New user scope: add all newly completed object policies and gradually reuse released GPUs. Current known additions are cuboid3/sphere2; largeclamp/cylinder5 are admitted only when their final report and checkpoint prove completion. Explicit GPU eligibility gates plus live compute ownership protect still-running training.
- Completion requires EVERY object–action pair to meet its fixed original-count×5 budget, plus no unresolved work/publication. The current inventory target remains74805;24006 verified rows exist, leaving50799 obligations. The old source-filter completion claim is withdrawn. Frozen data/hash reports remain valid as partial-data evidence, not as proof of goal completion.

## Constraints
Do not edit protected dev directly or alter any running training checkout/queue script.
Do not delete old worktrees, checkpoints, Lance datasets or ignored artifacts.
Do not terminate active training, synthesis GPU workers, or other users' processes. Management-only handoffs may replace this campaign's watch/supervisor. An owned legacy training-only heartbeat may be retired on a host handed to synthesis after verifying that no live training needs it; preserve its scripts and logs.
Subagent authorization in the current request is for monitoring; parent owns implementation in this isolated worktree.
Do not make an LLM sleep/poll indefinitely; server-side supervision owns persistent waiting, and monitoring-agent checks are bounded.
