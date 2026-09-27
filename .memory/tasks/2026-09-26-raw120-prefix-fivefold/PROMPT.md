## Objective
Produce five distinct, accepted synthetic trajectories for each retained raw trajectory covered by completed raw120 object policies. The user now permits stopping work on raw sources whose historical synthesis success rate is below 25%; preserve their already-published rows and explicitly report their exclusion and released quota. Gradually admit all genuinely idle GPUs on both servers, and automatically add newly completed object policies. Do not interrupt running training or existing synthesis. Benchmark accepted saved trajectories per GPU-hour before selecting production concurrency.

## Workbench
1. Align pre80 raw packages, checkpoint control semantics, prefix perturbations, and offline retreat.
2. Implement resumable per-source five-slot quotas and compact publication.
3. Compare real-asset GPU batch throughput and contact/capacity diagnostics.
4. Keep production supervised; expand onto idle GPUs and admit new final checkpoints with append-only host-local quotas and disjoint cross-host object ownership.

## Context
Feature branch: feat/raw120-prefix-fivefold; base dev 70153d8. Existing code: sim/manorl/approach_prefix.py, tools/run_manorl_prefix_only_coverage_plan_vectorized.py, tools/export_manorl_synthetic_lance.py, tools/build_manorl_full_retreat.py. Historical Far/Near contracts and old Lance readers must remain supported.
Current completed-model inventory: 16 objects, 113 object/action pairs, 13,540 raw sources; goal 67,700 accepted rows. Inventory is a dated observation, not a permanently fixed discovery rule.

## Task specifications
- The retained-source quota is 1:5 accepted saved rows, NOT five attempts. Each source owns five slots; duplicates or other sources cannot fill its quota. The latest user scope permits excluding raw sources below 25% historical success instead of retrying them indefinitely.
- Apply the threshold per raw source, not per object/action. Use at least 20 settled generation attempts before excluding; success means verified durable publication. Do not count unfinished work or infrastructure-abandoned attempts. Exactly 25% remains eligible. Keep all published rows, preserve original inventory targets, separately report skipped sources/slots and the effective remaining target, and never invalidate in-flight ownership or double-credit rows.
- Reference base is pre80; prefix executes in MJX-Warp physics with zero residual during its policy-free phase. Position displacement is 5–50 cm relative to the source reference start; stratify distance/direction/initial hand posture across sources and slots.
- Preserve the original manipulation reference after the prefix. Retreat is offline full-tail replacement anchored at last real contact +15 frames, not an in-simulation retreat.
- Use Cheyingtong right 28-DoF and 120/480 Hz clocks. Preserve provenance, raw source identity, physical hand, checkpoint, sampled perturbation, actual capacities, acceptance, and contact origin.
- Output the compact VLA schema without persisted rollout observations/actions/rewards. Final quota counts require successful postprocessing and verified durable row storage.
- Training CCD remains 8; requested inference CCD is 16. Measure overflow; capacity setting alone is not evidence of non-truncation. Do not silently alter trained controller/reward semantics to load a checkpoint.
- Benchmark comparable workloads with unchanged acceptance criteria; separately report warmup, simulation, contact extraction, postprocessing and disk costs.
- Workers on each host share a local recoverable task queue, use homogeneous-object batches and cache reusable runtime state. Never place SQLite on CIFS/NAS. A durable NAS object-owner registry prevents the same source quotas from being independently generated on both hosts. Persist success, failed attempts, retry position, worker heartbeats and process ownership.
- New user scope: add all newly completed object policies and gradually reuse released GPUs. Current known additions are cuboid3/sphere2; largeclamp/cylinder5 are admitted only when their final report and checkpoint prove completion. Explicit GPU eligibility gates plus live compute ownership protect still-running training.
- Low-success source exclusions retain their evidence and any accepted rows. Completion requires all retained sources at five accepted slots and no unresolved leases; excluded deficits are reported separately and are never relabeled as successful rows. The original 74,805-slot inventory is not the new completion denominator.

## Constraints
Do not edit protected dev directly or alter any running training checkout/queue script.
Do not delete old worktrees, checkpoints, Lance datasets or ignored artifacts.
Do not terminate active training, synthesis GPU workers, or other users' processes. Management-only handoffs may replace this campaign's watch/supervisor. An owned legacy training-only heartbeat may be retired on a host handed to synthesis after verifying that no live training needs it; preserve its scripts and logs.
Subagent authorization in the current request is for monitoring; parent owns implementation in this isolated worktree.
Do not make an LLM sleep/poll indefinitely; server-side supervision owns persistent waiting, and monitoring-agent checks are bounded.
