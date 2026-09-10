# Contact-template repair of all 59 source trajectories

## User goal
The user expanded scope from five additional examples to EVERY trajectory in source Lance v5: 59 rows total. Preserve and reuse the first five completed repairs and the five second-batch first-pass results. Focus effort on contact acquisition, stable holding and release; preserve non-contact motion unless a bounded transition is necessary. No RL, no training implementation, no forced object poses. Every source UUID must receive a status and an actual repaired motion record only after physical task validation.

## Scope and ownership
Work only in case/direct-replay-trajectory-repair/second-batch, parent feature feat/direct-replay-trajectory-repair. Source and first-batch NAS bundle are immutable. Owned surface: new task tools, generated second-batch outputs, task memory, contact patches/tests/docs only. Primary dev remains unedited.

## Coverage
First accepted source rows0/19/31/36/49 remain immutable. Second-batch rows50/51/54/55/58 all acquired and held for5s in first-pass replay; rows51/55 have measured posthold leveling candidates to check. Remaining source indices must all be processed, not substituted away. The parent in this case owns bowl groups (03/07/09), current progress and full-dataset integration. A separate isolated case may own bottle/pitcher template generation after explicit delegation; only one GPU writer/run stream at a time.

## Existing source of truth
- Dataset: dexgem_vla_demo_guangxue_cma2lance_20260909_060700.lance, version5,59rows.
- First bundle: dexgem_vla_demo_guangxue_astra_repair_5samples_20260910. tools/replay_repaired_capture.py and tools/export_repaired_motion.py are validated against unchanged sim/manorl/assets.py and trajectory.py in primary checkout.
- Donor patch: patches/guangxue_bowl09_row49_v3.json, recorded bowl-hold trajectory.
- Explicit solver elliptic/impratio100 is allowed from first batch; no further solver/mass/friction/gain changes.

## Latest user refinements
- The user questioned the excessive stationary pause after bowl lift. Long stability-test holds copied into task motion are not normal trajectory content. Remove synthetic constant-command waits from normal outputs, preserve necessary contact transitions and actual source hold intent, and rerun physics after changing timing. Keep extended stability tests as separate evidence, never silently as task data.
- The user explicitly permits larger grasp target closure (pinch tighter). Test modest joint-specific closing increments after acquisition and discharge before release. Do not blanket-add to all axes; do not change wrist targets, mass, friction coefficient or actuator gains merely to increase grip.

## Success criteria
All59 source UUIDs tracked; free-body lift and >=3s held tail for hold tasks, no floor/support contact during held phase, multi-finger contact and bounded drift/tilt. Moving tasks must complete transport/tilt/placement and safe hand withdrawal, with expected final support. Preserve source non-contact prefix/time where practical, and explicitly document changed phases. Judge behavior/geometry, not generated count.
Publish a new full-dataset directory, measured-motion Lance plus hand targets, source+patch replay, videos and per-row validation. Failed or untested rows remain explicit and may not be counted as repaired. Record real elapsed time and first-pass/template-reuse rate. Never overwrite original or first-batch NAS files.

## Operations
At 08:04+08 GPU0 idle except desktop,7.1GiB free. Prior GPU owner notified before bounded sequential one-world replays. No perpetual viewer or training. Use inspectable tmux jobs. Do not kill other workloads.
The global PROMPT.template.md is absent, so this task contract is initialized explicitly. User previously authorized bounded delegation for this ongoing repair objective; no nested delegation.
