## Phenomenon
The corrected v2.2 synthetic Lance rows are semantically valid but storage-heavy: a full checkpoint/runtime JSON is repeated identically in every row, and rollout intermediates are persisted even though target-DOF replay does not consume them.

## Mechanism
The exporter serializes an audit/training superset as the per-row contract. Lance stores each nested field per episode and does not deduplicate the identical checkpoint metadata blob. The direct replay consumer only needs six top-level fields and uses target DOF/object pose plus CCD scalar settings; visual consumers additionally need the MANO global pose, 48D hand pose, and 21x3 joint frames.

## Intervention
`synthetic_mano_target_replay_visual_v1` is now a separate schema. `build_compact_row()` projects only replay/visual fields and replaces repeated runtime JSON with a canonical metadata SHA256. Checkpoint metadata is written once in the synthesis manifest or an external catalog by `tools/compact_manorl_synthetic_lance.py`; Warp CCD iterations/contacts remain direct row scalars so replay does not parse the large JSON. `synthesize.sh` defaults to compact; `full` remains an explicit v2.2 audit mode.

## Supported claim
Compact rows preserve target/replayed DOF alignment, recorded object pose, MANO global pose/hand pose/21-joint visual frames, source/generated lineage, checkpoint identity, and CCD restoration inputs. They are a replay/visual contract, not an offline-training contract. Existing v2.2 rows remain readable by the replay decoder and are not rewritten.

## Evidence
- Focused project-environment tests: 30 passed across Lance schema/export, target replay, and shell entrypoints.
- Real current-NAS v2.2 row projected and decoded: 578 frames, compact contract, CCD 16/16; compact validator passed.
- One-row full fixture streamed through the compactor: compact row count/schema and external metadata catalog validated.
- Broad ManoRL suite is not a clean acceptance signal in this worktree because required materialized assets and historical datasets are absent; unrelated failures include submodule pin, device, and grasp tests.

## Remaining uncertainty
The compact projection deliberately excludes observations, actions, rewards, contact forces, and reference trajectories. If an offline-training consumer requires them, it needs a separately named training schema rather than expanding compact silently. The feature still needs coordinator review/integration into `dev`; published NAS data must remain unchanged until separately approved.
