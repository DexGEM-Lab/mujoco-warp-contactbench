## Phenomenon
The full synthetic Lance contract stores an audit/training superset per episode. A large checkpoint/runtime JSON is repeated identically in every row, while target-DOF replay does not consume contact, reference, or rollout intermediates. During implementation, `dev` also advanced the full contract from legacy v2.2 to clock-aware v2.3.

## Mechanism
Lance stores each nested field per row and does not deduplicate the repeated metadata blob. Replay needs target/recorded DOF, object pose, lineage, checkpoint identity, and CCD settings; visualization additionally needs MANO global pose, 48D hand pose, and 21x3 joint frames. Clock-aware v2.3 requires the replay controller to use per-row policy/physics clocks and substep counts rather than a hard-coded 200 Hz/2-step assumption.

## Intervention
`synthetic_mano_target_replay_visual_v1` is a separate schema. It projects either full v2.2 or v2.3 rows, stores canonical metadata SHA256 plus explicit Warp CCD scalars, and records source contract and reference/control/physics clock fields. Metadata is externalized once in the synthesis manifest or compactor catalog. `synthesize.sh` defaults to compact; `full` remains explicit. Target replay accepts legacy full v2.2, full v2.3, and compact rows, deriving its clock and physics substeps from the row.

## Supported claim
Compact rows preserve target/replayed DOF alignment, object pose, MANO visual frames, source/generated lineage, checkpoint identity, CCD restoration inputs, and clock semantics. They are replay/visual data, not an offline-training contract. Existing full v2.2/v2.3 datasets remain readable and are not rewritten.

## Evidence
- Focused project-environment tests: 30 passed across Lance schema/export, target replay, and shell entrypoints.
- Real one-row v2.2 full fixture projected, decoded, and validated: 578 frames, 200 Hz, 2 physics substeps, CCD 16/16.
- Synthetic one-row v2.3 120 Hz full fixture projected, decoded, and validated: 578 frames, 480 Hz physics, 4 substeps, CCD 16/16.
- Full v2.2 and full v2.3 validators passed one-episode fixtures with contact, MANO-frame, force, reward, lineage, and clock checks.
- Broad ManoRL suite remains environment-limited by absent materialized assets/historical datasets and unrelated pre-existing failures.

## Remaining uncertainty
Compact data deliberately excludes observations, actions, rewards, contact forces, and reference trajectories. Any offline-training consumer needs a separately named training schema. The feature is integrated into authoritative `dev`. Published NAS data remains untouched; any migration of the 5,425-row canonical dataset requires separate approval and guarded publication.
