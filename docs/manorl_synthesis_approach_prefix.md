# Default ManoRL Far/Near approach-prefix synthesis

## Production contract

The default accepted-parent synthesis operation is:

```text
canonical pre60 reference
  -> prepend one seeded Far or Near 4 cm approach prefix
  -> run the accepted-parent checkpoint through the complete original reference
  -> preserve the original reference tail unchanged
  -> no retreat suffix
```

The versioned contract is
`manorl_pre60_far_near_approach_prefix_only_complete_original_tail_v1`.
Historical retreat code and datasets remain identifiable, but default production
entrypoints do not request tail replacement.

## What is added

Far starts 0.30–1.00 m horizontally from the initial object, 0.08–0.30 m above
it, within ±30° of the initial object-to-hand direction. Near independently
samples a retreat-like start from accepted-parent movement-end+15 geometry and
maps it to the initial object. For Near, movement-end+15 chooses only the
approach start; it never cuts, replaces, or disables policy in the reference
tail.

Both modes copy raw source frame-0 right-hand `q_ref[3:28]` at the sampled start,
then smoothly reach pre60 frame 0. Wrist translation uses the established 4 cm
endpoint-smooth vertical arc. The prefix is policy-free and has zero processed
and cumulative residual. Solved right-hand/table or right-hand/object contact
above 0.2 N during the prefix rejects that attempt.

After the prefix, the accepted checkpoint runs across the complete original
reference. There is no retreat window, no tail policy shutdown, and no tail
residual discharge.

## Save gate

A candidate is written only when all production conditions hold:

1. it reaches the final reference state with termination reason 1;
2. final simulated/reference object intrinsic-XYZ mean wrapped error is <=35°;
3. at least 101 distinct persisted frames contain solved right-hand/target-object
   normal force strictly >0.2 N; and
4. the approach prefix passed its table/object collision gate.

The manifest records every attempt and sets `retreat_suffix` to null. Prefix-only
UUIDs use `manorl_synthesis_prefix_only_augmentation_identity_v4`; historical
prefix+retreat UUID semantics remain under v3.

## Single accepted parent

`./synthesize.sh` now defaults to prefix-only Far, five accepted rows within a
twelve-attempt bounded budget, with partial yield preserved. Supply the accepted
parent and canonical pre60 bundle:

```bash
MANORL_SYNTH_ACCEPTED_PARENT=/path/accepted-parent.json \
MANORL_PREDECODED_MANIFEST=/path/pre60-bundle/manifest.json \
CHECKPOINT=/path/exact-parent-checkpoint.pt \
./synthesize.sh banana 01 1 0
```

For Near:

```bash
MANORL_SYNTH_APPROACH_MODE=near \
MANORL_SYNTH_ACCEPTED_PARENT=/path/accepted-parent.json \
MANORL_PREDECODED_MANIFEST=/path/pre60-bundle/manifest.json \
CHECKPOINT=/path/exact-parent-checkpoint.pt \
./synthesize.sh banana 01 1 0
```

`MANORL_SYNTH_RETREAT_SUFFIX=true` is rejected by this default entrypoint.

## All-object production

```bash
python tools/run_manorl_all_objects_near_far.py \
  --output-dir /local/output \
  --gpus 0,1,2,3
```

Every verified source independently targets five Far and five Near rows, with at
most twelve attempts per mode. The runner consumes the canonical pre60 bundle,
never passes `--retreat-suffix`, and publishes bounded yield rather than changing
physics to meet a quota.

## Inspection and publication

The viewer defaults to prefix-only; `--retreat-suffix` is a historical inspection
option only. Before publication, run `tools/validate_manorl_synthetic_lance.py`.
For the prefix-only contract it verifies pre60, 4 cm Far/Near config,
`retreat_suffix=null`, v4 augmentation identity, the three-condition save gate,
and compact contact/reference/command mapping.

## Offline full-length retreat (default post-processing since 2026-08-27)

Every published synthesis dataset is now produced in two stages:

```text
stage 1 (simulation): prefix + complete base trajectory (this document's contract)
  -> base Lance with real solver contact (e.g. for_vla_..._20260826.lance, 1036 rows)

stage 2 (offline, kinematic, no solver):
  anchor = true last solved hand-object contact frame + 15
  retreat = replace the entire tail after the anchor (155-251 frames, median 209):
    - only hands[0].urdf_dof[:, :3] (right-wrist XYZ) and mano_global_pos change
    - smoothstep tail deformation (first two tail frames untouched, last three
      reach the endpoint); splice velocity/acceleration error = 0
    - direction = anchor -> original-final-wrist + extra horizontal 3-15 cm,
      +/-30 deg, Z +4-10 cm (historical retreat semantics)
    - length, timestamps, fingers, object, contact and reference preserved
  -> with_retreat Lance (same row length; contact remains real solver data)
```

The offline stage is deterministic (seed from base-row provenance) and uses the
persisted contact array to know the true contact-end frame, which runtime
synthesis cannot. Do not reintroduce in-simulation retreat (contact end unknown
at runtime) and do not use the short 29-frame retreat variant (a validation
skeleton, not a complete retreat).

Tool: `tools/build_manorl_full_retreat.py`
Published example: `for_vla_manorl_prefix_near_far_4pairs_20260826_with_retreat.lance`
(1035 rows, all with full-length retreat).

## Compact row semantics (v2_contact)

- MANO global translation is exactly `urdf_dof[:, :3]`; global axis-angle is
  derived from the URDF floating-root intrinsic `XYZ` composition `Rx @ Ry @ Rz`
  (`mano_global_frame_contract=urdf_floating_root_translation_intrinsic_XYZ_to_rotvec_v1`).
- `force_normal` contains the solved normal component with scale `1.0`; all
  force frames use a consistent hand-to-object direction. `pos_joint` and
  `total_force_joint` use the live collision-link transform rather than the
  historical wrist fallback.
- Shape metadata contains only the raw right-hand shape declared by
  `hand_names=["right"]`.
- Schema contract: `synthetic_mano_target_replay_visual_v2_contact`; source
  contract `synthetic_mano_28d_checkpoint_rollout_v2_3`. Schema metadata,
  row provenance, `data_fps`, and timestamps record the actual 100/120/200 Hz
  control clock together with its physics rate and substep count. The
  validator retains read support for fixed-200-Hz v2.2 datasets.
- Each compact row stores 28D physical and controller targets, 21 keypoints,
  reference frame indices, checkpoint SHA256, runtime sidecar, action
  contract, and source identity. Observations/actions/rewards are dropped in
  compact mode; full mode retains them as the explicit audit contract.

## Example delivery

The validated all-action cube2 delivery generated from contact-v2
`checkpoint-000800.pt` is:

```text
/mnt/nas-222-project/sunjieqiang/mujoco_synthetic/
cube2_all_actions_checkpoint800_ratio5_seed42_v22.lance
```

It contains 213 raw identities across actions `01,02,03,04,10,11`, five
accepted episodes per identity (1,065 rows), and adjacent manifest, validation,
and SHA256 checksum sidecars. All identities completed in five attempts; the
ten-attempt limit remained fail-closed and was not consumed.

## Scene-level augmentation (XY translate / Z rotate) — v2 standard

Since 2026-08-27, published synthesis datasets may be augmented with whole-scene
rigid transforms for VLA generalization. **Both `urdf_dof` AND `urdf_dof_target`
must be transformed** (they are isomorphic 28D rows: `[:3]` wrist XYZ, `[3:6]`
extrinsic-XYZ euler, `[6:]` fingers).

### Representations (verified, do not mix)

| Field | Representation |
|---|---|
| `objects[].rot_aa`, `reference.object_rot_aa` | axis-angle rotvec (`\|v\| <= pi`) |
| `hands[].urdf_dof[:,3:6]`, `urdf_dof_target[:,3:6]` | extrinsic-XYZ euler (scipy uppercase `"XYZ"` = fixed world axes) |
| `hands[].mano_global_rot_aa` | axis-angle rotvec (mirror of the euler) |

The contract name `...intrinsic_XYZ...` is historical/misleading; the actual
convention is extrinsic (verified: `from_euler("XYZ")` matches
`mano_global_rot_aa` to 1.2e-7, `"xyz"` intrinsic does not).

### Transform rules (v2, corrected)

- **positions** (hand XYZ, contact `pos_world`): `p' = Rz(p - o) + o` (row-vector `@ R.T`)
- **object orientation** (rotvec): `v' = log(Rz @ Rodrigues(v))` — compose `R @ mats`, NOT `mats @ R.T`
- **hand orientation**: `R' = Rz @ R` — write back euler in the SAME extrinsic-XYZ
  order AND update `mano_global_rot_aa` to the new rotvec
- **forces** (`force_normal`, `total_force_world`): `f' = Rz f`
- **unchanged**: `pos_wrist/joint/object` local frames, fingers, timestamps
- **`urdf_dof_target`**: same transform as `urdf_dof` (positions + euler), always

Tools: `tools/build_manorl_xy_translate.py` (v2), `tools/build_manorl_xy_rotate.py` (v3).
Sampling: Latin hypercube, 1:1:1 base:translate:rotate, translate +/-0.20 m,
rotate +/-30 deg about the per-frame object center (world Z).
