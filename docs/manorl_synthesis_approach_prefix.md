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
