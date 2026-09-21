# U1 exact800 compact export

Run from this worktree with the campaign's `manorl_mujoco` Python environment.
This command never runs physics or starts production. All five ledgers must
already have exactly160 selected slots and800 distinct campaign semantic UUIDs.
An exhausted, incomplete, changed, duplicate or identity-mismatched campaign is
rejected. No count override or append/replace mode exists.

```sh
PY=/home/jay/anaconda3/envs/manorl_mujoco/bin/python
# Set STAGING to the completed campaign root containing registry/, plan/, attempts/.
: "${STAGING:?set completed campaign root}"
OUT="$STAGING/.u1_exact800_native_v1"
$PY -m tools.export_u1_exact800 export --registry "$STAGING/registry/registry.json" \
  --plan "$STAGING/plan/plan.json" --attempts "$STAGING/attempts" \
  --output "$OUT" --execute-export
$PY -m tools.export_u1_exact800 validate --registry "$STAGING/registry/registry.json" \
  --plan "$STAGING/plan/plan.json" --attempts "$STAGING/attempts" --output "$OUT"
```

The hidden output directory must not exist. A failed export remains an incomplete
artifact, never automatically reused. Successful export includes a source-bound
full readback. Keep registry bundles, campaign ledgers and original attempt paths
available for subsequent validation. Validation deliberately requires the same
production implementation hashes and registered runtime versions.

## Representation

`u1_exact800_compact_native_v1` extends the existing compact Arrow layout with
float64 full physical qpos/qvel/applied controls, original per-frame native JSON
records, teacher qpos, all reference objects and explicit registry/source/donor
lineage. Existing compact fields remain float32. The extension requires a reader
that recognizes this version; older strict compact-contract readers will reject
it rather than silently dropping evidence. The existing compact schema is reused;
its writer cannot preserve these additional fields, so a one-row-batch Arrow
stream writes the extended schema.

Every row carries exactly one `trajectory_info.object_move` interval for the
registry active object. Formal/historical parents reuse their bounded movement
metadata; alternate005 and raw006 read the explicitly bound donor/source Lance
metadata. Export fails if the interval is missing, names the wrong object or
escapes the generated frame range.

Every physical field comes from the second replay. Time is frame/120; commands
map arrivals1..N−1. The inactive left slot is empty. All model objects retain
registry order. Keypoints use the existing16-link-plus5-tip convention and pinned
Cheyingtong right shape. Teacher indices refer to the registry's resampled120Hz
timeline, not original donor capture frame numbers. Full donor ancestry remains
in `lineage_json`, including historically rejected derivation evidence that the
accepted parent superseded.005 is row847 strong-pinch pick/place, **no deep
inversion**;006 retains release-v13 thumb-first lineage.

Native contacts retain six-component contact-frame wrenches, basis, positions,
friction, condim3, four pyramidal EFC addresses and world0 IDs. A basis stores
axes as rows: world force = basis.T @ force[:3]. Compact force vectors retain
the established **normal-only hand-to-object** contract: positive for hand geom1,
negative for hand geom2. Only named hand collision geoms against registry object
bodies/descendants contribute. Hand–hand and world contacts remain in raw native
evidence but not compact hand–object entries. CPU `mj_kinematics` at saved qpos
supplies body transforms and keypoints only; CPU normals or forces are never
substituted. Captured solver positions/bases/forces belong to the last480Hz
substep (frame0 is a native forward), not a re-solved arrival-state contact.

Manifest, rejection audit, registry/plan snapshots, setting snapshot and SHA256
sidecars accompany Lance. Readback rechecks source hashes, identities and quotas,
then reconstructs each complete row and compares Arrow-normalized content,
including timestamps, indices, shapes, contacts, lineage and exact full-state
trace parity. This is intentionally more expensive than sampling a few rows.
