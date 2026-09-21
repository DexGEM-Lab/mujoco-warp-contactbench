# Native U1 5×160 campaign

`python -m tools.run_u1_campaign` provides build-registry, plan, pilot,
run-action and status modes. It never writes Lance. Use an absolute caller-owned
staging path (including mounted NAS); one writer lock is held per action.

## Frozen inputs and selection

Build once from the approved local artifacts:

```sh
python -m tools.run_u1_campaign build-registry --staging /STAGING/registry
python -m tools.run_u1_campaign plan --registry /STAGING/registry/registry.json --staging /STAGING/plan
python -m tools.run_u1_campaign pilot --registry /STAGING/registry/registry.json --plan /STAGING/plan/plan.json --action 003 --staging /STAGING/pilots
```

Builder binds compiled binary models, native manifest arrays, exact initial
qpos/qvel, reference objects, teacher poses, accepted target/report and source
lineage. Historical loaders are needed only for building; their hashes are
recorded. All runtime bundle files and package versions are checked on load.
005 explicitly means row847 action04-derived strong-pinch alternate pick/place,
without deep inversion. Final registry status supersedes historical derivation
`accepted:false`; it does not erase that ancestry.

Each action has160 slots: five radii ×eight octants ×four roll/pitch orientations.
Each slot has16 ordered candidates total (primary normalized sign corner plus15
seeded same-octant directions). Radius and orientation never shrink on rejection.
First qualifying candidate wins after a separate-process frozen-control replay.
Interrupted attempts are rejected on resume, not rerolled. Exhaustion remains a
shortfall. UUIDs depend on semantic plan/parent/candidate identity, not trace luck.

Actual hand qpos[:6] and absolute target[:6] receive the same intrinsic-XYZ offset.
Taper C is first representative scene contact minus12, ending exactly at C−1.
Targets retain original dtype, fingers, timeline and byte-exact suffix.
`requested_target.npy` preserves that stream; `target.npy` contains actual executed
float32 native controls. Executed suffix must equal parent float32 controls exactly.
Objects and qvel are restored unchanged; all native buffers reset before each
attempt; the single-world graph persists for each action runtime.

Existing action-specific evaluate gates are supplemented with premerge scene
contact exclusion, sampled uncontrolled-flight exclusion, exact applied controls,
100ms sustained009 hold, and006 pour/spout/distal-force/release/settling checks.
Native per-frame geom pairs, contact-frame wrenches and contact positions remain
in `native_contacts.jsonl`; geometric contacts are separately labeled.

## Production command (not run)

```sh
for action in 003 005 006 007 009; do
  python -m tools.run_u1_campaign run-action \
    --registry /STAGING/registry/registry.json --plan /STAGING/plan/plan.json \
    --action "$action" --staging /STAGING/attempts --execute-production
done
```

Exactly160 selected unique slots/action is the output condition, not a promised
physical yield. Status uses the same registry/plan/staging/action arguments.
Selected trace artifacts are hash-checked on resume. Failures remain in place.

## Qualification and blockers

Local bounded pilots: zero and +++30mm/pitch−0.5° pass for003/005/007/009.
006 fails final full source-relative orientation<15° in both pilots (~59°).
Canonical release-v6 fresh02 already has59.73°: upright release does not certify
terminal yaw. Resolve the parent/gate conflict explicitly before production;
do not weaken the gate or silently switch parents.

These pilots are diagnostics, not independently accepted children. The new-process
second-pass command is implemented but not physically qualified in this bounded
run. Only one extreme cell/action was tested. Contact/constraint high-water checks
are at saved120Hz frames, not every480Hz substep; CCD overflow instrumentation is
not yet present. No21-keypoint compact-row exporter or Lance publisher is included.
These remaining qualification/instrumentation boundaries must be addressed before
claiming a production-ready800-row delivery. Existing datasets stay immutable.
