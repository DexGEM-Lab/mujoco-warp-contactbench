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

## Qualification

Local v3 bounded pilots under explicit every-480Hz-substep native stepping pass
for zero and +++30mm/pitch−0.5° on all five actions. Each extreme frozen-control
trace also passes a separate-process second replay. Observed high-water maxima
across these runs are42/1024 active contacts and196/4096 constraints; reaching a
capacity is a hard failure.

006 now uses release-v13: supported yaw registration, thumb-first clear, then
reverse same-U1 acquisition deltas for four-finger/wrist extraction. Two fresh
parent replays pass; v3 zero/extreme pilots finish with full orientation errors
2.21°/1.25° and upright settled release.

These pilots still cover only one extreme cell/action; production retains the
frozen16-candidate reserve order and explicit slot exhaustion. No compact-row
exporter or Lance publisher is included in this runner. Existing datasets stay
immutable; attempts must use caller-provided staging with adequate space.
