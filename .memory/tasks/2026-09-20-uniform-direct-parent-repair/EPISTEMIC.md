# Current exact640 U1 model

The active deliverable is a new, self-contained strict-U1 dataset for actions
003/006/007/009 only: exactly160 new trajectories/action and640 total. Action005
is paused and excluded because its valid full-pour U1 parent remains unsolved;
row847 pick/place cannot represent pouring.

## Supported mechanism and current claim

The accepted augmentation is the historical120-frame approach. Wrist XYZ/Euler
joins the physical teacher q0/q1 by the exact discrete-C1 rule whose final prefix
sample is `2*q0-q1`; XYZ includes the4cm endpoint-smooth vertical arc. Fingers
hold the parent frame0 target. The complete parent control suffix is byte-exact,
with no splice state reset. Prefix rejection uses solved native scene-contact
normal force above0.2N. Every selected child has two accepted frozen-control U1
replays from distinct positive PIDs and retains native contact-frame evidence.

Strict-C1 v3 production is complete:003/006/007/009 each have160 selected slots,
640 UUIDs are unique, and no slot exhausted.003 and006 shards are merged into
canonical ledgers;007/009 were canonical runs. The existing1,923-row dataset and
all invalid/diagnostic campaigns remain unchanged. Evidence: OPS.md entry
`2026-09-21T20:45:08+08:00`.

A hidden Lance v1 contains640 rows with exact action quotas, manifest order
parity and contract `u1_four_action_largepose640_c1_native_v1`. Enhanced
validation has passed all640 artifact sets/hashes, both replay streams and every
native frame, followed by exact Arrow-normalized reconstruction/readback of all
640 Lance rows. It is not yet published. The006 movement interval is source v4
row82 pitcherbase367..1092, shifted to487..1212 by the prefix; the other shifted
intervals are003 240..510,007 193..350,009 309..521.

## Audit model

Artifact movement and artifact proof are separate mechanisms. Same-filesystem
rename preserves bytes, so shard merging carries the shard-recorded hashes and
does not reread large traces over CIFS. Final validation must then perform the
single authoritative reread: exact selected-directory file coverage, all13
artifact SHA values, both replay traces/targets/results, every native-contact
frame, C1 target identity and byte-exact suffix, followed by full source-bound
Lance row equality.

Replay hash semantics are asymmetric by design. Replay1's requested hash binds
the analytic strict-C1 target. Replay2 loads replay1's executed control as its
frozen requested target, so its requested hash binds that frozen control. The
inspected003 row differs from the analytic target by only5.9442e-8, while the two
executed-control hashes are identical.

## Ruled out / isolated

- v1 large-pose artifacts are invalid due to runner/arc problems.
- v2's492 accepted diagnostics duplicate parent frame0 at the splice and cannot
  enter final640.
- The stopped scratch `tools/run_u1_largepose.py` is not the production runner.
- Geometry-only prefix touching cannot substitute for solved native force.
- Rehashing inside the shard mover is operationally pathological over CIFS and
  adds no evidence beyond the final authoritative hash audit.
- A validator that compares decoded content without checking ledger artifact
  coverage/SHA is insufficient and was stopped before producing validation.json.

## Remaining uncertainty and next discriminating result

All640 artifact and full-row checks pass. The mesh-accurate per-UUID nine-object
visualization solver is active after restoring the real778614d Git metadata on
Server1; its first launch failed before output because the server snapshot had
asset bytes but no Git identity. Publication is permitted only if the solver
reports zero unresolved clearances and the generated layout, registry/setting
snapshots, final hashes and atomic rename all pass. Any unresolved UUID/object is
the next question; there is no fallback to reduced perturbations, old children,
or partial output.
