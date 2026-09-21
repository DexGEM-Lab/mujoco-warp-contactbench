# Current strict-U1 five-action model

## Justified claim

The requested five-action deliverable is complete. The immutable strict-C1
exact640 remains published separately. A new exact800 contains160 rows each for
003/005/006/007/009 under setting
`c6db552f105d4de493880a52518772dbacad9496b105d2f4e43e87a907e692cd`.
The 640 inherited rows are source-equal to exact640; the 160 action005 rows are
new physical integrations from ten standardized parents,16 per parent.

Exact800 is published at
`/mnt/nas-222-project/mocap_v2/lance_datasets/manorl_cheyingtong_u1_largepose_5x160_c1_20260922`.
Lance v1 has800 unique UUIDs in003/005/006/007/009 block order. Full source-bound
readback reconstructed every row, the visualization layout has nine objects per
row with zero unresolved clearances, ten copied parent bundles load independently,
and all69 indexed final files rehash correctly. Final SHA-index digest:
`c4545e64d96eb983f5f49ebf3d45a9b32d7269a0af06f262a84e4813cc269777`.
Evidence: OPS.md `2026-09-22T04:17:27+08:00`.

## Physical augmentation mechanism

Every child uses the historical120-frame physical-teacher approach. Wrist
XYZ/Euler joins parent q0/q1 by the discrete-C1 endpoint `2*q0-q1`; XYZ includes
the4cm endpoint-smooth vertical arc. Fingers hold parent frame0 target. The
complete parent suffix follows without state reset. Prefix rejection uses solved
native hand-scene normal force above0.2N. Acceptance requires two complete
frozen-control physical processes with distinct positive PIDs; contact-driven
qpos/qvel need not be pointwise identical.

The action005 set exactly realizes the frozen global plan:144 base cells plus16
extras, radii5/10/15cm with totals54/53/53,20 slots per azimuth sector and signed
rotation totals27/27/27/27/26/26. Every parent receives16 slots and two per
sector. All160 slots passed; none exhausted16 deterministic same-cell reserves.
Fifty rejected candidates remain physical negative evidence. Canonical ledger
SHA: `f1b8d742b2f50694e854c9f6ade4ee83a0423f6b35c6da7fbd6eaabc5fa1f944`.

## Parent mechanism

All parents retain the ordered complete scene `mayonnaisebottle,bowl`. Rows0–8
execute source controls in identity order. Row7 showed that isolated contact
solver qvel spikes do not imply drift: direct terminal excursion was0.0311mm and
0.0170 degrees despite one3.83mm/s sample. Settlement therefore combines p95
linear/angular speed with direct terminal position/orientation excursion; maxima
remain diagnostics.

Row9 exposed a real mechanism rather than a proxy error. Its original frozen
controls passed once but independently tipped after final finger withdrawal and
settled at89.7 degrees. Repeating each existing frame520–560 twice halved
withdrawal speed while preserving every source target and order. Four independent
qualification processes then settled upright near1.1 degrees, and its16 assigned
children were robust. The original failure and intervention pilot remain in
staging diagnostics.

Parent registry digest:
`0bb86055fd2935cc638fd2ca887528d1679e272fe13f1fa6536d8722794a81c0`;
registry SHA:
`5587a8eceafd34488094ef7e55d42874da93d2d73def903e479f8493061f270a`.

## Representation and executable evidence

Action005 rows contain real float64 physical qpos/qvel/native controls, native
contact-frame wrench/basis/position/friction/dimension/EFC/world evidence,
physical teacher qpos, lineage and both reference object tracks. Missing
historical fields remain nullable; nothing is zero-filled.

Arrow-normalized readback compared all640 inherited rows to exact640 and rebuilt
all160 action005 rows from canonical second replays. The exact640 SHA-index stayed
`4004504f838aa866993680b901c0f767251cae2caef605154d27008630d9036a`.
Action005 required no new visualization offsets; all160 rows already cleared the
seven background objects. Existing exact640 offsets were reused unchanged.

A direct exported-row replay revealed two adapter assumptions before physics:
large-pose contracts were not routed, and the old viewer allocated128 contacts
while U1 requires capacities1024/256/4096. Large-pose provenance intentionally
leaves checkpoint/source-history fields nullable, so the adapter now derives the
active object from the sole explicit movement record and preserves null
checkpoint evidence. With frozen capacities, exact800 row160 (row9 parent) ran
all903 frames on GPU with no override and retained bottle+bowl. Maximum replay
error was7.56mm bottle position,0.04873rad rotation and0.02849rad generalized
position, all within the direct-replay contract.

## Ruled out and preserved boundaries

- Historical PID/U2 rows remain diagnostic and never enter the U1 publications.
- Row847 pick/place and its160 descendants cannot substitute for full-pour005.
- v1 large-pose outputs had runner/arc faults; v2 duplicated frame0 at the splice.
- Geometry-only prefix touch cannot substitute for solved native force.
- Single-sample terminal qvel cannot adjudicate physical settlement.
- Re-running a marginal target until lucky acceptance is invalid; row9 was
  causally repaired and its original independent failure preserved.
- The supplied ten-row minimal replay remains immutable input, not physical audit
  evidence. Its sibling minimal Lance omits unavailable audit columns rather than
  fabricating them.
- Background offsets are visualization-only and never alter recorded physics.
- Exact640 and the pre-existing1,923-row unified dataset were not rewritten.

No unresolved physical, quota, serialization, clearance, hash or publication
anomaly remains for the exact800 objective.
