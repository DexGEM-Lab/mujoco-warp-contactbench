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

The dataset is published at
`/mnt/nas-222-project/mocap_v2/lance_datasets/manorl_cheyingtong_u1_largepose_4x160_c1_20260921`.
Lance v1 contains640 rows with exact action quotas, manifest/layout order parity
and contract `u1_four_action_largepose640_c1_native_v1`. Enhanced validation
passed all640 artifact sets/hashes, both replay streams and every native frame,
then exact Arrow-normalized reconstruction/readback of all640 Lance rows. The006
movement interval is source v4 row82 pitcherbase367..1092, shifted to487..1212
by the prefix; the other shifted intervals are003 240..510,007 193..350,009
309..521.

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

## Final visualization and publication claim

Mesh-accurate per-UUID nine-object clearance uses the real pinned778614d asset
Git identity and file manifest. The initial15cm search left seven006 egg_cup
pairs unresolved. A discriminating extension kept the original16 directions and
candidate order but added18/20/25/30cm radii: five resolved at18cm and two at
20cm, while every prior offset remained unchanged. The final layout has640
UUIDs, nine objects each,480 rows with background offsets and zero unresolved
pairs. The initial seven failures remain in `background_offsets_initial.json`.

Publication generated registry/setting snapshots and a14-file SHA manifest,
then used a same-filesystem atomic rename. Final-path readback verified Lance v1,
640 unique UUIDs in manifest/layout order, action counts160 each, every recorded
file SHA, the SHA sidecar, and absence of the hidden staging source. The final
`sha256.json` digest is
`4004504f838aa866993680b901c0f767251cae2caef605154d27008630d9036a`.
Action005 remains intentionally outside this claim.

## Action005 minimal replay model

The user-updated u3300 action005 source is a compact replay dataset, not an
exact640 audit artifact. Its ten core Arrow fields already match the compact
base of the four-action export. The five exact640 extensions (`physical`,
`native_contacts`, `lineage_json`, `teacher_qpos`, and
`reference_objects_json`) prove campaign construction; they are not required
for the user's hand/object spatial playback. Zero-filling them would assert
false physical states or false absence of contact, so the accepted representation
omits them rather than imitating unavailable evidence.

The standardized sibling is
`outputs/manorl/dev_mayonnaisebottle05/synthesis-u3300-two-parents/mayonnaisebottle05-u3300-parents126-128-ratio5-replay-minimal.lance`.
It has ten rows and ten unique UUIDs. Every Arrow-normalized value equals the
user-supplied two-object source except row6 UUID, deterministically repaired
from the source identity, checkpoint, seed2003, episode/attempt and target hash.
Both `mayonnaisebottle` and `bowl` are frame-aligned in every row; no audit field
is zero-filled. The original supplied Lance remains unchanged.

The old viewer failure was a reader collapse: it interpreted `index.scene` as
one object and compiled only the active bottle. The replay reader now resolves
the active object from source lineage while retaining the ordered complete
scene from `index.scene`, `object_names`, and `objects`. A real GPU preflight
compiled `nq=42/nv=40`, mapped bowl and bottle to distinct free joints, and
reset all generalized coordinates exactly after stepping. DISPLAY=:1 shows the
bowl and bottle together while directly applying the stored28D targets. This
establishes replay usability only; it does not establish strict-U1 lineage or
formal action005 acceptance.

## Active ten-parent action005 augmentation model

All ten standardized rows are now the parent population. Each contributes
exactly16 accepted children; the complete160-child action005 set, rather than
each individual parent, carries the same144-base+16-extra large-pose distribution
as the published actions. Slot-to-parent assignment must be frozen and balanced,
and every child UUID must bind its exact repaired parent UUID.

The two-object model is not a new physics setting: pinned asset compilation
matches the existing U1 hash exactly. The compact row is still input rather than
acceptance evidence. Each row therefore needs a complete arrival-indexed nominal
replay and a separate frozen-control replay. Frame0 is prestep; its stored target
may be materially outside a joint limit and is never integrated. Replace only
that inert frame0 value with the native executed target. Retain frames1+ exactly:
a float32 endpoint can lie about1e-8 beyond the float64 limit while mapping back
to the identical float32 native control, so eligibility is executed equivalence
within1e-6 rather than a literal float64 inequality. Use the accepted first
replay's physical qpos as the strict-C1 teacher. Use its
physical object tracks as the child reference so the passive bowl and active
bottle share one actual two-object world.

Row0 supports this mechanism: full U1 replay matches the recorded bottle path
to1.341mm, passes pickup/place/release gates and reaches104.636deg tilt before
returning upright. Formal005 gates should preserve this observed mechanism:
multifinger airborne support, at least90deg pour, peak pose over and above the
bowl, upright supported release, exact controls and no solved prefix-force
contact. Settling is measured over the terminal200ms with separate physical
units (<2mm/s linear and<0.02rad/s angular for bottle and bowl); a mixed6D norm
was rejected because it conflates metres/second with radians/second and changed
with the final contact phase despite stable support.

The ten parents are now physically qualified, but the evidence changed the
parent mechanism in two important ways. Row7 proved that single-sample contact
qvel is not a valid settling measure: direct pose excursion stayed microscopic
while one process emitted a transient velocity spike. Settlement therefore uses
p95 speed together with direct terminal position/orientation excursion; maxima
remain reported as diagnostics. Row9 exposed a real instability instead: the
same frozen controls released an upright bottle in one process and a side-lying
bottle in the other. The divergence grew only after final finger withdrawal.
Repeating each existing control frame520–560 twice reduced withdrawal speed and
made four independent processes settle upright while preserving every source
control in order. This is a documented time schedule, not a target-value repair.

The formal registry binds ten unique complete-scene parents. Exact160 production
is complete: all global slots passed twice, each parent contributes16 children,
and no slot exhausted its deterministic same-cell reserves. Fifty rejected
candidates are concentrated in parent7/8 terminal stability and remain preserved;
row9's slowed withdrawal was robust across its assigned children. Hash-verified
same-filesystem merge produced one canonical ledger and artifact tree. Thus the
remaining uncertainty is no longer physical acceptance or sampling coverage. It
is representational: whether the new exact800 serialization, full source-bound
readback, and mesh-clearance layout preserve these traces without alteration.
