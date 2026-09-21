# Formal Cheyingtong compact replay

Canonical runner/semantics: `docs/formal_compact_replay.md` and
`tools/replay_formal_compact.py`. The formal no-augmentation September17 bundle
contains28multiobject Cheyingtong120/480Hz policy-generated records, distinct from
the earlier42historical repair/augmentation seeds.

Control targets in this bundle are ARRIVAL indexed: target[t] is repeated for all
four physics substeps producing state[t]. Source audit verifies every row against
ctrl_substeps; frame0 is prestep and velocities zero. The older single-object
TargetDofReplay parser/outgoing indexing must not be applied silently to it.
All scene objects and underscore-containing identities must be preserved. The
direct target-DOF reader now resolves the active object from the source identity
and requires `index.scene`, `trajectory_metadata.object_names`, and `objects` to
agree in order; it compiles and resets every declared scene body rather than
collapsing to the active object. Reuse native source model/CCD/servo settings,
not the unrelated repaired-input PID or elliptic/impratio100 recipe.

The strict-U1 large-pose contracts differ from checkpoint rollouts: historical
`source_identity` and checkpoint fields are intentionally nullable. Their single
`trajectory_info.object_move` record identifies the active object, including
006 where the active pitcher is not the first scene object. Replay treats the
current Lance row as the executable source, preserves null checkpoint evidence,
and uses the frozen U1 capacities ncon1024/nccd256/nj4096. Do not synthesize
checkpoint values or fall back to the old128-contact viewer allocation. The
five-action exact800 contract and publication are documented in
`docs/u1_exact800_export.md`.

Native-contact last-forward object poses and postintegration qpos differ by up to
one480Hz substep. Store/compare like boundaries. Acceptance uses source-specific
non-egg complete/XYZmean35deg/>100contactframes gates; eggs use actual released
bin entry and their shorter recorded horizon. Recorded actual-motion parity is a
separate measurement, not the task gate.

First fixed-target replay:25/28 pass (egg1/2,stir2/3,bowl3/3,pour19/20). Egg028
loses contact around3.808s before the recorded release; stir046 deviates while
returning its stick; cup122 returns tilted (source-reference XYZmean38.56deg).
Inputs/control match, early differences are small, later contact divergence is
large. Recorded policy success does not guarantee open-loop target replay. These
are observed single-run results, not estimates of repeated success probability.
