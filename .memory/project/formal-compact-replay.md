# Formal Cheyingtong compact replay

Canonical runner/semantics: `docs/formal_compact_replay.md` and
`tools/replay_formal_compact.py`. The formal no-augmentation September17 bundle
contains28multiobject Cheyingtong120/480Hz policy-generated records, distinct from
the earlier42historical repair/augmentation seeds.

Control targets in this bundle are ARRIVAL indexed: target[t] is repeated for all
four physics substeps producing state[t]. Source audit verifies every row against
ctrl_substeps; frame0 is prestep and velocities zero. The older single-object
TargetDofReplay parser/outgoing indexing must not be applied silently to it.
All scene objects and underscore-containing identities must be preserved. Reuse
native source model/CCD/servo settings, not the unrelated repaired-input PID or
elliptic/impratio100 recipe.

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
