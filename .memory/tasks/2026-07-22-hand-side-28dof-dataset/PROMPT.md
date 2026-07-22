# Hand-side and 28-DoF Dataset Runtime

Update ManoRL for the revised MANO URDF and the Lance dataset
`/mnt/nas-222-project/mocap_v2/lance_datasets/human_guangguan_p1_cma2lance_20260722_025541.lance`.

Scope:

- discover whether a dataset contains right-hand, left-hand, or both-hand
  trajectories and select the corresponding MANO asset automatically;
- update the pinned MANO submodule/URDF integration to the revised 28-DoF
  hand while preserving the existing mesh count and collision semantics;
- make action dimensions, action scaling, cumulative-action state, and any
  action-dependent observations support 28 DoF per hand;
- default to 56 actions for a two-hand dataset and 28 for a one-hand dataset;
  expose an explicit hand-selection override so one hand can be controlled
  while the other follows reference;
- keep left/right rewards equivalent and preserve existing single-right-hand
  behavior where possible.

Acceptance:

- dataset hand-side detection is data-driven and tested for left/right/both;
- revised URDF/submodule is pinned and compiles with the existing mesh count;
- one-hand and two-hand environments expose the correct action/observation
  shapes, scaling, cumulative-action and reference-following semantics;
- focused tests and CPU smoke checks pass; no generated Lance data is deleted
  or replaced.
