# Composite-scene active-object Lance adaptation

## Objective

Adapt ManoRL trajectory selection and decoding for `/mnt/nas-222-project/mocap_v2/lance_datasets/dexgem_vla_demo_guangxue_cma2lance_20260909_060700.lance`, whose rows list multiple comma-separated scene objects but exactly one manipulated object in `trajectory_metadata.trajectory_info.object_move`.

## Success criteria

- Derive each modern row's trainable object from its unique `object_move` entry.
- Map that object to the corresponding `objects` state using ordered scene names when `object_names` metadata is absent.
- Preserve existing single-object and legacy Lance behavior.
- Discover exactly `bowl:03`, `bowl:07`, `bowl:09`, `mayonnaisebottle:05`, and `pitcherbase:06` from dataset version 5.
- Run focused tests and a policy-free reference replay check.

## Constraints

- Work only on `feat/composite-scene-active-object` in its linked worktree.
- Do not modify or clean the dirty primary `dev` worktree.
- One active object per trajectory/environment; passive scene objects are not materialized in the training physics model.
- Fail explicitly for ambiguous composite rows with zero or multiple manipulated objects.
