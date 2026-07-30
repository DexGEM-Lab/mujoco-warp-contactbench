## Phenomenon
The old v2 exporter labeled the physical wrist/keypoint body pose as MANO global pose. The body carried an almost constant pi rotation relative to the URDF floating root and a translation offset up to 1.9 mm. Shape metadata also stored left/right shapes while declaring only right active.

## Mechanism
The refined MANO contract is defined by the exported 28D URDF state: translation is q[0:3], and global rotation is the axis-angle representation of intrinsic XYZ `Rx(q3) @ Ry(q4) @ Rz(q5)`. The physical body transform belongs to physics/contact computation, not the MANO global field. Shape rows follow active `hand_names`, while fixed `hand_slots` retain an empty left slot.

## Validated claim
`synthetic_mano_28d_checkpoint_rollout_v2_1` fixes this semantic boundary without changing physics, policy, 28D state/target, local MANO pose, keypoints, contact, reference, rollout, checkpoint, seed, or source lineage. Across all 49 delivered rows:
- `mano_global_pos` equals stored `urdf_dof[:, :3]` exactly;
- global rotation differs from intrinsic-XYZ reconstruction by at most 1.217e-7 rad (float32 serialization);
- `hand_names=[right]`, there is exactly one 10D shape, and it equals the raw row's right shape exactly;
- every seed UUID, raw row and source frame mapping remains unique;
- normal-only scale-1 force/frame invariants remain within 1.17e-6.

## Delivered state
The authoritative artifact is `/mnt/nas-222-project/sunjieqiang/mujoco_synthetic/cube2_02_checkpoint500_seed42_v21.lance`, with 49 rows and payload SHA256 `88d27432f025aa6ba1c3a1bdfd9e18d73a03bec6e31a890e16e9b94cb5b58cc2`. Superseded generated v2 Lance artifacts were deleted only after this dataset passed NAS readback and byte-hash verification.

## Remaining boundary
Server2 Lance 0.24.1 can nondeterministically crash during nested reads. Isolated per-row validation with bounded retries remains the required reader strategy on this host; one of 49 rows required a second attempt. This does not alter the persisted bytes or schema contract.
