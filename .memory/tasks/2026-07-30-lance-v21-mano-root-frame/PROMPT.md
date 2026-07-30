## Objective
Correct ManoRL synthetic Lance MANO global-frame semantics and regenerate the full cube2:02 checkpoint artifact.

## Contract changes
- `hands[0].mano_global_pos = hands[0].urdf_dof[:, :3]`.
- `hands[0].mano_global_rot_aa = Rotation.from_euler("XYZ", urdf_dof[:, 3:6]).as_rotvec()`, matching `Rx @ Ry @ Rz`.
- `trajectory_metadata.hand_names = ["right"]` and `mano_hand_shapes = [raw_right_shape]`.
- Preserve 28D state/target, MANO local pose, keypoints, contact, reference, rollout, force, checkpoint, seed, and identity semantics.
- Publish a new schema version so old and corrected coordinate meanings cannot share one ID.

## Inputs
- Source Lance: `/mnt/nas-222-project/mocap_v2/lance_datasets/human_p1_guangguan/human_p1_guangguan_clean.lance`, version 295.
- Pair: cube2:02, 49 valid right-hand identities.
- Checkpoint: `/home/jay/dexrobot/FromSSH/manoRL_mujoco/outputs/manorl/production_cube2_all_v5_u5000_n4096_g1/run-20260729T114123Z/training/checkpoint-000500.pt`.
- Checkpoint SHA256: `dafa2135a4de46e52dfec8c0ee264ca82fcd1af18db6f402024f1499d6e7c0ee`.

## Acceptance
- Focused tests prove exact root-position equality, intrinsic XYZ rotation equivalence, and one right-hand shape.
- One GPU smoke and the full 49-row export pass the isolated validator.
- New Lance is atomically copied to `/mnt/nas-222-project/sunjieqiang/mujoco_synthetic` with byte-identical hashes.
- Only after new NAS validation, delete this task's superseded v2 Lance outputs as authorized by the user.
- Integrate through the isolated feature branch; preserve primary-worktree unrelated files and existing production processes.
