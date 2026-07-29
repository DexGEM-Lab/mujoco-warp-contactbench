## Objective
Implement and run a GPU ManoRL checkpoint rollout exporter for cube2 action 02. Produce corrected v2 synthetic trajectory data in Lance format from the existing cube2 v5 checkpoint.

## Workbench
- Inspect current environment transition/contact ABI and existing Lance writers.
- Add a versioned v2 export contract with per-trajectory rows and provenance manifest.
- Export normal-only contact forces without the historical 0.5 scale.
- Verify the generated artifact with nested-field, frame, force, timestep, and checkpoint identity invariants.

## Context
- Primary repository: /home/jay/dexrobot/FromSSH/manoRL_mujoco
- Worker worktree: /home/jay/dexrobot/FromSSH/manoRL_mujoco-worktrees/feat-lance-v2-checkpoint-export
- Target selector: cube2:02, right hand.
- Existing checkpoint: /home/jay/dexrobot/FromSSH/manoRL_mujoco/outputs/manorl/production_cube2_all_v5_u5000_n4096_g1/run-20260729T114123Z/training/checkpoint-000500.pt
- Checkpoint SHA256: dafa2135a4de46e52dfec8c0ee264ca82fcd1af18db6f402024f1499d6e7c0ee
- Source trajectory Lance: /mnt/nas-222-project/mocap_v2/lance_datasets/human_p1_guangguan/human_p1_guangguan_clean.lance, version 295.
- Existing standard exemplar uses synthetic_mano_28dof_hand_slots_v1 but has fps/joint-force inconsistencies; this task explicitly creates v2_corrected.

## Task specifications
- Keep simulation and policy on GPU; host materialization is allowed for export serialization.
- v2 timestamp dt is 0.005 seconds and metadata data_fps is 200.
- Contact force_normal is the solved normal component only, in hand-to-object world direction, with no 0.5 multiplier.
- total_force_world is the sum of those per-contact normal forces.
- total_force_joint uses the actual hand collision-link rotation, not the historical wrist fallback.
- total_force_object uses object-frame force with an explicit v2 hand-to-object convention; document the sign in schema/manifest and use one consistent convention.
- Include 28D urdf_dof and urdf_dof_target, 21 keypoints, object pose, contact positions in world/joint/object frames, source identity, checkpoint SHA/path, runtime/environment/action contract, and deterministic mean-policy provenance.
- Avoid modifying main/dev directly; use this worker worktree and branch.
- Do not delete existing outputs. New generated output must use a unique path or explicit replace behavior.

## Acceptance
- Focused unit tests validate v2 schema construction, normal-force/no-scale semantics, frame transforms, and episode boundary handling.
- A real cube2:02 GPU export creates at least one complete independent trajectory and a Lance dataset plus manifest.
- A Lance reader can open the output and row-level invariants pass.
- No unrelated files or active processes are changed.
