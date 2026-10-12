# VoxMani 453 replacement

Owner: `feat/voxmani453`, ManoRL product line; base `0f8f441`.

Replace the policy with the complete VoxMani architecture from
`16a80fc6615c3f82194acd34e45b98d459359325`, reconstructing its 453 current-state
inputs from actual physics. Preserve the existing v11 reward, 28 actions,
persistent servo and directional antiwindup, MJX/Warp dynamics, reference
trajectories, reset/termination and PPO objective.

Goals remain structured streams outside the 453 ABI and are stored in PPO
states. No truncation or projection of the previous observation vector.

Deliver the full runnable project, assets and reference package under
`/home/cty/workplace/voxmani2contact_bench`. Deploy to `cty@192.168.9.220` and
validate real GPU physics, PPO updates, strict checkpoint resume and frozen
evaluation on a GPU with no compute processes. Never stop other users' jobs.

Owned files: the VoxMani package and model, version registration/contracts,
training CLI and structured-state collection, focused tests, delivery scripts
and architecture/validation documentation. Do not alter reward v11 or the PPO
objective. No subagents authorized.
