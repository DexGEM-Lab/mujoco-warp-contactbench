# Current Epistemic Model

## Phenomenon

The desired policy signal increases only the direct hand-object contact term. The distance terms currently reuse contact quality as a gate, so scaling a shared intermediate would incorrectly triple both mechanisms.

## Live mechanism

`raw_contact` is the force-quality value capped by `max_contact_reward=0.4`. Previously, windowed `contact` was both added directly to total reward and reused to construct `distance_gate`. The implemented split derives the unchanged gate from unscaled windowed quality and derives the reported/direct `contact` by applying the explicit scale only to that windowed quality.

## Current claim

The minimal implementation is a `RewardConfig.direct_contact_reward_scale` default of `3.0`. `windowed_contact` remains the unscaled quality used by `distance_gate`; only the direct `contact` diagnostic and total term apply the scale. The reward and raw-1.0x PPO contract identifiers advance coherently to v3 because old native checkpoints optimized a different environment reward.

The paired CPU computation with identical inputs and scales 1 and 3 confirms that `raw_contact`, `distance_gate`, and all distance components are exactly equal; scaled `contact` is 3x; and the total difference equals exactly twice the 1x direct term. Separate checkpoint assertions reject both immediate prior v2 contract identifiers.

## Unresolved question

None within the assigned boundary. A broader observation test needs a generated replay fixture absent from this linked worktree; that missing artifact is unrelated to reward semantics and does not weaken the paired reward evidence.
