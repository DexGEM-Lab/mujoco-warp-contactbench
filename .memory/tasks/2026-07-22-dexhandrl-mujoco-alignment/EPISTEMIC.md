# Epistemic Record

## Current model

The v3 trace schedule is settled on Gym's live task clock. Lance initializes the physical hand/object state at frame 61. Gym reset runs its second observation pass through `post_physics_step()`, so the observation presented to the policy is decision frame 62. Every trace row therefore records a policy decision in 62–692 and its post-step state in 63–693: 631 actions in the inclusive physical window 61–693. The reset marker belongs only to decision 62. Both producers enforce this contract; Gym reads `task_states["current_frame_idx"]` at the decision and after `env_step`, while MuJoCo initializes frame 61 then exposes frame 62 before its first action.

Episode time is policy action-count time, not a coordinate derived from the logical Lance frame: it is zero at reset and advances once per action (`n * dt / (633 * dt)`). This retains Gym's episode-counter semantics despite the policy's first reference being one frame after physical initialization.

Smoke5 provides the decisive shift evidence: its Gym row 0 exactly equals MuJoCo frame 62 and row 1 exactly equals MuJoCo frame 63. The old synthetic v2 labels 61–63 were consequently one frame early. They remain useful structural/runtime evidence but cannot support an aligned observation or physics comparison. Schema v3 prevents that interpretation by requiring the authoritative 62 decision start, contiguous decisions, and post-states through at most 693.

The remaining alignment model is unchanged: traces compare processed/RMS 461D observations by canonical slice, deterministic policy actions, residual offsets/targets, and then post-physics state. MuJoCo records the raw Lance active reference and the pre-coupling corrected active target; its applied target is that corrected target with explicit `post_filters=[]`. Gym is constructed source-read-only through its Hydra/Runner path, restores the checkpoint mapped to `player.device`, and captures the live ActionProcessor stages. Minimal Lance discovery is filtered to row 22 / `cube1` / `01` / sequence 1 / UUID `c21d4fa5-185a-5ea7-aa22-e1d34beae42f` before allocation.

## Next discriminating question

Does a fresh v3 Gym trace and v3 MuJoCo trace first diverge in a shared-input observation slice, action/control stage, or only post-physics state? Fresh traces are required because smoke5 carries v2's early synthetic labels; their comparison will localize the remaining mismatch without conflating reset scheduling with simulator divergence.
