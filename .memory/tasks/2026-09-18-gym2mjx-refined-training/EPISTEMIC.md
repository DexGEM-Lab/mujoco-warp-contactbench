# Epistemic Model

## Phenomenon
The refined RL rows preserve measured hand DoF state in `hands[].urdf_dof`, but the existing refinement dropped the source actuator command `urdf_dof_target`. ManoRL compiled measured state as `q_ref` and replayed it as a position-servo command. This applies a second dynamical lag to a track that was already the output of an upstream controller, so hand tracking falls behind before object tracking and terminal logic can be meaningful.

## Supported mechanism
All40 provenance-pinned source Lance/version datasets expose both state and target for all12,200 rows. On cylinder7 action03, hand-only replay with contacts disabled shows state-as-target step60 errors of3.84cm wrist XYZ,0.131rad joint MAE and0.509rad max joint error. Driving with the source target reduces them to2.21cm,0.084rad and0.301rad. The residual gap is expected because source and MuJoCo servo dynamics differ; target remains the correct command semantics.

Timestamps are authoritative. The inspected source declares data_fps100 but has exact0.005s intervals (200Hz). State, target and object pose must be jointly resampled by timestamp to120Hz; wrist rotation coordinates are unwrapped before interpolation and object orientation uses SLERP.

The correct runtime contract is dual-track: `q_ref` is actuator position target; `q_state_ref` is measured physical state. Initial qpos and tracking evaluation use state. Controller base target plus RL residual use target. Historical/raw trajectories explicitly retain state==target semantics; target120 artifacts use a new contract and MTP v3 so fallback cannot be silent.

## Ruled out
The observed hand lag is not principally object complexity or contact reaction: it remains with the object hidden and hand contacts disabled. It is not a simple120/480Hz arithmetic mismatch: reference/control are120Hz and physics uses four480Hz substeps. The data semantic mismatch—state replayed as target—is upstream of object terminal failures.

## Current claim
A one-row real-data target120 Lance smoke passed exact provenance joining,120Hz resampling, dual-track decode and pre60 windowing. All sources have target tracks, so full reconstruction is feasible without omissions. Current server training remains an intact state-as-target baseline until the new Lance, four MTPs and4096-env preflight pass.

## Most informative next observation
Run the full12,200-row streaming target120 build. Any UUID/timestamp/state/object mismatch is a contract failure and must stop publication. After publication, random20 target-driven replay should be compared against recorded state, followed by four-shard MTP compilation and4096-env preflight before switching training.
