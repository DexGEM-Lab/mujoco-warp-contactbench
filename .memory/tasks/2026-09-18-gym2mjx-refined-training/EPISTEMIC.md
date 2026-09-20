# Epistemic Model

## Phenomenon
The refined corpus had dropped source actuator targets and replayed measured `urdf_dof` state as a new position command, creating a second servo lag. This semantic error was upstream of hand/object tracking. A second independent difficulty remains: object reference begins at step60 while residual/deviation semantics begin at step120, so fast object trajectories can outrun physical contact before policy authority opens.

## Supported mechanism
All40 source Lance/version datasets contain `urdf_dof_target` for all12,200 rows. Target-driven hand-only replay reduces the critical cylinder7 step60 wrist error from3.84cm to2.21cm, joint MAE0.131rad to0.084rad, and max joint error0.509rad to0.301rad. The remaining mismatch reflects source-vs-MuJoCo servo differences.

The corrected contract is dual-track: measured `q_state_ref` initializes physical qpos and defines replay ground truth; actuator `q_ref` drives controller targets and receives RL residual. Timestamp is authoritative:11,062 rows are200Hz and1,138 are120Hz, jointly resampled with object pose to120Hz.

Persistent unified MJX-Warp workspaces support4096 env per shard. MTP v3 binds state/target semantics and is Lance-free at runtime. A5-update target preflight crossed checkpoints with22.2k transitions/s and no numerical/allocator faults.

## Ruled out
Hand lag is not caused only by object geometry/contact: it persists with object hidden and contacts disabled. It is not simple120/480Hz arithmetic error. Using target fixes hand-command semantics but does not automatically solve every object terminal because the early120/reference phase is independent.

## Current claim
The corrected target120 Lance and four MTP v3 shards are published and validated. The old state-as-target baseline is stopped with checkpoints preserved. Four fresh target-based policies are training from update0 at4096 env,pre60,early120,0.15m terminal,joint2x,with errors0 and online monitoring.

## Most informative next observation
Compare reward/contact/success trajectories over the first several hundred target-based updates against the preserved baseline. If early terminal concentration remains near step120, test a separate phase-alignment experiment (policy/deviation opening at movement start or delayed object reference) without mutating the current target-semantic run.
