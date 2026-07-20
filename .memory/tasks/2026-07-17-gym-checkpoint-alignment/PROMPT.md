# Gym Checkpoint Alignment

## Objective

Make the MuJoCo/skrl ManoRL runtime use the validated IsaacGym FiLM,
dynamic-point-cloud, action, timing, normalization, and reward semantics as its
production training default. Also provide an explicit conversion path for the
validated rl-games checkpoint family. The sibling IsaacGym repository and
original checkpoint are read-only.

## Context

- Branch: `feat/gym-checkpoint-alignment`
- Worktree: `/home/jay/dexrobot/FromSSH/manoRL_mujoco-worktrees/feat-gym-checkpoint-alignment`
- Parent integration branch at assignment: `dev` at `81de3b7`
- Source repository (read-only):
  `/home/jay/dexrobot/FromSSH/manohand_reconstruction/IsaacGymEnvs/isaacgymenvs`
- Validated source checkpoint (read-only):
  `/home/jay/dexrobot/FromSSH/manohand_reconstruction/checkpoint/18-09-37-36_MANOHand_all_all-new_setting-val099-new_obs/nn/MANOHand.pth`
- Source evaluation contract: FiLM enabled, PointNet enabled, 64 dynamic surface
  points, residual actions enabled, early phase 50, movement pre-padding 250.
- The saved checkpoint configuration records movement pre-padding 200, but the
  successful evaluation command inherited the current task default of 250.
- Validated Gym result on s02/cube1/01: return about 488.9614, 790/790 steps.

## Established evidence

- The target `use_film=True` model has a one-to-one 41-tensor mapping to the
  checkpoint model. After explicit namespace mapping, the same 476D input gave
  bitwise-identical actor `mu` and critic `value`. Do not rewrite PointNet or
  FiLM unless a failing executable parity test proves it necessary.
- Gym observation normalization uses epsilon `1e-5`; the aligned target uses the
  same formula and running-statistics layout.
- Dynamic point sampling uses the source surface-area CDF and square-root
  barycentric algorithm with PyTorch's global CUDA RNG. Fixed raw observations
  remain the network parity boundary.

## Required implementation

1. Add an explicit, auditable rl-games-to-skrl conversion path for this
   checkpoint family. It must map:
   - 41 model tensors: PointNet, condition encoder, FiLM actor, critic, actor
     head, and log_std;
   - observation normalizer running_mean/running_var/count plus shared XYZ
     pc_running_mean/pc_running_var/pc_count;
   - value normalizer running_mean/running_var/count.
2. Fail closed on missing keys, extra incompatible model keys, wrong shapes,
   unsupported non-floating dtypes/non-finite values, incompatible source
   containers, and output collisions. Supported floating inputs may be converted
   explicitly to target dtypes and the conversion must be recorded.
3. Align target observation-normalizer epsilon with rl-games (`1e-5`). Preserve
   normalizer state and frozen inference behavior. If training-update precision
   remains deliberately different, document it without weakening inference
   parity.
4. Serialize auditable provenance/compatibility metadata, including source
   checkpoint SHA256, source format, FiLM/PointNet dimensions, normalizer
   semantics, and the successful evaluation selection: early=50, movement
   pre-padding=250, dynamic_reset. Preserve the historical pre-padding 200 value
   as provenance rather than as the executable target contract.
5. Provide a safe CLI or executable tool under `tools/` that reads the source
   checkpoint and writes only a caller-selected target artifact under the target
   repository. It must never modify the source checkpoint and must refuse to
   overwrite outputs by default.
6. Make the aligned FiLM/dynamic/action/reward contract the only production
   inference and native-training default. Remove the legacy policy-contract CLI
   and reject checkpoints from the previous MuJoCo contract.
7. Add focused tests for exact key/shape mapping, provenance, fail-closed cases,
   normalizer epsilon/state, native checkpoint round trip, and fixed-observation
   parity. Use the real source checkpoint for a local acceptance check when
   available, but keep committed tests hermetic.
8. Update durable migration/usage documentation. Remove or update stale claims
   that conversion is out of scope.
9. Treat Gym-v1 PointNet, action, sampling, timing, reward, and evaluation values
   as the current default and versioned migration provenance, not permanent
   equality rules for every future checkpoint/runtime.

## Acceptance

- Fixed raw 476D observations show max absolute difference <= `1e-6` for
  normalized observations, deterministic actor `mu`, and critic `value` between
  the source checkpoint formula/runtime and converted target runtime.
- Converted native checkpoint restores model, observation normalizer, and value
  normalizer exactly into a fresh FiLM skrl runtime.
- Source checkpoint SHA256 is unchanged before/after conversion.
- Focused tests, relevant ManoRL checkpoint/model/runtime tests, py_compile,
  `uv lock --check`, and `git diff --check` pass.
- No generated output is deleted or overwritten. Any acceptance artifact belongs
  under a unique `outputs/` path and is reported, not cleaned.
- Commit the coherent implementation on this feature branch, report tests and
  residual risks, then integrate the reviewed feature into `dev`.

## Constraints

- Read `.memory/project/pi-orchestration.md`, this task memory, and the repository
  workflow skill before substantive work.
- Stay within this feature worktree. Preserve unrelated user files and outputs.
- Do not edit or commit anything in the sibling IsaacGym repository.
- MJX-Warp remains the only physical backend. Do not add a legacy CPU backend or
  JSON export path.
- Do not perform a broad physics/reward/action rewrite in this feature. If the
  full live MuJoCo rollout is blocked by pre-existing action/reward contract
  differences, finish checkpoint/policy parity, record the exact blocker, and
  do not hide it.
