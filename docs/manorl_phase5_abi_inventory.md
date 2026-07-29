# ManoRL Phase 5A ABI Inventory

This inventory began as the gate for Phase 5B and now records the completed
Gym-to-MuJoCo policy migration contract. It identifies the source authority,
the target implementation owner, the validated defaults, and the remaining
cross-simulator limits. It does not claim bitwise physics equivalence.

## Migration purpose and outcome

The migration has two concrete purposes:

1. Align IsaacGym and MuJoCo at the policy-semantic boundary: observation
   layout and normalization, PointNet/FiLM structure, deterministic action
   output, residual action authority, trajectory timing, reward parameters,
   reset behavior, and checkpoint state.
2. Establish one trustworthy MuJoCo training baseline. Before alignment,
   MuJoCo used a different network, smaller XYZ and thumb authority, different
   point sampling, timing, reward, and PPO reward scale. Poor training returns
   and checkpoints from that configuration could not distinguish a PPO or
   physics problem from a contract mismatch. Those results are not evidence
   against the aligned MuJoCo implementation.

The aligned contract is now the only production and training default. New
scratch training and checkpoint resume must use the current environment,
reward, PPO, model, and normalizer contracts; checkpoints from the previous
MuJoCo contract are rejected. Historical source traces remain useful as
read-only semantic evidence, but they are not loadable policy/checkpoint
formats.

## Implementation summary

| Code path | Key classes/functions | Concrete migration change | Why it matters |
| --- | --- | --- | --- |
| `sim/manorl/abi.py` | `ResidualActionConfig`, `process_residual_actions`, `early_phase_mask`, `check_termination` | Uses the current 28-DoF MuJoCo action mapping: default XYZ action scale is `0.003 m`, accumulated XYZ cap is `0.03 m`, joint scale/cap multipliers are `2.0`, XYZ/joint gamma is `0.9`, and early phase is 30. The 22 base joint scales/caps follow the six-joint thumb plus four-joint index-through-pinky order. Processing clips normalized actions, zeros inactive fingers and early-phase actions, accumulates residuals, applies limits, and adds the result to the mocap target. Deviation reset remains strictly above `0.10 m`. | A normalized policy action has one authoritative native command mapping. |
| `sim/manorl/observations.py` | `ObservationLayout`, `SOURCE_ALIGNED_COMPATIBILITY`, `build_observation` | Preserves the named field order with shape-derived 28-DoF layouts, `[-5, 5]` policy clipping, a hand-relative 64x3 point-cloud block, 30-step early phase, and movement pre-padding 100. One controlled hand produces 480 observations; two controlled hands produce 505. | Matching the total dimension is insufficient; the checkpoint requires the same internal feature meanings and trajectory frame alignment. |
| `sim/manorl/environment.py` | `EnvironmentConfig`, `_torch_global_surface_templates`, `_set_dynamic_templates`, `_reward_state`, `step` | Makes dynamic point templates the default. GPU sampling uses surface-area CDF face selection followed by square-root barycentric sampling with the global CUDA Torch RNG; construction/reset ordering preserves the source RNG sequence. CPU uses a deterministic fallback on the same MJX-Warp environment path. The environment passes the aligned residual and reward configs through the real physics step, observation, reward, and termination path. | PointNet receives the same sampling distribution as Gym, while viewer, evaluation, and training all execute one physical environment implementation. |
| `sim/manorl/model.py` | `PointNetEncoder`, `FiLMLayer`, `FiLMBlock`, `ManoActorCritic` | Uses the source FiLM actor by default. PointNet maps 64x3 points through `3 -> 64 -> 128 -> 256`, max-pools, then emits 64 features. Action, observation, and conditioning widths derive from the resolved one- or two-hand layout, producing 28 or 56 policy actions. An explicit `use_film=False` model remains available for isolated diagnostics. | One model implementation follows the selected 28-DoF hand layout without hard-coded legacy widths. |
| `sim/manorl/normalization.py` | `PointCloudAwareRunningStandardScaler`, `SourceRunningStandardScaler` | Uses float64 observation moments, shared 3D XYZ statistics across all 64 points, epsilon `1e-5`, and the native value-normalizer state layout. Native checkpoints restore these buffers instead of recomputing statistics during evaluation. | Identical raw observations can otherwise produce different actor inputs even when model weights match exactly. |
| `sim/manorl/rewards.py`, `sim/manorl/skrl_runtime.py`, `sim/manorl/rl_games_ppo.py` | `RewardConfig`, `compute_rewards`, `source_aligned_reward_shaper`, `ManoPPOConfig`, `RlGamesPPO` | Keeps the verified pair-filtered MuJoCo hand-object force calculation, aligns observation and reward contact gates at strict `0.2 N`, keeps direct contact scale at 1x and maximum contact quality at `0.4`, and applies the 0.5 PPO reward shaper. PPO retains 48-step rollouts, `gamma=0.99`, GAE `0.95`, three epochs, clipping `0.2`, and the FiLM model. | Native reward, PPO, and observation environment contracts are versioned for the 0.2 N production boundary. |
| `sim/manorl/checkpoint.py` | `checkpoint_runtime_metadata`, `load_skrl_checkpoint_for_inference`, `load_skrl_checkpoint` | Accepts only native MuJoCo skrl files, validates required modules, finite tensor state, sidecar schema, current reward/PPO/environment contracts, and the resolved hand/action signature. Isaac Gym/rl-games checkpoints are rejected without a conversion path. | A 26-DoF or wrong-hand checkpoint cannot silently enter the current 28/56-wide runtime. |
| `sim/manorl/view_environment.py`, `tools/train_manorl_cube1.py` | `_build_checkpoint_stepper`, `view_environment`, `run`, `_wandb_config` | Viewer and training construct the current default `EnvironmentConfig` and FiLM runtime. Training metadata derives the complete residual config from code; checkpoint sidecars, W&B, and Rerun record the active hand/action contracts. | Evaluation and training share the same current MuJoCo defaults. |
| `tests/manorl/` | ABI, environment, model, normalizer, checkpoint, viewer, and training-contract tests | Verifies exact 28-DoF action steps/caps, dynamic sampler behavior, normalizer state, native checkpoint rejection, hand-side signatures, reward scaling, metadata, and CPU/GPU execution. | Current MuJoCo contracts are tested without retaining an Isaac Gym checkpoint migration surface. |

## Authority and scope

Source authority is the local read-only sibling at
`/home/jay/dexrobot/FromSSH/manohand_reconstruction/IsaacGymEnvs/isaacgymenvs`.
It is historical semantic evidence only. The executable production contract is
MuJoCo-native: 28 DoF per controlled hand, early phase 30, movement pre-padding
100, and reset-local dynamic point templates. Prior checkpoint-sidecar profiles
are no longer exported or loadable.
The accepted `cube1_01_009` replay is a residual-off, deviation-disabled
diagnostic mode and must remain distinct from this training ABI.

The target environment and training runtime are implemented in the owner paths
listed below. Remaining audit items describe cross-simulator evidence gaps, not
missing production defaults.

## Vector-environment boundary

| Source semantic | Source evidence | Target owner | Status |
| --- | --- | --- | --- |
| Policy action space is `[-1, 1]^26`; input is clipped to `clipActions=1.0` before task control. | `cfg/task/MANOHand.yaml:79-91`; `tasks/base/vec_task.py:360-408` | `sim/manorl/abi.py` | Implemented in 5B. |
| Environment returns `{"obs": obs}` plus reward, reset mask, and `extras["time_outs"]`; observation is clamped to `[-5, 5]` before return. | `tasks/base/vec_task.py:360-408`; `cfg/task/MANOHand.yaml:85-86` | `sim/manorl/environment.py` | Implemented for the fixed cube1 source trajectory with a bounded batched MJX-Warp backend; it omits privileged state. |
| No `numStates` is declared by the task. `VecTask` defaults it to zero and omits `states`; there is no privileged actor-critic input in this configuration. | `tasks/base/vec_task.py:101-116, 344-346, 402-406`; no `numStates` in `cfg/task/MANOHand.yaml` | `sim/manorl/environment.py` | Owned: omit privileged output unless a future source configuration explicitly adds it. |
| One task control call performs one physics control interval (`controlFrequencyInv=1`). Source post-step increments progress before reward/termination. | `cfg/task/MANOHand.yaml:79-92`; `tasks/base/vec_task.py:374-394`; `tasks/mano_hand.py:3548-3559` | `sim/manorl/environment.py` | Implemented with two 2.5 ms MJX-Warp substeps and the source counter order. |

## Raw observation ABI

The ordered `observationKeys` in `cfg/task/MANOHand.yaml:7-53` are
checkpoint-bound. `ObservationSpaceConfig` computes dimensions from that order
(`tasks/components/observation_config.py:24-68, 173-212`), and
`ObservationEncoder` directly fills each segment
(`tasks/components/observation_encoder.py:277-393`). The raw observation is
exactly 476 dimensions:

| Slice | Key and source meaning | Target owner |
| --- | --- | --- |
| `0:6` | `wrist_pos`: MANO DOFs `0:6` | `sim/manorl/observations.py` |
| `6:26` | `finger_pos`: DOFs `6:26`, normalized by per-DOF limits to `[-1, 1]` | `sim/manorl/observations.py` |
| `26:30` | `hand_orientation`: task XYZW palm quaternion | `sim/manorl/observations.py` |
| `30:33` | `object_position`: world position | `sim/manorl/observations.py` |
| `33:37` | `object_orientation`: task XYZW quaternion | `sim/manorl/observations.py` |
| `37:40` | `hand_position`: world palm position | `sim/manorl/observations.py` |
| `40:55` | `finger_tip_position`: five true fingertip positions, each object-relative; link-local offsets are rotated by task XYZW quaternions | `sim/manorl/observations.py` |
| `55:58` | `target_object_position`: current mocap target | `sim/manorl/observations.py` |
| `58:62` | `target_object_orientation`: current mocap target XYZW quaternion | `sim/manorl/observations.py` |
| `62:65` | `cumulative_offset`: residual XYZ state | `sim/manorl/abi.py` / `observations.py` |
| `65:68` | `target_object_pos_next_5`: source-named lookahead tensor | `sim/manorl/trajectory.py` / `observations.py` |
| `68:74` | `table_clearance`: object minimum/center Z, hand minimum Z, fingertip minimum Z, reference object minimum/center Z, all relative to table height | `sim/manorl/observations.py` |
| `74:266` | `object_point_cloud_raw`: 64 x 3 coordinates | `sim/manorl/pointcloud.py` / `observations.py` |
| `266:316` | `action_types`: 50D one-hot action ID; action IDs are 1-based and use index `id - 1` | `sim/manorl/metadata.py` / `observations.py` |
| `316:328` | `object_geometry`: source 12D geometry encoding | `sim/manorl/object_features.py` / `observations.py` |
| `328:376` | `hand_keypoints`: 16 x 3 world keypoints minus object position | `sim/manorl/observations.py` |
| `376:392` | `contact_forces`: `tanh(0.025 * ||force||)` for each keypoint | `sim/manorl/contacts.py` / `observations.py` |
| `392:412` | `cumulative_joint_offset`: residual 20D finger state | `sim/manorl/abi.py` / `observations.py` |
| `412:460` | `contact_force_directions`: 16 x 3 unit vectors, exactly zero when magnitude is not above the contact threshold | `sim/manorl/contacts.py` / `observations.py` |
| `460:476` | `expected_contact_mask`: binary 16D expected-grasp mask | `sim/manorl/metadata.py` / `observations.py` |

The source transforms and feature order are explicit in
`tasks/components/observation_encoder.py:12-28, 289-393, 427-487, 565-615`.
The 16 keypoint order is already preserved by target
`sim/manorl/contracts.py:KEYPOINT_NAMES`; it is
`palm, index_mcp, index_pip, index_dip, middle_mcp, middle_pip, middle_dip,
ring_mcp, ring_pip, ring_dip, pinky_mcp, pinky_pip, pinky_dip, thumb_cmc,
thumb_mcp, thumb_ip`.

For the accepted cube1 trajectory, `sim/manorl/environment.py` now produces
these state fields from batched MJX-Warp data: named body transforms in source
keypoint order, five source fingertip offsets, deterministic seed-42 surface
templates, cube support points/geometry, source lookahead indexing, and world
contact-force aggregation. The Warp contact buffer is decoded from its pinned
3.10.0 pyramidal solver rows and contact frames because host conversion retains
padded contact records. A native MuJoCo contact fixture validates the
frame/sign convention. This establishes target-state production, not Isaac
contact-force equivalence or policy/normalizer compatibility.

## Residual action ABI

Each controlled-hand action block is wrist XYZ, wrist roll/pitch/yaw, six
thumb joints, then four-joint index, middle, ring, and pinky groups: `0:3`,
`3:6`, `6:12`, `12:16`, `16:20`, `20:24`, `24:28`. Bimanual actions concatenate
right then left blocks. Raw actions are
clipped before `ActionProcessor` (`vec_task.py:374-376`;
`action_processor.py:157-232, 358-386`). The processor:

1. zeros inactive finger actions and their stored offsets from the expected
   contact mask (`finger_mask_manager.py:68-158`);
2. zeros all actions in `[early_phase_start, early_phase_start + N)`;
3. applies default XYZ scales `(0.003, 0.003, 0.003)`, rotation scale
   `0.025 * 0.01`, and the configured per-joint base scales multiplied by `2.0`;
4. zeros all cumulative offsets on reset, in early phase, on the first step
   after early phase, or in non-residual mode; the first post-early step then
   immediately accumulates its scaled action from zero history, while later
   steps apply gamma `0.9`;
5. clips default XYZ offsets to `[-0.03, 0.03]`; the effective per-joint
   limits are the 22-entry base cap vector multiplied by `2.0`; and
6. adds `[cum_xyz, immediate_rotation, cum_joints]` to the mocap target,
   then clamps to physical DOF limits.

The executable scale, mask, transition, and target assembly are
`cfg/task/MANOHand.yaml:258-290`,
`tasks/components/action_processor.py:87-155, 205-232, 234-286, 388-496`, and
`tasks/mano_hand.py:3471-3524`. The target owner is `sim/manorl/abi.py`, and
the deterministic portion is implemented in this 5B slice. Source
`actionsMovingAverage=1` makes its moving-average assignment an identity for
the configured ABI. The target must not reuse the acceptance replay's
`RESIDUAL_ENABLED=False` as its training default. New v5 checkpoints and
training use `(0.003, 0.003, 0.003)`, `[-0.03, 0.03]`, and `2.0` joint
scale/cap multipliers. V4 checkpoint sidecars retain and restore their explicit
`0.002`/`0.02` and multiplier values.

The production mapping includes all 22 base joint scales and accumulated-offset
limits; the v5 defaults multiply both vectors by `2.0`. The six thumb base
scales are `(0.02, 0.02, 0.02, 0.02, 0.01, 0.005)` and
caps are `(0.2, 0.2, 0.2, 0.2, 0.1, 0.05)`; each other finger uses scales
`(0.01, 0.01, 0.015, 0.005)` and caps `(0.1, 0.1, 0.15, 0.1)`.

## Episode and reset ABI

`TerminationManager` resets an environment when either
`progress >= trajectory_length - 1`, or the Euclidean object-target distance
exceeds `0.10` after the early pure-mocap interval. The associated reward
penalty is `-25.0` only for the deviation condition
(`tasks/components/termination_manager.py:47-82, 85-134`). This is historical source
behavior; target terminal-enabled training uses the same strict predicate at `0.10 m`. A reset places the
object at the first trajectory frame, zeros object velocity, sets hand DOFs and
targets to frame-zero mocap, and zeros cumulative offsets, progress, reset,
episode reward, and statistics (`tasks/mano_hand.py:4423-4506`).

`episodeLength: 600` is configured, but the task's explicit termination uses
the trajectory length, not `max_episode_length`. `VecTask` computes a timeout
only after `post_physics_step`; that method resets done environments and clears
their progress/reset flag first (`vec_task.py:391-400`; `mano_hand.py:3548-3559,
3817-3827, 4483-4489`). The source therefore does not establish an independent
600-step reset semantic for this task. Target owner: `sim/manorl/abi.py` for
the task termination predicate and `sim/manorl/environment.py` for physical
reset. Both are implemented for the bounded cube1 MJX-Warp environment. Its
791-call CPU smoke test returns terminal source target 790, sets timeout
metadata at source completion, and restores frame-zero state on the following
post-step, matching delayed source reset ordering.

## Reward ABI

Reward is separate from termination. Outside the early phase it is distance,
rotation, action penalty, contact reward, post-contact stability, and survival;
in the early phase it is action penalty only
(`tasks/components/reward_calculator.py:79-145`).

- Axis distance is `scale_axis * exp(-40 * abs(object_axis-target_axis))` with
  `(x, y, z) = (0.5, 0.5, 2.0)`
  (`distance_reward_calculator.py:42-73, 76-108`).
- The source distance/contact equations are retained as historical evidence,
  but target contact eligibility is a distinct contract: it uses only strict
  pair-filtered hand-on-object forces, not the aggregate source contact tensor
  or the source object-force-versus-gravity gate.
- Production contract `source_aligned_hand_object_contact_1x_threshold_0p2n_v1` awards `0.4` times the
  weighted fraction of expected keypoints whose filtered world-force norm is
  strictly greater than `0.2 N`; no expected contacts produces zero. Contact
  remains gated by the inclusive movement window, and distance remains gated
  by the resulting proportional contact reward.
- Rotation uses XYZW angular error in degrees and the configured three-piece
  function, then `0.4 * value - 0.1`; the per-object/action disabled mask
  forces zero (`rotation_reward_calculator.py:46-83, 86-161`).
- Action penalty is negative weighted absolute XYZ centimeters plus active-joint
  normalized absolute residuals (`action_penalty_calculator.py:48-87, 90-157`).
- Post-window stability is `0.4 * exp(-(speed / 0.1)^2)` after a valid contact
  window (`reward_calculator.py:209-229`). Survival is `0.001`.

Target owner is `sim/manorl/rewards.py`. The deterministic equation layer is
implemented as a pure resolved-state calculation: it receives the strict
`hand_object_force_on_object_world_N` tensor, expected masks/weights,
trajectory-window values, and the `sim/manorl/abi.py` termination result.
Observations use aggregate source-order keypoint forces with the same strict
`0.2 N` feature threshold as reward eligibility. The source semantic
fixture verifies observation, action, and termination evidence; its reward
fields are reference-only because it lacks pair-filtered hand-object forces.
`tests/manorl/test_observations_rewards.py` fixes proportional weighting,
strict threshold, zero-expected, window, and deviation-penalty behavior.

## Policy, normalization, inference, and checkpoint ABI

With the configured 64-point PointNet, the model replaces raw `74:266` point
cloud values with 64 features. The network input becomes 348D: 74 base values,
64 point features, 50 action-type values, 12 geometry values, 128 contact/hand
values, and 20 joint offsets. FiLM condition input is the 62D action-type plus
geometry block; the source checkpoint uses a 32D embedding and FiLM on actor
layer zero. The source network extracts/reorders exactly as documented in
`learning/mano_network.py:109-228, 492-610`; PPO model parameters are in
`cfg/train/MANOHandPPO.yaml:4-94`.

The custom normalizer maintains per-dimension running statistics outside the
raw point cloud and shared XYZ statistics across all 64 points, then clips
normalized values to `[-5, 5]` (`learning/pointcloud_normalization.py:40-67,
119-171`). Network output is `(mu, log_std, value, None)`; fixed log std is
clamped to `[-10, 2]` (`mano_network.py:599-610`).

Both inspected checkpoints have top-level keys `env_state`, `epoch`, `frame`,
`last_mean_rewards`, `model`, `optimizer`, and `scaler`. Their `model` mapping
has exactly these 50 keys (the observed state-dict namespace):

```
value_mean_std.running_mean
value_mean_std.running_var
value_mean_std.count
running_mean_std.running_mean
running_mean_std.running_var
running_mean_std.count
running_mean_std.pc_running_mean
running_mean_std.pc_running_var
running_mean_std.pc_count
a2c_network.mano_net.log_std
a2c_network.mano_net.pointnet.point_mlp.{0,1,3,4,6,7}.{weight,bias}
a2c_network.mano_net.pointnet.global_mlp.{0,1}.{weight,bias}
a2c_network.mano_net.condition_encoder.encoder.0.{weight,bias}
a2c_network.mano_net.actor_backbone.0.linear.{weight,bias}
a2c_network.mano_net.actor_backbone.0.film.film_generator.{weight,bias}
a2c_network.mano_net.actor_backbone.{1,3,5}.{weight,bias}
a2c_network.mano_net.actor_head.{weight,bias}
a2c_network.mano_net.critic.{0,2,4,6,8}.{weight,bias}
```

The brace notation is an exact grouped enumeration: it expands to the 50
observed keys and adds no alternate namespace. The input running mean/variance
are `(476,)`, shared point-cloud mean/variance are `(3,)`, and `log_std` is
`(26,)`. `MANOHand.pth` records epoch 4872/frame 2873622528; the named final
checkpoint records epoch 6600/frame 3892838400. The optimizer has one parameter
group and 41 parameter-state entries. This is observed file content, not a
portable skrl format.

Target implementations are `sim/manorl/model.py`, `normalization.py`,
`gymnasium_env.py`, `skrl_runtime.py`, and `checkpoint.py`. The PointNet,
FiLM, actor/critic, fixed log-std, and normalizer preserve the source module
shapes and relevant state-dict tails: the FiLM generator is named
`actor_backbone.0.film.film_generator`, and the remaining actor linears retain
source indices `1`, `3`, and `5`. The normalizer retains per-observation
float64 moments plus shared XYZ point moments under the source buffer names.

The target runtime maps the source PPO rollout and KL/scheduler subset: 48
rollout steps, a resolved-run 4096-sample minibatch, three learning epochs,
`gamma=0.99`, `lambda=0.95`, clipping `0.2`, entropy `0.001`, learning rate
`3e-4`, KL threshold `0.016`, gradient norm `1.0`, the source `0.5x` reward
shaper, and timeout bootstrapping. Smaller vector batches automatically select
the largest divisor shared with 4096. The policy stores rollout mean/std and
uses the source `policy_kl` formula plus legacy per-minibatch adaptive schedule;
all minibatches complete without skrl's approximate-KL early stop. A two-sample
`optimizer_smoke` remains deliberately separate from training.

This change does not claim the remaining PPO implementation is fully identical:
the effective critic loss/clipping formula, source bounds loss, and raw sampled
action versus environment-clipped action boundary remain separately versioned
validation items. Gym also walks contiguous minibatch ranges while skrl's
current `RandomMemory` reshuffles samples each learning epoch, so update order
is not yet source-identical. The adapter exposes the source terminal observation,
uses Gymnasium `NEXT_STEP` autoreset semantics, and defines the target
evaluation choice as clipped normalized actor mean. That is a target contract,
not an inferred rl-games player behavior.

Target checkpoint I/O saves and reloads native skrl policy/value, optimizer,
and normalizer state with a configuration sidecar. Resume and inference both
require the current reward, PPO, environment, model-shape, and resolved
hand/action contracts. Raw rl-games top-level `model`/`env_state` checkpoints
are rejected and no converter is maintained; all new checkpoints originate
from the 28-DoF-per-hand MuJoCo runtime.

The aligned dynamic point-cloud sampler uses PyTorch's global CUDA RNG on GPU,
as the source does. It samples triangle faces from the surface-area CDF and then
uses the same square-root barycentric transform. The GPU runtime seeds the
global CUDA generator with 42 and preserves the source construction/reset
ordering. CPU execution remains on the same MJX-Warp environment path with its
deterministic sampling fallback.

## Unresolved-owner audit

The requested Phase 5A scope is closed because each row below has a production
owner and a concrete evidence boundary. “Unresolved” now means simulator-level
parity evidence is still missing; it does not mean the aligned production
default is absent.

| Required semantic | Source is sufficiently specified? | Target owner | Required evidence before it changes status |
| --- | --- | --- | --- |
| Raw 476D composition and slice order | Yes | `sim/manorl/observations.py` | Deterministic resolved-state fixtures now check every named slice and concatenated 476D layout. A source-versus-MuJoCo state fixture remains required once physical extractors exist. |
| Hand/object kinematics, point cloud, geometry, table clearance | Implemented for accepted cube1 MJX-Warp state production | `sim/manorl/environment.py` | Fixed-state Isaac-versus-MuJoCo component fixtures remain needed for cross-simulator parity. |
| Contact force aggregation and expected-mask lookup | Implemented for accepted cube1 MJX-Warp state production | `sim/manorl/environment.py` | Native sign/frame fixture and MJX buffer tests pass; Isaac force equivalence remains needed for parity. |
| 28D residual transform, active-joint masking, early phase | Yes | `sim/manorl/abi.py` | Deterministic tensor cases cover all 22 finger offsets, clipping, masking, transition step, accumulation, and limits. |
| Target application and MuJoCo batched stepping | Yes for the bounded cube1 MJX-Warp scene | `sim/manorl/environment.py` | Focused source-counter test and two-world CPU smoke cover action/target/two-substep ordering. |
| Reset, progress, completion, deviation penalty | Yes for the bounded cube1 MJX-Warp scene | `sim/manorl/abi.py`, `environment.py` | PPO records terminal observations, then explicitly resets completed rows before the next action. The physical environment retains its compatibility delayed-reset path; focused boundary fixtures and the 791-call CPU episode smoke cover both. |
| Rewards and contact-window timing | Aligned strict 0.2 N observation/reward, contact-1x production contract | `sim/manorl/environment.py`, `observations.py`, `rewards.py` | Source raw movement `[690,982]` maps to inclusive sliced window `[250,542]`; strict pair-filtered forces, weighted fractions, exact 0.2 N exclusion, observation-direction gating, and broad-contact isolation are tested. Cross-simulator return equality remains non-comparable because physics-derived forces differ. |
| Privileged/state inputs | Yes: absent in this configuration | `sim/manorl/environment.py` | Assert no `states` output until a source config declares `numStates > 0`. |
| Network, PointNet/FiLM preprocessing, and normalization | Yes | `sim/manorl/model.py`, `normalization.py` | Source module namespace/shape and shared-XYZ statistics are tested; frozen-normalizer and deterministic-mu equivalence on a captured source batch remain required. |
| Inference action selection and skrl adapter | No local Mano source defines the downstream rl-games player choice | `sim/manorl/gymnasium_env.py`, `skrl_runtime.py` | Target policy mode is explicitly clipped normalized mean; deterministic CPU/CUDA physics rollouts and a stochastic PPO update smoke pass. No rl-games evaluation equivalence is claimed. |
| Native checkpoint and optimizer persistence | Yes, for current MuJoCo 28/56-wide layouts only | `sim/manorl/checkpoint.py` | Native modules, current contracts, model shape, and resolved right/left hand signature are required. Isaac Gym/rl-games inputs and unsigned pre-28-DoF sidecars fail closed. |
| Current-source versus checkpoint-sidecar training settings | Yes: historical sidecar metadata differs from the successful invocation | `sim/manorl/observations.py`, `sim/manorl/view_environment.py` | Historical sidecar 50/dynamic/200 remains recorded evidence; successful 50/dynamic/250 evaluation is the only production runtime contract. |

## Gate disposition

Every requested semantic has a source citation and target owner. Items that
need simulator contact, point-cloud, geometry, reward, model, skrl, or
checkpoint work are explicitly unresolved with a validation path. This closes
the 5A ownership gate and permits only the isolated residual-control and
termination implementation below. It does not close Phases 5B-5E.
