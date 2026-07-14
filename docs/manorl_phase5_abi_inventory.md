# ManoRL Phase 5A ABI Inventory

This inventory is the gate for Phase 5B. It records what the checked-out
Isaac/rl-games implementation does, assigns a target owner, and names every
semantic that cannot yet be claimed by the MuJoCo target. It is not evidence
of training, policy, or checkpoint compatibility.

## Authority and scope

Source authority is the local read-only sibling at
`/home/jay/dexrobot/FromSSH/manohand_reconstruction/IsaacGymEnvs/isaacgymenvs`.
The checkpoint sidecar confirms the same 476D layout but is a recorded training
configuration, not the current executable task configuration. In particular it
has `earlyPhaseMocapSteps: 50`, `movementFramePrePadding: 200`, and dynamic
point templates, whereas the current task sets 100, 250, and non-dynamic
templates (`checkpoint/18-09-37-36_MANOHand_all_all-new_setting-val099-new_obs/config.yaml:1-24,142-154`;
`cfg/task/MANOHand.yaml:237-255`). Checkpoint inference compatibility must use
the sidecar provenance; new target training follows the current source unless a
later task explicitly selects the historical configuration.
The accepted `cube1_01_009` replay is a residual-off, deviation-disabled
diagnostic mode and must remain distinct from this training ABI.

The target has no pre-existing training environment. Planned owner paths below
are contracts for later slices, not claims that those modules already exist.

## Vector-environment boundary

| Source semantic | Source evidence | Target owner | Status |
| --- | --- | --- | --- |
| Policy action space is `[-1, 1]^26`; input is clipped to `clipActions=1.0` before task control. | `cfg/task/MANOHand.yaml:79-91`; `tasks/base/vec_task.py:360-408` | `sim/manorl/abi.py` | Implemented in 5B. |
| Environment returns `{"obs": obs}` plus reward, reset mask, and `extras["time_outs"]`; observation is clamped to `[-5, 5]` before return. | `tasks/base/vec_task.py:360-408`; `cfg/task/MANOHand.yaml:85-86` | `sim/manorl/environment.py` | Unresolved: full environment wrapper is later 5B work. |
| No `numStates` is declared by the task. `VecTask` defaults it to zero and omits `states`; there is no privileged actor-critic input in this configuration. | `tasks/base/vec_task.py:101-116, 344-346, 402-406`; no `numStates` in `cfg/task/MANOHand.yaml` | `sim/manorl/environment.py` | Owned: omit privileged output unless a future source configuration explicitly adds it. |
| One task control call performs one physics control interval (`controlFrequencyInv=1`). Source post-step increments progress before reward/termination. | `cfg/task/MANOHand.yaml:79-92`; `tasks/base/vec_task.py:374-394`; `tasks/mano_hand.py:3548-3559` | `sim/manorl/environment.py` | Unresolved: bind to batched MJX stepping after control core is validated. |

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

Unresolved observation owners are deliberately separated: batched MuJoCo
contact-to-keypoint force aggregation, surface-point sampling/hand-relative
coordinates, 12D source geometry encoding, table support-point reduction, and
the source `target_object_pos_next_5` indexing rule. Their validation is a
per-component source-state fixture comparison before a full environment is
advertised.

## Residual action ABI

The action ordering is wrist XYZ, wrist roll/pitch/yaw, then thumb, index,
middle, ring, pinky four-joint groups: `0:3`, `3:6`, `6:26`. Raw actions are
clipped before `ActionProcessor` (`vec_task.py:374-376`;
`action_processor.py:157-232, 358-386`). The processor:

1. zeros inactive finger actions and their stored offsets from the expected
   contact mask (`finger_mask_manager.py:68-158`);
2. zeros all actions in `[early_phase_start, early_phase_start + N)`;
3. applies XYZ scales `(0.005, 0.005, 0.005)`, rotation scale
   `0.025 * 0.01`, and the configured per-joint scales;
4. zeros all cumulative offsets on reset, in early phase, on the first step
   after early phase, or in non-residual mode; the first post-early step then
   immediately accumulates its scaled action from zero history, while later
   steps apply gamma `0.9`;
5. clips XYZ offsets to `[-0.05, 0.05]` and joint offsets to their per-joint
   limits; and
6. adds `[cum_xyz, immediate_rotation, cum_joints]` to the mocap target,
   then clamps to physical DOF limits.

The executable scale, mask, transition, and target assembly are
`cfg/task/MANOHand.yaml:258-290`,
`tasks/components/action_processor.py:87-155, 205-232, 234-286, 388-496`, and
`tasks/mano_hand.py:3471-3524`. The target owner is `sim/manorl/abi.py`, and
the deterministic portion is implemented in this 5B slice. Source
`actionsMovingAverage=1` makes its moving-average assignment an identity for
the configured ABI. The target must not reuse the acceptance replay's
`RESIDUAL_ENABLED=False` as its training default.

## Episode and reset ABI

`TerminationManager` resets an environment when either
`progress >= trajectory_length - 1`, or the Euclidean object-target distance
exceeds `0.10` after the early pure-mocap interval. The associated reward
penalty is `-25.0` only for the deviation condition
(`tasks/components/termination_manager.py:47-82, 85-134`). A reset places the
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
reset. The predicate is implemented in 5B; the physical reset remains open.

## Reward ABI

Reward is separate from termination. Outside the early phase it is distance,
rotation, action penalty, contact reward, post-contact stability, and survival;
in the early phase it is action penalty only
(`tasks/components/reward_calculator.py:79-145`).

- Axis distance is `scale_axis * exp(-40 * abs(object_axis-target_axis))` with
  `(x, y, z) = (0.5, 0.5, 2.0)`
  (`distance_reward_calculator.py:42-73, 76-108`).
- Distance components are gated by weighted expected contact reward during the
  inclusive contact window, use `maxContactReward=0.4` after it, and are zero
  before it (`reward_calculator.py:147-207`). Contact also requires the
  object contact magnitude to differ from gravity force by more than `0.25`.
- Contact reward is weighted-correct expected contacts divided by expected
  weight sum, multiplied by `0.4`; no expected contacts produces zero
  (`utils/contact_reward_jit.py:15-67`).
- Rotation uses XYZW angular error in degrees and the configured three-piece
  function, then `0.4 * value - 0.1`; the per-object/action disabled mask
  forces zero (`rotation_reward_calculator.py:46-83, 86-161`).
- Action penalty is negative weighted absolute XYZ centimeters plus active-joint
  normalized absolute residuals (`action_penalty_calculator.py:48-87, 90-157`).
- Post-window stability is `0.4 * exp(-(speed / 0.1)^2)` after a valid contact
  window (`reward_calculator.py:209-229`). Survival is `0.001`.

Target owner is `sim/manorl/rewards.py`. The deterministic equation layer is
implemented as a pure resolved-state calculation: it receives source-order
keypoint forces, expected masks/weights, object contact/gravity values,
trajectory-window values, and the `sim/manorl/abi.py` termination result.
`tests/manorl/test_observations_rewards.py` fixes the component equations,
window boundaries, historical/current early-phase variants, and the final
deviation-penalty ordering. It does not establish MuJoCo producers for those
inputs; contact aggregation and trajectory-window mapping remain unresolved.

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

Target owners are `sim/manorl/model.py`, `sim/manorl/normalization.py`, and
`sim/manorl/checkpoint.py`. They remain unresolved by design. Required evidence
before compatibility claims is: frozen-normalizer preprocessing equivalence,
deterministic mu equivalence on a captured 476D batch, explicit key/shape map,
and a documented optimizer-state migration or an explicit rejection. Exact
rl-games action sampling versus deterministic player selection is upstream
behavior not defined in the local Mano files inspected here; the future skrl
adapter must choose and test its evaluation mode rather than infer one.

## Unresolved-owner audit

The requested Phase 5A scope is closed only because each row below has a
specific destination or a concrete evidence blocker. “Unresolved” means the
target implementation does not exist; it does not mean that source behavior is
unknown.

| Required semantic | Source is sufficiently specified? | Target owner | Required evidence before it changes status |
| --- | --- | --- | --- |
| Raw 476D composition and slice order | Yes | `sim/manorl/observations.py` | Deterministic resolved-state fixtures now check every named slice and concatenated 476D layout. A source-versus-MuJoCo state fixture remains required once physical extractors exist. |
| Hand/object kinematics, point cloud, geometry, table clearance | Partly; source formula is known but target simulator data extraction is absent | `sim/manorl/observations.py`, `pointcloud.py`, `object_features.py` | Fixed-state MuJoCo-versus-source component fixtures, including coordinate frames and keypoint ordering. |
| Contact force aggregation and expected-mask lookup | Partly; source output formula and 16D ordering are known, target contact mapping is absent | `sim/manorl/contacts.py`, `metadata.py` | Keypoint-force and mask fixtures from a contact-bearing source episode. |
| 26D residual transform, active-joint masking, early phase | Yes | `sim/manorl/abi.py` | Deterministic tensor cases for clipping, masking, transition step, accumulation, and limits. |
| Target application and MuJoCo batched stepping | Partly; source target order is known, target vector runtime absent | `sim/manorl/environment.py` | One-world then batched action/target/physics ordering trace. |
| Reset, progress, completion, deviation penalty | Yes for predicates; physical state writes are simulator-specific | `sim/manorl/abi.py`, `environment.py` | Termination truth-table and reset-state fixture with a trajectory restart. |
| Rewards and contact-window timing | Yes for equations; target contact/window producers absent | `sim/manorl/rewards.py` | Component tensor oracles are implemented; a contact-window episode fixture remains required once the physical producers exist. |
| Privileged/state inputs | Yes: absent in this configuration | `sim/manorl/environment.py` | Assert no `states` output until a source config declares `numStates > 0`. |
| Network, PointNet/FiLM preprocessing, and normalization | Yes | `sim/manorl/model.py`, `normalization.py` | Frozen-normalizer and deterministic-mu equivalence on recorded 476D batches. |
| Inference action selection and skrl adapter | No local Mano source defines the downstream rl-games player choice | `sim/manorl/skrl_adapter.py` | Explicit policy-mode contract and an evaluated deterministic/stochastic test; no inference mode may be inferred. |
| Checkpoint and optimizer migration | File layout known; skrl mapping absent | `sim/manorl/checkpoint.py` | Key/shape conversion report plus loaded-model mu/value equivalence, or explicit unsupported-format rejection. |
| Current-source versus checkpoint-sidecar training settings | Yes: they differ in named fields above | `sim/manorl/observations.py` / future `config.py` | Pure compatibility variants explicitly select current (100/static/250) or historical checkpoint (50/dynamic/200); serialize the selection before an environment or checkpoint claim. |

## Gate disposition

Every requested semantic has a source citation and target owner. Items that
need simulator contact, point-cloud, geometry, reward, model, skrl, or
checkpoint work are explicitly unresolved with a validation path. This closes
the 5A ownership gate and permits only the isolated residual-control and
termination implementation below. It does not close Phases 5B-5E.
