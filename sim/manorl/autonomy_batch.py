"""Device-resident homogeneous cube2 autonomy batching.

This module is intentionally a thin port of the established v2 control path.
It keeps the measured-state rate map and clock unchanged, while replacing the
per-step host collision query with fixed offline MuJoCo witness endpoints.
Reference tables are immutable device inputs; they never enter the command map.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, NamedTuple, Sequence

import numpy as np
from numpy.typing import NDArray

from sim.manorl.autonomy import align_reference_trajectory, _quat_rotate
from sim.manorl.autonomy_contracts import (
    ACTION_DIM, ACTION_V3_CONTRACT_ID, AUTONOMY_V3_VERSION,
    CHECKPOINT_V3_FORMAT, OBSERVATION_V3_CONTRACT_ID, REWARD_V3_CONTRACT_ID,
    WITNESS_TABLE_CONTRACT_ID, rate_limited_command,
)
from sim.manorl.assets import compile_model, object_collision_vertices, object_runtime
from sim.manorl.contracts import FLOOR_TOP_Z, JOINT_DOF, KEYPOINT_NAMES, simulation_clock
from sim.manorl.device_runtime import (
    extract_mjx_physical_features, jax_to_torch_cuda, torch_to_jax_cuda,
)
from sim.manorl.environment import (
    MjxWarpPhysicalProducer, _FINGERTIP_LOCAL_OFFSETS, _FINGERTIP_NAMES,
    _build_masked_reset_data_fn, recommended_warp_contact_capacity,
)
from sim.manorl.trajectory import ReferenceTrajectory

V3_OBSERVATION_DIM = 538  # same width as v2; only fixed-witness semantics differ


@dataclass(frozen=True)
class ReferenceWitnessTables:
    """Offline rigid-body-local witness table in source keypoint order."""
    hand_endpoint_local: NDArray[np.float64]       # (T, 16, 3)
    object_endpoint_local: NDArray[np.float64]     # (T, 16, 3)
    signed_distance: NDArray[np.float64]            # (T, 16)
    confidence: NDArray[np.float64]                 # (T, 16)
    hand_geom_id: NDArray[np.int32]                 # (16,)
    object_geom_id: NDArray[np.int32]               # (T, 16)
    hand_body_id: NDArray[np.int32]                 # (16,)
    object_body_id: int
    support_shift: NDArray[np.float64]
    contract_id: str = WITNESS_TABLE_CONTRACT_ID
    confidence_rule: str = "exp(-abs(signed_distance_m)/0.01); penetration retained as intent"
    digest: str = ""

    def __post_init__(self) -> None:
        t = self.signed_distance.shape[0]
        if self.hand_endpoint_local.shape != (t, 16, 3) or self.object_endpoint_local.shape != (t, 16, 3):
            raise ValueError("witness endpoints must be (frames, 16, 3)")
        if self.confidence.shape != (t, 16) or self.object_geom_id.shape != (t, 16):
            raise ValueError("witness distance metadata must be (frames, 16)")
        if self.hand_geom_id.shape != (16,) or self.hand_body_id.shape != (16,):
            raise ValueError("source-order hand metadata must be (16,)")
        if self.support_shift.shape != (3,) or not np.all(np.isfinite(self.support_shift)):
            raise ValueError("support shift must be finite and 3-wide")
        for value in (self.hand_endpoint_local, self.object_endpoint_local, self.signed_distance, self.confidence):
            if not np.all(np.isfinite(value)):
                raise ValueError("witness table contains non-finite values")
        if np.any(self.confidence < 0) or np.any(self.confidence > 1):
            raise ValueError("witness confidence must be in [0, 1]")

    def payload_digest(self) -> str:
        digest = hashlib.sha256()
        for value in (self.hand_endpoint_local, self.object_endpoint_local, self.signed_distance,
                      self.confidence, self.hand_geom_id, self.object_geom_id, self.hand_body_id,
                      np.asarray([self.object_body_id]), self.support_shift):
            digest.update(np.ascontiguousarray(value).tobytes())
        return digest.hexdigest()


def reference_witness_tables(
    trajectory: ReferenceTrajectory, *, object_type: str = "cube2", table_height: float = FLOOR_TOP_Z
) -> ReferenceWitnessTables:
    """Compute fixed MuJoCo geom-distance witnesses once, outside runtime steps."""
    if trajectory.dof_dim != JOINT_DOF:
        raise ValueError("witness extraction requires the 28-DoF trajectory")
    mujoco, model = compile_model(object_type=object_type, hand_side="right", physics_timestep=1 / 480)
    producer = MjxWarpPhysicalProducer(mujoco, model, object_type=object_type, hand_sides=("right",))
    vertices = np.asarray(object_collision_vertices(object_type), dtype=np.float64)
    aligned_q, aligned_obj, shift = align_reference_trajectory(trajectory, vertices, table_height)
    object_body = producer.object_body_id
    hand_bodies = np.asarray(producer.keypoint_body_ids, dtype=np.int32)
    hand_geoms = np.asarray(producer.keypoint_geom_ids, dtype=np.int32)
    object_geoms = tuple(sorted(producer.object_geom_ids))
    hand_local = np.zeros((len(aligned_q), 16, 3), dtype=np.float64)
    object_local = np.zeros_like(hand_local)
    signed = np.zeros((len(aligned_q), 16), dtype=np.float64)
    object_ids = np.zeros((len(aligned_q), 16), dtype=np.int32)
    for frame, (q, obj, quat) in enumerate(zip(aligned_q, aligned_obj, trajectory.object_quat_xyzw, strict=True)):
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        data.qpos[:JOINT_DOF] = q
        data.qpos[producer.object_qpos_address:producer.object_qpos_address + 3] = obj
        data.qpos[producer.object_qpos_address + 3:producer.object_qpos_address + 7] = quat[[3, 0, 1, 2]]
        mujoco.mj_forward(model, data)
        for segment, hand_geom in enumerate(hand_geoms):
            best_distance = np.inf
            best_fromto = np.zeros(6, dtype=np.float64)
            best_object = -1
            for object_geom in object_geoms:
                fromto = np.zeros(6, dtype=np.float64)
                distance = float(mujoco.mj_geomDistance(model, data, int(hand_geom), int(object_geom), 1.0, fromto))
                if distance < best_distance:
                    best_distance, best_fromto, best_object = distance, fromto, object_geom
            body_rotation = np.asarray(data.xmat[hand_bodies[segment]], dtype=np.float64).reshape(3, 3)
            object_rotation = np.asarray(data.xmat[object_body], dtype=np.float64).reshape(3, 3)
            hand_local[frame, segment] = body_rotation.T @ (best_fromto[:3] - data.xpos[hand_bodies[segment]])
            object_local[frame, segment] = object_rotation.T @ (best_fromto[3:] - data.xpos[object_body])
            signed[frame, segment] = best_distance
            object_ids[frame, segment] = best_object
    confidence = np.exp(-np.abs(signed) / 0.01)
    result = ReferenceWitnessTables(
        hand_local, object_local, signed, confidence, hand_geoms, object_ids,
        hand_bodies, int(object_body), np.asarray(shift, dtype=np.float64)
    )
    object.__setattr__(result, "digest", result.payload_digest())
    for value in (result.hand_endpoint_local, result.object_endpoint_local, result.signed_distance,
                  result.confidence, result.hand_geom_id, result.object_geom_id, result.hand_body_id,
                  result.support_shift):
        value.setflags(write=False)
    return result


def _quat_rotate_device(quaternion: Any, vectors: Any) -> Any:
    import jax.numpy as jp
    qv, qw = quaternion[..., :3], quaternion[..., 3:4]
    return vectors * (2 * qw * qw - 1) + jp.cross(qv, vectors) * qw * 2 + qv * jp.sum(qv * vectors, axis=-1, keepdims=True) * 2


def rate_limited_command_batch(previous_command: Any, action: Any, lower: Any, upper: Any, rate_per_second: Any, measured_qpos: Any, max_tracking_error: Any, *, control_timestep: float = 1 / 120) -> Any:
    """JAX equivalent of v2 ``rate_limited_command`` for (B, 28) arrays."""
    import jax.numpy as jp
    previous, normalized = jp.asarray(previous_command), jp.asarray(action)
    lower, upper, rate, measured, envelope = map(jp.asarray, (lower, upper, rate_per_second, measured_qpos, max_tracking_error))
    if any(value.shape != (28,) for value in (lower, upper, rate, envelope)) or previous.ndim != 2 or normalized.shape != previous.shape or measured.shape != previous.shape or previous.shape[1] != 28:
        raise ValueError("batched command vectors require (batch, 28) and 28-wide limits")
    command = previous + jp.clip(normalized, -1, 1) * rate[None] * control_timestep
    return jp.clip(jp.clip(command, measured - envelope[None], measured + envelope[None]), lower, upper)


class DeviceWitnessFeatures(NamedTuple):
    hand_points: Any
    object_points: Any
    correspondence_error: Any
    proximity: Any
    confidence: Any
    signed_distance: Any
    object_endpoint_local: Any


def transform_fixed_witnesses(physical: Any, hand_endpoint_local: Any, object_endpoint_local: Any, signed_distance: Any, confidence: Any) -> DeviceWitnessFeatures:
    """Transform selected offline endpoints with actual body/object poses."""
    import jax.numpy as jp
    hand = physical.hand_keypoint_positions + _quat_rotate_device(physical.hand_keypoint_orientations_xyzw, hand_endpoint_local)
    obj = physical.object_position[:, None] + _quat_rotate_device(physical.object_orientation_xyzw[:, None], object_endpoint_local)
    error = jp.linalg.norm(hand - obj, axis=-1)
    # Signed d is retained for audit. Proximity is an unsigned metric because
    # penetration is geometrically zero separation plus a valid contact intent.
    proximity = jp.exp(-error / 0.01)
    return DeviceWitnessFeatures(hand, obj, error, proximity, confidence, signed_distance, object_endpoint_local)


class DeviceAutonomyReward(NamedTuple):
    total: Any
    object_motion: Any
    relation: Any
    correspondence: Any
    measured_contact: Any
    slip_proxy: Any
    release: Any
    smoothness: Any
    valid: Any


def compute_device_autonomy_reward_v3(*, object_position: Any, target_object_position: Any, object_velocity: Any, hand_object_relative: Any, reference_hand_object_relative: Any, witness: DeviceWitnessFeatures, reference_confidence: Any, hand_object_force: Any, relative_contact_motion: Any, action: Any, release_active: Any, measured_q: Any | None = None, reference_q: Any | None = None, object_quaternion: Any | None = None, reference_object_quaternion: Any | None = None, target_velocity: Any | None = None) -> DeviceAutonomyReward:
    """Device form of the current additive v2 terms.

    Optional state terms mirror ``dense_autonomous_reward`` exactly; omitted
    values are neutral only for pure kernel tests. There is no move gate and
    force magnitude is used only by measured-contact/slip terms.
    """
    import jax.numpy as jp
    motion = jp.linalg.norm(object_position - target_object_position, axis=1)
    relation = jp.linalg.norm(hand_object_relative - reference_hand_object_relative, axis=1)
    force_norm = jp.linalg.norm(hand_object_force, axis=-1)
    contact = jp.mean(jp.minimum(force_norm / 0.2, 1.0) * witness.proximity * reference_confidence, axis=1)
    slip = jp.mean(jp.linalg.norm(relative_contact_motion, axis=-1) * (force_norm > 0.02), axis=1)
    release = jp.where(jp.asarray(release_active), 1 - jp.mean(witness.proximity, axis=1), 0.)
    smooth = jp.mean(jp.square(action), axis=1)
    finger = jp.zeros((object_position.shape[0],), dtype=object_position.dtype)
    orientation = jp.ones_like(finger)
    velocity = jp.ones_like(finger)
    if measured_q is not None and reference_q is not None:
        finger_error = jp.linalg.norm(jp.asarray(measured_q)[:, 6:] - jp.asarray(reference_q)[:, 6:], axis=1) / jp.sqrt(22.)
        finger = jp.exp(-finger_error)
    if object_quaternion is not None and reference_object_quaternion is not None:
        dot = jp.abs(jp.sum(jp.asarray(object_quaternion) * jp.asarray(reference_object_quaternion), axis=1))
        orientation = jp.exp(-8. * (1. - dot))
    if target_velocity is not None:
        velocity = jp.exp(-8. * jp.linalg.norm(object_velocity - jp.asarray(target_velocity), axis=1))
    terms = (jp.exp(-20 * motion), jp.exp(-30 * relation), jp.mean(witness.proximity * reference_confidence, axis=1) * jp.exp(-jp.mean(witness.correspondence_error, axis=1) / .01), contact, -.02 * slip, release, -.001 * smooth, finger, orientation, velocity)
    total = sum(terms)
    valid = jp.all(jp.isfinite(total)) & jp.all(jp.isfinite(object_velocity)) & jp.all(jp.isfinite(hand_object_force))
    return DeviceAutonomyReward(total, terms[0], terms[1], terms[2], terms[3], terms[4], terms[5], terms[6], valid)


def masked_row_reset(data: Any, reset_mask: Any, reset_fn: Any) -> Any:
    """Apply only completed rows before their next actor action."""
    return reset_fn(data, reset_mask)


def build_device_autonomy_observation(*, physical: Any, contact: Any, witness: DeviceWitnessFeatures, reference_q: Any, reference_object: Any, previous_command: Any, action_ids: Any, index: Any, lower: Any, upper: Any, object_geometry: Any | None = None) -> Any:
    """Build the existing 538-wide v2 layout from device fields.

    The layout is unchanged for direct-port comparability. ``surface_proximity``
    is the only geometry-semantic difference: it is fixed-witness
    correspondence, never a runtime nearest-neighbor query.
    """
    import jax.numpy as jp
    from sim.manorl.autonomy_contracts import observation_slices
    b = physical.mano_dof_pos.shape[0]
    if reference_q.shape != (b, reference_q.shape[1], 28) or previous_command.shape != (b, 28):
        raise ValueError("device autonomy observation has inconsistent batch shapes")
    s = observation_slices(); out = jp.zeros((b, V3_OBSERVATION_DIM), dtype=jp.float32)
    lower, upper = jp.asarray(lower), jp.asarray(upper)
    out = out.at[:, s["measured_qpos_normalized"]].set(2 * (physical.mano_dof_pos - lower[None]) / jp.maximum(upper[None] - lower[None], 1e-6) - 1)
    # MJX qvel is not needed for the direct port's control map; velocity fields
    # remain actual device physics values where available.
    out = out.at[:, s["measured_qvel"]].set(jp.zeros((b, 28), dtype=out.dtype))
    out = out.at[:, s["object_position"]].set(physical.object_position)
    out = out.at[:, s["object_linear_velocity"]].set(physical.object_linear_velocity)
    out = out.at[:, s["hand_object_relative"]].set(physical.hand_keypoint_positions[:, 0] - physical.object_position)
    i = jp.minimum(index, reference_q.shape[1] - 1); world = jp.arange(b)
    out = out.at[:, s["reference_q_current"]].set(reference_q[world, i])
    out = out.at[:, s["reference_q_next"]].set(reference_q[world, jp.minimum(i + 1, reference_q.shape[1] - 1)])
    out = out.at[:, s["reference_q_velocity"]].set((reference_q[world, jp.minimum(i + 1, reference_q.shape[1] - 1)] - reference_q[world, i]) * 120.)
    out = out.at[:, s["reference_object_relative"]].set(reference_q[world, i, :3] - reference_object[world, i])
    out = out.at[:, s["reference_object_future_delta"]].set(reference_object[world, jp.minimum(i + 5, reference_q.shape[1] - 1)] - reference_object[world, i])
    out = out.at[:, s["previous_command_normalized"]].set(jp.tanh(previous_command))
    out = out.at[:, s["action_identity_one_hot"]].set(jp.eye(50, dtype=out.dtype)[jp.clip(jp.asarray(action_ids, dtype=jp.int32) - 1, 0, 49)])
    out = out.at[:, s["object_geometry"]].set(jp.broadcast_to(jp.asarray(object_geometry if object_geometry is not None else [.05, .05, .05, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=out.dtype), (b, 12)))
    out = out.at[:, s["measured_keypoint_relative"]].set((physical.hand_keypoint_positions - physical.object_position[:, None]).reshape(b, -1))
    out = out.at[:, s["surface_proximity"]].set(witness.proximity)
    out = out.at[:, s["surface_anchor_local"]].set(witness.object_endpoint_local.reshape(b, -1))
    out = out.at[:, s["contact_phase_confidence"]].set(jp.stack((index / jp.maximum(reference_q.shape[1] - 1, 1), jp.mean(witness.confidence, axis=1)), axis=1))
    out = out.at[:, s["reference_surface_proximity"]].set(jp.exp(-jp.abs(witness.signed_distance) / .01))
    out = out.at[:, s["reference_surface_anchor_local"]].set(witness.object_endpoint_local.reshape(b, -1))
    out = out.at[:, s["reference_contact_confidence"]].set(witness.confidence)
    out = out.at[:, s["measured_hand_object_force"]].set(contact.hand_object_forces.reshape(b, -1))
    out = out.at[:, s["supporting_object_net_force"]].set(jp.sum(contact.hand_object_forces, axis=1))
    out = out.at[:, s["relative_contact_motion"]].set(jp.zeros((b, 48), dtype=out.dtype))
    return jp.clip(out, -5., 5.)


class BatchedAutonomyRuntime:
    """Homogeneous cube2 MJX-Warp batch with device-only transition state."""
    def __init__(self, trajectory: ReferenceTrajectory, *, num_envs: int = 1, device: str = "gpu", seed: int = 0):
        if not isinstance(num_envs, int) or isinstance(num_envs, bool) or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        if trajectory.identity.identity.split("_")[0] != "cube2" or trajectory.dof_dim != 28:
            raise ValueError("batched autonomy currently requires homogeneous cube2 28-DoF references")
        if device not in {"cpu", "gpu"}:
            raise ValueError("device must be cpu or gpu")
        import jax
        import jax.numpy as jp
        from mujoco import mjx
        self.num_envs, self.device_name, self.seed = num_envs, device, int(seed)
        self.trajectory = trajectory
        self.clock = simulation_clock(120)
        self.mujoco, self.model = compile_model(object_type="cube2", hand_side="right", physics_timestep=1 / 480)
        self.jax, self.jp, self.mjx = jax, jp, mjx
        devices = jax.devices(device)
        if not devices: raise RuntimeError(f"no JAX {device} device available")
        self.device = devices[0]
        self.mjx_model = mjx.put_model(self.model, device=self.device, impl="warp")
        self.producer = MjxWarpPhysicalProducer(self.mujoco, self.model, object_type="cube2", hand_sides=("right",))
        self.witness = reference_witness_tables(trajectory)
        self.aligned_q_ref, self.aligned_object_pos, self.support_shift = align_reference_trajectory(trajectory, np.asarray(object_collision_vertices("cube2")))
        self.length = len(self.aligned_q_ref)
        self.reference_q = self._tile(self.aligned_q_ref)
        self.reference_obj = self._tile(self.aligned_object_pos)
        self.reference_quat = self._tile(np.asarray(trajectory.object_quat_xyzw, dtype=np.float32))
        self.reference_h = self._tile(self.witness.hand_endpoint_local)
        self.reference_o = self._tile(self.witness.object_endpoint_local)
        self.reference_d = self._tile(self.witness.signed_distance)
        self.reference_c = self._tile(self.witness.confidence)
        self.lower = np.asarray(self.model.jnt_range[:28, 0], dtype=np.float32)
        self.upper = np.asarray(self.model.jnt_range[:28, 1], dtype=np.float32)
        self.rate = np.r_[np.full(6, [0.5, 0.5, 0.5, 2., 2., 2.]), np.full(22, 4.)].astype(np.float32)
        self.envelope = np.r_[np.full(6, [0.02, 0.02, 0.02, .25, .25, .25]), np.full(22, .35)].astype(np.float32)
        data = self.mujoco.MjData(self.model); self._reset_host(data)
        capacity = recommended_warp_contact_capacity(num_envs, ("right",))
        single = mjx.put_data(self.model, data, device=self.device, impl="warp", naconmax=capacity, njmax=max(512, capacity * 2))
        self.data = jax.vmap(lambda _: single)(jp.arange(num_envs))
        masked_reset = _build_masked_reset_data_fn(jax=jax, jp=jp, reset_qpos=jp.broadcast_to(self.data.qpos, self.data.qpos.shape), reset_ctrl=jp.broadcast_to(self.data.ctrl, self.data.ctrl.shape))
        # Recompute derived poses on-device after indexed reset; terminal rows
        # therefore cannot leak stale x-position/quaternion features.
        self._reset_fn = jax.jit(lambda value, mask: mjx.forward(self.mjx_model, masked_reset(value, mask)))
        self.step_fn = jax.jit(lambda value: jax.vmap(lambda world: mjx.step(self.mjx_model, world))(value))
        self.indices = jp.zeros((num_envs,), dtype=jp.int32)
        self.pending_reset = jp.zeros((num_envs,), dtype=bool)
        self.previous_command = jp.asarray(np.broadcast_to(self._reset_q, (num_envs, 28)), dtype=jp.float32)
        self._observe()

    def _tile(self, value: NDArray[np.floating]) -> Any:
        return self.jax.device_put(np.broadcast_to(np.asarray(value), (self.num_envs, *value.shape)), self.device)

    def _reset_host(self, data: Any) -> None:
        self._reset_q = np.asarray(self.aligned_q_ref[0], dtype=np.float32)
        data.qpos[:28] = self._reset_q
        data.qpos[self.producer.object_qpos_address:self.producer.object_qpos_address + 3] = self.aligned_object_pos[0]
        data.qpos[self.producer.object_qpos_address + 3:self.producer.object_qpos_address + 7] = self.trajectory.object_quat_xyzw[0][[3, 0, 1, 2]]
        data.qvel[:] = 0; data.ctrl[:28] = self._reset_q; self.mujoco.mj_forward(self.model, data)

    def _observe(self) -> Any:
        jp = self.jp
        physical = extract_mjx_physical_features(qpos=self.data.qpos, qvel=self.data.qvel, xpos=self.data.xpos, xquat=self.data.xquat, hand_qpos_start=0, hand_dof=28, object_body_id=self.producer.object_body_id, object_qvel_address=self.producer.object_qvel_address, keypoint_body_ids=tuple(self.producer.keypoint_body_ids), fingertip_keypoint_ids=tuple(KEYPOINT_NAMES.index(name) for name in _FINGERTIP_NAMES), fingertip_local_offsets=_FINGERTIP_LOCAL_OFFSETS)
        contact = self.producer.device_contact_reduction(self.data)
        world = self.jp.arange(self.num_envs); index = self.jp.minimum(self.indices, self.length - 1)
        witness = transform_fixed_witnesses(physical, self.reference_h[world, index], self.reference_o[world, index], self.reference_d[world, index], self.reference_c[world, index])
        self.last_physical, self.last_contact, self.last_witness = physical, contact, witness
        self.observation = build_device_autonomy_observation(
            physical=physical, contact=contact, witness=witness,
            reference_q=self.reference_q, reference_object=self.reference_obj,
            previous_command=self.previous_command,
            action_ids=jp.full((self.num_envs,), int(self.trajectory.identity.identity.split("_")[1]), dtype=jp.int32),
            index=self.indices, lower=jp.asarray(self.lower), upper=jp.asarray(self.upper),
        )
        return self.observation

    def reset(self, mask: Any | None = None) -> Any:
        if mask is None: mask = self.jp.ones((self.num_envs,), dtype=bool)
        mask = self.jp.asarray(mask, dtype=bool)
        if mask.shape != (self.num_envs,): raise ValueError("reset mask must be (num_envs,)")
        self.data = masked_row_reset(self.data, mask, self._reset_fn)
        self.indices = self.jp.where(mask, 0, self.indices)
        self.previous_command = self.jp.where(mask[:, None], self.jp.asarray(self._reset_q)[None], self.previous_command)
        self.pending_reset = self.jp.where(mask, False, self.pending_reset)
        return self._observe()

    def step(self, actions: Any) -> tuple[Any, Any, Any, dict[str, Any]]:
        import jax.numpy as jp
        # Apply pending resets unconditionally with a row mask. This preserves
        # reset independence and avoids a host bool(any(done)) hot-path branch.
        self.reset(self.pending_reset)
        actions = jp.asarray(actions)
        if actions.shape != (self.num_envs, 28): raise ValueError("batched actions must be (num_envs, 28)")
        physical_before = self.last_physical
        command = rate_limited_command_batch(self.previous_command, actions, jp.asarray(self.lower), jp.asarray(self.upper), jp.asarray(self.rate), physical_before.mano_dof_pos, jp.asarray(self.envelope))
        self.data = self.data.replace(ctrl=command)
        for _ in range(4): self.data = self.step_fn(self.data)
        self.previous_command = command
        self.indices = self.indices + 1
        self._observe()
        world = jp.arange(self.num_envs); index = jp.minimum(self.indices, self.length - 1)
        target = self.reference_obj[world, index]
        force = self.last_contact.hand_object_forces
        relative = self.last_physical.hand_keypoint_positions[:, 0] - self.last_physical.object_position
        ref_current = self.reference_q[world, index]
        reference_relative = ref_current[:, :3] - target
        ref_next = self.reference_obj[world, self.jp.minimum(index + 1, self.length - 1)]
        target_velocity = (ref_next - target) * 120.
        reward = compute_device_autonomy_reward_v3(object_position=self.last_physical.object_position, target_object_position=target, object_velocity=self.last_physical.object_linear_velocity, hand_object_relative=relative, reference_hand_object_relative=reference_relative, witness=self.last_witness, reference_confidence=self.reference_c[world, index], hand_object_force=force, relative_contact_motion=jp.zeros_like(force), action=actions, release_active=self.indices >= self.length, measured_q=self.last_physical.mano_dof_pos, reference_q=ref_current, object_quaternion=self.last_physical.object_orientation_xyzw, reference_object_quaternion=self.reference_quat[world, index], target_velocity=target_velocity)
        done = self.indices >= self.length
        self.pending_reset = done
        valid = self.last_physical.valid & self.last_contact.valid & reward.valid
        return self.observation, reward.total.astype(jp.float32), done, {"valid": valid, "reason_code": jp.where(done, 1, 0), "command": command, "contract": AUTONOMY_V3_VERSION}


__all__ = ["ReferenceWitnessTables", "reference_witness_tables", "DeviceWitnessFeatures", "transform_fixed_witnesses", "DeviceAutonomyReward", "compute_device_autonomy_reward_v3", "rate_limited_command_batch", "BatchedAutonomyRuntime", "V3_OBSERVATION_DIM"]
