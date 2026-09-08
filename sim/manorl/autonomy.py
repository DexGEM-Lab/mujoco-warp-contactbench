"""Standalone autonomous cube2 MJX-Warp control path.

The actor emits bounded *rate* commands for all 28 actuators.  Demonstrations
condition the observation and reward only; they never enter the command map.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sim.manorl.assets import compile_model, object_collision_vertices, object_runtime
from sim.manorl.autonomy_contracts import (
    ACTION_DIM, OBSERVATION_DIM, ACTION_CONTRACT, OBSERVATION_CONTRACT,
    REWARD_CONTRACT, observation_slices, rate_limited_command,
)
from sim.manorl.contracts import JOINT_DOF, KEYPOINT_NAMES
from sim.manorl.trajectory import ReferenceTrajectory

PACKAGE_DEFAULT = Path("outputs/manorl/contact_conditioned_autonomy/cube2_02_v295_f120_pre180_post180")


def _quat_conjugate_rotate(q: NDArray[np.float64], v: NDArray[np.float64]) -> NDArray[np.float64]:
    qv = q[:3]
    return v * (2*q[3]*q[3]-1) + np.cross(qv, v)*q[3]*2 + qv*np.dot(qv, v)*2


def source_surface_points(object_type: str = "cube2", count: int = 128, seed: int = 0) -> NDArray[np.float64]:
    """Sample the pinned collision mesh surface, preserving physical units."""
    if count < 1:
        raise ValueError("surface sample count must be positive")
    triangles = object_collision_vertices(object_type).reshape(-1, 3, 3)
    edge_a, edge_b = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    areas = np.linalg.norm(np.cross(edge_a, edge_b), axis=1) * .5
    if np.any(areas <= 0) or not np.all(np.isfinite(areas)):
        raise ValueError("collision mesh contains degenerate triangles")
    rng = np.random.default_rng(seed)
    ids = np.searchsorted(np.cumsum(areas), rng.random(count) * areas.sum())
    base = triangles[ids, 0]
    u, v = rng.random((count, 1)), rng.random((count, 1))
    over = u + v > 1
    u[over], v[over] = 1-u[over], 1-v[over]
    return base + u * (triangles[ids, 1] - base) + v * (triangles[ids, 2] - base)


def surface_intent(
    keypoints_world: NDArray[np.floating], object_position: NDArray[np.floating],
    object_quat_xyzw: NDArray[np.floating], surface_local: NDArray[np.floating],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return proximity, local nearest-surface anchors, and confidence.

    The anchor is nearest-point geometry, not a force label. Confidence falls
    with distance and is therefore useful for distinguishing inferred intent
    from measured contact.
    """
    points = np.asarray(keypoints_world, dtype=np.float64)
    obj = np.asarray(object_position, dtype=np.float64)
    quat = np.asarray(object_quat_xyzw, dtype=np.float64)
    surface = np.asarray(surface_local, dtype=np.float64)
    if points.shape != (16, 3) or obj.shape != (3,) or quat.shape != (4,) or surface.ndim != 2 or surface.shape[1] != 3:
        raise ValueError("surface intent shapes must be (16,3), (3,), (4,), (N,3)")
    world_surface = np.stack([_quat_conjugate_rotate(quat, p) for p in surface], axis=0) + obj
    distances = np.linalg.norm(points[:, None, :] - world_surface[None, :, :], axis=-1)
    nearest = np.argmin(distances, axis=1)
    min_distance = distances[np.arange(16), nearest]
    anchors = np.stack([surface[index] for index in nearest], axis=0)
    proximity = np.exp(-min_distance / 0.01)
    confidence = np.clip(proximity, 0.0, 1.0)
    return proximity, anchors, confidence


def shared_support_alignment(
    hand_position: NDArray[np.floating], object_position: NDArray[np.floating],
    surface_local: NDArray[np.floating], table_height: float = -0.001,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Apply one common translation to hand/object references for table support."""
    hand = np.asarray(hand_position, dtype=np.float64)
    obj = np.asarray(object_position, dtype=np.float64)
    points = np.asarray(surface_local, dtype=np.float64)
    if hand.shape != (3,) or obj.shape != (3,) or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("alignment expects hand/object vectors and (N,3) surface")
    quat_identity = np.array([0., 0., 0., 1.])
    min_z = float((points[:, 2] + obj[2]).min())
    shift = np.array([0., 0., float(table_height) - min_z])
    return hand + shift, obj + shift, shift


def dense_autonomous_reward(
    object_position: NDArray[np.floating], target_object_position: NDArray[np.floating],
    object_velocity: NDArray[np.floating], hand_object_relative: NDArray[np.floating],
    reference_hand_object_relative: NDArray[np.floating], proximity: NDArray[np.floating],
    previous_proximity: NDArray[np.floating], action: NDArray[np.floating],
    *, phase: float, released: bool = False,
) -> dict[str, float]:
    """Dense additive physical objective, valid before object movement."""
    obj = np.asarray(object_position, dtype=float); target = np.asarray(target_object_position, dtype=float)
    vel = np.asarray(object_velocity, dtype=float); rel = np.asarray(hand_object_relative, dtype=float)
    ref_rel = np.asarray(reference_hand_object_relative, dtype=float)
    prox = np.asarray(proximity, dtype=float); prev = np.asarray(previous_proximity, dtype=float)
    if any(x.shape != (3,) for x in (obj, target, vel, rel, ref_rel)) or prox.shape != (16,) or prev.shape != (16,):
        raise ValueError("reward vectors have invalid shape")
    terms = {
        "object_motion": float(np.exp(-20.0 * np.linalg.norm(obj-target))),
        "reference_hand_object_relationship": float(np.exp(-30.0 * np.linalg.norm(rel-ref_rel))),
        "surface_proximity_contact": float(np.mean(prox) + .25*np.mean(prox-prev)),
        "stability": float(np.exp(-8.0 * np.linalg.norm(vel))),
        "release": float((1.0 - np.mean(prox)) if released else np.mean(prox)),
        "action_smoothness": float(-0.001 * np.mean(np.square(np.asarray(action, dtype=float)))),
        "phase": float(.05 * phase),
    }
    return terms

@dataclass
class AutonomousStep:
    observation: NDArray[np.float32]
    reward: float
    terms: dict[str, float]
    done: bool
    info: dict[str, Any]


class Cube2AutonomousMJX:
    """One-world real MJX-Warp autonomous environment."""
    def __init__(self, trajectory: ReferenceTrajectory, *, device: str = "cpu", seed: int = 0, near_contact: bool = False):
        if trajectory.dof_dim != JOINT_DOF or trajectory.identity.identity.split("_")[0] != "cube2":
            raise ValueError("autonomous milestone requires a 28-DoF cube2 trajectory")
        if device not in {"cpu", "gpu"}:
            raise ValueError("device must be cpu or gpu")
        import jax
        from mujoco import mjx
        self.trajectory, self.device_name, self.near_contact, self.seed = trajectory, device, near_contact, int(seed)
        self.mujoco, self.model = compile_model(object_type="cube2", hand_side="right")
        self.mjx, self.jax = mjx, jax
        devices = jax.devices(device)
        if not devices: raise RuntimeError(f"no JAX {device} device available")
        self.device = devices[0]
        self.mjx_model = mjx.put_model(self.model, device=self.device, impl="warp")
        if str(self.mjx_model.impl).lower().split(".")[-1] != "warp": raise RuntimeError("MJX-Warp implementation was not selected")
        self.step_fn = jax.jit(mjx.step)
        self.surface = source_surface_points("cube2", 128, seed)
        self.keypoint_body_ids = []
        for name in KEYPOINT_NAMES:
            body = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, name)
            if body < 0: raise ValueError(f"missing keypoint body {name}")
            self.keypoint_body_ids.append(body)
        object_joint = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, object_runtime("cube2").free_joint_name)
        self.object_qpos = int(self.model.jnt_qposadr[object_joint]); self.object_qvel = int(self.model.jnt_dofadr[object_joint])
        self.lower, self.upper = self.model.jnt_range[:JOINT_DOF,0].copy(), self.model.jnt_range[:JOINT_DOF,1].copy()
        self.rate = np.maximum((self.upper-self.lower) * .025, 1e-4)
        self.reset()

    def reset(self) -> NDArray[np.float32]:
        d = self.mujoco.MjData(self.model); self.mujoco.mj_resetData(self.model, d)
        q = np.asarray(self.trajectory.q_ref[0], dtype=float).copy(); obj = np.asarray(self.trajectory.object_pos[0], dtype=float).copy()
        if self.near_contact:
            q[:3] = q[:3] + (obj - q[:3]) * .25
        d.qpos[:JOINT_DOF], d.qpos[self.object_qpos:self.object_qpos+3] = q, obj
        d.qpos[self.object_qpos+3:self.object_qpos+7] = self.trajectory.object_quat_xyzw[0][[3,0,1,2]]
        d.qvel[:] = 0; d.ctrl[:JOINT_DOF] = q; self.mujoco.mj_forward(self.model, d)
        self.data = self.mjx.put_data(self.model, d, device=self.device, impl="warp", naconmax=256, njmax=512)
        self.previous_command, self.step_index, self.previous_proximity = q.copy(), 0, np.zeros(16)
        return self._observation()

    def _host(self):
        d = self.mjx.get_data(self.model, self.data)
        return d[0] if isinstance(d, list) else d

    def _state(self):
        d = self._host(); q = np.asarray(d.qpos[:JOINT_DOF]); qvel = np.asarray(d.qvel[:JOINT_DOF])
        obj = np.asarray(d.qpos[self.object_qpos:self.object_qpos+3]); quat = np.asarray(d.qpos[self.object_qpos+3:self.object_qpos+7])[[1,2,3,0]]
        keypoints = np.asarray(d.xpos[self.keypoint_body_ids]); return d, q, qvel, obj, quat, keypoints

    def _observation(self) -> NDArray[np.float32]:
        _, q, qvel, obj, quat, keypoints = self._state(); i = min(self.step_index, len(self.trajectory.q_ref)-1); j = min(i+1, len(self.trajectory.q_ref)-1); k = min(i+5, len(self.trajectory.q_ref)-1)
        aligned_hand, aligned_obj, _ = shared_support_alignment(self.trajectory.q_ref[i,:3], self.trajectory.object_pos[i], self.surface)
        proximity, anchors, confidence = surface_intent(keypoints, obj, quat, self.surface)
        s = observation_slices(); out = np.zeros(OBSERVATION_DIM, dtype=np.float64)
        out[s["measured_qpos_normalized"]] = 2*(q-self.lower)/(self.upper-self.lower)-1; out[s["measured_qvel"]] = np.clip(qvel, -20, 20)
        out[s["object_position"]], out[s["object_linear_velocity"]] = obj, np.asarray(self._host().qvel[self.object_qvel:self.object_qvel+3])
        out[s["hand_object_relative"]] = keypoints[0]-obj; out[s["reference_q_current"]] = self.trajectory.q_ref[i]; out[s["reference_q_next"]] = self.trajectory.q_ref[j]; out[s["reference_q_velocity"]] = (self.trajectory.q_ref[j]-self.trajectory.q_ref[i]) * 120
        out[s["reference_object_relative"]] = aligned_hand-aligned_obj; out[s["reference_object_future_delta"]] = self.trajectory.object_pos[k]-self.trajectory.object_pos[i]; out[s["previous_command_normalized"]] = 2*(self.previous_command-self.lower)/(self.upper-self.lower)-1
        action_id = int(self.trajectory.identity.identity.split("_")[1]); out[s["action_identity_one_hot"]][action_id-1] = 1.; out[s["object_geometry"]] = np.array([.05,.05,.05,0,0,0,0,0,0,0,0,0])
        out[s["measured_keypoint_relative"]] = (keypoints-obj).reshape(-1); out[s["surface_proximity"]] = proximity; out[s["surface_anchor_local"]] = anchors.reshape(-1); out[s["contact_phase_confidence"]] = (i/max(1,len(self.trajectory.q_ref)-1), float(np.mean(confidence)))
        return np.clip(out, -5, 5).astype(np.float32)

    def step(self, action: NDArray[np.floating]) -> AutonomousStep:
        action = np.asarray(action, dtype=float); command = rate_limited_command(self.previous_command, action, self.lower, self.upper, self.rate)
        before = self._state(); _, _, _, obj_before, _, key_before = before; prox_before, _, _ = surface_intent(key_before, obj_before, np.asarray(before[4]), self.surface)
        for _ in range(4): self.data = self.data.replace(ctrl=self.jax.numpy.asarray(command)); self.data = self.step_fn(self.mjx_model, self.data)
        self.previous_command, self.step_index = command, self.step_index + 1
        _, q, qvel, obj, quat, keypoints = self._state(); prox, _, _ = surface_intent(keypoints, obj, quat, self.surface)
        i = min(self.step_index, len(self.trajectory.q_ref)-1); target = self.trajectory.object_pos[i]; terms = dense_autonomous_reward(obj, target, np.asarray(self._host().qvel[self.object_qvel:self.object_qvel+3]), keypoints[0]-obj, self.trajectory.q_ref[i,:3]-self.trajectory.object_pos[i], prox, prox_before, action, phase=i/max(1,len(self.trajectory.q_ref)-1))
        self.previous_proximity = prox; done = self.step_index >= len(self.trajectory.q_ref)-1
        return AutonomousStep(self._observation(), float(sum(terms.values())), terms, done, {"command": command.copy(), "step": self.step_index, "identity": self.trajectory.identity.identity, "object_position": obj.copy(), "surface_proximity": prox.copy(), "qpos": q.copy(), "qvel": qvel.copy()})
