"""MJX-Warp-only cube2 autonomy v4 runtime.

The contact arena belongs to the complete Warp batch.  Only qpos/qvel/ctrl are
per-world reset state; derived Warp contact/constraint buffers are regenerated
by a post-reset forward and are never row-masked.
"""
from __future__ import annotations
from typing import Any, NamedTuple
import numpy as np
from sim.manorl.autonomy_contracts import ACTION_DIM, AUTONOMY_VERSION, OBSERVATION_DIM
from sim.manorl.autonomy_v4 import (
    ReferenceCacheV4, ReferenceBankV4, V4Contact, compile_reference_cache_v4, extract_v4_physical,
    reduce_pyramidal_contacts_v4, build_raw_observation, compute_reward,
    DOF_RATE, ANTIWINDUP_ERROR,
)

V4_OBSERVATION_DIM = OBSERVATION_DIM
# Deliberate import-only migration alias. It is raw v4, never the legacy 538 ABI.
V3_OBSERVATION_DIM = V4_OBSERVATION_DIM


class AutonomyTransitionState(NamedTuple):
    data: Any
    indices: Any
    pending_reset: Any
    previous_command: Any


class BatchedAutonomyRuntime:
    """Four Warp substeps, one post-integration forward, raw 957 observations.

    Structural B>1 support is intentional.  Dynamic validation remains N=1:
    no caller should infer multi-world physics throughput from this class.
    """
    @staticmethod
    def _capacity_contract(
        num_envs: int, ccd_contacts_per_world: int | None, recommended_capacity: int,
    ) -> tuple[int, int | None]:
        ccd_capacity = (
            None if ccd_contacts_per_world is None else ccd_contacts_per_world * num_envs
        )
        return (
            max(recommended_capacity, 0 if ccd_capacity is None else ccd_capacity),
            ccd_capacity,
        )

    def __init__(
        self, trajectory, *, num_envs: int = 1, device: str = "cpu", seed: int = 0,
        persistent_ccd_workspace: bool = False, ccd_contacts_per_world: int | None = None,
        full_horizon_diagnostic: bool = False, **_: Any,
    ):
        if not isinstance(num_envs, int) or isinstance(num_envs, bool) or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        if device not in {"cpu", "gpu"}:
            raise ValueError("device must be cpu or gpu")
        if not isinstance(persistent_ccd_workspace, bool):
            raise TypeError("persistent_ccd_workspace must be bool")
        if persistent_ccd_workspace and device != "gpu":
            raise ValueError("persistent_ccd_workspace requires device='gpu'")
        if persistent_ccd_workspace and ccd_contacts_per_world is None:
            raise ValueError("persistent_ccd_workspace requires ccd_contacts_per_world")
        if ccd_contacts_per_world is not None and (
            not isinstance(ccd_contacts_per_world, int)
            or isinstance(ccd_contacts_per_world, bool)
            or ccd_contacts_per_world < 1
        ):
            raise ValueError("ccd_contacts_per_world must be a positive integer")
        if not isinstance(full_horizon_diagnostic, bool):
            raise TypeError("full_horizon_diagnostic must be bool")

        import jax
        import jax.numpy as j
        from mujoco import mjx
        from sim.manorl.assets import compile_model_metadata_only, object_collision_vertices
        from sim.manorl.environment import (
            MjxWarpPhysicalProducer, _build_masked_reset_data_fn,
            recommended_warp_contact_capacity,
        )

        self.jax, self.jp, self.mjx = jax, j, mjx
        self.num_envs, self.device_name, self.seed = num_envs, device, int(seed)
        trajectories = tuple(trajectory) if isinstance(trajectory, (list, tuple)) else None
        if trajectories is not None and not trajectories:
            raise ValueError("reference trajectories must not be empty")
        if trajectories is not None and any(t.identity.identity.split("_")[:2] != ["cube2", "02"] for t in trajectories):
            raise ValueError("reference bank runtime requires cube2:02 trajectories")
        self.trajectories = trajectories
        trajectory = trajectories[0] if trajectories is not None else trajectory
        self.trajectory, self.full_horizon_diagnostic = trajectory, full_horizon_diagnostic
        self.mujoco, self.model = compile_model_metadata_only(
            object_type="cube2", hand_side="right", physics_timestep=1 / 480
        )
        self.device = jax.devices(device)[0]
        self.mjx_model = mjx.put_model(self.model, device=self.device, impl="warp")
        if getattr(self.mjx_model, "_impl", None) is None:
            raise RuntimeError("v4 requires mjx.put_model(..., impl='warp')")
        self.producer = MjxWarpPhysicalProducer(
            self.mujoco, self.model, object_type="cube2", hand_sides=("right",)
        )
        if int(self.model.opt.cone) != int(self.mujoco.mjtCone.mjCONE_PYRAMIDAL):
            raise RuntimeError("v4 supports only pyramidal contact cone")
        self.cache = compile_reference_cache_v4(trajectory, device=device)
        first_cache = self.cache
        self.env_ref = None
        if trajectories is not None:
            self.cache = ReferenceBankV4([first_cache] + [compile_reference_cache_v4(t, device=device) for t in trajectories[1:]], device=self.device)
            self.env_ref = j.arange(num_envs, dtype=j.int32) % len(trajectories)
        self.length = len(first_cache.q_feasible) if trajectories is None else self.cache.max_length
        self.lengths = j.full((num_envs,), self.length, j.int32) if self.env_ref is None else self.cache.lengths[self.env_ref]
        self.lower = np.asarray(self.model.jnt_range[:28, 0], np.float32)
        self.upper = np.asarray(self.model.jnt_range[:28, 1], np.float32)
        self.rate, self.envelope, self.antiwindup = DOF_RATE.copy(), ANTIWINDUP_ERROR.copy(), ANTIWINDUP_ERROR.copy()
        self.clock_metadata = {
            "control_timestep": self.cache.control_timestep,
            "physics_timestep": 1 / 480,
            "physics_substeps": 4,
            "full_horizon_diagnostic": full_horizon_diagnostic,
        }
        self.table_metadata = {"height": self.cache.table_height, "object_type": "cube2"}

        # naconmax is one global contact arena, not a per-world capacity.
        # Explicit CCD scratch shares the same global allocation contract:
        # MJX-Warp requires naccdmax <= naconmax.
        self.warp_contact_capacity, self.warp_ccd_naccdmax = self._capacity_contract(
            num_envs,
            ccd_contacts_per_world,
            recommended_warp_contact_capacity(num_envs, ("right",)),
        )
        self.warp_constraint_capacity = 512
        make_kwargs = dict(
            device=self.device, impl="warp", naconmax=self.warp_contact_capacity,
            njmax=self.warp_constraint_capacity,
        )
        if self.warp_ccd_naccdmax is not None:
            make_kwargs["naccdmax"] = self.warp_ccd_naccdmax
        base = mjx.make_data(self.model, **make_kwargs)
        qpos = np.asarray(base.qpos).copy()
        qpos[:28] = first_cache.q_feasible[0]
        qpos[self.producer.object_qpos_address:self.producer.object_qpos_address + 3] = first_cache.object_origin[0]
        qpos[self.producer.object_qpos_address + 3:self.producer.object_qpos_address + 7] = first_cache.object_quat_xyzw[0, (3, 0, 1, 2)]
        ctrl = np.asarray(base.ctrl).copy()
        ctrl[:28] = first_cache.q_feasible[0]
        self.initial = base.replace(qpos=j.asarray(qpos), qvel=j.zeros_like(base.qvel), ctrl=j.asarray(ctrl))
        self._forward_batch = jax.jit(jax.vmap(lambda row: mjx.forward(self.mjx_model, row)))
        self._step_batch = jax.jit(jax.vmap(lambda row: mjx.step(self.mjx_model, row)))
        self._reset_qpos = j.broadcast_to(self.initial.qpos, (num_envs, self.initial.qpos.shape[0]))
        self._reset_ctrl = j.broadcast_to(self.initial.ctrl, (num_envs, self.initial.ctrl.shape[0]))
        if self.env_ref is not None:
            self._reset_qpos = self._reset_qpos.at[:, :28].set(self.cache.q_feasible[self.env_ref, 0])
            address = self.producer.object_qpos_address
            self._reset_qpos = self._reset_qpos.at[:, address:address+3].set(self.cache.object_origin[self.env_ref, 0])
            self._reset_qpos = self._reset_qpos.at[:, address+3:address+7].set(self.cache.object_quat_xyzw[self.env_ref, 0][:, (3,0,1,2)])
            self._reset_ctrl = self._reset_ctrl.at[:, :28].set(self.cache.q_feasible[self.env_ref, 0])
        self._masked_reset_data_fn = _build_masked_reset_data_fn(
            jax=jax, jp=j, reset_qpos=self._reset_qpos, reset_ctrl=self._reset_ctrl
        )
        self._reset_and_forward = jax.jit(
            lambda data, mask: jax.lax.cond(
                j.any(mask),
                lambda value: self._forward_batch(self._masked_reset_data_fn(value, mask)),
                lambda value: value,
                data,
            )
        )
        self.data = self._forward_batch(jax.vmap(lambda _: self.initial)(j.arange(num_envs)).replace(qpos=self._reset_qpos, ctrl=self._reset_ctrl))
        self.persistent_ccd_workspace = None
        self.persistent_solver_workspace = None
        if persistent_ccd_workspace:
            from sim.manorl.mjx_warp_workspace import (
                install_persistent_ccd_workspace, install_persistent_solver_workspace,
                warp_device_ordinal,
            )
            impl = self.mjx_model._impl
            self.persistent_ccd_workspace = install_persistent_ccd_workspace(
                device_ordinal=warp_device_ordinal(self.device),
                naccdmax=self.warp_ccd_naccdmax,
                epa_iterations=int(self.model.opt.ccd_iterations),
                nmaxpolygon=int(impl.nmaxpolygon), nmaxmeshdeg=int(impl.nmaxmeshdeg),
            )
            self.persistent_solver_workspace = install_persistent_solver_workspace(
                device_ordinal=warp_device_ordinal(self.device), nworld=num_envs,
                nv=int(self.model.nv), nv_pad=int(impl.nv_pad),
                njmax=int(self.data._impl.njmax), solver_type=int(self.model.opt.solver),
            )
        self.indices = j.zeros((num_envs,), j.int32)
        self.pending_reset = j.zeros((num_envs,), bool)
        self.previous_command = self._reset_ctrl[:, :28]
        self.object_vertices = j.asarray(object_collision_vertices("cube2"), j.float32)
        self._transition_fn = jax.jit(self._transition)
        self._refresh(False, j.zeros((num_envs, 28), j.float32))

    def _physical(self, data, previous):
        from sim.manorl.autonomy_v4 import quat_rotate
        objq = data.xquat[:, self.producer.object_body_id][..., (1, 2, 3, 0)]
        bottom = self.jp.min(
            data.xpos[:, self.producer.object_body_id, None, 2]
            + quat_rotate(objq[:, None], self.object_vertices)[:, :, 2], axis=1
        )
        return extract_v4_physical(
            qpos=data.qpos, qvel=data.qvel, xpos=data.xpos, xquat=data.xquat,
            xipos=data.xipos, subtree_com=data.subtree_com, cvel=data.cvel,
            body_rootid=self.model.body_rootid, hand_dof_address=0,
            object_body_id=self.producer.object_body_id,
            palm_body_id=self.producer.keypoint_body_ids[0],
            region_body_ids=self.producer.keypoint_body_ids, object_bottom=bottom,
            lower=self.jp.asarray(self.lower), upper=self.jp.asarray(self.upper),
            previous_command=previous, dof_rate=self.jp.asarray(self.rate),
            antiwindup=self.jp.asarray(self.antiwindup),
        )

    def _contact(self, data, physical):
        x = data._impl
        required = (
            "nacon", "nefc", "contact__geom", "contact__worldid", "contact__dim",
            "contact__efc_address", "contact__friction", "contact__frame",
            "contact__pos", "efc__force",
        )
        if any(not hasattr(x, name) for name in required):
            raise RuntimeError("pinned MJX-Warp contact ABI unavailable")
        allf, pair, torque, count, slip, valid = reduce_pyramidal_contacts_v4(
            nacon=x.nacon, nefc=x.nefc, geom=x.contact__geom, world=x.contact__worldid,
            dimension=x.contact__dim, addresses=x.contact__efc_address,
            friction=x.contact__friction, frame=x.contact__frame, position=x.contact__pos,
            constraint_force=x.efc__force, ngeom=self.model.ngeom,
            hand_geom_ids=self.producer.keypoint_geom_ids,
            object_geom_ids=tuple(self.producer.object_geom_ids), object_com=physical.object_com,
            hand_com=physical.region_com, hand_v_com=physical.region_v_com,
            hand_w=physical.region_w, object_v_com=physical.object_v_com,
            object_w=physical.object_w,
        )
        return V4Contact(allf, pair, torque, count, slip, valid)

    def _observe(self, data, index, previous):
        physical = self._physical(data, previous)
        contact = self._contact(data, physical)
        raw = build_raw_observation(physical, contact, self.cache, index, previous, self.env_ref)
        return physical, contact, raw, raw

    def _transition(self, data, index, previous, action, execute):
        import jax
        physical = self._physical(data, previous)

        def command_for_action(_):
            delta = self.jp.clip(action, -1, 1) * self.jp.asarray(self.rate)[None] * self.cache.control_timestep
            error = previous - physical.q_raw
            # Do not re-anchor a loaded servo target to measured q.
            delta = self.jp.where(
                (self.jp.abs(error) >= self.jp.asarray(self.envelope)) & (error * delta > 0),
                0., delta,
            )
            return self.jp.clip(previous + delta, self.jp.asarray(self.lower), self.jp.asarray(self.upper))

        command = jax.lax.cond(execute, command_for_action, lambda _: previous, operand=None)
        stepped = data.replace(ctrl=command)

        def advance(value):
            for _ in range(4):
                value = self._step_batch(value)
            return self._forward_batch(value)

        next_data = jax.lax.cond(execute, advance, lambda value: value, stepped)
        next_index = index + self.jp.asarray(execute, self.jp.int32)
        physical_next, contact, raw, observation = self._observe(next_data, next_index, command)
        reward = compute_reward(physical_next, contact, self.cache, next_index, self.jp.clip(action, -1, 1), self.env_ref)
        valid = physical_next.valid & contact.valid & reward.valid
        return next_data, next_index, command, raw, observation, reward, valid, contact, physical_next

    def _refresh(self, execute, action):
        (
            self.data, self.indices, self.previous_command, self.raw_observation,
            self.observation, self.last_reward, self.last_valid, self.last_contact,
            self.last_physical,
        ) = self._transition_fn(self.data, self.indices, self.previous_command, action, execute)
        # Diagnostic rollouts may continue past a normal terminal, but this
        # option is runtime-only and is intentionally absent from training.
        self.last_done = self.jp.zeros_like(self.last_reward.done) if self.full_horizon_diagnostic else self.last_reward.done
        self.last_reason = self.last_reward.reason
        self.last_command = self.previous_command

    def reset(self, mask=None):
        if mask is None:
            mask = self.jp.ones((self.num_envs,), dtype=bool)
        mask = self.jp.asarray(mask, dtype=bool)
        if mask.shape != (self.num_envs,):
            raise ValueError("reset mask must be (num_envs,)")
        self.data = self._reset_and_forward(self.data, mask)
        self.indices = self.jp.where(mask, 0, self.indices)
        initial_command = self._reset_ctrl[:, :28]
        self.previous_command = self.jp.where(mask[:, None], initial_command, self.previous_command)
        # Every raw cache row is rebuilt from one forwarded state; no stale
        # global contact arena is exposed after a subset reset.
        self._refresh(False, self.jp.zeros((self.num_envs, 28), self.jp.float32))
        return self.observation

    def prepare_action(self):
        """Return terminal state first; reset exactly completed worlds afterward."""
        return self.reset(self.last_done)

    def step(self, actions):
        action = self.jp.asarray(actions)
        if action.shape != (self.num_envs, ACTION_DIM):
            raise ValueError("v4 actions must be (num_envs,28)")
        self._refresh(True, action)
        return self.observation, self.last_reward.total, self.last_done, {
            "valid": self.last_valid, "reason_code": self.last_reason,
            "command": self.last_command, "raw_observation": self.raw_observation,
            "contact": self.last_contact, "last_physical": self.last_physical,
            "clock": self.clock_metadata, "cache": self.cache.content_hash,
            "table": self.table_metadata, "contact_capacity": self.warp_contact_capacity,
            "constraint_capacity_per_world": self.warp_constraint_capacity,
            "contract": AUTONOMY_VERSION,
        }


__all__ = [
    "ReferenceCacheV4", "compile_reference_cache_v4", "V4Contact",
    "V4_OBSERVATION_DIM", "V3_OBSERVATION_DIM", "AutonomyTransitionState",
    "BatchedAutonomyRuntime",
]
