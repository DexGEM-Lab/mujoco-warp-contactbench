"""Unified fixed-topology MJX batch physics for mixed ManoRL objects.

The kernel keeps every registered collision mesh in one compiled model.  Each
world selects one object by its free-joint state; all other naturally falling
objects stay far above and outside the bounded workspace.  This avoids
per-object MJX/Warp routes without changing the active object's mesh, mass,
collision masks, or contact solver semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sim.manorl.assets import compile_unified_model, object_runtime
from sim.manorl.contracts import JOINT_DOF, ServoConfig, normalize_hand_side
from sim.manorl.environment import (
    UnifiedMjxWarpPhysicalProducer,
    minimum_warp_contact_capacity,
)


INACTIVE_OBJECT_ORIGIN = 1000.0


@dataclass(frozen=True)
class UnifiedBatchConfig:
    """Static settings for one unified physics batch."""

    object_types: tuple[str, ...]
    active_object_indices: tuple[int, ...]
    device: str = "cpu"
    impl: str = "warp"
    contact_capacity: int = 128
    constraint_capacity: int = 512
    hand_side: str = "right"

    def __post_init__(self) -> None:
        if not self.object_types:
            raise ValueError("unified batch requires at least one object type")
        if len(set(self.object_types)) != len(self.object_types):
            raise ValueError("unified batch object types must be unique")
        if not self.active_object_indices:
            raise ValueError("unified batch requires at least one world")
        if any(index < 0 or index >= len(self.object_types) for index in self.active_object_indices):
            raise ValueError("active object index is outside the unified object set")
        if self.device not in {"cpu", "gpu"}:
            raise ValueError("unified batch device must be cpu or gpu")
        if self.impl != "warp":
            raise ValueError("unified batch requires the pinned MJX-Warp implementation")
        if self.contact_capacity < 2 or self.constraint_capacity < 2:
            raise ValueError("unified batch capacities must be at least two")
        object.__setattr__(
            self,
            "hand_side",
            normalize_hand_side(self.hand_side, allow_auto=False),
        )


class UnifiedBatchPhysics:
    """One model and one batched MJX data state for mixed object worlds."""

    def __init__(
        self,
        config: UnifiedBatchConfig,
        *,
        servo: ServoConfig = ServoConfig(),
    ) -> None:
        try:
            import jax
            from mujoco import mjx
        except ImportError as exc:
            raise RuntimeError("jax and mujoco-mjx are required for unified MJX physics") from exc
        platform = "cuda" if config.device == "gpu" else "cpu"
        devices = jax.devices(platform)
        if not devices:
            raise RuntimeError(f"no JAX {platform} device is available")
        self.jax = jax
        self.jp = jax.numpy
        self.mjx = mjx
        self.device = devices[0]
        self.config = config
        self.object_types = config.object_types
        self.active_object_indices = np.asarray(config.active_object_indices, dtype=np.int64)
        self.num_envs = len(self.active_object_indices)
        self.mujoco, self.model = compile_unified_model(
            servo,
            object_types=self.object_types,
            hand_side=self.config.hand_side,
        )
        self.action_dim = int(self.model.nu)
        if self.action_dim not in (JOINT_DOF, 2 * JOINT_DOF):
            raise ValueError(
                f"compiled unified model has unsupported actuator width {self.action_dim}"
            )
        self.hand_side = self.config.hand_side
        model_sides = (
            ("right",)
            if self.hand_side == "right"
            else (("left",) if self.hand_side == "left" else ("right", "left"))
        )
        minimum_contact_capacity = minimum_warp_contact_capacity(
            self.num_envs, model_sides
        )
        if self.config.contact_capacity < minimum_contact_capacity:
            raise ValueError(
                f"contact_capacity {self.config.contact_capacity} is below "
                f"{minimum_contact_capacity} required for {self.num_envs} "
                f"unified worlds with {len(model_sides)} hand(s)"
            )
        self.producer = UnifiedMjxWarpPhysicalProducer(
            self.mujoco,
            self.model,
            object_types=self.object_types,
            hand_sides=model_sides,
            primary_hand_side="right" if "right" in model_sides else "left",
        )
        self.producer.set_active_objects(self.active_object_indices)
        self.mjx_model = mjx.put_model(
            self.model,
            device=self.device,
            impl=config.impl,
        )
        if str(self.mjx_model.impl).lower().split(".")[-1] != config.impl:
            raise RuntimeError(
                f"MJX selected {self.mjx_model.impl}, expected implementation {config.impl}"
            )
        self._step_fn = jax.jit(
            jax.vmap(lambda world: mjx.step(self.mjx_model, world))
        )
        self._forward_fn = jax.jit(
            jax.vmap(lambda world: mjx.forward(self.mjx_model, world))
        )
        self._qpos_addresses = np.asarray(
            [
                self.model.jnt_qposadr[
                    self.mujoco.mj_name2id(
                        self.model,
                        self.mujoco.mjtObj.mjOBJ_JOINT,
                        object_runtime(name).free_joint_name,
                    )
                ]
                for name in self.object_types
            ],
            dtype=np.int64,
        )
        self._qvel_addresses = self.producer.object_qvel_addresses_by_type.copy()
        self._reset_qpos = self._build_reset_qpos()
        self.data = self._put_batch_data()

    def _build_reset_qpos(self) -> NDArray[np.float64]:
        qpos = np.zeros((self.num_envs, self.model.nq), dtype=np.float64)
        for object_index, address in enumerate(self._qpos_addresses):
            qpos[:, address + 3] = 1.0
            qpos[:, address : address + 3] = (
                INACTIVE_OBJECT_ORIGIN + 10.0 * object_index,
                0.0,
                INACTIVE_OBJECT_ORIGIN,
            )
        active_object_addresses = self._qpos_addresses[self.active_object_indices]
        for world_id, address in enumerate(active_object_addresses):
            qpos[world_id, address : address + 3] = 0.0
            qpos[world_id, address + 3 : address + 7] = (1.0, 0.0, 0.0, 0.0)
        return qpos

    def _host_data(self, world_id: int) -> Any:
        data = self.mujoco.MjData(self.model)
        self.mujoco.mj_resetData(self.model, data)
        data.qpos[:] = self._reset_qpos[world_id]
        data.qvel[:] = 0.0
        data.ctrl[:] = 0.0
        data.qfrc_applied[:] = 0.0
        self.mujoco.mj_forward(self.model, data)
        return data

    def _put_batch_data(self) -> Any:
        # Warp's private contact buffers carry custom batching metadata.  A
        # plain tree_map(stack) loses that metadata and makes the FFI see a
        # second batch axis; replicate the repository's proven vmap pattern.
        single_data = self.mjx.put_data(
            self.model,
            self._host_data(0),
            device=self.device,
            impl=self.config.impl,
            naconmax=self.config.contact_capacity,
            njmax=self.config.constraint_capacity,
        )
        batch_index = self.jax.device_put(self.jp.arange(self.num_envs), self.device)
        data = self.jax.vmap(lambda _: single_data)(batch_index)
        return data.replace(
            qpos=self.jax.device_put(self.jp.asarray(self._reset_qpos), self.device),
            qvel=self.jax.device_put(self.jp.zeros((self.num_envs, self.model.nv)), self.device),
            ctrl=self.jax.device_put(
                self.jp.zeros((self.num_envs, self.action_dim)), self.device
            ),
            qfrc_applied=self.jax.device_put(
                self.jp.zeros((self.num_envs, self.model.nv)), self.device
            ),
        )

    def reset(self) -> None:
        self.data = self._put_batch_data()

    def step(self, controls: NDArray[np.floating[Any]]) -> Any:
        values = np.asarray(controls, dtype=np.float64)
        if values.shape != (self.num_envs, self.action_dim) or not np.all(np.isfinite(values)):
            raise ValueError(
                f"controls must be finite with shape ({self.num_envs}, {self.action_dim})"
            )
        self.data = self.data.replace(
            ctrl=self.jax.device_put(self.jp.asarray(values), self.device),
            qfrc_applied=self.jax.device_put(
                self.jp.zeros((self.num_envs, self.model.nv)), self.device
            ),
        )
        self.data = self._step_fn(self.data)
        return self.data

    def forward(self) -> Any:
        self.data = self._forward_fn(self.data)
        return self.data

    def physical(self):
        """Materialize source-order physical fields for the active objects."""

        return self.producer.extract(self.data)

    def host_data_batch(self) -> list[Any]:
        values = self.mjx.get_data(self.model, self.data)
        if not isinstance(values, list) or len(values) != self.num_envs:
            raise RuntimeError("unified MJX batch did not produce one host state per world")
        return values
