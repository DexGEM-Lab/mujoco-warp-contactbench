"""Rerun observer for target-native ManoRL environment transitions."""

from __future__ import annotations

from dataclasses import fields, replace
import json
import os
from pathlib import Path
import shutil
import warnings

import numpy as np

from sim.manorl.abi import ENVIRONMENT_CONTRACT_ID
from sim.manorl.contracts import CONTROL_TIMESTEP, JOINT_NAMES, KEYPOINT_NAMES, PHYSICS_SUBSTEPS_PER_TARGET
from sim.manorl.environment import MujocoManoEnvironment, TransitionSnapshot
from sim.manorl.observations import CONTACT_FORCE_THRESHOLD, OBSERVATION_SLICES, quat_rotate_xyzw
from sim.manorl.rewards import (
    PPO_REWARD_CONTRACT_ID,
    PPO_REWARD_SCALE,
    REWARD_CONTRACT_ID,
    REWARD_HAND_OBJECT_THRESHOLD_N,
)

_FORCE_ARROW_SCALE = 0.002
_GEOMETRY_FORCE_PATHS = (
    ("magnitude_N", "Net magnitude"),
    ("x_N", "World F_x"),
    ("y_N", "World F_y"),
    ("z_N", "World F_z"),
)
_HAND_OBJECT_FORCE_PATH = "contact/hand_object_force/on_object/magnitude_N"
_OBJECT_GRAVITY_MAGNITUDE_PATH = "contact/object/gravity/world/magnitude_N"


class ManoRerunRecorder:
    """Publish reset-completed selected-world episodes to one stable Rerun path."""

    def __init__(
        self,
        environment: MujocoManoEnvironment,
        output: str | Path,
        *,
        env_id: int = 0,
        grpc_url: str | None = None,
        archive_dir: str | Path | None = None,
        archive_threshold: float | None = None,
        archive_following: int = 5,
    ) -> None:
        if not isinstance(environment, MujocoManoEnvironment):
            raise TypeError("environment must be a MujocoManoEnvironment")
        if not 0 <= env_id < environment.config.num_envs:
            raise ValueError("env_id must be within the configured environment batch")
        try:
            import rerun as rr
        except ImportError as exc:
            raise RuntimeError("rerun-sdk is required for ManoRL Rerun recording") from exc
        self.environment = environment
        self.env_id = env_id
        self.rr = rr
        self.output = Path(output)
        self.grpc_url = grpc_url
        self.streaming = grpc_url is not None
        if archive_dir is not None and self.streaming:
            raise ValueError("high-return Rerun archives require a local .rrd recording")
        if archive_threshold is not None and not np.isfinite(archive_threshold):
            raise ValueError("archive_threshold must be finite")
        if archive_threshold is not None and archive_dir is None:
            raise ValueError("archive_threshold requires archive_dir")
        if archive_following < 0:
            raise ValueError("archive_following must be non-negative")
        self.archive_dir = None if archive_dir is None else Path(archive_dir)
        self.archive_threshold = archive_threshold
        self.archive_following = archive_following
        self.output.parent.mkdir(parents=True, exist_ok=True)
        if self.archive_dir is not None:
            self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.episode_id = 0
        # This accumulator is deliberately local to the selected env. The
        # delayed reset snapshot has already cleared environment.episode_returns,
        # so the completed return must be captured before publishing the file.
        self._episode_return = 0.0
        self._archive_remaining = 0
        self._pending_high_env_id: int | None = None
        self._replay_enabled = self.archive_dir is not None and self.archive_threshold is not None
        self._queued_replay_keys: set[tuple[int, int]] = set()
        self._queued_replays: list[tuple[int, int, float, np.ndarray, np.ndarray]] = []
        if self._replay_enabled:
            batch_size = environment.config.num_envs
            max_episode_steps = max(
                int(np.max(environment.trajectory_lengths)),
                int(environment.config.episode_length),
            ) + 2
            self._replay_action_history = np.zeros(
                (batch_size, max_episode_steps, 26), dtype=np.float64
            )
            self._replay_episode_steps = np.zeros(batch_size, dtype=np.int64)
            self._replay_episode_numbers = np.zeros(batch_size, dtype=np.int64)
            self._replay_episode_templates = environment.object_point_cloud_local().copy()
        else:
            self._replay_action_history = None
            self._replay_episode_steps = None
            self._replay_episode_numbers = None
            self._replay_episode_templates = None
        suffix = self.output.suffix or ".rrd"
        stem = self.output.stem if self.output.suffix else self.output.name
        self.active_path = self.output.with_name(f".{stem}.active{suffix}")
        self._closed = False
        self._published = False
        self._recording_open = False
        self._close_result: Path | None = None
        self.geometry_table = self._geometry_table()
        self.geometry_series_names = tuple(row["series_name"] for row in self.geometry_table)
        self.hand_object_force_series_table = self._hand_object_force_series_table()
        self.hand_object_force_series_names = tuple(
            row["series_name"] for row in self.hand_object_force_series_table
        )
        self.hand_meshes = self._hand_meshes()
        self._start_episode()

    def _default_blueprint(self):
        blueprint = self.rr.blueprint
        return blueprint.Blueprint(
            blueprint.Vertical(
                blueprint.Spatial3DView(
                    name="Object, hand, and point cloud",
                    contents=["world/**", "target/**"],
                    eye_controls=blueprint.EyeControls3D(
                        kind=blueprint.components.Eye3DKind.Orbital,
                        tracking_entity="world/object",
                        speed=0.1,
                    ),
                    line_grid=True,
                ),
                blueprint.Horizontal(
                    blueprint.TimeSeriesView(
                        name="Reward and termination",
                        contents=["reward/**", "termination/**", "reset/**"],
                    ),
                    blueprint.TimeSeriesView(
                        name="Contacts and actions",
                        contents=[
                            "contact/count",
                            "contact/object_force_magnitude",
                            "contact/keypoint_force_total",
                            "action/**",
                            "episode/**",
                        ],
                    ),
                ),
                blueprint.Tabs(
                    *(
                        blueprint.TimeSeriesView(
                            name=view_name,
                            contents=[f"contact/geometry_force/world/{component}"],
                        )
                        for component, view_name in _GEOMETRY_FORCE_PATHS
                    ),
                    blueprint.TimeSeriesView(
                        name="Object gravity",
                        contents=["contact/object/gravity/world/**"],
                    ),
                    name="Collision geometry contact forces",
                ),
                blueprint.TimeSeriesView(
                    name="ManoHand-object contact forces",
                    contents=[_HAND_OBJECT_FORCE_PATH, _OBJECT_GRAVITY_MAGNITUDE_PATH],
                    overrides={
                        _OBJECT_GRAVITY_MAGNITUDE_PATH: self.rr.SeriesLines.from_fields(names="Object gravity"),
                    },
                ),
                row_shares=[3.0, 1.0, 2.0, 1.0],
            ),
            blueprint.TimePanel(timeline="step"),
            auto_layout=False,
            auto_views=False,
            collapse_panels=False,
        )

    def _start_episode(self) -> None:
        if self._recording_open:
            raise RuntimeError("cannot start an episode while a recording stream is open")
        # A previous process may have left its private stream behind. The
        # stable output is replaced only after a complete episode is flushed,
        # so removing this stale temporary file cannot discard a published run.
        if not self.streaming:
            self.active_path.unlink(missing_ok=True)
        self.recording = self.rr.RecordingStream("manorl_mujoco")
        if self.streaming:
            self.recording.connect_grpc(self.grpc_url)
        else:
            self.recording.save(self.active_path)
        self._recording_open = True
        self.recording.send_blueprint(self._default_blueprint(), make_active=True, make_default=True)
        self._log_static_metadata()

    def _publish_episode(self, *, archive_path: Path | None = None) -> None:
        if not self._recording_open:
            raise RuntimeError("cannot publish without an open recording stream")
        self.recording.flush()
        self.recording.disconnect()
        self._recording_open = False
        if not self.streaming:
            if archive_path is not None:
                shutil.copy2(self.active_path, archive_path)
            os.replace(self.active_path, self.output)
            self._published = True

    def _high_return_archive_path(self, completed_return: float) -> Path | None:
        if self.archive_dir is None or self.archive_threshold is None:
            return None
        threshold_hit = completed_return >= self.archive_threshold
        if threshold_hit:
            self._archive_remaining = max(self._archive_remaining, self.archive_following)
        if not threshold_hit and self._archive_remaining <= 0:
            return None
        if not threshold_hit:
            self._archive_remaining -= 1
        return self.archive_dir / (
            f"env_{self.env_id:04d}_episode_{self.episode_id:06d}_return_{completed_return:.6f}.rrd"
        )

    def _discard_episode(self) -> None:
        if not self._recording_open:
            return
        self.recording.flush()
        self.recording.disconnect()
        self._recording_open = False
        if not self.streaming:
            self.active_path.unlink(missing_ok=True)

    def _switch_to_high_return_env(self, env_id: int, *, initial_return: float | None = None) -> None:
        if env_id == self.env_id:
            return
        self._discard_episode()
        self.env_id = env_id
        self.episode_id += 1
        self._episode_return = 0.0 if initial_return is None else initial_return
        self._archive_remaining = max(self._archive_remaining, self.archive_following)
        self._start_episode()

    def _capture_replay_action(self, snapshot: TransitionSnapshot) -> None:
        if not self._replay_enabled:
            return
        assert self._replay_action_history is not None
        assert self._replay_episode_steps is not None
        assert self._replay_episode_numbers is not None
        assert self._replay_episode_templates is not None
        reset_envs = np.flatnonzero(np.asarray(snapshot.reset_applied, dtype=bool))
        starting_envs = np.flatnonzero(self._replay_episode_steps == 0)
        if reset_envs.size:
            self._replay_episode_steps[reset_envs] = 0
            self._replay_episode_numbers[reset_envs] += 1
        if starting_envs.size or reset_envs.size:
            template_envs = np.unique(np.concatenate((starting_envs, reset_envs)))
            templates = self.environment.object_point_cloud_local()
            self._replay_episode_templates[template_envs] = templates[template_envs]
        steps = self._replay_episode_steps
        if np.any(steps >= self._replay_action_history.shape[1]):
            raise RuntimeError("high-return replay action history exceeded the configured episode capacity")
        env_indices = np.arange(self.environment.config.num_envs)
        self._replay_action_history[env_indices, steps] = np.asarray(snapshot.raw_actions, dtype=np.float64)
        self._replay_episode_steps += 1

    def _queue_high_return_replays(
        self, snapshot: TransitionSnapshot, high_envs: np.ndarray
    ) -> None:
        if not self._replay_enabled:
            return
        assert self._replay_action_history is not None
        assert self._replay_episode_steps is not None
        assert self._replay_episode_numbers is not None
        assert self._replay_episode_templates is not None
        for env_id in high_envs.tolist():
            env_id = int(env_id)
            episode_number = int(self._replay_episode_numbers[env_id])
            key = (env_id, episode_number)
            if key in self._queued_replay_keys:
                continue
            step_count = int(self._replay_episode_steps[env_id])
            if step_count < 1:
                continue
            self._queued_replay_keys.add(key)
            self._queued_replays.append(
                (
                    env_id,
                    episode_number,
                    float(snapshot.episode_return[env_id]),
                    self._replay_action_history[env_id, :step_count].copy(),
                    self._replay_episode_templates[env_id].copy(),
                )
            )

    def _replay_high_return_episodes(self) -> None:
        queued_replays = getattr(self, "_queued_replays", ())
        archive_dir = getattr(self, "archive_dir", None)
        if not queued_replays or archive_dir is None:
            return
        base_config = self.environment.config
        for env_id, episode_number, episode_return, actions, template in queued_replays:
            output = archive_dir / (
                f"env_{env_id:04d}_episode_{episode_number:06d}_return_{episode_return:.6f}_full.rrd"
            )
            try:
                replay_config = replace(
                    base_config,
                    num_envs=1,
                    contact_capacity=max(128, 31),
                    device_resident_controls=False,
                    capture_transition_diagnostics=True,
                )
                replay_environment = MujocoManoEnvironment(
                    self.environment.trajectories[env_id], replay_config
                )
                if replay_config.compatibility.point_template_mode == "dynamic_reset":
                    if replay_environment._dynamic_templates is None:
                        raise RuntimeError("replay environment did not initialize dynamic point templates")
                    replay_environment._dynamic_templates[0] = template
                replay_recorder = ManoRerunRecorder(replay_environment, output, env_id=0)
                try:
                    for action in actions:
                        replay_environment.step(action.reshape(1, 26))
                        replay_recorder.record_transition()
                    # The source applies delayed resets on the following step;
                    # this flushes the terminal episode into the stable .rrd.
                    replay_environment.step(np.zeros((1, 26), dtype=np.float64))
                    replay_recorder.record_transition()
                finally:
                    artifact = replay_recorder.close()
                if artifact is None:
                    raise RuntimeError("replayed high-return episode did not publish a complete Rerun artifact")
            except Exception as exc:  # pragma: no cover - GPU/JAX replay failure is runtime-specific
                warnings.warn(
                    f"failed to materialize full high-return Rerun episode env={env_id} episode={episode_number}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def _geometry_table(self) -> list[dict[str, int | str | None]]:
        environment = self.environment
        model = environment.model
        mujoco = environment.mujoco
        mesh_type = int(mujoco.mjtGeom.mjGEOM_MESH)
        rows: list[dict[str, int | str | None]] = []
        for geom_id in range(model.ngeom):
            geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom_{geom_id}"
            body_id = int(model.geom_bodyid[geom_id])
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"
            geom_type = int(model.geom_type[geom_id])
            mesh_name = None
            if geom_type == mesh_type:
                mesh_id = int(model.geom_dataid[geom_id])
                mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id) if mesh_id >= 0 else None
            rows.append(
                {
                    "id": geom_id,
                    "label": geom_name,
                    "series_name": f"{geom_id:03d}:{geom_name}",
                    "geom_type": mujoco.mjtGeom(geom_type).name,
                    "body_id": body_id,
                    "body_name": body_name,
                    "mesh_name": mesh_name,
                }
            )
        return rows

    def _hand_object_force_series_table(self) -> list[dict[str, int | str]]:
        producer = self.environment.producer
        return [
            {
                "keypoint_name": keypoint_name,
                "geom_id": geom_id,
                "geom_name": f"{keypoint_name}_collision",
                "series_name": f"{geom_id:03d}:{keypoint_name}_collision",
            }
            for keypoint_name, geom_id in zip(KEYPOINT_NAMES, producer.keypoint_geom_ids, strict=True)
        ]

    def _hand_meshes(self) -> list[dict[str, object]]:
        """Extract the URDF-resolved hand meshes in their owning body frames."""

        environment = self.environment
        model = environment.model
        mujoco = environment.mujoco
        mesh_type = int(mujoco.mjtGeom.mjGEOM_MESH)
        meshes: list[dict[str, object]] = []
        for keypoint_name, geom_id, body_id in zip(
            KEYPOINT_NAMES,
            environment.producer.keypoint_geom_ids,
            environment.producer.keypoint_body_ids,
            strict=True,
        ):
            if int(model.geom_type[geom_id]) != mesh_type:
                raise RuntimeError(f"MANO geom {geom_id} is not a mesh")
            if int(model.geom_bodyid[geom_id]) != body_id:
                raise RuntimeError(f"MANO geom {geom_id} is not attached to body {body_id}")
            mesh_id = int(model.geom_dataid[geom_id])
            if mesh_id < 0:
                raise RuntimeError(f"MANO geom {geom_id} has no mesh data")
            vertex_start = int(model.mesh_vertadr[mesh_id])
            vertex_count = int(model.mesh_vertnum[mesh_id])
            face_start = int(model.mesh_faceadr[mesh_id])
            face_count = int(model.mesh_facenum[mesh_id])
            vertices = np.asarray(
                model.mesh_vert[vertex_start : vertex_start + vertex_count], dtype=np.float64
            ).copy()
            faces = np.asarray(model.mesh_face[face_start : face_start + face_count], dtype=np.int32).copy()
            geom_position = np.asarray(model.geom_pos[geom_id], dtype=np.float64)
            geom_quaternion_wxyz = np.asarray(model.geom_quat[geom_id], dtype=np.float64)
            geom_quaternion_xyzw = geom_quaternion_wxyz[[1, 2, 3, 0]]
            vertices = quat_rotate_xyzw(
                np.broadcast_to(geom_quaternion_xyzw, (len(vertices), 4)),
                vertices,
            ) + geom_position
            if not np.all(np.isfinite(vertices)) or not np.all(np.isfinite(faces)):
                raise RuntimeError(f"MANO mesh {keypoint_name} contains non-finite geometry")
            meshes.append(
                {
                    "keypoint_name": keypoint_name,
                    "geom_id": int(geom_id),
                    "body_id": int(body_id),
                    "mesh_id": mesh_id,
                    "path": f"world/mano_hand/{keypoint_name}",
                    "vertices": vertices,
                    "faces": faces,
                    "vertex_count": vertex_count,
                    "triangle_count": face_count,
                }
            )
        return meshes

    def _log_static_metadata(self) -> None:
        environment = self.environment
        if not hasattr(self, "hand_meshes"):
            self.hand_meshes = self._hand_meshes()
        trajectory = environment.trajectories[self.env_id]
        identity_parts = trajectory.identity.identity.split("_")
        metadata = {
            "schema": "manorl.rerun.v2",
            "reward_contract": REWARD_CONTRACT_ID,
            "ppo_reward_contract": PPO_REWARD_CONTRACT_ID,
            "ppo_reward_scale": PPO_REWARD_SCALE,
            "environment_contract": ENVIRONMENT_CONTRACT_ID,
            "env_id": self.env_id,
            "episode_id": self.episode_id,
            "trajectory_identity": trajectory.identity.identity,
            "object_type": identity_parts[0],
            "gesture": identity_parts[1],
            "trajectory_uuid": trajectory.identity.uuid,
            "trajectory_source_slice": [int(trajectory.source_indices[0]), int(trajectory.source_indices[-1]) + 1],
            "trajectory_length": int(environment.trajectory_lengths[self.env_id]),
            "joint_names": list(JOINT_NAMES),
            "keypoint_names": list(KEYPOINT_NAMES),
            "hand_meshes": [
                {
                    key: mesh[key]
                    for key in (
                        "keypoint_name",
                        "geom_id",
                        "body_id",
                        "mesh_id",
                        "path",
                        "vertex_count",
                        "triangle_count",
                    )
                }
                for mesh in self.hand_meshes
            ],
            "hand_mesh_source": "compiled MuJoCo meshes resolved from mano_hand.urdf",
            "collision_geometries": self.geometry_table,
            "hand_object_force_on_object_world_N": {
                "direction": "net world-frame force exerted on the object by each hand collision geom",
                "filter": "only contact rows with exactly one source-mapped hand collision geom and one object collision geom",
                "series": self.hand_object_force_series_table,
            },
            "object_gravity_world_N": environment.object_gravity_world_force.tolist(),
            "control_timestep_s": CONTROL_TIMESTEP,
            "physics_substeps": PHYSICS_SUBSTEPS_PER_TARGET,
            "residual_enabled": environment.config.residual_enabled,
            "residual_action": {
                "position_scale": list(environment.config.residual_action.position_scale),
                "gamma_xy": environment.config.residual_action.gamma_xy,
                "gamma_z": environment.config.residual_action.gamma_z,
                "max_position_offset": list(environment.config.residual_action.max_position_offset),
            },
            "compatibility": environment.config.compatibility.point_template_mode,
            "thresholds": {
                "max_deviation_distance": environment.config.max_deviation_distance,
                "deviation_penalty": environment.config.deviation_penalty,
                "observation_contact_threshold_N": CONTACT_FORCE_THRESHOLD,
                "reward_hand_object_threshold_N": REWARD_HAND_OBJECT_THRESHOLD_N,
                "contact_start_frame": int(environment.contact_start_frames[self.env_id]),
                "contact_end_frame": int(environment.contact_end_frames[self.env_id]),
                "contact_capacity": environment.config.contact_capacity,
                "constraint_capacity": environment.config.constraint_capacity,
            },
        }
        self.recording.log("run/metadata", self.rr.TextDocument(json.dumps(metadata, indent=2, sort_keys=True)), static=True)
        for name, value in metadata["thresholds"].items():
            self.recording.log(f"threshold/{name}", self.rr.Scalars(float(value)), static=True)
        for component, _ in _GEOMETRY_FORCE_PATHS:
            self.recording.log(
                f"contact/geometry_force/world/{component}",
                self.rr.SeriesLines(names=self.geometry_series_names),
                static=True,
            )
        self.recording.log(
            _HAND_OBJECT_FORCE_PATH,
            self.rr.SeriesLines(names=self.hand_object_force_series_names),
            static=True,
        )
        for mesh in self.hand_meshes:
            self.recording.log(
                mesh["path"],
                self.rr.Mesh3D(
                    vertex_positions=mesh["vertices"],
                    triangle_indices=mesh["faces"],
                    albedo_factor=(0.88, 0.58, 0.46, 1.0),
                ),
                static=True,
            )

    def record_transition(self) -> None:
        if self._closed:
            raise RuntimeError("cannot record after close")
        snapshot = self.environment.last_transition
        if snapshot is None:
            raise RuntimeError("record_transition requires one completed environment.step call")
        self._capture_replay_action(snapshot)
        if self.archive_dir is not None and self.archive_threshold is not None:
            terminal_returns = np.asarray(snapshot.episode_return, dtype=np.float64)
            terminal_mask = np.asarray(snapshot.termination.reset, dtype=bool)
            high_envs = np.flatnonzero(terminal_mask & (terminal_returns >= self.archive_threshold))
            self._queue_high_return_replays(snapshot, high_envs)
            if high_envs.size and self._pending_high_env_id is None:
                self._pending_high_env_id = int(high_envs[0])
        switched_to_high_env = False
        switched_high_return: float | None = None
        if self._pending_high_env_id is not None and not bool(snapshot.reset_applied[self._pending_high_env_id]):
            pending_env_id = self._pending_high_env_id
            pending_return = float(snapshot.episode_return[pending_env_id])
            if bool(snapshot.termination.reset[pending_env_id]) and pending_env_id != self.env_id:
                self._pending_high_env_id = None
                self._switch_to_high_return_env(pending_env_id, initial_return=pending_return)
                switched_to_high_env = True
                switched_high_return = pending_return
        if (
            self._pending_high_env_id is not None
            and bool(snapshot.reset_applied[self._pending_high_env_id])
        ):
            pending_env_id = self._pending_high_env_id
            self._pending_high_env_id = None
            if pending_env_id != self.env_id:
                self._switch_to_high_return_env(pending_env_id)
                switched_to_high_env = True
        # The transition after terminal state has physically applied the delayed
        # reset. Atomically replace the user-visible recording before writing
        # the new reset state into a fresh private active file.
        if bool(snapshot.reset_applied[self.env_id]) and not switched_to_high_env:
            completed_return = float(self._episode_return)
            archive_path = self._high_return_archive_path(completed_return)
            self._publish_episode(archive_path=archive_path)
            self.episode_id += 1
            self._start_episode()
            self._episode_return = 0.0
        self._record(snapshot)
        if switched_high_return is None:
            self._episode_return += float(snapshot.reward.total[self.env_id])
        else:
            # The terminal snapshot already contains the complete return; do
            # not add its final reward a second time before archive naming.
            self._episode_return = switched_high_return

    def _record(self, snapshot: TransitionSnapshot) -> None:
        env_id = self.env_id
        physical = snapshot.physical
        if not hasattr(self, "hand_meshes"):
            self.hand_meshes = self._hand_meshes()
        reward = snapshot.reward
        termination = snapshot.termination
        index = int(snapshot.target_indices[env_id])
        object_position = physical.object_position[env_id]
        hand_position = physical.hand_position[env_id]
        object_cloud_local = self.environment.object_point_cloud_local()[env_id]
        keypoint_forces = physical.hand_keypoint_contact_forces[env_id]
        geometry_forces = physical.geom_contact_force_world_N[env_id]
        hand_object_forces = physical.hand_object_force_on_object_world_N[env_id]
        if geometry_forces.shape != (len(self.geometry_series_names), 3):
            raise RuntimeError("physical geometry force shape no longer matches static Rerun labels")
        if hand_object_forces.shape != (len(KEYPOINT_NAMES), 3) or not np.all(np.isfinite(hand_object_forces)):
            raise RuntimeError("physical hand-object force shape no longer matches the source keypoint contract")
        force_magnitudes = np.linalg.norm(keypoint_forces, axis=1)
        hand_object_force_magnitudes = np.linalg.norm(hand_object_forces, axis=1)
        target_next = min(index + 5, int(self.environment.trajectory_lengths[env_id]) - 1)
        target_position = self.environment.reference_object_pos[env_id, index]
        target_orientation = self.environment.reference_object_quat_xyzw[env_id, index]
        trajectory_complete = bool(termination.reset[env_id] and not termination.deviation_reset[env_id])
        target_distance = float(np.linalg.norm(object_position - target_position))
        self.recording.set_time("step", sequence=snapshot.control_call)
        self.recording.set_time("simulation", duration=snapshot.control_call * CONTROL_TIMESTEP)

        self.recording.log(
            "world/object",
            self.rr.Transform3D(
                translation=object_position,
                quaternion=self.rr.Quaternion(xyzw=physical.object_orientation_xyzw[env_id]),
            ),
        )
        self.recording.log("world/object/center", self.rr.Points3D([[0.0, 0.0, 0.0]], colors=[(45, 190, 100)], radii=[0.012]))
        self.recording.log("world/hand", self.rr.Points3D([hand_position], colors=[(70, 140, 230)], radii=[0.010]))
        hand_positions = physical.hand_keypoint_positions[env_id]
        hand_orientations = physical.hand_keypoint_orientations_xyzw[env_id]
        if hand_positions.shape != (len(self.hand_meshes), 3) or hand_orientations.shape != (
            len(self.hand_meshes),
            4,
        ):
            raise RuntimeError("physical hand transform shapes no longer match the MANO mesh contract")
        orientation_norms = np.linalg.norm(hand_orientations, axis=1)
        if (
            not np.all(np.isfinite(hand_positions))
            or not np.all(np.isfinite(hand_orientations))
            or not np.allclose(orientation_norms, 1.0, rtol=0.0, atol=1e-6)
        ):
            raise RuntimeError("physical hand transforms must contain finite positions and unit quaternions")
        for mesh, position, orientation in zip(
            self.hand_meshes, hand_positions, hand_orientations, strict=True
        ):
            self.recording.log(
                mesh["path"],
                self.rr.Transform3D(
                    translation=position,
                    quaternion=self.rr.Quaternion(xyzw=orientation),
                ),
            )
        self.recording.log("world/hand_keypoints", self.rr.Points3D(physical.hand_keypoint_positions[env_id], colors=[(255, 180, 70)], radii=0.004))
        self.recording.log("world/fingertips", self.rr.Points3D(physical.fingertip_positions[env_id], colors=[(250, 100, 100)], radii=0.006))
        self.recording.log("world/object/point_cloud", self.rr.Points3D(object_cloud_local, colors=[(80, 245, 255)], radii=0.005))
        self.recording.log("world/contact_force", self.rr.Arrows3D(origins=physical.hand_keypoint_positions[env_id], vectors=keypoint_forces * _FORCE_ARROW_SCALE, colors=[(240, 80, 220)], radii=0.0015))
        self.recording.log("target/object", self.rr.Points3D([target_position], colors=[(240, 80, 80)], radii=[0.010]))

        scalar_values: dict[str, float] = {
            "episode/id": float(self.episode_id),
            "episode/progress": float(snapshot.progress[env_id]),
            "episode/trajectory_step": float(snapshot.trajectory_steps[env_id]),
            "episode/trajectory_length": float(self.environment.trajectory_lengths[env_id]),
            "episode/return": float(snapshot.episode_return[env_id]),
            "episode/source_reference_index": float(self.environment.reference_source_indices[env_id, index]),
            "world/object_speed": float(np.linalg.norm(physical.object_linear_velocity[env_id])),
            "contact/count": float(physical.contact_count[env_id]),
            "contact/object_force_magnitude": float(np.linalg.norm(physical.object_contact_force[env_id])),
            "contact/keypoint_force_total": float(force_magnitudes.sum()),
            "termination/requested": float(termination.reset[env_id]),
            "termination/trajectory_complete": float(trajectory_complete),
            "termination/deviation": float(termination.deviation_reset[env_id]),
            "termination/object_target_distance": target_distance,
            "reset/applied": float(snapshot.reset_applied[env_id]),
        }
        for field in fields(reward):
            scalar_values[f"reward/{field.name}"] = float(getattr(reward, field.name)[env_id])
        for name, value in scalar_values.items():
            self.recording.log(name, self.rr.Scalars(value))
        geometry_series = {
            "magnitude_N": np.linalg.norm(geometry_forces, axis=1),
            "x_N": geometry_forces[:, 0],
            "y_N": geometry_forces[:, 1],
            "z_N": geometry_forces[:, 2],
        }
        for component, _ in _GEOMETRY_FORCE_PATHS:
            self.recording.log(
                f"contact/geometry_force/world/{component}",
                self.rr.Scalars(geometry_series[component]),
            )
        self.recording.log(_HAND_OBJECT_FORCE_PATH, self.rr.Scalars(hand_object_force_magnitudes))
        gravity = self.environment.object_gravity_world_force
        for component, value in zip(("x_N", "y_N", "z_N"), gravity, strict=True):
            self.recording.log(f"contact/object/gravity/world/{component}", self.rr.Scalars(float(value)))
        self.recording.log(
            _OBJECT_GRAVITY_MAGNITUDE_PATH,
            self.rr.Scalars(float(np.linalg.norm(gravity))),
        )
        self.recording.log(
            "state/arrays",
            self.rr.AnyValues(
                object_orientation_xyzw=physical.object_orientation_xyzw[env_id],
                hand_orientation_xyzw=physical.hand_orientation_xyzw[env_id],
                mano_dof_pos=physical.mano_dof_pos[env_id],
                target_object_orientation_xyzw=target_orientation,
                target_object_position_t_plus_5=self.environment.reference_object_pos[env_id, target_next],
                raw_action_26=snapshot.raw_actions[env_id],
                processed_target_26=snapshot.processed_targets[env_id],
                controller_target_26=snapshot.controller_targets[env_id],
                command_target_26=snapshot.command_targets[env_id],
                cumulative_position_offset=snapshot.observation.raw[env_id, OBSERVATION_SLICES["cumulative_offset"]],
                cumulative_joint_offset=snapshot.observation.raw[env_id, OBSERVATION_SLICES["cumulative_joint_offset"]],
                keypoint_force_components_48=keypoint_forces.reshape(-1),
                keypoint_force_magnitude=force_magnitudes,
                hand_object_force_on_object_world_N_components_48=hand_object_forces.reshape(-1),
                hand_object_force_on_object_magnitude_N=hand_object_force_magnitudes,
                object_force_xyz=physical.object_contact_force[env_id],
                expected_contact_mask=self.environment.expected_contact_mask[env_id],
                raw_observation_476=snapshot.observation.raw[env_id],
            ),
        )

    def close(self) -> Path | None:
        if not self._closed:
            if self._recording_open:
                self.recording.flush()
                self.recording.disconnect()
                self._recording_open = False
            self.active_path.unlink(missing_ok=True)
            self._close_result = self.output if self._published else None
            self._replay_high_return_episodes()
            self._closed = True
        return self._close_result
