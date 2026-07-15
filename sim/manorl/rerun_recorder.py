"""Rerun observer for target-native ManoRL environment transitions."""

from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path

import numpy as np

from sim.manorl.contracts import CONTROL_TIMESTEP, JOINT_NAMES, KEYPOINT_NAMES, PHYSICS_SUBSTEPS_PER_TARGET
from sim.manorl.environment import MujocoManoEnvironment, TransitionSnapshot
from sim.manorl.observations import OBSERVATION_SLICES
from sim.manorl.rewards import CONTACT_FORCE_THRESHOLD

_FORCE_ARROW_SCALE = 0.002


class ManoRerunRecorder:
    """Write one immutable Rerun artifact for each episode of a selected world."""

    def __init__(self, environment: MujocoManoEnvironment, output: str | Path, *, env_id: int = 0) -> None:
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
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.episode_id = 0
        self.episode_paths: list[Path] = []
        self._closed = False
        self._start_episode()

    def _episode_path(self) -> Path:
        suffix = self.output.suffix or ".rrd"
        stem = self.output.stem if self.output.suffix else self.output.name
        return self.output.with_name(f"{stem}.episode_{self.episode_id:04d}{suffix}")

    def _default_blueprint(self):
        blueprint = self.rr.blueprint
        return blueprint.Blueprint(
            blueprint.Vertical(
                blueprint.Spatial3DView(
                    name="Object, hand, and point cloud",
                    contents=[
                        "world/object", "target/object", "world/object_point_cloud",
                        "world/hand", "world/hand_keypoints", "world/fingertips", "world/contact_force",
                    ],
                    eye_controls=blueprint.EyeControls3D(
                        kind=blueprint.components.Eye3DKind.Orbital,
                        tracking_entity="world/object",
                        speed=0.1,
                    ),
                    line_grid=False,
                ),
                blueprint.Horizontal(
                    blueprint.TimeSeriesView(
                        name="Reward and termination",
                        contents=["reward/**", "termination/**", "reset/**"],
                    ),
                    blueprint.TimeSeriesView(
                        name="Contacts and actions",
                        contents=["contact/**", "action/**", "episode/**"],
                    ),
                ),
                row_shares=[3.0, 1.0],
            ),
            auto_layout=False,
            auto_views=False,
            collapse_panels=False,
        )

    def _start_episode(self) -> None:
        self.current_path = self._episode_path()
        self.recording = self.rr.RecordingStream("manorl_mujoco")
        self.recording.save(self.current_path, default_blueprint=self._default_blueprint())
        self.episode_paths.append(self.current_path)
        self._log_static_metadata()

    def _finish_episode(self) -> None:
        self.recording.flush()
        self.recording.disconnect()

    def _log_static_metadata(self) -> None:
        environment = self.environment
        trajectory = environment.trajectories[self.env_id]
        identity_parts = trajectory.identity.identity.split("_")
        metadata = {
            "schema": "manorl.rerun.v2",
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
            "control_timestep_s": CONTROL_TIMESTEP,
            "physics_substeps": PHYSICS_SUBSTEPS_PER_TARGET,
            "residual_enabled": environment.config.residual_enabled,
            "compatibility": environment.config.compatibility.point_template_mode,
            "thresholds": {
                "max_deviation_distance": environment.config.max_deviation_distance,
                "deviation_penalty": environment.config.deviation_penalty,
                "contact_force_threshold": CONTACT_FORCE_THRESHOLD,
                "contact_start_frame": int(environment.contact_start_frames[self.env_id]),
                "contact_end_frame": int(environment.contact_end_frames[self.env_id]),
                "contact_capacity": environment.config.contact_capacity,
                "constraint_capacity": environment.config.constraint_capacity,
            },
        }
        self.recording.log("run/metadata", self.rr.TextDocument(json.dumps(metadata, indent=2, sort_keys=True)), static=True)
        for name, value in metadata["thresholds"].items():
            self.recording.log(f"threshold/{name}", self.rr.Scalars(float(value)), static=True)

    def record_transition(self) -> None:
        if self._closed:
            raise RuntimeError("cannot record after close")
        snapshot = self.environment.last_transition
        if snapshot is None:
            raise RuntimeError("record_transition requires one completed environment.step call")
        # The transition after terminal state has physically applied the delayed
        # reset. Finalize the prior episode before writing the new reset state.
        if bool(snapshot.reset_applied[self.env_id]):
            self._finish_episode()
            self.episode_id += 1
            self._start_episode()
        self._record(snapshot)

    def _record(self, snapshot: TransitionSnapshot) -> None:
        env_id = self.env_id
        physical = snapshot.physical
        reward = snapshot.reward
        termination = snapshot.termination
        index = int(snapshot.target_indices[env_id])
        object_position = physical.object_position[env_id]
        hand_position = physical.hand_position[env_id]
        raw_cloud = snapshot.observation.raw[env_id, OBSERVATION_SLICES["object_point_cloud_raw"]].reshape(-1, 3)
        object_cloud_world = raw_cloud + hand_position
        keypoint_forces = physical.hand_keypoint_contact_forces[env_id]
        force_magnitudes = np.linalg.norm(keypoint_forces, axis=1)
        target_next = min(index + 5, int(self.environment.trajectory_lengths[env_id]) - 1)
        target_position = self.environment.reference_object_pos[env_id, index]
        target_orientation = self.environment.reference_object_quat_xyzw[env_id, index]
        trajectory_complete = bool(termination.reset[env_id] and not termination.deviation_reset[env_id])
        target_distance = float(np.linalg.norm(object_position - target_position))
        self.recording.set_time("control_call", sequence=snapshot.control_call)
        self.recording.set_time("simulation", duration=snapshot.control_call * CONTROL_TIMESTEP)

        self.recording.log("world/object", self.rr.Points3D([object_position], colors=[(45, 190, 100)], radii=[0.012]))
        self.recording.log("world/hand", self.rr.Points3D([hand_position], colors=[(70, 140, 230)], radii=[0.010]))
        self.recording.log("world/hand_keypoints", self.rr.Points3D(physical.hand_keypoint_positions[env_id], colors=[(255, 180, 70)], radii=0.004))
        self.recording.log("world/fingertips", self.rr.Points3D(physical.fingertip_positions[env_id], colors=[(250, 100, 100)], radii=0.006))
        self.recording.log("world/object_point_cloud", self.rr.Points3D(object_cloud_world, colors=[(120, 220, 240)], radii=0.0025))
        self.recording.log("world/contact_force", self.rr.Arrows3D(origins=physical.hand_keypoint_positions[env_id], vectors=keypoint_forces * _FORCE_ARROW_SCALE, colors=[(240, 80, 220)], radii=0.0015))
        self.recording.log("target/object", self.rr.Points3D([target_position], colors=[(240, 80, 80)], radii=[0.010]))

        scalar_values: dict[str, float] = {
            "episode/id": float(self.episode_id),
            "episode/progress": float(snapshot.progress[env_id]),
            "episode/trajectory_step": float(snapshot.trajectory_steps[env_id]),
            "episode/trajectory_length": float(self.environment.trajectory_lengths[env_id]),
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
                object_force_xyz=physical.object_contact_force[env_id],
                expected_contact_mask=self.environment.expected_contact_mask[env_id],
                raw_observation_476=snapshot.observation.raw[env_id],
            ),
        )

    def close(self) -> Path:
        if not self._closed:
            self._finish_episode()
            self._closed = True
        return self.current_path
