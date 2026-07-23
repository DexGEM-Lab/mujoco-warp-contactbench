#!/usr/bin/env python3
"""Write an exact, deterministic MuJoCo DexHand decision/post-state trace."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from sim.dexhandrl.contracts import DexHandPhysicsConfig, resolve_physics_config
from sim.dexhandrl.obs_layout import OBS_COMPONENT_SLICES, OBS_DIM_021PRO
from sim.dexhandrl.view_environment import (
    PassiveDexHandViewer as PassiveTraceViewer,
    passive_dexhand_viewer as _passive_trace_viewer,
)
from sim.dexhandrl.trace_contract import (
    EXACT_IDENTITY,
    FINAL_STATE_FRAME,
    MAX_ACTIONS,
    RESET_DECISION_FRAME,
    SCHEMA,
    load_trace,
    validate_trace,
)

CHECKPOINT = Path("/home/jay/dexrobot/FromSSH/dexhand_reconstruction/dexrobot_isaac/last_Dexhand021proReconstruction_ep_20000_rew_101.06729.pth")
LANCE = Path("/mnt/nas-222-project/mocap/for_retargeting/dexhand021pro/results_all_mano/all/lance_new_all_generated_mano/lance_result_archive_full/dexhand021pro_lance_new_all_generated_mano_retarget_result_archive.lance")
ISAAC = Path("/home/jay/dexrobot/FromSSH/dexhand_reconstruction/dexrobot_isaac")
CONTROL_TRACE_ARRAYS = (
    "policy_mean", "final_action", "base_offset_before", "base_offset_after",
    "joint_offset_before", "joint_offset_after", "reference_active_target",
    "corrected_active_target", "applied_active_target", "full_target",
)
CONTROL_TRACE_ATOL = 2e-6


class ObservationTraceIntervention:
    """Fail-closed injection of selected Gym observation components into policy input.

    MuJoCo remains the source of every live state, residual/control stage, and
    physics transition.  Only named slices of the 461D policy observation are
    replaced immediately before RMS normalization and policy evaluation.
    """

    def __init__(self, source_path: Path, metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray],
                 components: tuple[str, ...], intervention_end_frame: int | None = None) -> None:
        if metadata.get("backend") != "gym":
            raise ValueError("--observation-trace must be a Gym trace")
        if not components:
            raise ValueError("--observation-trace requires at least one --replace-observation-component")
        invalid = [name for name in components if name not in OBS_COMPONENT_SLICES]
        if invalid:
            raise ValueError(
                "--replace-observation-component must name exact OBS_COMPONENT_SLICES keys; "
                f"invalid: {', '.join(invalid)}"
            )
        self.source_path = source_path.resolve()
        self.metadata = dict(metadata)
        self.arrays = {name: np.asarray(value) for name, value in arrays.items()}
        self.count = validate_trace(self.metadata, self.arrays)
        self.components = components
        self.intervention_end_frame = intervention_end_frame
        self.source_observations = np.asarray(self.arrays["processed_observation"], dtype=np.float32)
        if not np.all(np.isfinite(self.source_observations)):
            raise ValueError("observation trace requires finite processed_observation")
        self.source_sha256 = hashlib.sha256(self.source_path.read_bytes()).hexdigest()
        self._row = 0

    @classmethod
    def load(cls, path: Path, steps: int, components: tuple[str, ...],
             intervention_end_frame: int | None = None) -> "ObservationTraceIntervention":
        metadata, arrays = load_trace(str(path))
        intervention = cls(path, metadata, arrays, components, intervention_end_frame)
        if intervention.count < steps:
            raise ValueError(
                f"--observation-trace has {intervention.count} actions, fewer than requested --steps={steps}"
            )
        return intervention

    def apply(self, live_observation: np.ndarray, target_frame: int) -> np.ndarray:
        """Return one hybrid policy input, consuming exactly one aligned source row."""
        live = np.asarray(live_observation, dtype=np.float32)
        if live.shape != (OBS_DIM_021PRO,) or not np.all(np.isfinite(live)):
            raise RuntimeError(f"live raw observation must be finite shape ({OBS_DIM_021PRO},), got {live.shape}")
        if self._row >= self.count:
            raise RuntimeError("observation trace exhausted before MuJoCo policy decision")
        source_frame = int(self.arrays["decision_frame"][self._row])
        if source_frame != target_frame:
            raise RuntimeError(
                f"observation trace decision frame {source_frame} does not equal live target frame {target_frame}"
            )
        self._row += 1
        if self.intervention_end_frame is not None and target_frame > self.intervention_end_frame:
            return live.copy()
        hybrid = live.copy()
        source = self.source_observations[self._row - 1]
        for name in self.components:
            start, end = OBS_COMPONENT_SLICES[name]
            hybrid[start:end] = source[start:end]
        return hybrid

    def metadata_record(self) -> dict[str, Any]:
        return {
            "source_path": str(self.source_path),
            "source_sha256": self.source_sha256,
            "source_backend": "gym",
            "components": list(self.components),
            "intervention_end_frame": self.intervention_end_frame,
            "switch_semantics": (
                "Source rows are alignment-validated and consumed at every decision; named slices are "
                "replaced through intervention_end_frame inclusive, then the untouched live observation is used."
                if self.intervention_end_frame is not None else
                "Named slices are replaced for every traced decision."
            ),
            "semantics": (
                "Only named Gym processed_observation slices are imported into MuJoCo policy input; "
                "no target, action, residual/control stage, or physics state is imported."
            ),
        }


class ActionTraceIntervention:
    """Fail-closed injection of aligned Gym final actions into a live MuJoCo step.

    The source contributes exactly one final_action vector per decision.  MuJoCo
    retains its observation/RMS-policy computation, residual target path, and
    physics transition.
    """

    def __init__(self, source_path: Path, metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> None:
        if metadata.get("backend") != "gym":
            raise ValueError("--action-trace must be a Gym trace")
        self.source_path = source_path.resolve()
        self.metadata = dict(metadata)
        self.arrays = {name: np.asarray(value) for name, value in arrays.items()}
        self.count = validate_trace(self.metadata, self.arrays)
        self.final_actions = np.asarray(self.arrays["final_action"], dtype=np.float32)
        if not np.all(np.isfinite(self.final_actions)):
            raise ValueError("action trace requires finite final_action")
        self.source_sha256 = hashlib.sha256(self.source_path.read_bytes()).hexdigest()
        self._row = 0

    @classmethod
    def load(cls, path: Path, steps: int) -> "ActionTraceIntervention":
        metadata, arrays = load_trace(str(path))
        intervention = cls(path, metadata, arrays)
        if intervention.count < steps:
            raise ValueError(
                f"--action-trace has {intervention.count} actions, fewer than requested --steps={steps}"
            )
        return intervention

    def action(self, target_frame: int) -> np.ndarray:
        """Consume and return the one source action aligned to target_frame."""
        if self._row >= self.count:
            raise RuntimeError("action trace exhausted before MuJoCo policy decision")
        source_frame = int(self.arrays["decision_frame"][self._row])
        if source_frame != target_frame:
            raise RuntimeError(
                f"action trace decision frame {source_frame} does not equal live target frame {target_frame}"
            )
        action = self.final_actions[self._row].copy()
        self._row += 1
        return action

    def metadata_record(self) -> dict[str, Any]:
        return {
            "source_path": str(self.source_path),
            "source_sha256": self.source_sha256,
            "source_backend": "gym",
            "semantics": (
                "Only aligned Gym final_action vectors are imported into MuJoCo steps; "
                "MuJoCo processed/RMS observations, checkpoint policy_mean, residual/control targets, "
                "and post-physics state remain live."
            ),
        }


class ControlTraceIntervention:
    """Fail-closed source-control injection for the preregistered Gym replay.

    It replaces only the environment's action-to-target transformation. MuJoCo
    still generates the observation and post-step physical state.
    """

    def __init__(self, source_path: Path, metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> None:
        if metadata.get("backend") != "gym":
            raise ValueError("--control-trace must be a Gym trace")
        self.source_path = source_path.resolve()
        self.metadata = dict(metadata)
        self.arrays = {name: np.asarray(value) for name, value in arrays.items()}
        self.count = validate_trace(self.metadata, self.arrays)
        for name in CONTROL_TRACE_ARRAYS:
            value = self.arrays.get(name)
            if value is None or not np.all(np.isfinite(value)):
                raise ValueError(f"control trace requires finite {name}")
        self.source_sha256 = hashlib.sha256(self.source_path.read_bytes()).hexdigest()
        self._row = 0

    @classmethod
    def load(cls, path: Path, steps: int) -> "ControlTraceIntervention":
        metadata, arrays = load_trace(str(path))
        intervention = cls(path, metadata, arrays)
        if intervention.count < steps:
            raise ValueError(
                f"--control-trace has {intervention.count} actions, fewer than requested --steps={steps}"
            )
        return intervention

    def current(self, target_frame: int) -> dict[str, np.ndarray]:
        if self._row >= self.count:
            raise RuntimeError("control trace exhausted before MuJoCo step")
        source_frame = int(self.arrays["decision_frame"][self._row])
        if source_frame != target_frame:
            raise RuntimeError(
                f"control trace decision frame {source_frame} does not equal live target frame {target_frame}"
            )
        return {name: np.asarray(self.arrays[name][self._row], dtype=np.float32).copy()
                for name in CONTROL_TRACE_ARRAYS}

    def apply(self, env: Any, target_frame: int) -> np.ndarray:
        """Assert live offset state, then inject one source control row."""
        source = self.current(target_frame)
        for label, live, expected in (
            ("base_position_offset", env.base_position_offset, source["base_offset_before"]),
            ("joint_offset", env.joint_offset, source["joint_offset_before"]),
        ):
            live = np.asarray(live, dtype=np.float32)
            if not np.all(np.isfinite(live)) or not np.allclose(live, expected, rtol=0.0, atol=CONTROL_TRACE_ATOL):
                maximum = float(np.max(np.abs(live - expected))) if live.shape == expected.shape else float("inf")
                raise RuntimeError(
                    f"control trace {label} mismatch at frame {target_frame}: max_abs_error={maximum}"
                )
        env.base_position_offset[...] = source["base_offset_after"]
        env.joint_offset[...] = source["joint_offset_after"]
        env.active_prev_targets = source["applied_active_target"].copy()
        env._last_action_trace = {
            "decision_frame": target_frame,
            "reference_active_target": source["reference_active_target"].copy(),
            "corrected_active_target": source["corrected_active_target"].copy(),
            "applied_active_target": source["applied_active_target"].copy(),
            "full_target": source["full_target"].copy(),
        }
        self._row += 1
        return source["full_target"].copy()

    @contextmanager
    def patch(self, env: Any) -> Iterator[None]:
        """Inject source targets only while the diagnostic trace executes."""
        original = env._apply_residual_action

        def injected(_action: np.ndarray, target_frame: int, _current_full: np.ndarray | None) -> np.ndarray:
            return self.apply(env, target_frame)

        env._apply_residual_action = injected
        try:
            yield
        finally:
            env._apply_residual_action = original

    def metadata_record(self) -> dict[str, Any]:
        return {
            "source_path": str(self.source_path),
            "source_sha256": self.source_sha256,
            "source_backend": "gym",
            "semantics": (
                "Gym source offsets, policy/action records, and action-stage targets are imported; "
                "only MuJoCo processed/RMS observations and post-physics state are live."
            ),
        }


def _validate_trace_interventions(control_trace: Path | None, action_trace: Path | None,
                                  observation_trace: Path | None, components: tuple[str, ...],
                                  intervention_end_frame: int | None = None) -> None:
    traces = (control_trace, action_trace, observation_trace)
    if sum(trace is not None for trace in traces) > 1:
        raise ValueError("--control-trace, --action-trace, and --observation-trace are mutually exclusive")
    if (observation_trace is None) != (not bool(components)):
        raise ValueError("--observation-trace and --replace-observation-component are required together")
    invalid = [name for name in components if name not in OBS_COMPONENT_SLICES]
    if invalid:
        raise ValueError(
            "--replace-observation-component must name exact OBS_COMPONENT_SLICES keys; "
            f"invalid: {', '.join(invalid)}"
        )
    if intervention_end_frame is not None:
        if observation_trace is None or not components:
            raise ValueError("--observation-intervention-end-frame requires --observation-trace and --replace-observation-component")
        if not RESET_DECISION_FRAME <= intervention_end_frame <= FINAL_STATE_FRAME:
            raise ValueError(
                f"--observation-intervention-end-frame must be in {RESET_DECISION_FRAME}..{FINAL_STATE_FRAME}"
            )


def _policy_observation(live_observation: np.ndarray, decision_frame: int,
                        intervention: ObservationTraceIntervention | None) -> np.ndarray:
    """Select the authoritative policy input without changing normal tracing."""
    return intervention.apply(live_observation, decision_frame) if intervention is not None else live_observation


def _ordered_qvel(env) -> np.ndarray:
    qvel = np.asarray(env._dx.qvel, dtype=np.float32)
    return np.asarray([qvel[env.ids.joint_dof_addr[name]] for name in env.ids.joint_dof_addr], dtype=np.float32)


def _append(store: dict[str, list[np.ndarray]], **items: np.ndarray) -> None:
    for key, value in items.items():
        store[key].append(np.asarray(value))


def run(output: Path, steps: int, checkpoint: Path, control_trace: Path | None = None,
        hand_friction: str | None = None, headless: bool = True,
        finger_kp: float | None = None, finger_kv: float | None = None,
        base_rot_kp: float | None = None, base_rot_kv: float | None = None,
        base_rot_force: float | None = None, observation_trace: Path | None = None,
        replace_observation_component: tuple[str, ...] = (),
        observation_intervention_end_frame: int | None = None,
        action_trace: Path | None = None, *, physics: DexHandPhysicsConfig | None = None,
        dt: float | None = None, substeps: int | None = None,
        base_pos_kp: float | None = None, base_pos_kv: float | None = None, base_pos_force: float | None = None,
        finger_force: float | None = None, object_friction: str | None = None,
        ground_friction: str | None = None, collision: str | None = None,
        naconmax: int | None = None, njmax: int | None = None) -> Path:
    physics = resolve_physics_config(
        physics, dt=dt, substeps=substeps,
        base_pos_kp=base_pos_kp, base_pos_kv=base_pos_kv, base_pos_force=base_pos_force,
        base_rot_kp=base_rot_kp, base_rot_kv=base_rot_kv, base_rot_force=base_rot_force,
        finger_kp=finger_kp, finger_kv=finger_kv, finger_force=finger_force,
        hand_friction=hand_friction, object_friction=object_friction, ground_friction=ground_friction,
        collision=collision, naconmax=naconmax, njmax=njmax,
    )
    if not 1 <= steps <= MAX_ACTIONS:
        raise ValueError(f"--steps must be in 1..{MAX_ACTIONS}")
    _validate_trace_interventions(
        control_trace, action_trace, observation_trace, replace_observation_component, observation_intervention_end_frame
    )
    intervention = ControlTraceIntervention.load(control_trace, steps) if control_trace is not None else None
    action_intervention = ActionTraceIntervention.load(action_trace, steps) if action_trace is not None else None
    observation_intervention = (ObservationTraceIntervention.load(
        observation_trace, steps, replace_observation_component, observation_intervention_end_frame
    ) if observation_trace is not None else None)
    import torch
    from sim.dexhandrl.hand_mapping import full_dof_from_qpos
    from sim.dexhandrl.rlgames_checkpoint import load_rlgames_checkpoint
    from sim.dexhandrl.skrl_env import DexHandRLMJXEnv, DexHandRLMJXEnvConfig
    from sim.dexhandrl.skrl_train import SkrlLauncherConfig, _build_agent
    cfg = DexHandRLMJXEnvConfig(lance_path=LANCE, isaac_source_root=ISAAC, object_name="cube1", action="01", sequence_index=1,
        uuid=EXACT_IDENTITY["uuid"], device="cpu", use_residual=True, seed=0, object_position_terminal_threshold=1e9,
        physics=physics)
    env = DexHandRLMJXEnv(cfg)
    if env.trajectory.row_id != 22 or env.start_frame != 61 or env.end_frame != 693:
        raise RuntimeError(f"exact trajectory isolation failed: row={env.trajectory.row_id}, window={env.start_frame}..{env.end_frame}")
    launcher = SkrlLauncherConfig(device="cpu", num_envs=1, rlgames_checkpoint=checkpoint)
    agent, _ = _build_agent(env, launcher)
    load_rlgames_checkpoint(agent, checkpoint)
    set_mode = getattr(agent, "set_running_mode", None)
    if callable(set_mode):
        set_mode("eval")
    names = ["decision_frame", "state_frame", "reset", "terminated", "truncated", "processed_observation", "rms_observation", "policy_mean", "final_action", "base_offset_before", "base_offset_after", "joint_offset_before", "joint_offset_after", "reference_active_target", "corrected_active_target", "applied_active_target", "full_target", "hand_qpos", "hand_qvel", "object_position", "object_quaternion_xyzw", "object_linear_velocity", "target_object_position", "target_object_quaternion_xyzw"]
    if observation_intervention is not None:
        names.append("live_processed_observation")
    obs, _ = env.reset(); store: dict[str, list[np.ndarray]] = {name: [] for name in names}
    if int(env._current_frame) != RESET_DECISION_FRAME:
        raise RuntimeError(f"MuJoCo reset policy frame must be {RESET_DECISION_FRAME}, got {env._current_frame}")
    patch = intervention.patch(env) if intervention is not None else nullcontext()
    with _passive_trace_viewer(env, headless=headless) as viewer, patch:
        if viewer is not None:
            viewer.sync(env)
        for _step in range(steps):
            decision_frame = int(env._current_frame)
            source = intervention.current(decision_frame) if intervention is not None else None
            imported_action = action_intervention.action(decision_frame) if action_intervention is not None else None
            live_processed_observation = np.asarray(obs, dtype=np.float32).copy()
            policy_observation = _policy_observation(live_processed_observation, decision_frame, observation_intervention)
            tensor = torch.as_tensor(policy_observation[None], dtype=torch.float32, device=agent.policy.device)
            normalized = agent._observation_preprocessor(tensor)
            rms_observation = normalized.detach().cpu().numpy()[0]
            if source is None:
                before_base, before_joint = env.base_position_offset.copy(), env.joint_offset.copy()
                with torch.no_grad():
                    mean, _ = agent.policy.compute({"observations": normalized}, role="policy")
                policy_mean = mean.detach().cpu().numpy()[0]
                local_action = torch.clamp(mean, -1, 1).detach().cpu().numpy()[0]
                action = imported_action if imported_action is not None else local_action
            else:
                before_base, before_joint = source["base_offset_before"], source["joint_offset_before"]
                policy_mean, action = source["policy_mean"], source["final_action"]
            next_obs, _, terminated, truncated, info = env.step(action)
            trace = env._last_action_trace
            if trace is None or trace["decision_frame"] != decision_frame:
                raise RuntimeError("environment did not expose authoritative action trace for this decision")
            state_frame = int(env._current_frame)
            if source is not None:
                for name in ("reference_active_target", "corrected_active_target", "applied_active_target", "full_target"):
                    if not np.array_equal(np.asarray(trace[name], dtype=np.float32), source[name]):
                        raise RuntimeError(f"environment altered injected source {name}")
            _append(store, decision_frame=np.asarray(decision_frame), state_frame=np.asarray(state_frame), reset=np.asarray(decision_frame == RESET_DECISION_FRAME), terminated=np.asarray(terminated), truncated=np.asarray(truncated),
              processed_observation=policy_observation, rms_observation=rms_observation, policy_mean=policy_mean, final_action=action,
              **({"live_processed_observation": live_processed_observation} if observation_intervention is not None else {}),
              base_offset_before=before_base, base_offset_after=(source["base_offset_after"] if source is not None else env.base_position_offset), joint_offset_before=before_joint, joint_offset_after=(source["joint_offset_after"] if source is not None else env.joint_offset),
              reference_active_target=(source["reference_active_target"] if source is not None else np.asarray(trace["reference_active_target"], dtype=np.float32)), corrected_active_target=(source["corrected_active_target"] if source is not None else np.asarray(trace["corrected_active_target"], dtype=np.float32)), applied_active_target=(source["applied_active_target"] if source is not None else np.asarray(trace["applied_active_target"], dtype=np.float32)), full_target=(source["full_target"] if source is not None else np.asarray(trace["full_target"], dtype=np.float32)),
              hand_qpos=full_dof_from_qpos(np.asarray(env._dx.qpos), env.ids), hand_qvel=_ordered_qvel(env), object_position=np.asarray(info["object_position"]), object_quaternion_xyzw=np.asarray(env._dx.xquat[env.object_body_id], dtype=np.float32)[[1, 2, 3, 0]], object_linear_velocity=np.asarray(env._dx.qvel[env.ids.object_free_dof_addr:env.ids.object_free_dof_addr + 3]), target_object_position=np.asarray(info["target_object_position"]), target_object_quaternion_xyzw=np.asarray(env.trajectory.object_quat_xyzw[state_frame]))
            obs = next_obs
            if viewer is not None:
                viewer.sync(env)
            if terminated or truncated:
                break
    scalar_types = {"decision_frame": np.int64, "state_frame": np.int64, "reset": np.bool_, "terminated": np.bool_, "truncated": np.bool_}
    arrays = {key: np.asarray(values, dtype=scalar_types.get(key, np.float32)) for key, values in store.items()}
    metadata: dict[str, Any] = {"schema": SCHEMA, "backend": "mujoco", "identity": EXACT_IDENTITY, "post_filters": ["velocity_clamp", "position_clamp"], "raw_policy_observation": "processed 461D only; component schema is obs_layout.py", "checkpoint": str(checkpoint), "physics_contract": physics.record(), "friction": {"hand": cfg.hand_friction, "object": cfg.object_friction, "ground": cfg.ground_friction}, "finger_pd": {"kp": cfg.finger_kp, "kv": cfg.finger_kv, "force": cfg.finger_force}, "base_rotation_pd": {"kp": cfg.base_rot_kp, "kv": cfg.base_rot_kv, "force": cfg.base_rot_force}, "viewer": {"headless": headless, "semantics": "native passive viewer mirrors completed MJX qpos/qvel via mj_forward and never advances environment physics", "env_count": 1}}
    if intervention is not None:
        metadata["control_trace_intervention"] = intervention.metadata_record()
        metadata["policy_action_provenance"] = "imported from Gym control trace; policy/action records do not arise from MuJoCo observation"
    if observation_intervention is not None:
        metadata["observation_trace_intervention"] = observation_intervention.metadata_record()
        metadata["policy_action_provenance"] = "MuJoCo policy evaluated hybrid processed_observation with selected Gym slices"
    if action_intervention is not None:
        metadata["action_trace_intervention"] = action_intervention.metadata_record()
        metadata["policy_action_provenance"] = "MuJoCo checkpoint policy_mean evaluated from live MuJoCo RMS observation; final_action imported from Gym action trace"
    validate_trace(metadata, arrays); output.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(output, metadata=np.asarray(json.dumps(metadata)), **arrays); return output


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=3, help=f"actions to trace (1..{MAX_ACTIONS})")
    p.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    p.add_argument("--control-trace", type=Path, help="validated Gym trace whose exact control stages are injected")
    p.add_argument("--action-trace", type=Path, help="validated Gym trace whose final_action alone is injected into live MuJoCo steps")
    p.add_argument("--observation-trace", type=Path, help="validated Gym trace supplying selected policy-observation slices")
    p.add_argument("--replace-observation-component", action="append", default=[], metavar="NAME",
                   help="exact OBS_COMPONENT_SLICES key to replace; repeat for multiple components")
    p.add_argument("--observation-intervention-end-frame", type=int, metavar="INT",
                   help=f"inclusive replacement end frame ({RESET_DECISION_FRAME}..{FINAL_STATE_FRAME}); requires observation intervention")
    p.add_argument("--hand-friction", default="0.5 0.01 0.001")
    p.add_argument("--finger-kp", type=float, default=15.0)
    p.add_argument("--finger-kv", type=float, default=0.0)
    p.add_argument("--base-rot-kp", type=float, default=10000.0)
    p.add_argument("--base-rot-kv", type=float, default=125.0)
    p.add_argument("--base-rot-force", type=float, default=100.0)
    p.add_argument("--finger-force", type=float, default=None)
    p.add_argument("--dt", type=float, default=None)
    p.add_argument("--substeps", type=int, default=None)
    p.add_argument("--base-pos-kp", type=float, default=None)
    p.add_argument("--base-pos-kv", type=float, default=None)
    p.add_argument("--base-pos-force", type=float, default=None)
    p.add_argument("--object-friction", default=None)
    p.add_argument("--ground-friction", default=None)
    p.add_argument("--collision", choices=("none", "disable_within_finger", "all_disabled"), default=None)
    p.add_argument("--naconmax", type=int, default=None)
    p.add_argument("--njmax", type=int, default=None)
    p.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    return p


def main() -> int:
    parser = _parser()
    a = parser.parse_args()
    try:
        _validate_trace_interventions(a.control_trace, a.action_trace, a.observation_trace,
                                      tuple(a.replace_observation_component), a.observation_intervention_end_frame)
    except ValueError as error:
        parser.error(str(error))
    extra_physics = {
        "dt": a.dt, "substeps": a.substeps, "base_pos_kp": a.base_pos_kp,
        "base_pos_kv": a.base_pos_kv, "base_pos_force": a.base_pos_force,
        "finger_force": a.finger_force, "object_friction": a.object_friction,
        "ground_friction": a.ground_friction, "collision": a.collision,
        "naconmax": a.naconmax, "njmax": a.njmax,
    }
    call_kwargs = {name: value for name, value in extra_physics.items() if value is not None}
    print(run(a.output, a.steps, a.checkpoint, a.control_trace, a.hand_friction, a.headless,
              a.finger_kp, a.finger_kv, a.base_rot_kp, a.base_rot_kv, a.base_rot_force,
              a.observation_trace, tuple(a.replace_observation_component), a.observation_intervention_end_frame,
              a.action_trace, **call_kwargs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
