#!/usr/bin/env python3
"""Write an exact, deterministic MuJoCo DexHand decision/post-state trace."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from sim.dexhandrl.trace_contract import (
    EXACT_IDENTITY,
    MAX_ACTIONS,
    RESET_DECISION_FRAME,
    SCHEMA,
    validate_trace,
)

CHECKPOINT = Path("/home/jay/dexrobot/FromSSH/dexhand_reconstruction/dexrobot_isaac/last_Dexhand021proReconstruction_ep_20000_rew_101.06729.pth")
LANCE = Path("/mnt/nas-222-project/mocap/for_retargeting/dexhand021pro/results_all_mano/all/lance_new_all_generated_mano/lance_result_archive_full/dexhand021pro_lance_new_all_generated_mano_retarget_result_archive.lance")
ISAAC = Path("/home/jay/dexrobot/FromSSH/dexhand_reconstruction/dexrobot_isaac")


def _ordered_qvel(env) -> np.ndarray:
    qvel = np.asarray(env._dx.qvel, dtype=np.float32)
    return np.asarray([qvel[env.ids.joint_dof_addr[name]] for name in env.ids.joint_dof_addr], dtype=np.float32)


def _append(store: dict[str, list[np.ndarray]], **items: np.ndarray) -> None:
    for key, value in items.items():
        store[key].append(np.asarray(value))


def run(output: Path, steps: int, checkpoint: Path) -> Path:
    if not 1 <= steps <= MAX_ACTIONS:
        raise ValueError(f"--steps must be in 1..{MAX_ACTIONS}")
    import torch
    from sim.dexhandrl.hand_mapping import full_dof_from_qpos
    from sim.dexhandrl.rlgames_checkpoint import load_rlgames_checkpoint
    from sim.dexhandrl.skrl_env import DexHandRLMJXEnv, DexHandRLMJXEnvConfig
    from sim.dexhandrl.skrl_train import SkrlLauncherConfig, _build_agent
    cfg = DexHandRLMJXEnvConfig(lance_path=LANCE, isaac_source_root=ISAAC, object_name="cube1", action="01", sequence_index=1,
        uuid=EXACT_IDENTITY["uuid"], device="cpu", use_residual=True, seed=0, object_position_terminal_threshold=1e9)
    env = DexHandRLMJXEnv(cfg)
    if env.trajectory.row_id != 22 or env.start_frame != 61 or env.end_frame != 693:
        raise RuntimeError(f"exact trajectory isolation failed: row={env.trajectory.row_id}, window={env.start_frame}..{env.end_frame}")
    launcher = SkrlLauncherConfig(device="cpu", num_envs=1, rlgames_checkpoint=checkpoint)
    agent, _ = _build_agent(env, launcher); load_rlgames_checkpoint(agent, checkpoint)
    set_mode = getattr(agent, "set_running_mode", None)
    if callable(set_mode): set_mode("eval")
    names = ("decision_frame", "state_frame", "reset", "terminated", "truncated", "processed_observation", "rms_observation", "policy_mean", "final_action", "base_offset_before", "base_offset_after", "joint_offset_before", "joint_offset_after", "reference_active_target", "corrected_active_target", "applied_active_target", "full_target", "hand_qpos", "hand_qvel", "object_position", "object_quaternion_xyzw", "object_linear_velocity", "target_object_position", "target_object_quaternion_xyzw")
    obs, _ = env.reset(); store: dict[str, list[np.ndarray]] = {name: [] for name in names}
    if int(env._current_frame) != RESET_DECISION_FRAME:
        raise RuntimeError(
            f"MuJoCo reset policy frame must be {RESET_DECISION_FRAME}, got {env._current_frame}"
        )
    for _step in range(steps):
        decision_frame = int(env._current_frame)
        before_base, before_joint = env.base_position_offset.copy(), env.joint_offset.copy()
        tensor = torch.as_tensor(obs[None], dtype=torch.float32, device=agent.policy.device)
        normalized = agent._observation_preprocessor(tensor)
        with torch.no_grad(): mean, _ = agent.policy.compute({"observations": normalized}, role="policy")
        action = torch.clamp(mean, -1, 1).detach().cpu().numpy()[0]
        next_obs, _, terminated, truncated, info = env.step(action)
        trace = env._last_action_trace
        if trace is None or trace["decision_frame"] != decision_frame:
            raise RuntimeError("environment did not expose authoritative action trace for this decision")
        state_frame = int(env._current_frame)
        full_target = np.asarray(trace["full_target"], dtype=np.float32)
        corrected = np.asarray(trace["corrected_active_target"], dtype=np.float32)
        applied = np.asarray(trace["applied_active_target"], dtype=np.float32)
        _append(store, decision_frame=np.asarray(decision_frame), state_frame=np.asarray(state_frame), reset=np.asarray(decision_frame == RESET_DECISION_FRAME), terminated=np.asarray(terminated), truncated=np.asarray(truncated),
          processed_observation=obs, rms_observation=normalized.detach().cpu().numpy()[0], policy_mean=mean.detach().cpu().numpy()[0], final_action=action,
          base_offset_before=before_base, base_offset_after=env.base_position_offset, joint_offset_before=before_joint, joint_offset_after=env.joint_offset,
          reference_active_target=np.asarray(trace["reference_active_target"], dtype=np.float32), corrected_active_target=corrected, applied_active_target=applied, full_target=full_target,
          hand_qpos=full_dof_from_qpos(np.asarray(env._dx.qpos), env.ids), hand_qvel=_ordered_qvel(env), object_position=np.asarray(info["object_position"]), object_quaternion_xyzw=np.asarray(env._dx.xquat[env.object_body_id], dtype=np.float32)[[1, 2, 3, 0]], object_linear_velocity=np.asarray(env._dx.qvel[env.ids.object_free_dof_addr:env.ids.object_free_dof_addr + 3]), target_object_position=np.asarray(info["target_object_position"]), target_object_quaternion_xyzw=np.asarray(env.trajectory.object_quat_xyzw[state_frame]))
        obs = next_obs
        if terminated or truncated: break
    scalar_types = {"decision_frame": np.int64, "state_frame": np.int64, "reset": np.bool_, "terminated": np.bool_, "truncated": np.bool_}
    arrays = {key: np.asarray(values, dtype=scalar_types.get(key, np.float32)) for key, values in store.items()}
    metadata = {"schema": SCHEMA, "backend": "mujoco", "identity": EXACT_IDENTITY, "post_filters": ["velocity_clamp", "position_clamp"], "raw_policy_observation": "processed 461D only; component schema is obs_layout.py", "checkpoint": str(checkpoint)}
    validate_trace(metadata, arrays); output.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(output, metadata=np.asarray(json.dumps(metadata)), **arrays); return output


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--output", type=Path, required=True); p.add_argument("--steps", type=int, default=3, help=f"actions to trace (1..{MAX_ACTIONS})"); p.add_argument("--checkpoint", type=Path, default=CHECKPOINT); a = p.parse_args()
    print(run(a.output, a.steps, a.checkpoint)); return 0
if __name__ == "__main__": raise SystemExit(main())
