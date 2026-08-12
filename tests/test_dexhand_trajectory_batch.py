from pathlib import Path

import numpy as np
import pytest

import sim.dexhandrl.lance_loader as lance_loader
import sim.dexhandrl.skrl_train as skrl_train
from sim.dexhandrl.lance_loader import DexHandTrajectory
from sim.dexhandrl.skrl_train import SkrlLauncherConfig, _build_wrapped_env, _resolve_trajectory_specs
from sim.dexhandrl.trajectory_batch import DEXHAND_ALL_ACTIONS, pack_trajectory_batch


def _trajectory(row_id: int, action: str, frames: int, start: int, end: int, raw: bool = True) -> DexHandTrajectory:
    values = np.arange(frames, dtype=np.float32)
    return DexHandTrajectory(
        lance_path=Path("synthetic.lance"), uuid=f"uuid-{row_id}", row_id=row_id,
        object_name="cube1", action=action,
        hand_dof=np.repeat(values[:, None], 22, axis=1),
        hand_finger_dof_full_raw=np.repeat(values[:, None], 20, axis=1) if raw else None,
        object_pos=np.repeat(values[:, None], 3, axis=1),
        object_quat_xyzw=np.repeat(values[:, None], 4, axis=1),
        expected_contact_mask=np.ones((frames, 16), dtype=np.float32),
        movement_start_frame=start, movement_end_frame=end, metadata={},
    )


def test_bulk_loader_uses_stable_lance_row_ids_not_logical_offsets(monkeypatch) -> None:
    row_id = 120_259_084_300

    class FakeTable:
        def to_pylist(self):
            return [{"index": {"uuid": "stable-row"}}]

    class FakeDataset:
        schema = type("Schema", (), {"names": []})()

        def __init__(self):
            self.requested_row_ids = None

        def _take_rows(self, row_ids, *, columns):
            self.requested_row_ids = row_ids
            assert columns == ["index", "trajectory_metadata", "hands", "objects"]
            return FakeTable()

        def take(self, *args, **kwargs):
            raise AssertionError("public take interprets logical offsets and must not be called")

    dataset = FakeDataset()
    monkeypatch.setattr(lance_loader, "_import_lance", lambda: type("Lance", (), {"dataset": staticmethod(lambda _: dataset)})())
    monkeypatch.setattr(
        lance_loader,
        "_trajectory_from_row",
        lambda path, row, stable_row_id: (stable_row_id, row["index"]["uuid"]),
    )

    loaded = lance_loader.load_trajectories_by_row_ids(
        "synthetic.lance", [row_id], expected_uuids={row_id: "stable-row"}
    )

    assert dataset.requested_row_ids == [row_id]
    assert loaded == [(row_id, "stable-row")]


def test_bulk_loader_fails_closed_without_stable_row_id_api(monkeypatch) -> None:
    class FakeDataset:
        schema = type("Schema", (), {"names": []})()

    monkeypatch.setattr(
        lance_loader,
        "_import_lance",
        lambda: type("Lance", (), {"dataset": staticmethod(lambda _: FakeDataset())})(),
    )

    with pytest.raises(RuntimeError, match=r"stable row-ID bulk loading \(_take_rows\)"):
        lance_loader.load_trajectories_by_row_ids("synthetic.lance", [120_259_084_300])


def test_n8_evaluation_assignment_duplicates_world_zero_exactly() -> None:
    rows = []
    for action_index, action in enumerate(DEXHAND_ALL_ACTIONS):
        rows.append({
            "object_name": "cube1",
            "action": action,
            "uuid": f"uuid-{action}",
            "seed_uuid": f"seed-{action}",
            "row_id": 10 + action_index,
        })
    from sim.dexhandrl.trajectory_batch import resolve_balanced_action_assignments

    specs = resolve_balanced_action_assignments(
        rows,
        object_name="cube1",
        num_envs=8,
        duplicate_first_for_extra=True,
    )
    assert [spec["action"] for spec in specs[:7]] == list(DEXHAND_ALL_ACTIONS)
    assert specs[7]["row_id"] == specs[0]["row_id"]
    assert specs[7]["uuid"] == specs[0]["uuid"]
    assert specs[7]["sequence_index"] == specs[0]["sequence_index"]
    assert specs[7]["determinism_duplicate_of"] == 0


def test_n8_training_uses_identical_duplicate_identity_contract(monkeypatch) -> None:
    rows = [
        {
            "object_name": "cube1", "action": action,
            "uuid": f"uuid-{action}", "seed_uuid": f"seed-{action}",
            "row_id": 10 + action_index,
        }
        for action_index, action in enumerate(DEXHAND_ALL_ACTIONS)
    ]
    monkeypatch.setattr(skrl_train, "list_trajectories", lambda *args, **kwargs: rows)

    specs = _resolve_trajectory_specs(SkrlLauncherConfig(
        lance_path=Path("unused"), action="all", num_envs=8, sequence_index=0,
    ))

    assert [spec["action"] for spec in specs[:7]] == list(DEXHAND_ALL_ACTIONS)
    assert specs[7]["uuid"] == specs[0]["uuid"]
    assert specs[7]["row_id"] == specs[0]["row_id"]
    assert specs[7]["determinism_duplicate_of"] == 0


def test_all_action_assignment_is_balanced_sorted_and_deterministic(monkeypatch) -> None:
    rows = []
    for action_index, action in enumerate(DEXHAND_ALL_ACTIONS):
        # Deliberately out of order: assignment order must sort row_id then uuid.
        rows += [
            {"object_name": "cube1", "action": action, "uuid": f"z-{action}", "row_id": 90 + action_index},
            {"object_name": "cube1", "action": action, "uuid": f"a-{action}", "row_id": 10 + action_index},
        ]
    monkeypatch.setattr(skrl_train, "list_trajectories", lambda *args, **kwargs: rows)
    cfg = SkrlLauncherConfig(lance_path=Path("unused"), action="all", num_envs=1024, sequence_index=0)
    specs = _resolve_trajectory_specs(cfg)
    assert [spec["action"] for spec in specs[:7]] == list(DEXHAND_ALL_ACTIONS)
    counts = {action: sum(spec["action"] == action for spec in specs) for action in DEXHAND_ALL_ACTIONS}
    assert sorted(counts.values()) == [146, 146, 146, 146, 146, 147, 147]
    assert [specs[env_id]["row_id"] for env_id in (0, 7, 14)] == [10, 90, 10]
    assert specs == _resolve_trajectory_specs(cfg)


def test_packed_windows_preserve_sources_and_padding() -> None:
    root = Path("assets/isaac_source_root")
    trajectories = [_trajectory(1, "01", 5, 1, 3), _trajectory(2, "02", 7, 0, 2, raw=False)]
    assignments = [
        {"object_name": "cube1", "action": "01", "uuid": "uuid-1", "row_id": 1},
        {"object_name": "cube1", "action": "02", "uuid": "uuid-2", "row_id": 2},
    ]
    batch = pack_trajectory_batch(
        trajectories, assignments, pre_contact_frames=1, post_contact_frames=1,
        contact_bodies_file=root / "assets/all_assets/Assets/object_grasps_simple.yaml",
        task_config_path=root / "dexhand_env/cfg/task/Dexhand021proReconstruction.yaml",
    )
    assert batch.hand_active.shape == (2, 5, 22)
    assert batch.lengths.tolist() == [5, 4]
    assert batch.start_frames.tolist() == [0, 0]
    assert batch.movement_start_local.tolist() == [1, 0]
    assert batch.movement_end_local.tolist() == [3, 2]
    assert batch.hand_active[1, 4, 0] == batch.hand_active[1, 3, 0] == 3
    assert batch.object_pos[1, 4, 0] == 3
    assert not batch.lance_contact_annotation[1, 4].any()
    assert batch.hand_finger_dof_full_raw is not None
    assert batch.full_raw_validity[0].all() and not batch.full_raw_validity[1].any()
    assert batch.action_onehot50.shape == (2, 50)
    assert batch.task_expected_contact_masks[0].sum() > 0
    assert not np.array_equal(batch.task_expected_contact_masks[0], batch.lance_contact_annotation[0, 0])
    assert batch.active_joint_masks.shape == (2, 16)
    assert batch.active_fingertip_masks.shape == (2, 5)


def test_batch_fails_closed_without_contact_config() -> None:
    trajectory = _trajectory(1, "01", 4, 1, 2)
    with pytest.raises(RuntimeError, match="contact-body config is missing"):
        pack_trajectory_batch(
            [trajectory], [{"object_name": "cube1", "action": "01", "uuid": "uuid-1", "row_id": 1}],
            pre_contact_frames=0, post_contact_frames=0,
            contact_bodies_file=Path("missing.yaml"), task_config_path=Path("missing-task.yaml"),
        )


def test_all_action_sync_vector_route_is_explicitly_rejected() -> None:
    with pytest.raises(RuntimeError, match="device-resident DexHand trajectory-batch runtime"):
        _build_wrapped_env(SkrlLauncherConfig(action="all"))


def test_launcher_preserves_legacy_timestep_default_and_resolves_updates_explicitly() -> None:
    defaults = SkrlLauncherConfig()
    assert (defaults.timesteps, defaults.updates) == (250_000, None)

    divisible_timesteps = SkrlLauncherConfig(timesteps=144_000, rollouts=48)
    assert (divisible_timesteps.timesteps, divisible_timesteps.updates) == (144_000, 3000)

    args = skrl_train.build_arg_parser().parse_args(["--updates", "3000"])
    explicit_updates = SkrlLauncherConfig(
        timesteps=args.timesteps, updates=args.updates, rollouts=args.rollouts
    )
    assert (explicit_updates.updates, explicit_updates.timesteps) == (3000, 144_000)

    with pytest.raises(ValueError, match=r"updates \* rollouts"):
        SkrlLauncherConfig(updates=3, rollouts=48, timesteps=100)
