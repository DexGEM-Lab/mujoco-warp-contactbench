from pathlib import Path

import sim.dexhandrl.skrl_train as skrl_train
from sim.dexhandrl.skrl_train import SkrlLauncherConfig, _resolve_trajectory_specs


def test_sequence_index_preserves_selected_lance_row(monkeypatch) -> None:
    rows = [
        {"object_name": "cube1", "action": "01", "uuid": "seq-0", "row_id": 16},
        {"object_name": "cube1", "action": "01", "uuid": "seq-1", "row_id": 22},
        {"object_name": "cube1", "action": "01", "uuid": "seq-2", "row_id": 56},
    ]
    monkeypatch.setattr(skrl_train, "list_trajectories", lambda *args, **kwargs: rows)
    cfg = SkrlLauncherConfig(
        lance_path=Path("unused.lance"),
        object_name="cube1",
        action="01",
        sequence_index=1,
        num_envs=1,
    )

    specs = _resolve_trajectory_specs(cfg)

    assert specs == [
        {"object_name": "cube1", "action": "01", "uuid": "seq-1", "row_id": 22, "sequence_index": 1}
    ]


def test_sequence_routing_wraps_across_vector_environments(monkeypatch) -> None:
    rows = [
        {"object_name": "cube1", "action": "01", "uuid": "seq-0", "row_id": 16},
        {"object_name": "cube1", "action": "01", "uuid": "seq-1", "row_id": 22},
    ]
    monkeypatch.setattr(skrl_train, "list_trajectories", lambda *args, **kwargs: rows)
    cfg = SkrlLauncherConfig(
        lance_path=Path("unused.lance"),
        object_name="cube1",
        action="01",
        sequence_index=1,
        num_envs=3,
    )

    specs = _resolve_trajectory_specs(cfg)

    assert [spec["uuid"] for spec in specs] == ["seq-1", "seq-0", "seq-1"]
    assert [spec["sequence_index"] for spec in specs] == [1, 0, 1]
