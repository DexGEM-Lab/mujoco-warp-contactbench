from __future__ import annotations

from pathlib import Path

import pytest

import sim.manorl.trajectory as trajectory_module
from sim.manorl.trajectory import TrajectorySelection, load_assigned_trajectory_batch


@pytest.mark.parametrize("version", (0, -1, True, 1.5, "69"))
def test_trajectory_selection_rejects_invalid_dataset_version(version: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        TrajectorySelection(expected_dataset_version=version)  # type: ignore[arg-type]


def test_assigned_loader_opens_exact_lance_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import lance

    path = tmp_path / "growing.lance"
    path.mkdir()
    calls: list[tuple[str, int | None]] = []

    class FakeDataset:
        version = 69

    def fake_dataset(uri: str, *, version: int | None = None) -> FakeDataset:
        calls.append((uri, version))
        return FakeDataset()

    class DiscoveryReached(Exception):
        pass

    def stop_after_open(*_args, **_kwargs):
        raise DiscoveryReached

    monkeypatch.setattr(lance, "dataset", fake_dataset, raising=False)
    monkeypatch.setattr(
        trajectory_module, "_discover_trajectory_candidates", stop_after_open
    )

    selection = TrajectorySelection(
        dataset_path=path,
        expected_dataset_version=69,
    )
    with pytest.raises(DiscoveryReached):
        load_assigned_trajectory_batch(selection, num_envs=1)

    assert calls == [(str(path), 69)]
