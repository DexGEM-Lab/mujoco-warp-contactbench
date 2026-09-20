from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.supervise_manorl_v5_v6_evaluation import (
    EvaluationConfig,
    Experiment,
    discover_periodic_checkpoints,
    probe_video,
    split_identities,
)


def _config(tmp_path: Path) -> EvaluationConfig:
    return EvaluationConfig(
        deployment=tmp_path / "deployment",
        output_root=tmp_path / "outputs",
        package=tmp_path / "package",
        python=tmp_path / "python",
        dexstream_root=tmp_path / "assets",
        poll_seconds=1,
        checkpoint_interval=50,
        final_update=200,
    )


def test_periodic_checkpoint_discovery_is_exact_and_ordered(tmp_path: Path) -> None:
    config = _config(tmp_path)
    experiment = Experiment("v5", "v5", "v5-formal", "session")
    root = config.output_root / experiment.output_name
    root.mkdir(parents=True)
    for update in (200, 50, 150, 100):
        (root / f"policy.update{update:06d}.pt").write_bytes(b"x")
    found = discover_periodic_checkpoints(config, experiment)
    assert [path.name for path in found] == [
        "policy.update000050.pt",
        "policy.update000100.pt",
        "policy.update000150.pt",
        "policy.update000200.pt",
    ]


def test_periodic_checkpoint_discovery_rejects_missing_updates(tmp_path: Path) -> None:
    config = _config(tmp_path)
    experiment = Experiment("v5", "v5", "v5-formal", "session")
    root = config.output_root / experiment.output_name
    root.mkdir(parents=True)
    for update in (50, 150, 200):
        (root / f"policy.update{update:06d}.pt").write_bytes(b"x")
    with pytest.raises(RuntimeError, match=r"missing=\[100\]"):
        discover_periodic_checkpoints(config, experiment)


def test_split_identities_uses_checkpoint_provenance_indices(tmp_path: Path) -> None:
    trace = tmp_path / "eval.json"
    trace.write_text(
        json.dumps(
            {
                "provenance": {
                    "checkpoint": {
                        "reference_assignment": {
                            "identities": ["train", "validation", "test"],
                        },
                        "identity_split": {
                            "validation_indices": [1],
                            "test_indices": [2],
                        },
                    }
                }
            }
        )
    )
    assert split_identities(trace) == {
        "validation": ("validation",),
        "test": ("test",),
    }


def test_video_probe_falls_back_to_full_imageio_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import imageio.v2 as imageio
    import numpy as np

    video = tmp_path / "video.mp4"
    with imageio.get_writer(video, fps=10, codec="libx264", macro_block_size=1) as writer:
        for value in range(3):
            writer.append_data(np.full((16, 24, 3), value, dtype=np.uint8))
    monkeypatch.setattr(
        "tools.supervise_manorl_v5_v6_evaluation.shutil.which",
        lambda _: None,
    )
    result = probe_video(video)
    assert result["verification_backend"] == "imageio"
    assert result["stream"]["width"] == 24
    assert result["stream"]["height"] == 16
    assert result["stream"]["nb_frames"] == 3
