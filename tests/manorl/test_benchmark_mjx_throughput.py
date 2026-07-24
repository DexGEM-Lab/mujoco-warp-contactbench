from __future__ import annotations

import pytest

from sim.manorl.contracts import PHYSICS_SUBSTEPS_PER_TARGET
from tools import benchmark_manorl_mjx_throughput as benchmark


def test_throughput_reports_control_and_substep_rates() -> None:
    metrics = benchmark._throughput(elapsed_seconds=2.0, control_steps=10, num_envs=4)

    assert metrics["batch_control_steps_per_second"] == 5.0
    assert metrics["per_env_control_steps_per_second"] == 5.0
    assert metrics["aggregate_world_control_steps_per_second"] == 20.0
    assert metrics["aggregate_physics_substeps_per_second"] == 20.0 * PHYSICS_SUBSTEPS_PER_TARGET
    assert metrics["physics_substeps_per_control_step"] == PHYSICS_SUBSTEPS_PER_TARGET


def test_physics_loop_executes_exactly_two_substeps_per_control_step() -> None:
    calls: list[int] = []
    synchronized: list[int] = []
    ticks = iter((100.0, 102.5))

    def step_fn(data: int) -> int:
        calls.append(data)
        return data + 1

    data, elapsed = benchmark._run_physics_loop(
        0,
        step_fn=step_fn,
        synchronize=synchronized.append,
        control_steps=3,
        clock=lambda: next(ticks),
    )

    assert data == 3 * PHYSICS_SUBSTEPS_PER_TARGET
    assert len(calls) == 3 * PHYSICS_SUBSTEPS_PER_TARGET
    assert synchronized == [0, data]
    assert elapsed == 2.5


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--num-envs", "0"], "num-envs"),
        (["--measurement-steps", "0"], "measurement-steps"),
        (["--dataset-version", "0"], "dataset-version"),
        (["--warp-persistent-ccd-workspace"], "unified-object-batch"),
        (
            ["--unified-object-batch", "--warp-persistent-ccd-workspace"],
            "warp-ccd-contacts-per-world",
        ),
        (
            [
                "--device", "cpu", "--unified-object-batch", "--warp-persistent-ccd-workspace",
                "--warp-ccd-contacts-per-world", "8",
            ],
            "device gpu",
        ),
    ],
)
def test_cli_validation_rejects_invalid_benchmark_config(arguments: list[str], message: str) -> None:
    parser = benchmark._parser()
    args = parser.parse_args([*arguments, "--output", "result.json"])

    with pytest.raises(ValueError, match=message):
        benchmark._validate_args(args)


def test_validation_accepts_persistent_unified_gpu_config() -> None:
    parser = benchmark._parser()
    args = parser.parse_args(
        [
            "--device", "gpu", "--selector", "all", "--unified-object-batch",
            "--warp-persistent-ccd-workspace", "--warp-ccd-contacts-per-world", "8",
            "--output", "result.json",
        ]
    )

    benchmark._validate_args(args)
