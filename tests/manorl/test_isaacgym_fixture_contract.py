from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


def _fixture_tool():
    path = Path(__file__).resolve().parents[2] / "tools" / "isaacgym_reference_trace.py"
    spec = importlib.util.spec_from_file_location("isaacgym_reference_trace", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop(spec.name, None)


def test_semantic_fixture_uses_current_source_residual_contract() -> None:
    tool = _fixture_tool()
    trace_updates = tool.task_updates_for_mode(semantic_only=False)
    semantic_updates = tool.task_updates_for_mode(semantic_only=True)

    assert trace_updates["task.env.useResidualActions"] is False
    assert trace_updates["task.env.earlyPhaseMocapSteps"] == 0
    assert semantic_updates["task.env.useResidualActions"] is True
    assert semantic_updates["task.env.earlyPhaseMocapSteps"] == 100
    assert semantic_updates["task.env.maxDeviationDistance"] == tool.REFERENCE_MAX_DEVIATION_DISTANCE


def test_semantic_output_prefix_stays_target_absolute_after_source_chdir(tmp_path: Path) -> None:
    tool = _fixture_tool()
    output = tmp_path / "target" / "semantic_fixture"
    paths = tool.artifact_paths(output)
    assert paths.prefix.is_absolute()
    assert tool.SOURCE_REPO not in paths.prefix.parents


def test_semantic_fixture_schema_brackets_boundaries_and_requires_full_state() -> None:
    tool = _fixture_tool()
    assert {249, 250, 542, 543, 789, 790}.issubset(tool.SEMANTIC_REGULAR_CALLS)
    assert tool.SEMANTIC_REGULAR_CALLS.index(99) + 1 == tool.SEMANTIC_REGULAR_CALLS.index(100)
    required_fields = {
        "point_cloud_local_template",
        "point_cloud_scale",
        "object_support_points",
        "table_surface_height",
        "table_clearance_support_point_cap",
        "expected_contact_weights",
        "rotation_disabled_mask",
        "early_phase_start",
        "action_id",
        "active_joint_mask",
        "action_cumulative_offset",
        "controller_target",
        "sample_kind",
        "pulse_case",
    }
    assert required_fields.issubset(tool.SEMANTIC_ARRAY_SPECS)
    assert {99, 100, 101, 102}.issubset(tool.SEMANTIC_REGULAR_CALLS)

    arrays = {
        name: np.zeros((tool.SEMANTIC_SAMPLE_COUNT, *shape), dtype=dtype)
        for name, (shape, dtype) in tool.SEMANTIC_ARRAY_SPECS.items()
    }
    arrays["physical_call"][:] = [*tool.SEMANTIC_REGULAR_CALLS, tool.SEMANTIC_DELAYED_RESET_CALL]
    arrays["sample_kind"][-1] = tool.SEMANTIC_SAMPLE_KIND_DELAYED_RESET_POST
    for call, pulse in {
        99: tool.PULSE_EARLY_MASK,
        100: tool.PULSE_TRANSITION,
        101: tool.PULSE_ACCUMULATE_POSITIVE,
        102: tool.PULSE_ACCUMULATE_NEGATIVE,
    }.items():
        arrays["pulse_case"][list(tool.SEMANTIC_REGULAR_CALLS).index(call)] = pulse
    metadata = {
        "schema": tool.SEMANTIC_FIXTURE_SCHEMA,
        "regular_sample_calls": list(tool.SEMANTIC_REGULAR_CALLS),
        "trajectory_identity": {"identity": tool.TRAJECTORY_NAME},
        "termination_discriminator": {
            "cases": {name: {} for name in (
                "below_threshold", "above_threshold", "early_above_threshold", "trajectory_complete"
            )}
        },
        "configuration": {"useResidualActions": True, "earlyPhaseMocapSteps": 100},
        "pulse_cases": {
            str(tool.PULSE_NONE): "zero",
            str(tool.PULSE_EARLY_MASK): "early",
            str(tool.PULSE_TRANSITION): "transition",
            str(tool.PULSE_ACCUMULATE_POSITIVE): "positive",
            str(tool.PULSE_ACCUMULATE_NEGATIVE): "negative",
        },
    }
    tool.validate_semantic_fixture_schema(metadata, arrays)
    missing = dict(arrays)
    missing.pop("action_id")
    with pytest.raises(ValueError, match="missing"):
        tool.validate_semantic_fixture_schema(metadata, missing)
