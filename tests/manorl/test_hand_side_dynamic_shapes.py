"""Focused contracts for side discovery and revised dynamic ManoRL widths."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sim.manorl.contracts import JOINT_DOF
from sim.manorl.environment import (
    minimum_warp_contact_capacity,
    recommended_warp_contact_capacity,
)
from sim.manorl.observations import observation_layout, observation_layout_for_dimension
from sim.manorl.trajectory import (
    TrajectorySelection,
    detect_hand_sides,
    resolve_hand_selection,
    trajectory_from_lance_row,
)


def _modern_row(*, hand_names: list[str], dof: int = JOINT_DOF) -> dict[str, object]:
    """Build a small in-memory Lance row with deliberately distinct sides."""

    frames = 4
    timestamps = np.arange(frames, dtype=np.float64) * 0.01
    hands = [
        {"urdf_dof": np.full((frames, dof), float(index + 1), dtype=np.float64)}
        for index, _ in enumerate(hand_names)
    ]
    return {
        "index": {"scene": "cube1", "gesture": "001-Pinch", "uuid": "synthetic"},
        "trajectory_metadata": {
            "object_names": ["cube1"],
            "hand_names": hand_names,
            "total_frames": frames,
            "data_fps": 100,
            "trajectory_info": {
                "object_move": [{"object_name": "cube1", "start_frame": 0, "end_frame": frames - 1}]
            },
        },
        "timestamp": timestamps,
        "hands": hands,
        "objects": [
            {
                "pos": np.repeat([[0.0, 0.0, 1.0]], frames, axis=0),
                "rot_aa": np.zeros((frames, 3), dtype=np.float64),
            }
        ],
    }


@pytest.mark.parametrize(
    ("names", "expected"),
    [
        (["right"], ("right",)),
        (["left"], ("left",)),
        (["r_hand", "lhand"], ("left", "right")),
    ],
)
def test_detect_hand_sides_is_metadata_driven_and_canonical(names, expected) -> None:
    assert detect_hand_sides({"trajectory_metadata": {"hand_names": names}}) == expected


def test_detect_hand_sides_rejects_ambiguous_metadata() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        detect_hand_sides({"trajectory_metadata": {"hand_names": ["left", "lhand"]}})
    with pytest.raises(ValueError, match="unsupported"):
        detect_hand_sides({"trajectory_metadata": {"hand_names": ["middle"]}})
    with pytest.raises(ValueError, match="non-empty"):
        detect_hand_sides({"trajectory_metadata": {"hand_names": []}})


@pytest.mark.parametrize(
    ("available", "selection", "expected"),
    [
        (("right",), "auto", ("right",)),
        (("left", "right"), "auto", ("left", "right")),
        (("left", "right"), "both", ("left", "right")),
        (("left", "right"), "r", ("right",)),
    ],
)
def test_hand_selection_supports_auto_both_and_explicit_side(available, selection, expected) -> None:
    assert resolve_hand_selection(available, selection) == expected
    if selection == "both":
        with pytest.raises(ValueError, match="dataset"):
            resolve_hand_selection(("right",), selection)


def test_trajectory_side_mapping_does_not_assume_hands_list_order() -> None:
    # Metadata says the first row is left, even though the list is left/right;
    # values make an accidental right-first interpretation observable.
    row = _modern_row(hand_names=["left", "right"])
    trajectory = trajectory_from_lance_row(row, dataset_version=7, row_index=3)
    assert len(trajectory.q_ref) == 4
    assert trajectory.identity.source_stop == 4
    assert trajectory.hand_sides == ("left", "right")
    assert trajectory.selected_hand_sides == ("left", "right")
    assert trajectory.action_layout.action_dim == 2 * JOINT_DOF
    np.testing.assert_array_equal(trajectory.q_ref_for("left"), 1.0)
    np.testing.assert_array_equal(trajectory.q_ref_for("right"), 2.0)
    # The historical primary q_ref remains right-hand when both are present.
    np.testing.assert_array_equal(trajectory.q_ref, trajectory.q_ref_for("right"))

    left_only = trajectory_from_lance_row(row, dataset_version=7, hand_side="left")
    assert left_only.selected_hand_sides == ("left",)
    assert left_only.action_layout.action_dim == JOINT_DOF
    assert left_only.action_layout.reference_sides == ("right",)


def test_modern_row_without_movement_metadata_uses_the_full_inclusive_range() -> None:
    row = _modern_row(hand_names=["right"])
    row["trajectory_metadata"]["trajectory_info"]["object_move"] = []

    trajectory = trajectory_from_lance_row(row, dataset_version=7)

    assert trajectory.identity.source_start == 0
    assert trajectory.identity.source_stop == 4
    assert trajectory.identity.movement_start_raw == 0
    assert trajectory.identity.movement_end_raw == 3


def test_modern_row_rejects_out_of_bounds_inclusive_movement_range() -> None:
    row = _modern_row(hand_names=["right"])
    row["trajectory_metadata"]["trajectory_info"]["object_move"][0][
        "end_frame"
    ] = 4

    with pytest.raises(ValueError, match="inclusive interval"):
        trajectory_from_lance_row(row, dataset_version=7)


@pytest.mark.parametrize(
    ("names", "selection", "selected", "action_dim", "reference_sides"),
    [
        (["left"], "auto", ("left",), JOINT_DOF, ()),
        (["right"], "auto", ("right",), JOINT_DOF, ()),
        (["left", "right"], "auto", ("left", "right"), 2 * JOINT_DOF, ()),
        (["left", "right"], "right", ("right",), JOINT_DOF, ("left",)),
        (["left", "right"], "left", ("left",), JOINT_DOF, ("right",)),
    ],
)
def test_synthetic_rows_resolve_controlled_and_reference_following_sides(
    names,
    selection,
    selected,
    action_dim,
    reference_sides,
) -> None:
    trajectory = trajectory_from_lance_row(
        _modern_row(hand_names=names),
        dataset_version=7,
        hand_side=selection,
    )
    assert trajectory.selected_hand_sides == selected
    assert trajectory.action_layout.action_dim == action_dim
    assert trajectory.action_layout.reference_sides == reference_sides


def test_trajectory_selection_normalizes_string_dataset_path() -> None:
    selection = TrajectorySelection(dataset_path="/tmp/synthetic.lance")
    assert selection.dataset_path == Path("/tmp/synthetic.lance")


def test_contact_capacity_scales_with_every_available_hand() -> None:
    assert minimum_warp_contact_capacity(1, ("right",)) == 64
    assert minimum_warp_contact_capacity(1, ("left",)) == 64
    assert minimum_warp_contact_capacity(1, ("right", "left")) == 128
    assert recommended_warp_contact_capacity(1, ("right",)) == 128
    assert recommended_warp_contact_capacity(1, ("right", "left")) == 192
    assert recommended_warp_contact_capacity(4, ("right", "left")) == 576


@pytest.mark.parametrize(
    ("dof", "hands", "expected"),
    [(26, 1, 476), (26, 2, 499), (28, 1, 480), (28, 2, 505)],
)
def test_observation_layout_resolves_one_and_two_hand_widths(dof, hands, expected) -> None:
    layout = observation_layout(dof, cumulative_joint_dim=(dof - 6) * hands)
    assert layout.dimension == expected
    assert layout.hand_count == hands
    assert (
        layout.slices["cumulative_offset"].stop
        - layout.slices["cumulative_offset"].start
        == 3 * hands
    )
    assert observation_layout_for_dimension(expected) == layout
    points = layout.slices["object_point_cloud_raw"]
    assert points.stop - points.start == 64 * 3


@pytest.mark.parametrize(("observation_dim", "action_dim"), [(480, 28), (505, 56)])
def test_gym_model_and_normalizer_follow_dynamic_spaces(observation_dim: int, action_dim: int) -> None:
    gym = pytest.importorskip("gymnasium")
    pytest.importorskip("skrl")
    torch = pytest.importorskip("torch")

    from sim.manorl.gymnasium_env import ManoGymnasiumVectorEnv
    from sim.manorl.model import ManoActorCritic
    from sim.manorl.normalization import PointCloudAwareRunningStandardScaler

    class FakePhysicalEnvironment:
        """Minimal typed stand-in; no simulator/data files are needed here."""

        # The adapter checks this concrete class, so inherit without invoking
        # its expensive MJX constructor.
        pass

    # Construct a real environment instance without running physics.  This
    # keeps the test focused on Gym's shape boundary and works in CPU CI.
    from sim.manorl.environment import MujocoManoEnvironment

    physical = object.__new__(MujocoManoEnvironment)
    physical.config = SimpleNamespace(num_envs=2, device="cpu")
    physical.action_dim = action_dim
    physical.observation_dim = observation_dim
    adapter = ManoGymnasiumVectorEnv(physical)
    assert adapter.single_action_space.shape == (action_dim,)
    assert adapter.single_observation_space.shape == (observation_dim,)

    model = ManoActorCritic(
        gym.spaces.Box(-5.0, 5.0, shape=(observation_dim,), dtype=np.float32),
        None,
        gym.spaces.Box(-1.0, 1.0, shape=(action_dim,), dtype=np.float32),
        device="cpu",
    )
    observations = torch.zeros((2, observation_dim), dtype=torch.float32)
    observations[:, 266] = 1.0
    policy, _ = model.compute({"observations": observations}, role="policy")
    value, _ = model.compute({"observations": observations}, role="value")
    assert tuple(policy.shape) == (2, action_dim)
    assert tuple(value.shape) == (2, 1)

    layout = observation_layout_for_dimension(observation_dim)
    scaler = PointCloudAwareRunningStandardScaler(size=observation_dim)
    points = torch.arange(64 * 3, dtype=torch.float32).reshape(64, 3)
    observations[0, layout.slices["object_point_cloud_raw"]] = points.flatten()
    observations[1, layout.slices["object_point_cloud_raw"]] = (points + 1).flatten()
    normalized = scaler(observations, train=True)
    assert tuple(normalized.shape) == (2, observation_dim)
    assert torch.isfinite(normalized).all()
    np.testing.assert_allclose(
        scaler.running_mean[layout.slices["object_point_cloud_raw"]]
        .reshape(64, 3)
        .detach()
        .cpu()
        .numpy(),
        np.broadcast_to(
            scaler.pc_running_mean.detach().cpu().numpy(),
            (64, 3),
        ),
    )


@pytest.mark.parametrize(
    ("observation_dim", "action_dim"),
    [(476, 28), (480, 26), (499, 56), (505, 28)],
)
def test_model_rejects_mismatched_action_and_observation_layouts(
    observation_dim: int, action_dim: int
) -> None:
    gym = pytest.importorskip("gymnasium")
    pytest.importorskip("skrl")

    from sim.manorl.model import ManoActorCritic

    with pytest.raises(ValueError, match="action and observation layouts do not match"):
        ManoActorCritic(
            gym.spaces.Box(
                -5.0, 5.0, shape=(observation_dim,), dtype=np.float32
            ),
            None,
            gym.spaces.Box(-1.0, 1.0, shape=(action_dim,), dtype=np.float32),
            device="cpu",
        )
