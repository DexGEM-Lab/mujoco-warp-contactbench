"""Hand-side discovery and action routing shared by trajectory and Gym code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from sim.manorl.contracts import (
    ACTION_SIDE_ORDER,
    JOINT_DOF,
    canonical_hand_sides,
    normalize_hand_side,
)


@dataclass(frozen=True)
class HandActionLayout:
    """Deterministic action slots for one- or two-hand trajectories.

    Slots are always emitted in right-then-left order, independent of the
    order used by a Lance row.  ``hand_side='auto'`` controls every available
    side; an explicit side leaves all other hands reference-following.
    """

    available_sides: tuple[str, ...]
    controlled_sides: tuple[str, ...]
    dof_per_hand: int = JOINT_DOF

    def __post_init__(self) -> None:
        available = canonical_hand_sides(self.available_sides)
        controlled = canonical_hand_sides(self.controlled_sides)
        if not set(controlled).issubset(available):
            missing = sorted(set(controlled) - set(available))
            raise ValueError(f"controlled hand side(s) are absent from dataset: {missing}")
        if self.dof_per_hand not in (26, JOINT_DOF):
            raise ValueError("dof_per_hand must be 26 or 28")
        object.__setattr__(self, "available_sides", available)
        # Preserve action order right,left while dropping unavailable slots.
        object.__setattr__(
            self,
            "controlled_sides",
            tuple(side for side in ACTION_SIDE_ORDER if side in controlled),
        )

    @classmethod
    def from_dataset(
        cls,
        available_sides: object,
        *,
        hand_side: str = "auto",
        dof_per_hand: int = JOINT_DOF,
    ) -> "HandActionLayout":
        available = canonical_hand_sides(available_sides)
        selection = normalize_hand_side(hand_side)
        if selection == "both" and set(available) != {"left", "right"}:
            raise ValueError(f"requested both hands; dataset has {available}")
        controlled = available if selection in {"auto", "both"} else (selection,)
        return cls(available, controlled, dof_per_hand=dof_per_hand)

    @property
    def action_dim(self) -> int:
        return len(self.controlled_sides) * self.dof_per_hand

    @property
    def cumulative_dim(self) -> int:
        return len(self.controlled_sides) * (self.dof_per_hand - 6)

    @property
    def reference_sides(self) -> tuple[str, ...]:
        return tuple(side for side in self.available_sides if side not in self.controlled_sides)

    def side_slice(self, side: str) -> slice:
        normalized = normalize_hand_side(side, allow_auto=False, allow_both=False)
        if normalized not in self.controlled_sides:
            raise KeyError(f"hand side {normalized!r} has no action slot")
        index = self.controlled_sides.index(normalized)
        start = index * self.dof_per_hand
        return slice(start, start + self.dof_per_hand)

    def cumulative_slice(self, side: str) -> slice:
        normalized = normalize_hand_side(side, allow_auto=False, allow_both=False)
        if normalized not in self.controlled_sides:
            raise KeyError(f"hand side {normalized!r} has no cumulative-action slot")
        width = self.dof_per_hand - 6
        index = self.controlled_sides.index(normalized)
        start = index * width
        return slice(start, start + width)

    def split(self, actions: NDArray[object]) -> dict[str, NDArray[np.float64]]:
        values = np.asarray(actions, dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != self.action_dim or not np.all(np.isfinite(values)):
            raise ValueError(f"actions must be finite with shape (batch, {self.action_dim})")
        return {side: values[:, self.side_slice(side)].copy() for side in self.controlled_sides}

    def merge(self, actions_by_side: Mapping[str, NDArray[object]], *, batch: int | None = None) -> NDArray[np.float64]:
        if batch is None:
            first = next(iter(actions_by_side.values()), None)
            if first is None:
                raise ValueError("actions_by_side cannot be empty")
            batch = int(np.asarray(first).shape[0])
        result = np.zeros((batch, self.action_dim), dtype=np.float64)
        for side in self.controlled_sides:
            if side not in actions_by_side:
                raise ValueError(f"missing action slot for {side}")
            values = np.asarray(actions_by_side[side], dtype=np.float64)
            if values.shape != (batch, self.dof_per_hand) or not np.all(np.isfinite(values)):
                raise ValueError(f"action slot {side} must have shape ({batch}, {self.dof_per_hand})")
            result[:, self.side_slice(side)] = values
        return result


def action_dim_for_sides(available_sides: object, *, hand_side: str = "auto", dof_per_hand: int = JOINT_DOF) -> int:
    """Convenience function used by config/Gym callers and tests."""

    return HandActionLayout.from_dataset(
        available_sides, hand_side=hand_side, dof_per_hand=dof_per_hand
    ).action_dim
