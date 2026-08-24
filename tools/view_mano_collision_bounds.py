#!/usr/bin/env python3
"""Inspect a MANO MJCF's MuJoCo collision boundaries interactively.

The selected geom group is recolored red and shown together with the MANO skin.
For mesh geoms, MuJoCo's ``mjVIS_CONVEXHULL`` option shows the convex hull used
by collision detection rather than the source mesh surface.

Example:

    DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority \
    python tools/view_mano_collision_bounds.py \
      /path/to/mano_left_hand_floating.xml

Controls:
    1   collision boundary only
    2   collision boundary + MANO skin (default)
    3   MANO skin only
    H   toggle MuJoCo convex hull / source mesh surface
    W   toggle wireframe rendering
    S   toggle MANO skin
    C   toggle collision geoms
    R   reset the hand pose
    P   print the current qpos
    Q   close the viewer
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import os
from pathlib import Path
import threading
import time
from typing import Any

import mujoco
import numpy as np


DEFAULT_COLLISION_GROUP = 3
DEFAULT_COLLISION_RGBA = (1.0, 0.12, 0.04, 0.78)
DEFAULT_SKIN_ALPHA = 0.24


@dataclass(frozen=True)
class ModelSummary:
    """Collision geometry facts extracted from a compiled MuJoCo model."""

    xml_path: Path
    collision_group: int
    collision_ids: tuple[int, ...]
    collision_names: tuple[str, ...]
    mesh_collision_count: int
    contact_disabled_count: int


def _geom_name(model: mujoco.MjModel, geom_id: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    return name if name is not None else f"geom_{geom_id}"


def inspect_model(
    model: mujoco.MjModel,
    *,
    xml_path: Path,
    collision_group: int,
) -> ModelSummary:
    """Identify the collision geoms selected by the MJCF visualization group."""

    collision_ids = tuple(
        int(geom_id)
        for geom_id in np.flatnonzero(model.geom_group == collision_group)
    )
    if not collision_ids:
        groups = sorted({int(group) for group in model.geom_group})
        raise ValueError(
            f"no geoms belong to group {collision_group}; present geom groups: {groups}"
        )

    mesh_type = int(mujoco.mjtGeom.mjGEOM_MESH)
    mesh_count = sum(int(model.geom_type[geom_id]) == mesh_type for geom_id in collision_ids)
    contact_disabled_count = sum(
        int(model.geom_contype[geom_id]) == 0
        and int(model.geom_conaffinity[geom_id]) == 0
        for geom_id in collision_ids
    )
    return ModelSummary(
        xml_path=xml_path,
        collision_group=collision_group,
        collision_ids=collision_ids,
        collision_names=tuple(_geom_name(model, geom_id) for geom_id in collision_ids),
        mesh_collision_count=mesh_count,
        contact_disabled_count=contact_disabled_count,
    )


def _print_summary(model: mujoco.MjModel, summary: ModelSummary) -> None:
    print(f"MJCF: {summary.xml_path}")
    print(
        "Compiled model: "
        f"{model.nbody} bodies, {model.njnt} joints, {model.ngeom} geoms, "
        f"{model.nskin} skins"
    )
    print(
        f"Collision group {summary.collision_group}: "
        f"{len(summary.collision_ids)} geoms "
        f"({summary.mesh_collision_count} mesh geoms)"
    )
    print("Collision geoms: " + ", ".join(summary.collision_names))
    if summary.contact_disabled_count:
        print(
            "Contact-mask note: "
            f"{summary.contact_disabled_count}/{len(summary.collision_ids)} selected geoms "
            "have contype=0 and conaffinity=0. Their boundaries are visible here, "
            "but those geoms cannot generate contacts in this MJCF."
        )


@dataclass
class _ViewState:
    collision_visible: bool = True
    skin_visible: bool = True
    convex_hull: bool = True
    wireframe: bool = False
    reset_requested: bool = False
    print_qpos_requested: bool = False
    close_requested: bool = False
    revision: int = 0


class _Controls:
    """Thread-safe state shared by the viewer key callback and main loop."""

    def __init__(self, state: _ViewState) -> None:
        self._state = state
        self._lock = threading.Lock()

    def on_key(self, keycode: int) -> None:
        try:
            key = chr(keycode).upper()
        except (ValueError, OverflowError):
            return
        with self._lock:
            state = self._state
            if key == "1":
                state.collision_visible = True
                state.skin_visible = False
            elif key == "2":
                state.collision_visible = True
                state.skin_visible = True
            elif key == "3":
                state.collision_visible = False
                state.skin_visible = True
            elif key == "H":
                state.convex_hull = not state.convex_hull
            elif key == "W":
                state.wireframe = not state.wireframe
            elif key == "S":
                state.skin_visible = not state.skin_visible
            elif key == "C":
                state.collision_visible = not state.collision_visible
            elif key == "R":
                state.reset_requested = True
            elif key == "P":
                state.print_qpos_requested = True
            elif key == "Q":
                state.close_requested = True
            else:
                return
            state.revision += 1

    def snapshot(self) -> _ViewState:
        with self._lock:
            state = self._state
            snapshot = _ViewState(**vars(state))
            state.reset_requested = False
            state.print_qpos_requested = False
            return snapshot


def _apply_view_state(
    viewer: Any,
    *,
    collision_group: int,
    state: _ViewState,
) -> None:
    viewer.opt.geomgroup[collision_group] = int(state.collision_visible)
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_SKIN] = int(state.skin_visible)
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = int(state.convex_hull)
    viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = int(state.wireframe)


def _describe_view(state: _ViewState) -> str:
    geometry = "convex hull" if state.convex_hull else "source mesh"
    return (
        f"view: collision={'on' if state.collision_visible else 'off'}, "
        f"skin={'on' if state.skin_visible else 'off'}, "
        f"geometry={geometry}, wireframe={'on' if state.wireframe else 'off'}"
    )


def _require_graphical_session() -> None:
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return
    raise RuntimeError(
        "MuJoCo viewer needs an X11 or Wayland session; set DISPLAY "
        "(for example DISPLAY=:1) before launching"
    )


def view_collision_bounds(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    summary: ModelSummary,
    collision_rgba: tuple[float, float, float, float],
    skin_alpha: float,
    azimuth: float,
    elevation: float,
    distance: float | None,
    duration: float | None,
) -> None:
    """Open a passive MuJoCo viewer configured for boundary inspection."""

    from mujoco import viewer as mujoco_viewer

    _require_graphical_session()

    collision_ids = np.asarray(summary.collision_ids, dtype=np.int64)
    model.geom_rgba[collision_ids] = collision_rgba
    if model.nskin:
        model.skin_rgba[:, 3] = skin_alpha

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    controls = _Controls(_ViewState())
    print(
        "Controls: 1 collision | 2 overlay | 3 skin | H hull/mesh | "
        "W wireframe | S skin | C collision | R reset | P qpos | Q close",
        flush=True,
    )

    started = time.monotonic()
    with mujoco_viewer.launch_passive(
        model,
        data,
        key_callback=controls.on_key,
        show_left_ui=True,
        show_right_ui=True,
    ) as viewer:
        # Hide unrelated geom groups so the selected collision representation is
        # unambiguous. Skin visibility is controlled separately by mjVIS_SKIN.
        with viewer.lock():
            viewer.opt.geomgroup[:] = 0
            state = controls.snapshot()
            _apply_view_state(
                viewer,
                collision_group=summary.collision_group,
                state=state,
            )
            viewer.cam.azimuth = azimuth
            viewer.cam.elevation = elevation
            viewer.cam.distance = (
                distance if distance is not None else max(0.25, 1.6 * model.stat.extent)
            )
            viewer.cam.lookat[:] = model.stat.center
        viewer.sync()
        print(_describe_view(state), flush=True)
        applied_revision = state.revision

        while viewer.is_running():
            state = controls.snapshot()
            if state.close_requested:
                viewer.close()
                break
            if duration is not None and time.monotonic() - started >= duration:
                viewer.close()
                break

            if state.revision != applied_revision:
                with viewer.lock():
                    if state.reset_requested:
                        mujoco.mj_resetData(model, data)
                        mujoco.mj_forward(model, data)
                    _apply_view_state(
                        viewer,
                        collision_group=summary.collision_group,
                        state=state,
                    )
                viewer.sync()
                if state.print_qpos_requested:
                    print(
                        "qpos="
                        + np.array2string(
                            data.qpos,
                            precision=5,
                            suppress_small=True,
                            max_line_width=240,
                        ),
                        flush=True,
                    )
                else:
                    print(_describe_view(state), flush=True)
                applied_revision = state.revision
            else:
                # sync() applies camera/UI/perturbation changes made in the native
                # viewer without advancing this gravity-free inspection model.
                viewer.sync()
            time.sleep(1.0 / 60.0)


def _unit_interval(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be a finite number in [0, 1]")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xml", type=Path, help="MANO MJCF/XML model to inspect")
    parser.add_argument(
        "--collision-group",
        type=int,
        default=DEFAULT_COLLISION_GROUP,
        choices=range(6),
        metavar="{0..5}",
        help="MJCF geom group containing collision geoms (default: 3)",
    )
    parser.add_argument(
        "--collision-alpha",
        type=_unit_interval,
        default=DEFAULT_COLLISION_RGBA[3],
        help="collision boundary opacity (default: 0.78)",
    )
    parser.add_argument(
        "--skin-alpha",
        type=_unit_interval,
        default=DEFAULT_SKIN_ALPHA,
        help="MANO skin opacity in overlay mode (default: 0.24)",
    )
    parser.add_argument("--azimuth", type=float, default=135.0)
    parser.add_argument("--elevation", type=float, default=-20.0)
    parser.add_argument(
        "--distance",
        type=_positive_float,
        default=None,
        help="initial camera distance (default: derived from model extent)",
    )
    parser.add_argument(
        "--duration",
        type=_positive_float,
        default=None,
        help="close automatically after this many seconds (default: run until closed)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="compile and inspect the model without opening a window",
    )
    args = parser.parse_args(argv)
    args.xml = args.xml.expanduser().resolve()
    if not args.xml.is_file():
        parser.error(f"MJCF does not exist: {args.xml}")
    for name in ("azimuth", "elevation"):
        if not math.isfinite(getattr(args, name)):
            parser.error(f"--{name} must be finite")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model = mujoco.MjModel.from_xml_path(str(args.xml))
    data = mujoco.MjData(model)
    summary = inspect_model(
        model,
        xml_path=args.xml,
        collision_group=args.collision_group,
    )
    _print_summary(model, summary)
    if args.check:
        return 0

    view_collision_bounds(
        model,
        data,
        summary=summary,
        collision_rgba=(
            DEFAULT_COLLISION_RGBA[0],
            DEFAULT_COLLISION_RGBA[1],
            DEFAULT_COLLISION_RGBA[2],
            args.collision_alpha,
        ),
        skin_alpha=args.skin_alpha,
        azimuth=args.azimuth,
        elevation=args.elevation,
        distance=args.distance,
        duration=args.duration,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
