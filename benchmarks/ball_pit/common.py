from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

VIDEO_WIDTH = 960
VIDEO_HEIGHT = 544
VIDEO_FPS = 30
BALL_PALETTE_RGBA = (
    (0.05, 0.35, 0.95, 1.0),
    (0.95, 0.20, 0.16, 1.0),
    (0.10, 0.70, 0.25, 1.0),
    (0.96, 0.72, 0.08, 1.0),
    (0.75, 0.25, 0.95, 1.0),
    (0.00, 0.70, 0.78, 1.0),
)


@dataclass(frozen=True)
class BallPitSpec:
    case_name: str = "ball_pit_filled_tank_120_rigid_hand_sweep_10s"
    scenario: str = "filled_tank"
    ball_count: int = 120
    ball_radius: float = 0.03
    ball_mass: float = 0.003
    container_size: tuple[float, float, float] = (0.52, 0.36, 0.24)
    container_center: tuple[float, float, float] = (-0.08, 0.0, 0.12)
    seed: int = 42
    fps: float = 100.0
    substeps: int = 2
    settle_frames: int = 60
    rollout_frames: int = 1000
    duration_seconds: float = 10.0
    hand_base_start: tuple[float, float, float] = (0.030, -0.060, 0.030)
    hand_base_end: tuple[float, float, float] = (-0.130, 0.060, 0.030)
    hand_y_amplitude: float = 0.020
    hand_z_amplitude: float = 0.020
    hand_roll_amplitude: float = 0.18
    hand_pitch_amplitude: float = 0.12
    hand_yaw_amplitude: float = 0.65
    video_start_seconds: float = 0.0
    video_duration_seconds: float | None = None


def spec_from_args(args: Any) -> BallPitSpec:
    fps = float(args.fps)
    duration_seconds = float(getattr(args, "duration_seconds", BallPitSpec.duration_seconds))
    requested_frames = int(args.rollout_frames)
    if duration_seconds > 0.0:
        requested_frames = int(math.ceil(duration_seconds * fps))
    raw_video_duration = getattr(args, "video_duration_seconds", None)
    video_duration_seconds = None if raw_video_duration is None else max(0.0, float(raw_video_duration))
    return BallPitSpec(
        scenario=str(getattr(args, "scenario", BallPitSpec.scenario)),
        ball_count=int(args.ball_count),
        ball_radius=float(getattr(args, "ball_radius", BallPitSpec.ball_radius)),
        fps=fps,
        substeps=max(1, int(args.substeps)),
        settle_frames=int(args.settle_frames),
        rollout_frames=requested_frames,
        duration_seconds=max(0.0, duration_seconds),
        video_start_seconds=max(0.0, float(getattr(args, "video_start_seconds", 0.0))),
        video_duration_seconds=video_duration_seconds,
    )


def _lcg(seed: int) -> Iterable[float]:
    state = int(seed) & 0x7FFFFFFF
    while True:
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        yield state / 0x80000000


def ball_initial_positions(spec: BallPitSpec) -> list[tuple[float, float, float]]:
    """Return deterministic non-overlapping-ish ball starts inside the container."""
    sx, sy, sz = spec.container_size
    cx, cy, cz = spec.container_center
    r = spec.ball_radius
    margin = r * 1.15
    rng = _lcg(spec.seed)
    positions: list[tuple[float, float, float]] = []

    if spec.scenario == "filled_tank":
        spacing = 2.04 * r
        x_min = cx - sx / 2.0 + 1.15 * r
        x_max = cx + sx / 2.0 - 1.15 * r
        y_min = cy - sy / 2.0 + 1.15 * r
        y_max = cy + sy / 2.0 - 1.15 * r
        z_min = cz - sz / 2.0 + 1.15 * r
        z_max = cz + sz / 2.0 - 1.15 * r

        def axis_values(v_min: float, v_max: float) -> list[float]:
            count = max(1, int((v_max - v_min) // spacing) + 1)
            if count == 1:
                return [(v_min + v_max) * 0.5]
            used = spacing * (count - 1)
            start = (v_min + v_max - used) * 0.5
            return [start + i * spacing for i in range(count)]

        xs = axis_values(x_min, x_max)
        ys = axis_values(y_min, y_max)
        zs = axis_values(z_min, z_max)
        grid: list[tuple[float, float, float]] = []
        for z in zs:
            for y in ys:
                for x in xs:
                    grid.append((x, y, z))
        if spec.ball_count >= len(grid):
            return grid[: spec.ball_count]

        # Uniformly subsample the full grid so lower ball counts still cover the box.
        if spec.ball_count <= 1:
            return [grid[len(grid) // 2]]
        for idx in range(spec.ball_count):
            grid_idx = round(idx * (len(grid) - 1) / (spec.ball_count - 1))
            positions.append(grid[grid_idx])
        return positions

    # For large-ball visual checks, start balls as a sparse 3D cloud instead of
    # letting them settle into a floor layer before the hand arrives.
    if spec.ball_count <= 16 or spec.ball_radius >= 0.025:
        # Hand-picked irregular 3D offsets for visual debugging. These avoid a
        # grid/corner pattern so the balls do not appear collinear from oblique views.
        coords = [
            (-0.190, -0.110, 0.060),
            (-0.060, -0.030, -0.050),
            (0.085, -0.120, 0.020),
            (-0.135, 0.055, -0.015),
            (0.035, 0.105, 0.055),
            (0.155, 0.020, -0.045),
            (-0.015, -0.095, 0.045),
            (0.115, 0.115, -0.005),
            (-0.205, 0.010, 0.015),
            (0.000, 0.000, 0.065),
            (0.165, -0.055, 0.050),
            (-0.090, 0.120, -0.060),
            (0.055, -0.010, -0.025),
            (-0.160, -0.060, 0.030),
            (0.135, 0.075, 0.035),
            (-0.025, 0.070, -0.040),
        ]
        x_min = cx - sx / 2.0 + 1.35 * r
        x_max = cx + sx / 2.0 - 1.35 * r
        y_min = cy - sy / 2.0 + 1.35 * r
        y_max = cy + sy / 2.0 - 1.35 * r
        z_min = cz - sz / 2.0 + 1.35 * r
        z_max = cz + sz / 2.0 - 1.35 * r
        for idx in range(spec.ball_count):
            xoff, yoff, zoff = coords[idx % len(coords)]
            x = min(x_max, max(x_min, cx + xoff + (next(rng) - 0.5) * 0.06 * r))
            y = min(y_max, max(y_min, cy + yoff + (next(rng) - 0.5) * 0.06 * r))
            z = min(z_max, max(z_min, cz + zoff))
            positions.append((x, y, z))
        return positions

    # Default smoke path: compact 3D mound instead of filling the floor first.
    cols = max(2, min(5, math.ceil(spec.ball_count ** (1.0 / 3.0)) + 1))
    rows = max(2, min(5, math.ceil(spec.ball_count / (cols * 2))))
    layer_capacity = cols * rows
    spacing_xy = 2.15 * r
    spacing_z = 1.85 * r
    for idx in range(spec.ball_count):
        layer = idx // layer_capacity
        rem = idx % layer_capacity
        layer_cols = max(1, cols - layer // 2)
        layer_rows = max(1, rows - layer // 2)
        rem = rem % (layer_cols * layer_rows)
        col = rem % layer_cols
        row = rem // layer_cols
        grid_w = max(0.0, (layer_cols - 1) * spacing_xy)
        grid_h = max(0.0, (layer_rows - 1) * spacing_xy)
        jitter_x = (next(rng) - 0.5) * 0.45 * r
        jitter_y = (next(rng) - 0.5) * 0.45 * r
        x = cx - grid_w / 2.0 + col * spacing_xy + jitter_x
        y = cy - grid_h / 2.0 + row * spacing_xy + jitter_y
        z = cz - sz / 2.0 + margin + layer * spacing_z
        positions.append((x, y, z))
    return positions


def physics_dt(spec: BallPitSpec) -> float:
    return 1.0 / (float(spec.fps) * float(spec.substeps))


def rollout_duration_seconds(spec: BallPitSpec) -> float:
    return float(spec.rollout_frames) / float(spec.fps)


def video_frame_indices(spec: BallPitSpec) -> set[int]:
    if spec.rollout_frames <= 0:
        return set()
    rollout_seconds = rollout_duration_seconds(spec)
    start_seconds = min(max(0.0, spec.video_start_seconds), max(0.0, rollout_seconds))
    requested_seconds = spec.video_duration_seconds
    if requested_seconds is None or requested_seconds <= 0.0:
        requested_seconds = max(0.0, rollout_seconds - start_seconds)
    end_seconds = min(rollout_seconds, start_seconds + requested_seconds)
    segment_seconds = max(0.0, end_seconds - start_seconds)
    start_frame = min(spec.rollout_frames - 1, int(round(start_seconds * spec.fps)))
    end_frame = min(spec.rollout_frames - 1, max(start_frame, int(round(end_seconds * spec.fps)) - 1))
    segment_frames = end_frame - start_frame + 1
    video_frames = max(1, int(math.ceil(segment_seconds * VIDEO_FPS)))
    if video_frames >= segment_frames:
        return set(range(start_frame, end_frame + 1))
    if video_frames == 1:
        return {start_frame}
    return {
        start_frame + int(round(i * (segment_frames - 1) / (video_frames - 1)))
        for i in range(video_frames)
    }


def ball_color_rgba(index: int) -> tuple[float, float, float, float]:
    return BALL_PALETTE_RGBA[int(index) % len(BALL_PALETTE_RGBA)]


def container_wall_boxes(spec: BallPitSpec) -> list[dict[str, Any]]:
    sx, sy, sz = spec.container_size
    cx, cy, cz = spec.container_center
    wall = 0.006
    floor_z = cz - sz / 2.0
    wall_height = max(sz, 0.48) if spec.scenario == "filled_tank" else sz
    wall_center_z = floor_z + wall_height / 2.0
    wall_half_z = wall_height / 2.0
    return [
        {"name": "pit_floor", "pos": (cx, cy, floor_z - wall), "size": (sx / 2.0, sy / 2.0, wall), "rgba": (0.62, 0.66, 0.68, 0.30)},
        {"name": "pit_x_min", "pos": (cx - sx / 2.0 - wall, cy, wall_center_z), "size": (wall, sy / 2.0, wall_half_z), "rgba": (0.80, 0.88, 0.95, 0.08)},
        {"name": "pit_x_max", "pos": (cx + sx / 2.0 + wall, cy, wall_center_z), "size": (wall, sy / 2.0, wall_half_z), "rgba": (0.80, 0.88, 0.95, 0.08)},
        {"name": "pit_y_min", "pos": (cx, cy - sy / 2.0 - wall, wall_center_z), "size": (sx / 2.0, wall, wall_half_z), "rgba": (0.80, 0.88, 0.95, 0.05)},
        {"name": "pit_y_max", "pos": (cx, cy + sy / 2.0 + wall, wall_center_z), "size": (sx / 2.0, wall, wall_half_z), "rgba": (0.80, 0.88, 0.95, 0.05)},
    ]


def camera_pose(spec: BallPitSpec) -> dict[str, Any]:
    cx, cy, cz = spec.container_center
    return {
        "pos": (cx + 0.62, cy - 0.66, cz + 0.54),
        "lookat": (cx - 0.02, cy, cz + 0.08),
        "up": (0.0, 0.0, 1.0),
        "fovy": 54.0,
        "width": VIDEO_WIDTH,
        "height": VIDEO_HEIGHT,
        "fps": VIDEO_FPS,
    }


def write_video(path: Path, frames: Sequence[Any], fps: int = VIDEO_FPS) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=fps, macro_block_size=16)
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


def hand_pose_at(frame: float, spec: BallPitSpec) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return base translation and xyz Euler angles for one exported frame."""
    n = max(1, spec.rollout_frames - 1)
    u = min(1.0, max(0.0, float(frame) / float(n)))
    smooth = 0.5 - 0.5 * math.cos(math.pi * u)
    if spec.scenario == "filled_tank":
        cx, cy, cz = spec.container_center
        sx, sy, sz = spec.container_size
        insert_z = cz - 0.035
        entry_z = cz + sz * 0.82
        exit_z = cz + sz * 0.72
        # Start above the settled pile, insert deep into the tank, sweep through
        # the balls, then lift out.
        if u < 0.32:
            v = 0.5 - 0.5 * math.cos(math.pi * (u / 0.32))
            x = cx + 0.045
            y = cy - 0.025
            z = entry_z + (insert_z - entry_z) * v
        elif u < 0.76:
            v = (u - 0.32) / 0.44
            x = cx + 0.045 - 0.13 * v
            y = cy - 0.025 + 0.085 * math.sin(math.pi * v)
            z = insert_z + 0.012 * math.sin(2.0 * math.pi * v)
        else:
            v = 0.5 - 0.5 * math.cos(math.pi * ((u - 0.76) / 0.24))
            x = cx - 0.085
            y = cy - 0.025
            z = insert_z + (exit_z - insert_z) * v
        roll = 0.22 * math.sin(math.pi * u)
        pitch = 0.18 * math.sin(2.0 * math.pi * u + 0.3)
        yaw = 0.55 * math.sin(2.0 * math.pi * u)
        return (x, y, z), (roll, pitch, yaw)

    sx, sy, sz = spec.hand_base_start
    ex, ey, ez = spec.hand_base_end
    x = sx + (ex - sx) * smooth
    y = sy + (ey - sy) * smooth + spec.hand_y_amplitude * math.sin(2.0 * math.pi * u)
    z = sz + (ez - sz) * smooth + spec.hand_z_amplitude * math.sin(math.pi * u)
    roll = spec.hand_roll_amplitude * math.sin(math.pi * u)
    pitch = spec.hand_pitch_amplitude * math.sin(2.0 * math.pi * u + 0.5 * math.pi)
    yaw = spec.hand_yaw_amplitude * math.sin(2.0 * math.pi * u)
    return (x, y, z), (roll, pitch, yaw)


def euler_xyz_to_quat_wxyz(rx: float, ry: float, rz: float) -> tuple[float, float, float, float]:
    cr = math.cos(rx * 0.5)
    sr = math.sin(rx * 0.5)
    cp = math.cos(ry * 0.5)
    sp = math.sin(ry * 0.5)
    cy = math.cos(rz * 0.5)
    sy = math.sin(rz * 0.5)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def quat_wxyz_to_matrix(quat: Sequence[float]) -> list[list[float]]:
    w, x, y, z = [float(v) for v in quat]
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 0.0:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ]


def norm3(value: Sequence[float]) -> float:
    return math.sqrt(sum(float(v) * float(v) for v in value[:3]))


def summarize_contact_payload(contact: list[list[dict[str, Any]]]) -> dict[str, Any]:
    entries = sum(len(frame) for frame in contact)
    pairs = sum(len(entry["contact_pairs"]) for frame in contact for entry in frame)
    nonempty = sum(1 for frame in contact if frame)
    touched_links = sorted({entry["joint_name"] for frame in contact for entry in frame})
    touched_objects = sorted({entry["object_name"] for frame in contact for entry in frame})
    normal_values = [
        norm3(pair["force_normal"])
        for frame in contact
        for entry in frame
        for pair in entry["contact_pairs"]
    ]
    tangential_values = [
        norm3(pair["force_tangential"])
        for frame in contact
        for entry in frame
        for pair in entry["contact_pairs"]
    ]
    return {
        "frames": len(contact),
        "nonempty_frames": nonempty,
        "contact_entries": entries,
        "contact_pairs": pairs,
        "touched_hand_links": touched_links,
        "touched_objects": touched_objects[:16],
        "touched_object_count": len(touched_objects),
        "peak_normal_force": max(normal_values) if normal_values else 0.0,
        "mean_normal_force": sum(normal_values) / len(normal_values) if normal_values else 0.0,
        "peak_tangential_force": max(tangential_values) if tangential_values else 0.0,
        "mean_tangential_force": sum(tangential_values) / len(tangential_values) if tangential_values else 0.0,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _project(point: Sequence[float], bounds: tuple[float, float, float, float], size: tuple[int, int]) -> tuple[float, float]:
    x_min, x_max, y_min, y_max = bounds
    width, height = size
    x = (float(point[0]) - x_min) / max(1e-9, x_max - x_min) * width
    y = height - (float(point[1]) - y_min) / max(1e-9, y_max - y_min) * height
    return x, y


def render_svg(
    path: Path,
    *,
    spec: BallPitSpec,
    backend: str,
    ball_positions: Sequence[Sequence[float]],
    hand_trace: Sequence[Sequence[float]],
    contact_points: Sequence[Sequence[float]],
    summary: dict[str, Any],
) -> None:
    """Write a simple top/side-view SVG so Docker images need no plotting deps."""
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 1100
    height = 560
    panel_w = 500
    panel_h = 390
    sx, sy, sz = spec.container_size
    cx, cy, cz = spec.container_center
    top_bounds = (cx - sx / 2.0, cx + sx / 2.0, cy - sy / 2.0, cy + sy / 2.0)
    side_bounds = (cx - sx / 2.0, cx + sx / 2.0, cz - sz / 2.0, cz + sz / 2.0)
    r_px = max(2.0, spec.ball_radius / sx * panel_w)

    def circle(point: Sequence[float], panel_x: int, panel_y: int, mode: str, color: str, radius: float, opacity: float) -> str:
        if mode == "top":
            px, py = _project((point[0], point[1]), top_bounds, (panel_w, panel_h))
        else:
            px, py = _project((point[0], point[2]), side_bounds, (panel_w, panel_h))
        return f'<circle cx="{panel_x + px:.1f}" cy="{panel_y + py:.1f}" r="{radius:.1f}" fill="{color}" opacity="{opacity:.2f}" />'

    def polyline(points: Sequence[Sequence[float]], panel_x: int, panel_y: int, mode: str, color: str) -> str:
        coords = []
        for point in points:
            if mode == "top":
                px, py = _project((point[0], point[1]), top_bounds, (panel_w, panel_h))
            else:
                px, py = _project((point[0], point[2]), side_bounds, (panel_w, panel_h))
            coords.append(f"{panel_x + px:.1f},{panel_y + py:.1f}")
        return f'<polyline points="{" ".join(coords)}" fill="none" stroke="{color}" stroke-width="3" opacity="0.85" />'

    items = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f7f4" />',
        f'<text x="32" y="34" font-family="Arial" font-size="22" font-weight="700">Ball-Pit Contact Benchmark: {backend}</text>',
        f'<text x="32" y="62" font-family="Arial" font-size="14" fill="#444">balls={spec.ball_count}, entries={summary.get("contact_entries", 0)}, pairs={summary.get("contact_pairs", 0)}, touched_links={len(summary.get("touched_hand_links", []))}</text>',
        '<text x="52" y="104" font-family="Arial" font-size="16" font-weight="700">Top view: x-y</text>',
        '<text x="590" y="104" font-family="Arial" font-size="16" font-weight="700">Side view: x-z</text>',
        '<rect x="40" y="120" width="500" height="390" fill="#ffffff" stroke="#222" stroke-width="1.5" />',
        '<rect x="570" y="120" width="500" height="390" fill="#ffffff" stroke="#222" stroke-width="1.5" />',
    ]
    for ball in ball_positions:
        items.append(circle(ball, 40, 120, "top", "#68a9cf", r_px, 0.62))
        items.append(circle(ball, 570, 120, "side", "#68a9cf", r_px, 0.62))
    if len(hand_trace) >= 2:
        items.append(polyline(hand_trace, 40, 120, "top", "#222222"))
        items.append(polyline(hand_trace, 570, 120, "side", "#222222"))
    for point in contact_points[-300:]:
        items.append(circle(point, 40, 120, "top", "#d13f32", 3.2, 0.72))
        items.append(circle(point, 570, 120, "side", "#d13f32", 3.2, 0.72))
    items.extend(
        [
            '<text x="48" y="534" font-family="Arial" font-size="13" fill="#333">blue: final balls, black: hand base trajectory, red: exported hand-ball contact points</text>',
            '</svg>',
        ]
    )
    path.write_text("\n".join(items), encoding="utf-8")


def _draw_rect(img: Any, x0: float, y0: float, x1: float, y1: float, color: Sequence[int]) -> None:
    h, w = img.shape[:2]
    ix0 = max(0, min(w, int(round(min(x0, x1)))))
    ix1 = max(0, min(w, int(round(max(x0, x1)))))
    iy0 = max(0, min(h, int(round(min(y0, y1)))))
    iy1 = max(0, min(h, int(round(max(y0, y1)))))
    if ix1 > ix0 and iy1 > iy0:
        img[iy0:iy1, ix0:ix1, :3] = color[:3]


def _draw_circle(img: Any, cx: float, cy: float, radius: float, color: Sequence[int], alpha: float = 1.0) -> None:
    import numpy as np

    h, w = img.shape[:2]
    r = max(1, int(round(radius)))
    x0 = max(0, int(round(cx)) - r)
    x1 = min(w - 1, int(round(cx)) + r)
    y0 = max(0, int(round(cy)) - r)
    y1 = min(h - 1, int(round(cy)) + r)
    if x1 < x0 or y1 < y0:
        return
    yy, xx = np.ogrid[y0 : y1 + 1, x0 : x1 + 1]
    mask = (xx - cx) * (xx - cx) + (yy - cy) * (yy - cy) <= radius * radius
    region = img[y0 : y1 + 1, x0 : x1 + 1]
    if alpha >= 1.0:
        region[mask, :3] = color[:3]
    else:
        region[mask, :3] = (region[mask, :3].astype(np.float32) * (1.0 - alpha) + np.asarray(color[:3], dtype=np.float32) * alpha).astype(np.uint8)


def _draw_line(img: Any, p0: Sequence[float], p1: Sequence[float], color: Sequence[int], width: int = 2) -> None:
    import numpy as np

    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    steps = max(1, int(max(abs(x1 - x0), abs(y1 - y0))))
    for t in np.linspace(0.0, 1.0, steps + 1):
        _draw_circle(img, x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, width, color, 1.0)


def render_state_video_frames(
    *,
    spec: BallPitSpec,
    ball_positions_by_frame: Sequence[Sequence[Sequence[float]]],
    hand_trace: Sequence[Sequence[float]],
    contact_points_by_frame: Sequence[Sequence[Sequence[float]]] | None = None,
) -> list[Any]:
    """Render a lightweight diagnostic MP4 from simulated state, not backend RGB camera."""
    import numpy as np

    frames: list[Any] = []
    indices = sorted(video_frame_indices(spec))
    if not indices:
        return frames

    width, height = VIDEO_WIDTH, VIDEO_HEIGHT
    margin = 28
    gap = 24
    panel_w = (width - margin * 2 - gap) // 2
    panel_h = height - margin * 2
    left_x = margin
    right_x = margin + panel_w + gap
    panel_y = margin
    sx, sy, sz = spec.container_size
    cx, cy, cz = spec.container_center
    top_bounds = (cx - sx / 2.0, cx + sx / 2.0, cy - sy / 2.0, cy + sy / 2.0)
    side_bounds = (cx - sx / 2.0, cx + sx / 2.0, cz - sz / 2.0, cz + max(sz / 2.0, 0.30))
    ball_radius_px = max(3.0, spec.ball_radius / sx * panel_w)

    def panel_project(point: Sequence[float], mode: str) -> tuple[float, float]:
        if mode == "top":
            px, py = _project((point[0], point[1]), top_bounds, (panel_w, panel_h))
            return left_x + px, panel_y + py
        px, py = _project((point[0], point[2]), side_bounds, (panel_w, panel_h))
        return right_x + px, panel_y + py

    for frame_idx in indices:
        img = np.full((height, width, 3), 246, dtype=np.uint8)
        _draw_rect(img, left_x, panel_y, left_x + panel_w, panel_y + panel_h, (255, 255, 255))
        _draw_rect(img, right_x, panel_y, right_x + panel_w, panel_y + panel_h, (255, 255, 255))
        for x in (left_x, left_x + panel_w, right_x, right_x + panel_w):
            _draw_rect(img, x - 1, panel_y, x + 1, panel_y + panel_h, (45, 45, 45))
        for x0 in (left_x, right_x):
            _draw_rect(img, x0, panel_y - 1, x0 + panel_w, panel_y + 1, (45, 45, 45))
            _draw_rect(img, x0, panel_y + panel_h - 1, x0 + panel_w, panel_y + panel_h + 1, (45, 45, 45))

        balls = ball_positions_by_frame[min(frame_idx, len(ball_positions_by_frame) - 1)] if ball_positions_by_frame else ball_initial_positions(spec)
        for i, ball in enumerate(balls):
            rgba = ball_color_rgba(i)
            color = tuple(int(255 * c) for c in rgba[:3])
            _draw_circle(img, *panel_project(ball, "top"), ball_radius_px, color, 0.78)
            _draw_circle(img, *panel_project(ball, "side"), ball_radius_px, color, 0.78)

        trace_end = min(frame_idx + 1, len(hand_trace))
        recent_trace = hand_trace[:trace_end]
        for p0, p1 in zip(recent_trace, recent_trace[1:]):
            _draw_line(img, panel_project(p0, "top"), panel_project(p1, "top"), (20, 20, 20), 2)
            _draw_line(img, panel_project(p0, "side"), panel_project(p1, "side"), (20, 20, 20), 2)
        if recent_trace:
            _draw_circle(img, *panel_project(recent_trace[-1], "top"), 6, (0, 0, 0), 1.0)
            _draw_circle(img, *panel_project(recent_trace[-1], "side"), 6, (0, 0, 0), 1.0)

        if contact_points_by_frame:
            start = max(0, frame_idx - int(spec.fps * 0.25))
            for contact_frame in contact_points_by_frame[start : frame_idx + 1]:
                for point in contact_frame[-64:]:
                    _draw_circle(img, *panel_project(point, "top"), 3, (210, 40, 35), 0.75)
                    _draw_circle(img, *panel_project(point, "side"), 3, (210, 40, 35), 0.75)
        frames.append(img)
    return frames


def render_perspective_state_video_frames(
    *,
    spec: BallPitSpec,
    ball_positions_by_frame: Sequence[Sequence[Sequence[float]]],
    hand_trace: Sequence[Sequence[float]],
    contact_points_by_frame: Sequence[Sequence[Sequence[float]]] | None = None,
) -> list[Any]:
    """Render a 3D perspective diagnostic video from simulated state."""
    import numpy as np

    frames: list[Any] = []
    indices = sorted(video_frame_indices(spec))
    if not indices:
        return frames

    width, height = VIDEO_WIDTH, VIDEO_HEIGHT
    cam = camera_pose(spec)
    eye = np.asarray(cam["pos"], dtype=np.float64)
    target = np.asarray(cam["lookat"], dtype=np.float64)
    up_hint = np.asarray(cam["up"], dtype=np.float64)
    forward = target - eye
    forward = forward / max(1e-9, float(np.linalg.norm(forward)))
    right = np.cross(forward, up_hint)
    right = right / max(1e-9, float(np.linalg.norm(right)))
    up = np.cross(right, forward)
    focal = 0.5 * height / math.tan(math.radians(float(cam["fovy"])) * 0.5)

    sx, sy, sz = spec.container_size
    cx, cy, cz = spec.container_center
    floor_z = cz - sz / 2.0
    box_h = max(sz, 0.48)
    corners = [
        np.asarray((cx + dx * sx / 2.0, cy + dy * sy / 2.0, floor_z + dz * box_h), dtype=np.float64)
        for dx in (-1.0, 1.0)
        for dy in (-1.0, 1.0)
        for dz in (0.0, 1.0)
    ]
    edges = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]

    def project(point: Sequence[float]) -> tuple[float, float, float] | None:
        p = np.asarray(point, dtype=np.float64) - eye
        x = float(np.dot(p, right))
        y = float(np.dot(p, up))
        z = float(np.dot(p, forward))
        if z <= 1e-4:
            return None
        return width * 0.5 + focal * x / z, height * 0.52 - focal * y / z, z

    def shaded(color: Sequence[int], depth: float) -> tuple[int, int, int]:
        shade = max(0.48, min(1.0, 1.18 - 0.22 * depth))
        return tuple(max(0, min(255, int(c * shade))) for c in color[:3])

    floor_points = [
        (cx - sx / 2.0, cy - sy / 2.0, floor_z),
        (cx + sx / 2.0, cy - sy / 2.0, floor_z),
        (cx + sx / 2.0, cy + sy / 2.0, floor_z),
        (cx - sx / 2.0, cy + sy / 2.0, floor_z),
    ]

    for frame_idx in indices:
        img = np.full((height, width, 3), (232, 236, 239), dtype=np.uint8)
        for y in range(height):
            t = y / max(1, height - 1)
            img[y, :, :] = np.asarray((220 + 25 * t, 226 + 20 * t, 232 + 12 * t), dtype=np.uint8)

        projected_floor = [project(p) for p in floor_points]
        for a, b in zip(projected_floor, projected_floor[1:] + projected_floor[:1]):
            if a and b:
                _draw_line(img, (a[0], a[1]), (b[0], b[1]), (120, 128, 132), 2)

        projected_corners = [project(corner) for corner in corners]
        for a_idx, b_idx in edges:
            a = projected_corners[a_idx]
            b = projected_corners[b_idx]
            if a and b:
                _draw_line(img, (a[0], a[1]), (b[0], b[1]), (130, 170, 190), 1)

        balls = ball_positions_by_frame[min(frame_idx, len(ball_positions_by_frame) - 1)] if ball_positions_by_frame else ball_initial_positions(spec)
        draw_items: list[tuple[float, Sequence[float], tuple[int, int, int], float]] = []
        for i, ball in enumerate(balls):
            projected = project(ball)
            if projected is None:
                continue
            px, py, depth = projected
            radius_px = max(3.0, focal * spec.ball_radius / depth)
            rgba = ball_color_rgba(i)
            color = tuple(int(255 * c) for c in rgba[:3])
            draw_items.append((depth, (px, py), shaded(color, depth), radius_px))

        for depth, pos2, color, radius_px in sorted(draw_items, reverse=True):
            shadow = (pos2[0] + 0.24 * radius_px, pos2[1] + 0.34 * radius_px)
            _draw_circle(img, shadow[0], shadow[1], radius_px * 0.72, (45, 45, 45), 0.18)
            _draw_circle(img, pos2[0], pos2[1], radius_px, color, 0.96)
            _draw_circle(img, pos2[0] - radius_px * 0.30, pos2[1] - radius_px * 0.35, max(1.0, radius_px * 0.22), (255, 255, 255), 0.45)

        trace_end = min(frame_idx + 1, len(hand_trace))
        recent_trace = hand_trace[:trace_end]
        projected_trace = [project(p) for p in recent_trace]
        for a, b in zip(projected_trace, projected_trace[1:]):
            if a and b:
                _draw_line(img, (a[0], a[1]), (b[0], b[1]), (18, 18, 18), 3)
        if projected_trace and projected_trace[-1]:
            px, py, depth = projected_trace[-1]
            hand_r = max(8.0, focal * 0.035 / depth)
            _draw_circle(img, px, py, hand_r, (18, 18, 18), 1.0)
            _draw_circle(img, px - hand_r * 0.28, py - hand_r * 0.35, hand_r * 0.22, (210, 210, 210), 0.35)

        if contact_points_by_frame:
            start = max(0, frame_idx - int(spec.fps * 0.20))
            for contact_frame in contact_points_by_frame[start : frame_idx + 1]:
                for point in contact_frame[-80:]:
                    projected = project(point)
                    if projected:
                        _draw_circle(img, projected[0], projected[1], 3, (220, 35, 30), 0.70)
        frames.append(img)
    return frames

