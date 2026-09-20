#!/usr/bin/env python3
"""Observe an existing U1 workspace from a terminal; never advance or alter physics.

Telemetry follows published state.json snapshots (not every physics substep).
Contacts are CPU geometry witnesses, not measured native contact forces.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time

import numpy as np

FOUR_FINGERS = {"index", "middle", "ring", "pinky"}


def relative_error(state):
    actual = state["hand_object_transform"]
    teacher = state["teacher_hand_object_transform"]
    dp = np.asarray(actual["position"]) - np.asarray(teacher["position"])
    dr = np.asarray(teacher["rotation"]).T @ np.asarray(actual["rotation"])
    angle = math.degrees(math.acos(float(np.clip((np.trace(dr) - 1) / 2, -1, 1))))
    return float(np.linalg.norm(dp) * 1000), angle


def edit_text(edit):
    joint = edit["joint"]
    value = edit["value"] * (1000 if joint < 3 else 180 / math.pi)
    unit = "mm" if joint < 3 else "deg"
    return (f"j{joint}={value:+.2f}{unit}@{edit['start']}:{edit['end']}"
            f"/r{edit['ramp']}")


def format_state(state, previous=None):
    links = sorted(state["hand_links"])
    prefixes = {name.split("_")[0] for name in links}
    contacts = state["contacts"]
    hand_distances = [float(c["distance"]) for c in contacts if c["link"] in links]
    penetration = max((max(0.0, -d) for d in hand_distances), default=0.0) * 1000
    floor_contact = any(c["link"] == "world" for c in contacts)
    xyz = np.asarray(state["object_position"], dtype=float)
    delta = "--"
    if previous is not None and state["frame"] > previous["frame"]:
        delta = f"{np.linalg.norm(xyz-np.asarray(previous['object_position']))*1000:.2f}"
    relative_mm, relative_deg = relative_error(state)
    servo_mm = (np.asarray(state["applied_ctrl"])[:3]
                - np.asarray(state["current_28d"])[:3]) * 1000
    mode = "PLAY" if state["running"] else "STEP" if state["remaining_steps"] else "PAUSE"
    rewind = " REWIND" if previous is not None and state["frame"] < previous["frame"] else ""
    text = (
        f"f={state['frame']:04d} t={state['physics_seconds']:6.3f}s {mode}{rewind} "
        f"checkpoint={state.get('checkpoint_frame')} remaining={state['remaining_steps']}\n"
        f"  object xyz=({xyz[0]:+.4f},{xyz[1]:+.4f},{xyz[2]:+.4f})m "
        f"delta_since_sample={delta}mm tilt={state['object_tilt_deg']:.2f}deg "
        f"floor_geometry={'touch' if floor_contact else 'clear'}\n"
        f"  hand_contacts={state['hand_contact_count']} "
        f"nonthumb_groups={len(prefixes & FOUR_FINGERS)}/4 thumb={'yes' if 'thumb' in prefixes else 'no'} "
        f"palm={'yes' if 'palm' in prefixes else 'no'} penetration_max={penetration:.2f}mm "
        f"links={','.join(links) or '-'}\n"
        f"  relative_pose_vs_source={relative_mm:.2f}mm/{relative_deg:.2f}deg "
        f"servo_ctrl_minus_q_xyz=({servo_mm[0]:+.1f},{servo_mm[1]:+.1f},{servo_mm[2]:+.1f})mm "
        f"source_links={','.join(state.get('teacher_hand_links', [])) or '-'}"
    )
    if 'native_hand_force_on_object_N' in state:
        fhand=state['native_hand_force_on_object_N'][2]
        fother=state['native_other_force_on_object_N'][2]
        weight=state['object_weight_N']
        text += f'\n  native_last_substep hand_Fz={fhand:+.2f}N other_Fz={fother:+.2f}N weight={weight:.2f}N'
    if 'fingertip_object_positions_m' in state:
        tips=state['fingertip_object_positions_m']
        depths=[tips[n][1]*1000 for n in ('index_dip','middle_dip','ring_dip','pinky_dip')]
        text += ('\n  FK_four_tip_object_y_mm='+','.join(f'{v:+.1f}' for v in depths)
                 + f" thumb_index_tip_distance={state['thumb_index_tip_distance_m']*1000:.1f}mm")
    if previous is None or state["edits"] != previous["edits"]:
        text += "\n  edits=" + ("; ".join(edit_text(e) for e in state["edits"]) or "none")
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=float, default=0.1)
    args = parser.parse_args()
    if not 0 < args.interval <= 10:
        parser.error("--interval must be in (0,10]")
    path = args.workspace / "state.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    log = None
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        log = args.log.open("a", buffering=1)

    def emit(text):
        line = time.strftime("%H:%M:%S ") + text
        print(line, flush=True)
        if log:
            log.write(line + "\n")

    history = args.workspace / "command_history"
    seen = set(history.glob("*.response.json"))
    previous = None
    signature = None
    stale = False
    unreadable = False
    emit("U1 LIVE TELEMETRY | observed state snapshots only; no physics/control writes")
    emit("contacts/floor/penetration = CPU geometry; force if present = native last substep. "
         "Neither contact counts nor FK tip coordinates prove a stable grasp. "
         "relative_pose compares live to source; servo error is ctrl minus q, NOT tracking error")
    try:
        while True:
            try:
                state = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as error:
                if args.once:
                    raise
                if not unreadable:
                    emit(f"STATE_UNREADABLE: {error}")
                unreadable = True
                time.sleep(args.interval)
                continue
            if unreadable:
                emit("STATE_READ_RECOVERED")
            unreadable = False
            current_signature = json.dumps(state, sort_keys=True)
            if current_signature != signature:
                emit(format_state(state, previous))
                previous = state
                signature = current_signature
            for response in sorted(set(history.glob("*.response.json")) - seen):
                try:
                    result = json.loads(response.read_text())
                except (OSError, json.JSONDecodeError):
                    continue  # the writer may not have completed this response yet
                emit(f"COMMAND {response.name}: {json.dumps(result, ensure_ascii=False)}")
                seen.add(response)
            if args.once:
                return
            pid = state.get("pid")
            if pid:
                try:
                    os.kill(int(pid), 0)
                except ProcessLookupError:
                    emit(f"SIMULATOR_EXITED pid={pid}")
                    return
            age = time.time() - path.stat().st_mtime
            if age > 5 and not stale:
                emit(f"STATE_STALE age={age:.1f}s; frame progression is not known")
                stale = True
            elif age <= 5 and stale:
                emit("STATE_LIVE_AGAIN")
                stale = False
            time.sleep(args.interval)
    finally:
        if log:
            log.close()


if __name__ == "__main__":
    main()
