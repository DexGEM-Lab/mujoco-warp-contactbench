# A_row035 live U1 repair

Run `python -m tools.view_u1_repair --bundle BUNDLE --asset-root ASSETS --target target.npy --output NEW_DIRECTORY --checkpoint 1030` on `DISPLAY=:1`. The selected absolute target stream physically replays from frame0. Use an earlier checkpoint (or the GUI frame entry) to search the dynamically acquired grasp before carry/release. The old successful trace supplies displayed hand-relative object transforms only; it never drives state or forces.

The viewer and Tk editor follow the egg-repair workspace's session/command-inbox pattern. Red spheres mark pitcher contacts; hand, pitcher and bowl are the live native model. Contact labels/counts are CPU geometry witnesses at the GPU state, not native constraint-force measurements. `ray_count` is the existing validation convention: distinct contacting hand-link prefixes, not geometric ray casts. Rotation matrices in the relative transforms map object axes into wrist coordinates. XYZ offsets are metres; Euler and finger offsets are radians in joint coordinates.

Controls: Space play/pause, N step1, K save checkpoint, R restore, E export (viewer focus). GUI includes step12, reset/clear, arbitrary checkpoint replay, and all 28 joint offsets. Playback is quarter speed. For StepN, put `{"action":"step","frames":24}` in the command inbox. Pause before editing a diagnostic branch.

Offsets are additive to the selected stream, zero at inclusive window endpoints, smoothstep ramp in/out, constant between ramps. They never modify a target at or before the current cursor. Example: `{"action":"offset","joint":6,"value":0.05,"start":1030,"end":1150,"ramp":24}`. This is an exploratory target edit, not an established opening direction. Wrist XYZ/Euler use indices 0–5. Inspect the displayed joint names/axes before larger changes. Joint-limit violations are rejected, not silently clipped.

Publish each JSON command by atomic rename into `commands/UNIQUE.json`. Supported actions: `play` (toggle), `pause`, `step` (`frames`), `save`, `restore`, `reset`, `goto` (`frame`), `offset` (as above), `export`. Responses and original commands are in `command_history/`. `goto` keeps edits and physically replays from frame0, then automatically validates/saves the new checkpoint. `reset` clears edits and returns to frame0. Restore reinstates the saved edits as well as physics, so save before changing targets. GUI entries are edit proposals, not the stored target; applied targets/edits are authoritative in `state.json`.

`state.json` contains frame/cursor, actual/desired/applied 28D controls, contact names, object pose/tilt, live and teacher hand-object transforms, and edits. `checkpoints/TIMESTAMP/state.npz` contains every native Warp array (including nested contact/constraint workspaces, qpos/qvel/ctrl/qacc_warmstart/time) plus targets; `workspace.json` holds frame and edits. In-process restore uses these same complete snapshots without replacing captured-graph device addresses. `restore_validation.json` reports a repeated 12-frame qpos/qvel continuation and its tolerance. Failure is explicit and stops the UI's automatic rollout.

Export writes immutable `candidates/TIMESTAMP/target.npy` and provenance. Exported targets remain **unaccepted**: run the existing `tools.u1_placement_workspace` full frame0 U1 evaluation and independent passing replay before publishing anything. No force/attachment/teleport controls exist. Current Lance/NAS and previous trial outputs are never changed.

## Live launch (2026-09-20)

- tmux: `feat-uniform-direct-parent-repair-gui`, window `live` (ordinary persistent tmux)
- DISPLAY: `:1`
- Workspace: `outputs/u1_live_A_row035_ready/`
- Log: `/tmp/u1-live-A-row035-ready.log`
- Initial target: `outputs/u1_placement_v1/A_row035/release01/target.npy`
- Default checkpoint: 1030, reached by native U1 frame0 replay, not archived state injection.

Next interaction: inspect thumb_ip/CMC/palm contact at the checkpoint. Save, add a small thumb abduction window, step12, compare contact links and object motion, restore. Do not retract the wrist until thumb/handle clearance is visible. For the grasp anchor instead, use the checkpoint frame entry to replay to a pregrasp frame and compare live versus teacher hand-object transforms. Neither this workspace nor the failed release01 target is a repaired trajectory.

The first GPU restore check reproduced all 155 saved device arrays exactly but diverged by 4.37e-6 in combined qpos/qvel after 12 steps. The continuation check therefore separates pose (1e-6 bound) and velocity (1e-4 bound), rather than demanding bitwise deterministic GPU reduction order. Every result is retained.
