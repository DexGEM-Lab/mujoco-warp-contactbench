"""Full-capture, source-hand-matched right-only physical replay, without a policy.

Objects receive recorded poses only at initialization. The sole trajectory
normalization is a common scene grounding shift and equivalent wrist-angle
wrapping/joint-limit enforcement by the existing position controller.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import json
from pathlib import Path
import time

import lance
import numpy as np
from scipy.spatial.transform import Rotation

from sim.manorl.contracts import FLOOR_TOP_Z, OBJECT_CLEARANCE

COLUMNS = ["index", "trajectory_metadata", "timestamp", "hands", "objects"]


PINNED_MANIFEST_SHA256 = "e686d91931979c444c923d4f962ddd135ce8204e0de9df6fd7368bfdb855e29b"
PINNED_ASSET_COMMIT = "778614d09e917deffed0bff3f357aa237efa762d"
PINNED_OPERATOR = "cheyingtong"


def activate_hand_profile(operator: str, asset_root: Path, asset_manifest: Path) -> dict:
    """Bind the formal profile before physical imports; never generate a fallback."""
    root = asset_root.expanduser().resolve(strict=True)
    path = asset_manifest.expanduser().resolve(strict=True)
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != PINNED_MANIFEST_SHA256:
        raise ValueError("asset manifest SHA256 differs from the pinned formal profile")
    manifest = json.loads(payload)
    if operator != PINNED_OPERATOR or manifest.get("hand_operator") != operator:
        raise ValueError("source/manifest hand operator differs from the pinned profile")
    if manifest.get("source_commit") != PINNED_ASSET_COMMIT:
        raise ValueError("asset manifest source commit differs from the pinned profile")
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if commit != PINNED_ASSET_COMMIT:
        raise ValueError("asset root source commit differs from the pinned profile")
    os.environ["MANORL_ASSET_MANIFEST"] = str(path)
    from sim.manorl import assets

    assets.DEXSTREAM_ROOT = root
    assets.EXPLICIT_ASSET_MANIFEST = str(path)
    assets.ASSET_MANIFEST = path
    assets.MANO_OPERATOR = operator
    assets._asset_manifest.cache_clear()
    assets.object_collision_vertices.cache_clear()
    assets.validate_asset_manifest(hand_side="right")
    return manifest


def source_arrays(row: dict, manifest: dict, fps: int) -> dict:
    from sim.manorl import assets

    metadata = row["trajectory_metadata"]
    hand_names = metadata["hand_names"]
    if hand_names.count("right") != 1:
        raise ValueError("source must identify exactly one right-hand slot")
    slot = hand_names.index("right")
    betas = np.asarray(metadata["mano_hand_shapes"][slot], dtype=np.float64)
    if not np.allclose(betas, manifest["hands"]["right"]["betas"], atol=1e-6, rtol=0):
        raise ValueError("recorded right-hand shape differs from the selected source hand")
    names = tuple(name.strip().casefold() for name in row["index"]["scene"].split(","))
    declared = metadata.get("object_names")
    if declared is not None and tuple(name.casefold() for name in declared) != names:
        raise ValueError("scene and metadata disagree on ordered object slots")
    if len(names) != len(row["objects"]) or len(set(names)) != len(names):
        raise ValueError("source scene must uniquely identify every object slot")
    moves = metadata["trajectory_info"]["object_move"]
    if len(moves) != 1 or moves[0]["object_name"].casefold() not in names:
        raise ValueError("source must identify one manipulated object")
    active = moves[0]["object_name"].casefold()
    q = np.asarray(row["hands"][slot]["urdf_dof"], dtype=np.float64)
    timestamps = np.asarray(row["timestamp"], dtype=np.float64)
    frames = int(metadata["total_frames"])
    if q.shape != (frames, 28) or timestamps.shape != (frames,) or frames < 2:
        raise ValueError("source requires complete Nx28 commands and timestamps")
    actual_fps = (frames - 1) / (timestamps[-1] - timestamps[0])
    if np.any(np.diff(timestamps) <= 0) or abs(actual_fps - fps) > .5:
        raise ValueError("source clock does not match the explicitly selected frame-command rate")
    positions = np.asarray([obj["pos"] for obj in row["objects"]], dtype=np.float64).transpose(1, 0, 2)
    rotvecs = np.asarray([obj["rot_aa"] for obj in row["objects"]], dtype=np.float64).transpose(1, 0, 2)
    if positions.shape != (frames, len(names), 3) or rotvecs.shape != positions.shape:
        raise ValueError("object tracks must cover every source frame")
    if not all(np.all(np.isfinite(value)) for value in (q, timestamps, positions, rotvecs)):
        raise ValueError("source trajectories must be finite")
    lowest = min(
        (assets.object_collision_vertices(name) @ Rotation.from_rotvec(rotvecs[0, i]).as_matrix().T
         + positions[0, i])[:, 2].min()
        for i, name in enumerate(names)
    )
    shift = float(FLOOR_TOP_Z + OBJECT_CLEARANCE - lowest)
    original_commands = q.copy()
    q = q.copy(); q[:, 2] += shift
    positions = positions.copy(); positions[:, :, 2] += shift
    return dict(names=names, active=active, commands=q, original_commands=original_commands,
                positions=positions, rotvecs=rotvecs, timestamps=timestamps,
                shift=shift, actual_fps=float(actual_fps), movement=moves[0])


def longest_duration(mask: np.ndarray, fps: int) -> float:
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    return float(lengths.max(initial=0) / fps)


def render_trace(output: Path, source: dict, trace: dict, fps: int) -> None:
    from sim.manorl import assets

    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    import mujoco as mj

    names = tuple(sorted(source["names"]))
    _, model = assets.compile_unified_model(object_types=names, object_collisions=True,
                                             hand_side="right", visual_meshes=True,
                                             physics_timestep=1 / (4 * fps))
    if model.nq != trace["qpos"].shape[1]:
        raise ValueError("render model differs from recorded physics model")
    data = mj.MjData(model); reference = mj.MjData(model)
    renderer = mj.Renderer(model, height=480, width=640)
    camera = mj.MjvCamera()
    active_slot = source["names"].index(source["active"])
    camera.lookat[:] = np.median(source["positions"][:, active_slot], axis=0) + [0, 0, .04]
    camera.distance = max(.65, float(np.ptp(source["positions"][0], axis=0).max()) * 1.65 + .25)
    camera.azimuth = 100; camera.elevation = -30
    option = mj.MjvOption(); option.geomgroup[assets.COLLISION_GEOM_GROUP] = 0
    peak = int(np.argmax(source["positions"][:, active_slot, 2]))
    snapshots = {peak, len(source["commands"]) - 1, int(source["movement"]["start_frame"])}
    stride = max(1, fps // 30)
    writer = imageio.get_writer(output / "comparison.mp4", fps=fps / stride, codec="libx264", quality=7)
    try:
        for frame in sorted(set(range(0, len(source["commands"]), stride)) | snapshots):
            data.qpos[:] = trace["qpos"][frame]
            reference.qpos[:] = trace["reference_qpos"][frame]
            mj.mj_forward(model, data); mj.mj_forward(model, reference)
            renderer.update_scene(reference, camera=camera, scene_option=option)
            left = renderer.render().copy()
            renderer.update_scene(data, camera=camera, scene_option=option)
            right = renderer.render().copy()
            panel = Image.new("RGB", (1280, 512), "white")
            panel.paste(Image.fromarray(left), (0, 32)); panel.paste(Image.fromarray(right), (640, 32))
            draw = ImageDraw.Draw(panel)
            draw.text((12, 9), f"SOURCE POSES (kinematic comparison) | frame {frame}", fill="black")
            draw.text((652, 9), "FREE PHYSICS | right hand only | NO RL / NO REPAIR", fill="black")
            if frame % stride == 0:
                writer.append_data(np.asarray(panel))
            if frame in snapshots:
                panel.save(output / f"frame_{frame:04d}.jpg", quality=92)
    finally:
        writer.close(); renderer.close()


def replay(args: argparse.Namespace) -> dict:
    if not 0 < args.ccd <= args.contacts:
        raise ValueError("MJX-Warp requires 0 < CCD capacity <= contact capacity")
    import warp as wp
    from mujoco.mjx.third_party import mujoco_warp as mw

    args.output.mkdir(parents=True, exist_ok=False)
    dataset = lance.dataset(str(args.dataset), version=args.version)
    row = dataset.take([args.row], columns=COLUMNS).to_pylist()[0]
    operator = row["index"]["operator"]
    manifest = activate_hand_profile(operator, args.asset_root, args.asset_manifest)
    from sim.manorl import assets
    from sim.manorl.mjx_sim import command_target
    source = source_arrays(row, manifest, args.fps)
    names = tuple(sorted(source["names"]))
    mj, model = assets.compile_unified_model(object_types=names, object_collisions=True,
                                             hand_side="right", physics_timestep=1 / (4 * args.fps))
    if model.nu != 28 or model.nq != 28 + 7 * len(names):
        raise ValueError("replay must contain only one 28-DoF right hand and the source objects")
    body_names = [model.body(i).name for i in range(model.nbody)]
    if any(name.startswith("left_") for name in body_names):
        raise ValueError("left hand was unexpectedly loaded")
    addresses = {name: int(model.joint(name + "_free").qposadr[0]) for name in names}
    data = mj.MjData(model)
    reference_qpos = np.zeros((len(source["commands"]), model.nq))
    reference_qpos[:, :28] = source["commands"]
    for name, adr in addresses.items():
        slot = source["names"].index(name)
        reference_qpos[:, adr:adr + 3] = source["positions"][:, slot]
        reference_qpos[:, adr + 3:adr + 7] = Rotation.from_rotvec(source["rotvecs"][:, slot]).as_quat(scalar_first=True)
    lower, upper = model.actuator_ctrlrange.T
    data.qpos[:] = reference_qpos[0]
    data.ctrl[:] = command_target(source["commands"][0], data.qpos[:28], lower, upper)
    mj.mj_forward(model, data)
    initial_qpos, initial_ctrl = data.qpos.copy(), data.ctrl.copy()
    settings = dict(schema="manorl.raw_right_capture_replay.v1", dataset=str(args.dataset),
                    dataset_version=int(dataset.version), row=args.row, index=row["index"],
                    hand_operator=operator, hand_sides=["right"], actuators=model.nu,
                    asset_root=str(assets.DEXSTREAM_ROOT), asset_manifest=str(assets.ASSET_MANIFEST),
                    asset_provenance=assets.asset_provenance(), objects=list(names),
                    object_qpos_addresses=addresses, body_names=body_names,
                    frames=len(source["commands"]), source_mean_fps=source["actual_fps"],
                    command_fps=args.fps, physics_fps=4 * args.fps,
                    source_ground_shift_m=source["shift"], movement=source["movement"],
                    backend="mjx-warp", device=args.device,
                    cone="pyramidal" if int(model.opt.cone) == 0 else "elliptic", impratio=float(model.opt.impratio),
                    mujoco_version=mj.__version__, warp_version=wp.__version__,
                    no_learned_policy=True, no_residual=True, no_repair=True,
                    no_object_forcing_after_initialization=True,
                    contact_capacity=args.contacts, ccd_capacity=args.ccd)
    (args.output / "manifest.json").write_text(json.dumps(settings, indent=2, ensure_ascii=False))
    print(json.dumps({"row":args.row,"operator":operator,"objects":names,"ngeom":model.ngeom,"nu":model.nu,"frames":len(source["commands"])},ensure_ascii=False),flush=True)
    wp.init(); wp.set_device(args.device)
    wm = mw.put_model(model)
    wd = mw.put_data(model, data, nworld=1, nconmax=args.contacts, nccdmax=args.ccd, njmax=65536)
    for _ in range(4): mw.step(wm, wd)
    with wp.ScopedCapture() as capture:
        for _ in range(4): mw.step(wm, wd)
    # Compilation warmup is discarded. The measured episode starts from the
    # recorded frame-zero state, and no object coordinates are set afterward.
    wd.qpos.assign(initial_qpos[None].astype(np.float32)); wd.qvel.zero_()
    wd.qacc_warmstart.zero_(); wd.time.zero_(); wd.ctrl.assign(initial_ctrl[None].astype(np.float32))
    active_body = model.body(source["active"]).id
    scene_bodies = {model.body(name).id for name in names}
    hand_bodies = set(range(1, model.nbody)) - scene_bodies
    qpos=[]; qvel=[]; controls=[]; ncontacts=[]; hand_contacts=[]; support_contacts=[]; finger_contacts=[]
    started=time.monotonic()
    for frame, reference_command in enumerate(source["commands"]):
        current = wd.qpos.numpy()[0]
        target = command_target(reference_command, current[:28], lower, upper)
        wd.ctrl.assign(target[None].astype(np.float32))
        wp.capture_launch(capture.graph)
        pose = wd.qpos.numpy()[0].copy(); velocity = wd.qvel.numpy()[0].copy()
        if not np.all(np.isfinite(pose)) or not np.all(np.isfinite(velocity)):
            raise RuntimeError(f"non-finite physical state at frame {frame}")
        count = int(wd.nacon.numpy().reshape(-1)[0])
        if count >= args.contacts:
            raise RuntimeError(f"contact capacity reached at frame {frame}: {count}")
        geoms = wd.contact.geom.numpy()[:count]
        bodies = np.asarray(model.geom_bodyid)[geoms]
        selected = bodies[np.any(bodies == active_body, axis=1)]
        other = selected[:, 0] + selected[:, 1] - active_body
        hc = np.isin(other, list(hand_bodies))
        fc = [sum(body_names[int(b)].startswith(prefix) for b in other[hc])
              for prefix in ("palm", "thumb", "index", "middle", "ring", "pinky")]
        qpos.append(pose); qvel.append(velocity); controls.append(target)
        ncontacts.append(count); hand_contacts.append(int(hc.sum())); support_contacts.append(int((~hc).sum())); finger_contacts.append(fc)
        if frame % 240 == 0:
            adr=addresses[source["active"]]
            print(f"row={args.row} frame={frame} contacts={count} hand={hc.sum()} support={(~hc).sum()} xyz={pose[adr:adr+3].round(4).tolist()}",flush=True)
    trace = dict(qpos=np.asarray(qpos), qvel=np.asarray(qvel), ctrl=np.asarray(controls),
                 original_commands=source["original_commands"], reference_qpos=reference_qpos,
                 timestamps=source["timestamps"], initial_qpos=initial_qpos,
                 ncontacts=np.asarray(ncontacts), hand_contacts=np.asarray(hand_contacts),
                 support_contacts=np.asarray(support_contacts), finger_contacts=np.asarray(finger_contacts))
    np.savez_compressed(args.output / "trace.npz", **trace)
    adr=addresses[source["active"]]
    actual=trace["qpos"][:,adr:adr+3]; expected=reference_qpos[:,adr:adr+3]
    lifted=actual[:,2] - initial_qpos[adr+2]
    free_grip=(lifted > .02) & (trace["hand_contacts"] > 0) & (trace["support_contacts"] == 0)
    peak=int(np.argmax(expected[:,2]))
    demand=expected[:,2] - expected[0,2] > .02
    summary=dict(row=args.row, uuid=row["index"]["uuid"], gesture=row["index"]["gesture"], active_object=source["active"],
                 frames=len(qpos), simulation_seconds=len(qpos)/args.fps,
                 wall_seconds=time.monotonic()-started, source_peak_frame=peak,
                 source_max_lift_m=float(np.max(expected[:,2]-expected[0,2])),
                 actual_max_lift_m=float(lifted.max()), actual_lift_at_source_peak_m=float(lifted[peak]),
                 longest_free_grip_seconds=longest_duration(free_grip,args.fps),
                 free_grip_fraction_when_source_lifted=float(np.mean(free_grip[demand])) if demand.any() else None,
                 object_error_at_source_peak_m=float(np.linalg.norm(actual[peak]-expected[peak])),
                 final_object_position_error_m=float(np.linalg.norm(actual[-1]-expected[-1])),
                 wrist_position_rmse_m=float(np.sqrt(np.mean(np.sum((trace['qpos'][:,:3]-reference_qpos[:,:3])**2,axis=1)))),
                 max_contact_count=int(max(ncontacts)),
                 note="Free grip proxy: >2 cm lift, actual MJX hand contact, no other-body contact; inspect trace/video for task success.")
    (args.output / "summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print("REPLAY_COMPLETE",json.dumps(summary,ensure_ascii=False),flush=True)
    if args.render: render_trace(args.output, source, trace, args.fps)
    return summary


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root",type=Path,required=True,
                        help="Existing DexStream checkout at the pinned formal commit")
    parser.add_argument("--asset-manifest",type=Path,required=True,
                        help="Existing SHA-pinned Cheyingtong formal manifest (never regenerated)")
    parser.add_argument("--dataset",type=Path,required=True)
    parser.add_argument("--version",type=int,required=True)
    parser.add_argument("--row",type=int,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--fps",type=int,default=120,choices=(100,120))
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--contacts",type=int,default=8192)
    parser.add_argument("--ccd",type=int,default=8192)
    parser.add_argument("--render",action="store_true")
    replay(parser.parse_args())


if __name__ == "__main__":
    main()
