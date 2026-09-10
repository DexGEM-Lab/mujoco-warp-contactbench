"""Replay a versioned human-capture repair with explicit MJX-Warp solver settings.

The hand receives position-actuator targets. All scene objects remain free
bodies; no object pose, velocity, force, or constraint is overwritten after reset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import threading
import time

import lance
import numpy as np
from scipy.spatial.transform import Rotation
import warp as wp
from mujoco.mjx.third_party import mujoco_warp as mw

from replay_pose_edits import apply_replay_edits
from sim.manorl.assets import COLLISION_GEOM_GROUP, asset_provenance, compile_unified_model
from sim.manorl.mjx_sim import command_target
from sim.manorl.trajectory import trajectory_from_lance_row, resample_reference_trajectory


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--patch', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--post-padding', type=int, default=180)
    parser.add_argument('--solver', choices=('recorded', 'default'), default='recorded')
    parser.add_argument('--view', action='store_true')
    parser.add_argument('--loop', action='store_true')
    parser.add_argument('--speed', type=float, default=1.)
    args = parser.parse_args()
    if args.post_padding < 0 or not np.isfinite(args.speed) or args.speed <= 0:
        parser.error('post-padding must be nonnegative and speed must be positive')
    if args.loop and not args.view:
        parser.error('--loop requires --view')
    return args


def load_command_track(patch_path, record, reference_count):
    """Validate immutable actuator targets and their source-frame mapping."""
    directory = patch_path.resolve().parent
    command_path = (directory / record['path']).resolve()
    if not command_path.is_relative_to(directory):
        raise ValueError('command track must stay inside the patch directory')
    if hashlib.sha256(command_path.read_bytes()).hexdigest() != record['sha256']:
        raise ValueError('command-track hash differs from patch')
    with np.load(command_path, allow_pickle=False) as stored:
        commands = stored['ctrl'].copy()
        frame_map = stored['reference_index'].copy()
    if commands.shape != (int(record['frames']),28) or not np.all(np.isfinite(commands)):
        raise ValueError('command track must contain finite Nx28 actuator targets')
    if (frame_map.shape != (len(commands),)
            or not np.issubdtype(frame_map.dtype,np.integer)
            or np.any(frame_map<0) or np.any(frame_map>=reference_count)
            or np.any(np.diff(frame_map)<0)):
        raise ValueError('command track requires monotonic valid source-reference indices')
    return commands, frame_map


def main():
    args = parse_args()
    recipe = json.loads(args.patch.read_text())
    if recipe.get('schema') != 'direct_capture_repair.v1':
        raise ValueError('patch requires schema direct_capture_repair.v1')
    source = recipe['source']
    dataset = lance.dataset(args.dataset, version=int(source['version']))
    if args.dataset.name != source['dataset_name']:
        raise ValueError('dataset basename differs from patch source')
    row = dataset.take([int(source['row'])], columns=[
        'index', 'trajectory_metadata', 'timestamp', 'hands', 'objects',
    ]).to_pylist()[0]
    if row['index']['uuid'] != source['uuid']:
        raise ValueError('source row UUID differs from patch')
    if int(row['trajectory_metadata']['total_frames']) != int(source['frames']):
        raise ValueError('source frame count differs from patch')
    base = resample_reference_trajectory(
        trajectory_from_lance_row(row, dataset.version, row_index=source['row'],
                                  hand_side='right', pre_padding=180,
                                  post_padding=args.post_padding),
        reference_fps=100, control_fps=100,
    )
    trajectory = apply_replay_edits(base, recipe['edit'])
    commands = trajectory.q_ref
    frame_map = np.arange(len(commands), dtype=np.int64)
    if 'command_track' in recipe:
        commands, frame_map = load_command_track(
            args.patch, recipe['command_track'], len(trajectory.q_ref),
        )
    names = tuple(sorted(trajectory.scene_object_types))
    active_name = trajectory.scene_object_types[trajectory.identity.object_index]
    if active_name != recipe['active_object']:
        raise ValueError('patch active object differs from the source selection')
    mj, model = compile_unified_model(object_types=names, object_collisions=True,
                                      physics_timestep=.0025)
    solver = {'cone': 'pyramidal', 'impratio': 1.}
    if args.solver == 'recorded':
        solver = recipe['solver']
    if solver['cone'] not in ('pyramidal', 'elliptic'):
        raise ValueError('unsupported friction cone')
    impratio = float(solver['impratio'])
    if not np.isfinite(impratio) or impratio <= 0:
        raise ValueError('impratio must be finite and positive')
    model.opt.cone = (mj.mjtCone.mjCONE_ELLIPTIC if solver['cone'] == 'elliptic'
                      else mj.mjtCone.mjCONE_PYRAMIDAL)
    model.opt.impratio = impratio
    data = mj.MjData(model)
    object_qposadr = {}
    for name in names:
        source_index = trajectory.scene_object_types.index(name)
        adr = int(model.joint(name + '_free').qposadr[0])
        object_qposadr[name] = adr
        data.qpos[adr:adr+3] = trajectory.scene_object_initial_pos[source_index]
        data.qpos[adr+3:adr+7] = trajectory.scene_object_initial_quat_xyzw[source_index][[3,0,1,2]]
    lower, upper = model.jnt_range[:28].T
    initial_hand = np.asarray(recipe.get('initial_hand', trajectory.q_ref[0]), dtype=np.float64)
    if initial_hand.shape != (28,) or not np.all(np.isfinite(initial_hand)):
        raise ValueError('initial_hand must contain 28 finite coordinates')
    data.qpos[:28] = initial_hand
    data.ctrl[:] = command_target(commands[0], data.qpos[:28], lower, upper)
    mj.mj_forward(model, data)
    initial_qpos = data.qpos.copy()
    initial_ctrl = data.ctrl.copy()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        'schema': 'direct_capture_repair_run.v1', 'source': source,
        'source_gesture': row['index']['gesture'],
        'dataset_path': str(args.dataset.resolve()), 'patch': recipe,
        'solver': solver, 'model_asset_provenance': asset_provenance(),
        'mujoco_version': mj.__version__, 'warp_version': wp.__version__,
        'reference_hz': 100, 'control_hz': 100, 'physics_hz': 400,
        'post_padding': args.post_padding, 'objects': names,
        'object_qpos_addresses': object_qposadr,
        'no_object_pose_forcing_after_reset': True,
        'no_object_external_forces': True, 'no_learned_policy': True,
        'same_masses_friction_coefficients_and_actuator_parameters': True,
        'source_ground_shift_m': trajectory.object_z_shift,
        'movement_steps': [trajectory.movement_start_step, trajectory.movement_end_step],
        'termination': 'complete selected input and explicit held tail; no deviation resets',
    }
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps({'source': source, 'solver': solver, 'output': str(args.output)}, ensure_ascii=False), flush=True)
    wp.init()
    wp.set_device('cuda:0')
    wm = mw.put_model(model)
    wd = mw.put_data(model, data, nworld=1, nconmax=512, nccdmax=512, njmax=4000)
    # Compile once, then capture only the fixed four-substep physics transition.
    for _ in range(4):
        mw.step(wm, wd)
    with wp.ScopedCapture() as capture:
        for _ in range(4):
            mw.step(wm, wd)

    viewer = None
    render_thread = None
    render_data = None
    if args.view:
        from mujoco import viewer as mj_viewer
        _, visual = compile_unified_model(object_types=names, object_collisions=True,
                                           physics_timestep=.0025, visual_meshes=True)
        if visual.nq != model.nq or [visual.joint(i).name for i in range(visual.njnt)] != [model.joint(i).name for i in range(model.njnt)]:
            raise ValueError('viewer joint ABI differs from physics model')
        render_data = mj.MjData(visual)
        render_data.qpos[:] = initial_qpos
        mj.mj_forward(visual, render_data)
        existing_threads = set(threading.enumerate())
        viewer = mj_viewer.launch_passive(visual, render_data, show_left_ui=False, show_right_ui=False)
        # launch_passive uses a daemon render thread on Linux. Its close()
        # requests exit but does not join; process teardown can otherwise destroy
        # the X11 drawable while that thread is still swapping buffers.
        render_threads = [thread for thread in threading.enumerate()
                          if thread not in existing_threads and '_launch_internal' in thread.name]
        if len(render_threads) == 1:
            render_thread = render_threads[0]
        else:
            viewer.close()
            raise RuntimeError('cannot identify the passive viewer render thread for orderly shutdown')
        with viewer.lock():
            viewer.opt.geomgroup[COLLISION_GEOM_GROUP] = 0
            viewer.cam.lookat[:] = trajectory.object_pos[0] + [0.,0.,.12]
            viewer.cam.distance = .72
            viewer.cam.azimuth = 100
            viewer.cam.elevation = -30
            viewer.sync()
    saved = False
    try:
        while True:
            # Reset all integrator state that can affect the captured transition.
            wd.qpos.assign(initial_qpos[None].astype(np.float32))
            wd.qvel.zero_(); wd.qacc_warmstart.zero_(); wd.time.zero_()
            wd.ctrl.assign(initial_ctrl[None].astype(np.float32))
            qpos, qvel, controls, reference_indices = [], [], [], []
            for index in range(len(commands)):
                if viewer is not None and not viewer.is_running():
                    break
                started = time.perf_counter()
                current = wd.qpos.numpy()[0]
                target = command_target(commands[index], current[:28], lower, upper)
                wd.ctrl.assign(target[None].astype(np.float32))
                wp.capture_launch(capture.graph)
                pose = wd.qpos.numpy()[0]
                velocity = wd.qvel.numpy()[0]
                qpos.append(pose.copy()); qvel.append(velocity.copy())
                controls.append(target.copy()); reference_indices.append(int(frame_map[index]))
                if index % 100 == 0:
                    adr = object_qposadr[active_name]
                    print(f'frame={index} source={trajectory.source_indices[frame_map[index]]} object_xyz={pose[adr:adr+3].round(4).tolist()}', flush=True)
                if viewer is not None:
                    with viewer.lock():
                        render_data.qpos[:] = pose
                        render_data.qvel[:] = velocity
                        render_data.ctrl[:] = target
                        render_data.time = (index+1)/100.
                        mj.mj_forward(visual, render_data)
                        viewer.sync()
                    time.sleep(max(0., .01/args.speed-(time.perf_counter()-started)))
            if not saved:
                indices = np.asarray(reference_indices, dtype=np.int64)
                poses = np.asarray(qpos)
                if len(poses) == 0:
                    raise RuntimeError('viewer closed before any frame was simulated')
                np.savez_compressed(args.output/'trajectory.npz',
                    qpos=poses, qvel=qvel, ctrl=controls, initial_qpos=initial_qpos,
                    scene_object_names=np.asarray(names),
                    scene_object_qpos_addresses=np.asarray([object_qposadr[n] for n in names]),
                    scene_object_pos=np.stack([poses[:,object_qposadr[n]:object_qposadr[n]+3] for n in names],axis=1),
                    scene_object_rot_aa=np.stack([Rotation.from_quat(poses[:,object_qposadr[n]+3:object_qposadr[n]+7][:,[1,2,3,0]]).as_rotvec() for n in names],axis=1),
                    source_frame=trajectory.source_indices[indices], source_ground_shift=trajectory.object_z_shift,
                    reference_indices=indices, reference_hand=trajectory.q_ref[indices],
                    reference_object_pos=trajectory.object_pos[indices],
                    reference_object_quat_xyzw=trajectory.object_quat_xyzw[indices],
                    time_seconds=(np.arange(len(indices))+1)/100.,
                )
                adr = object_qposadr[active_name]
                active_pos = poses[:,adr:adr+3]
                lift = active_pos[:,2]-trajectory.object_pos[0,2]
                hold = indices >= trajectory.movement_end_step
                result = {
                    'complete': len(poses) == len(commands),
                    'frames': len(poses), 'maximum_lift_m': float(lift.max()),
                    'last_lift_m': float(lift[-1]),
                    'hold_minimum_lift_m': float(lift[hold].min()) if np.any(hold) else None,
                    'last_position_error_m': float(np.linalg.norm(active_pos[-1]-trajectory.object_pos[indices[-1]])),
                    'last_second_displacement_m': (active_pos[-1]-active_pos[max(0,len(active_pos)-101)]).tolist(),
                }
                (args.output/'result.json').write_text(json.dumps(result, indent=2))
                print(json.dumps(result), flush=True)
                saved = True
            if not args.loop or not viewer.is_running():
                break
    finally:
        if viewer is not None:
            viewer.close()
            if render_thread is not None:
                render_thread.join(timeout=10.)
                if render_thread.is_alive():
                    raise RuntimeError('viewer render thread failed to stop after close')


if __name__ == '__main__':
    main()
