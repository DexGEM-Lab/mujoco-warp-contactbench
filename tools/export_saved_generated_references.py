#!/usr/bin/env python3
"""CPU-only export of selected saved 120Hz physical traces; never integrates."""
from collections import defaultdict
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native_state(trace, decoded, *, expected_contact_mode):
    import mujoco as mj
    import numpy as np
    from scipy.spatial.transform import Rotation as R
    from sim.manorl.assets import compile_unified_model, object_collision_vertices, object_runtime
    from sim.manorl.contracts import KEYPOINT_NAMES, FLOOR_TOP_Z
    from sim.manorl.environment import (MjxWarpPhysicalProducer, _FINGERTIP_NAMES,
        _FINGERTIP_LOCAL_OFFSETS, _dynamic_surface_template, _expected_keypoint_ids)
    from sim.manorl.observations import (ObservationState, PointCloudTemplate,
        SOURCE_ALIGNED_COMPATIBILITY, build_observation, geometry_encoding)
    names = trace['scene_object_names'].tolist()
    active = decoded['active_name']
    ai = names.index(active)
    _, model = compile_unified_model(object_types=names, object_collisions=True, physics_timestep=1 / 480)
    model.opt.cone = mj.mjtCone.mjCONE_ELLIPTIC
    model.opt.impratio = 100.
    producer = MjxWarpPhysicalProducer(mj, model, object_type=active)
    data = mj.MjData(model)
    assert model.nq == trace['qpos'].shape[1]
    for name, address in zip(names, trace['scene_object_qpos_addresses']):
        assert int(model.joint(name + '_free').qposadr[0]) == address
    assert [model.joint(i).name for i in range(28)] == decoded['joint_names']
    np.testing.assert_array_equal(model.actuator_trnid[:, 0], np.arange(28))
    kpids = producer.keypoint_body_ids
    tipids = [KEYPOINT_NAMES.index(k) for k in _FINGERTIP_NAMES]
    contacts, raw_contacts, keypoints, fingertips, palmq, forces = [], [], [], [], [], []
    for frame, pose in enumerate(trace['qpos']):
        data.qpos[:] = pose
        data.qvel[:] = trace['qvel'][frame]
        data.ctrl[:] = trace['ctrl'][frame]
        mj.mj_forward(model, data)
        kp = data.xpos[kpids].copy()
        tips = kp[tipids] + np.einsum('nij,nj->ni', data.xmat[np.asarray(kpids)[tipids]].reshape(-1,3,3), _FINGERTIP_LOCAL_OFFSETS)
        keypoints.append(kp)
        fingertips.append(tips)
        palmq.append(data.xquat[kpids[0]][[1,2,3,0]].copy())
        grouped = defaultdict(list)
        force = np.zeros((16,3))
        raw = []
        ob = producer.object_body_id
        orot = data.xmat[ob].reshape(3,3)
        wr = data.xmat[kpids[0]].reshape(3,3)
        for ci in range(data.ncon):
            c = data.contact[ci]
            g1,g2 = int(c.geom1),int(c.geom2)
            if g1 in producer.geom_to_keypoint and g2 in producer.object_geom_ids:
                hg, sign = g1, 1.
            elif g2 in producer.geom_to_keypoint and g1 in producer.object_geom_ids:
                hg, sign = g2, -1.
            else:
                continue
            wrench = np.zeros(6)
            mj.mj_contactForce(model,data,ci,wrench)
            ki = producer.geom_to_keypoint[hg]
            hb = int(model.geom_bodyid[hg])
            jr = data.xmat[hb].reshape(3,3)
            f = sign * wrench[0] * c.frame.reshape(3,3)[0]
            force[ki] += f
            grouped[ki].append(dict(force_normal=f.tolist(), pos_world=c.pos.tolist(),
                pos_wrist=(wr.T@(c.pos-data.xpos[kpids[0]])).tolist(),
                pos_joint=(jr.T@(c.pos-data.xpos[hb])).tolist(),
                pos_object=(orot.T@(c.pos-data.xpos[ob])).tolist()))
        entries=[]
        for ki,pairs in sorted(grouped.items()):
            f=force[ki]
            jr=data.xmat[kpids[ki]].reshape(3,3)
            entries.append(dict(hand_name='right',joint_name=KEYPOINT_NAMES[ki],object_name=active,
                total_force_world=f.tolist(),total_force_wrist=(wr.T@f).tolist(),
                total_force_joint=(jr.T@f).tolist(),total_force_object=(orot.T@f).tolist(),contact_pairs=pairs))
        contacts.append(entries)
        raw_contacts.append(raw)
        forces.append(force)
    n=len(keypoints)
    kp=np.asarray(keypoints); tips=np.asarray(fingertips)
    vertices=object_collision_vertices(active)
    action=decoded['action']
    mask=np.zeros((n,16)); mask[:,_expected_keypoint_ids(active,f'{action:02d}', mode=expected_contact_mode)]=1
    # No residual policy exists in this replay; its cumulative residual is exactly zero.
    state=ObservationState(mano_dof_pos=trace['qpos'][:,:28],
        mano_dof_lower=model.jnt_range[:28,0],mano_dof_upper=model.jnt_range[:28,1],
        hand_position=kp[:,0],hand_orientation_xyzw=np.asarray(palmq),
        object_position=trace['actual_object_pos'][:,ai],object_orientation_xyzw=trace['actual_object_quat_xyzw'][:,ai],
        target_object_position=decoded['scene_pos'][:,ai],target_object_orientation_xyzw=decoded['scene_quat_xyzw'][:,ai],
        target_object_pos_next_5=decoded['scene_pos'][np.minimum(np.arange(n)+5,n-1),ai],
        cumulative_offset=np.zeros((n,3)),cumulative_joint_offset=np.zeros((n,22)),
        point_cloud=PointCloudTemplate(np.broadcast_to(_dynamic_surface_template(np.random.default_rng(0),active),(n,64,3)).copy(),mode='dynamic_reset'),
        object_geometry=np.broadcast_to(geometry_encoding(object_name=active,geometry_type=object_runtime(active).geometry_type,dimensions=np.ptp(vertices,axis=0)),(n,12)),
        hand_keypoint_positions=kp,fingertip_positions=tips,hand_keypoint_contact_forces=np.asarray(forces),
        expected_contact_mask=mask,action_ids=np.full(n,action,dtype=np.int64),object_support_points=vertices,table_surface_height=FLOOR_TOP_Z)
    obs=build_observation(state,compatibility=SOURCE_ALIGNED_COMPATIBILITY).policy_input
    assert obs.shape==(n,480) and np.any(obs)
    return np.concatenate([kp,tips],axis=1),contacts,raw_contacts,obs,model.jnt_range[:28].copy()



def assemble(entry, *, expected_contact_mode, raw_cache, model_profile, software_commit):
    import lance
    import numpy as np
    from scipy.spatial.transform import Rotation as R
    from sim.manorl import assets
    from sim.manorl.contracts import TrajectoryIdentity
    from sim.manorl.local_contact_repair import MODEL_FIELDS, array_sha
    from sim.manorl.trajectory import ReferenceTrajectory, trajectory_from_lance_row
    from sim.manorl.lance_v2 import build_v2_row
    folder = Path(entry['run_path'])
    run = folder if (folder/'trace.npz').is_file() else folder/'replay'
    result = load(folder/'result.json')
    if not (result.get('accepted') is True or result.get('physical_pose_pass') is True):
        raise ValueError(f"unaccepted physical trace: {entry['id']}")
    if sha(run/'trace.npz') != result['trace_sha256']:
        raise ValueError('physical trace hash mismatch')
    metrics = load(run/'metrics.json')
    src = metrics['source']
    source_key = (src['dataset'], src['version'], src['row'])
    if source_key not in raw_cache:
        dataset = lance.dataset(src['dataset'], version=src['version'])
        raw_cache[source_key] = dataset.take([src['row']], columns=['index','trajectory_metadata']).to_pylist()[0]
    raw = raw_cache[source_key]
    if raw['index']['uuid'] != src['uuid']:
        raise ValueError('raw capture UUID mismatch')
    with np.load(run/'trace.npz', allow_pickle=False) as z:
        trace = {k:z[k] for k in z.files}
    n = len(trace['qpos'])
    if trace['ctrl_substeps'].shape != (n-1,4,28):
        raise ValueError('expected prestep frame0 and N-1 four-substep intervals')
    runtime = metrics['runtime']
    if runtime.get('control_hz') != 120 or runtime.get('physics_hz') != 480 or runtime.get('frame_zero_integrated') is not False:
        raise ValueError('invalid physical clock')
    active = metrics['source_metadata']['active_object']
    names = trace['scene_object_names'].tolist(); ai=names.index(active)
    action = int(metrics['source_metadata']['source_gesture'][:3])
    decoded = dict(active_name=active, action=action, joint_names=list(assets.hand_joint_names('right')),
                   scene_pos=trace['source_object_pos'], scene_quat_xyzw=trace['source_object_quat_xyzw'])
    _, model = assets.compile_unified_model(object_types=names, object_collisions=True, physics_timestep=1/480)
    for key in MODEL_FIELDS:
        if array_sha(getattr(model,key)) != runtime['model_array_hashes'][key]:
            raise ValueError(f'export model does not match physical trace: {key}')
    joint,contacts,_,obs,limits = native_state(trace,decoded,expected_contact_mode=expected_contact_mode)
    movement = metrics['source_metadata']['object_move'][0]
    start = int(np.rint(movement['start_frame']*1.2));end=min(n-1,int(np.rint(movement['end_frame']*1.2)))
    identity = TrajectoryIdentity(src['dataset'],src['version'],src['row'],ai,src['uuid'],src['uuid'],
                                  f'{active}_{action:02d}_{src["row"]}',0,n,start,end)
    trajectory = ReferenceTrajectory(identity,src['version'],np.arange(n),np.arange(n)/120,
                                    trace['desired'],trace['source_object_pos'][:,ai],trace['source_object_pos'][:,ai],
                                    trace['source_object_quat_xyzw'][:,ai],0.,reference_fps=120,control_fps=120,
                                    movement_start_step=start,movement_end_step=end)
    command=trace['ctrl'][1:];z=np.zeros(n-1)
    rollout=dict(observation_t=obs[:-1],next_observation=obs[1:],policy_mean_action=command,processed_action=command,
                 cumulative_position_residual=np.zeros((n-1,3)),cumulative_joint_residual=np.zeros((n-1,22)),
                 command_reference_index=np.arange(1,n),command_source_frame_index=np.arange(1,n),
                 reference_target=trace['desired'][1:],processed_target=np.clip(trace['desired'],limits[:,0],limits[:,1])[1:],
                 controller_target=command,reward=z,raw_contact_reward=z,contact_reward=z,
                 terminated=np.arange(n-1)==n-2,termination_reason_code=(np.arange(n-1)==n-2).astype(np.int32))
    recipe = load(folder/'provenance.json')
    recipe_sha = sha(folder/'provenance.json')
    metadata=dict(kind='saved_free_object_start_augmentation',frame_zero_integrated=False,
                  physical_hand_profile=model_profile,raw_capture=raw,source=src,
                  predecessor_replay_hand='sunke',saved_run_provenance=recipe,
                  trace_sha256=sha(run/'trace.npz'),substep_control_shape=list(trace['ctrl_substeps'].shape),
                  retained_trace='per-variant NPZ holds finger_target_delta, donor envelopes and480Hz controls',
                  original_source_frame_coordinates=trace['source_frame'].tolist(),
                  source_frame_semantics='canonical indices name120Hz reference grid; original fractional100Hz coordinates stored separately',
                  observations=dict(expected_contact_mode=expected_contact_mode,contact_forces='CPU mj_forward normal-only, not Warp force telemetry',
                                    object_reference='saved source-object reference',point_cloud_seed=0,policy_residual='zero; no learned residual policy'),
                  actions='absolute arrival-frame final-substep actuator targets, not normalized PPO actions or exact480Hz replay',
                  reward='zero, not computed',termination='recipe completion only',physical_acceptance=result)
    physical_index=dict(raw['index'],operator='cheyingtong')
    physical_metadata=dict(raw['trajectory_metadata'],hand_names=['right'],mano_hand_shapes=[model_profile['betas']],
                           gesture=metrics['source_metadata']['source_gesture'])
    generation = hashlib.sha256(json.dumps(dict(id=entry['id'],trace=metadata['trace_sha256'],recipe=recipe_sha,
                                                profile=model_profile,clock=[120,480]),sort_keys=True).encode()).hexdigest()
    row=build_v2_row(trajectory=trajectory,source_index=physical_index,source_metadata=physical_metadata,
                    states=dict(urdf_dof=trace['qpos'][:,:28],urdf_dof_target=trace['ctrl'],mano_joint_pos=joint,
                                object_position=trace['actual_object_pos'][:,ai],object_orientation_xyzw=trace['actual_object_quat_xyzw'][:,ai]),
                    contacts=contacts,rollout=rollout,provenance=dict(checkpoint_path='generation-recipe:'+entry['id'],
                      checkpoint_sha256=generation,checkpoint_update=0,checkpoint_metadata=metadata,software_commit=software_commit,
                      seed=2026091701,episode_index=0,generation_attempt=1,augmentation_identity=generation))
    row['provenance']['policy_mode']='replay_contact_repair'
    order=[ai]+[i for i in range(len(names))if i!=ai]
    row['trajectory_metadata']['object_names']=[names[i] for i in order]
    row['objects']=[dict(pos=trace['actual_object_pos'][:,i].tolist(),rot_aa=R.from_quat(trace['actual_object_quat_xyzw'][:,i]).as_rotvec().tolist())for i in order]
    decoded_row=trajectory_from_lance_row(row,dataset_version=1,row_index=0,hand_side='right',generated_reference=True)
    np.testing.assert_array_equal(decoded_row.q_ref,trace['qpos'][:,:28].astype(np.float32))
    np.testing.assert_array_equal(decoded_row.object_pos,trace['actual_object_pos'][:,ai])
    assert len(decoded_row.q_ref)==n and decoded_row.object_z_shift==0
    return row,dict(id=entry['id'],uuid=row['index']['uuid'],source_uuid=src['uuid'],frames=n,run_path=str(folder.resolve()),
                    trace_sha256=metadata['trace_sha256'],raw_capture_operator=raw['index'].get('operator'),physical_hand='cheyingtong')


def main():
    import sys
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('selection','manifest','asset-root','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--expected-contact-mode',choices=('source_mapping','five_fingertips'),required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    manifest=load(args.manifest)
    expected='e686d91931979c444c923d4f962ddd135ce8204e0de9df6fd7368bfdb855e29b'
    if sha(args.manifest)!=expected or manifest['hand_operator']!='cheyingtong':raise ValueError('physical profile mismatch')
    os.environ['MANORL_ASSET_MANIFEST']=str(args.manifest.resolve())
    from sim.manorl import assets
    assets.DEXSTREAM_ROOT=args.asset_root.resolve()
    from sim.manorl.lance_v2 import write_v2_lance
    from sim.manorl.trajectory import TrajectorySelection,_discover_trajectory_candidates,trajectory_from_lance_row
    import lance
    import numpy as np
    from scipy.spatial.transform import Rotation as R
    entries=load(args.selection)
    if isinstance(entries,dict):entries=entries['rows']
    if not entries or len({e['id']for e in entries})!=len(entries):raise ValueError('empty or duplicate selection')
    raw_cache={}; records=[]; uuids=set()
    profile=dict(operator='cheyingtong',hand_side='right',betas=manifest['hands']['right']['betas'],
                 manifest_sha256=expected,source_commit=manifest['source_commit'])
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    for index,entry in enumerate(entries):
        row,record=assemble(entry,expected_contact_mode=args.expected_contact_mode,raw_cache=raw_cache,model_profile=profile,software_commit=commit)
        if record['uuid']in uuids:raise ValueError('generated UUID collision')
        uuids.add(record['uuid'])
        write_v2_lance([row],output=args.output,observation_dim=480,action_dim=28,append=index>0)
        # Check actual canonical Arrow/Lance float32 serialization, not only the Python mapping.
        saved=lance.dataset(str(args.output)).take([index]).to_pylist()[0]
        for key in ('urdf_dof','urdf_dof_target'):
            np.testing.assert_array_equal(np.asarray(saved['hands'][0][key]),np.asarray(row['hands'][0][key],dtype=np.float32))
        for left,right in zip(saved['objects'],row['objects']):
            np.testing.assert_array_equal(np.asarray(left['pos']),np.asarray(right['pos'],dtype=np.float32))
            np.testing.assert_allclose(R.from_rotvec(left['rot_aa']).as_matrix(),R.from_rotvec(right['rot_aa']).as_matrix(),atol=2e-7,rtol=0)
        for key in ('observation_t','next_observation','controller_target','reference_target'):
            np.testing.assert_array_equal(np.asarray(saved['rollout'][key]),np.asarray(row['rollout'][key],dtype=np.float32))
        decoded=trajectory_from_lance_row(saved,dataset_version=index+1,row_index=index,hand_side='right',generated_reference=True)
        np.testing.assert_array_equal(decoded.q_ref,np.asarray(saved['hands'][0]['urdf_dof']))
        assert len(decoded.q_ref)==record['frames'] and decoded.object_z_shift==0
        records.append(record)
        args.output.with_suffix('.progress.json').write_text(json.dumps(dict(completed=len(records),planned=len(entries),last=record),indent=2)+'\n')
        print('EXPORTED',index+1,len(entries),entry['id'],record['frames'],flush=True)
    dataset=lance.dataset(str(args.output))
    selection=TrajectorySelection(selector='all',dataset_path=args.output,expected_dataset_version=dataset.version,
                                  generated_reference=True,reference_fps=120,pre_padding=0,post_padding=0,hand_side='right')
    pairs,candidates=_discover_trajectory_candidates(dataset,selection)
    assert sum(map(len,candidates.values()))==len(records)
    try:
        _discover_trajectory_candidates(dataset,TrajectorySelection(selector='all',dataset_path=args.output,pre_padding=0,post_padding=0))
    except LookupError:
        default_excludes=True
    else:
        raise AssertionError('raw default unexpectedly accepts generated rows')
    output=dict(schema='manorl.generated-reference-export.v1',rows=records,row_count=len(records),dataset_version=int(dataset.version),
                expected_contact_mode=args.expected_contact_mode,physical_profile=profile,default_raw_excludes=default_excludes,
                opt_in_discovered=len(records),pairs=[p.canonical for p in pairs],full_episode_roundtrip=True,
                source_hand_profile_semantics='sunke is predecessor physical replay, raw capture operator preserved separately',
                retained_controller_artifact='input per-variant trace.npz with480Hz substeps; Lance final targets alone are not equivalent replay')
    args.output.with_suffix('.manifest.json').write_text(json.dumps(output,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in output.items()if k!='rows'},ensure_ascii=False))


if __name__=='__main__':
    main()
