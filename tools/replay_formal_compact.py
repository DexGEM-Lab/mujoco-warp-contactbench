#!/usr/bin/env python3
"""Replay formal multi-object compact controls, preserving arrival-frame timing.

No policy inference or physical-state forcing after initialization. Native Warp
contacts and last-forward poses are measured at the source recording boundary.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.spatial.transform import Rotation

PROFILE_SHA = 'e686d91931979c444c923d4f962ddd135ce8204e0de9df6fd7368bfdb855e29b'
ASSET_COMMIT = '778614d09e917deffed0bff3f357aa237efa762d'
CONTRACT = 'synthetic_mano_target_replay_visual_v2_contact'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def arrival_target(targets, frame):
    """State0 is prestep; target[t] was applied to generate state[t]."""
    if not 1 <= frame < len(targets):
        raise ValueError('transition arrival frame must be1..N-1')
    return targets[frame]


def metadata_digest(extra):
    return hashlib.sha256(json.dumps(extra, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def decode_row(row):
    metadata, provenance = row['trajectory_metadata'], row['provenance']
    if provenance['contract'] != CONTRACT or row['index']['is_generated'] is not True:
        raise ValueError('not a formal compact generated row')
    if (metadata['data_fps'], provenance['control_fps'], provenance['physics_fps'],
            provenance['physics_substeps_per_control']) != (120,120,480,4):
        raise ValueError('the requested120/480Hz clock differs from the row')
    if metadata['hand_names'] != ['right'] or metadata['hand_slots'] != ['right','left']:
        raise ValueError('expected canonical right-only hand slots')
    hands = row['hands']
    if len(hands)!=2 or hands[0]['hand_name']!='right' or hands[1]['hand_name'] is not None:
        raise ValueError('hand slot mismatch')
    n = metadata['total_frames']; names = metadata['object_names']
    if len(set(names))!=len(names) or len(names)!=len(row['objects']):
        raise ValueError('scene object ordering mismatch')
    move = metadata['trajectory_info']['object_move']
    if len(move)!=1 or move[0]['object_name'] not in names:
        raise ValueError('expected one named active object')
    active = move[0]['object_name']
    identity = provenance['source_identity']
    if identity.rsplit('_',2)[0] != active:
        raise ValueError('active object differs from source identity')
    time = np.asarray(row['timestamp'],dtype=float)
    if time.shape!=(n,) or not np.allclose(time,np.arange(n)/120,atol=1e-10,rtol=0):
        raise ValueError('invalid frame clock')
    state = np.asarray(hands[0]['urdf_dof'],dtype=float)
    target = np.asarray(hands[0]['urdf_dof_target'],dtype=float)
    pos = np.stack([x['pos']for x in row['objects']],axis=1).astype(float)
    quat = np.stack([Rotation.from_rotvec(x['rot_aa']).as_quat()for x in row['objects']],axis=1)
    if state.shape!=(n,28) or target.shape!=(n,28) or pos.shape!=(n,len(names),3):
        raise ValueError('malformed state/target arrays')
    reference = row['reference']
    arrays = dict(recorded_hand=state, targets=target, recorded_object_pos=pos, recorded_object_quat_xyzw=quat,
                  reference_hand=np.asarray(reference['hand_urdf_dof'],dtype=float),
                  reference_object_pos=np.asarray(reference['object_pos'],dtype=float),
                  reference_object_quat_xyzw=Rotation.from_rotvec(reference['object_rot_aa']).as_quat(),
                  source_frame=np.asarray(reference['source_frame_index']), time=time,
                  command_reference_index=np.asarray(row['command_reference_index'],dtype=np.int64))
    if arrays['command_reference_index'].shape!=(n-1,) or arrays['reference_object_pos'].shape!=(n,3):
        raise ValueError('reference or command indices not aligned')
    for key,value in arrays.items():
        if not np.isfinite(value).all(): raise ValueError('nonfinite '+key)
    info = dict(uuid=row['index']['uuid'], source_uuid=row['index']['seed_uuid'], identity=identity,
                frames=n, names=names, active=active, action=int(identity.rsplit('_',2)[1]),
                movement=move[0], provenance=provenance, operator=row['index']['operator'],
                betas=metadata['mano_hand_shapes'][0])
    return info,arrays


def prepare(a):
    import lance
    root=a.dataset.parent
    dataset=lance.dataset(str(a.dataset),version=1)
    if dataset.count_rows()!=28:raise ValueError('expected the specified28-row formal dataset')
    profile=json.loads((root/'asset_manifest.json').read_text())
    if sha(root/'asset_manifest.json')!=PROFILE_SHA or profile['hand_operator']!='cheyingtong':
        raise ValueError('physical hand manifest mismatch')
    metadata=json.loads((root/'row_checkpoint_metadata.json').read_text())
    bundle=json.loads((root/'manifest.json').read_text())
    catalog={r['uuid']:r for r in bundle['catalog']}
    a.output.mkdir(parents=True,exist_ok=False)
    prepared=a.output/'inputs';prepared.mkdir()
    rows=[]
    for index in range(dataset.count_rows()):
        row=dataset.take([index]).to_pylist()[0]
        info,arrays=decode_row(row)
        entry=catalog[info['uuid']];extra=metadata[info['uuid']]
        if metadata_digest(extra)!=info['provenance']['checkpoint_metadata_sha256']:
            raise ValueError('external row metadata digest differs')
        if extra['frame_zero_integrated'] is not False or extra['physical_hand_profile']['manifest_sha256']!=PROFILE_SHA:
            raise ValueError('physical profile or frame0 metadata differs')
        np.testing.assert_allclose(info['betas'],profile['hands']['right']['betas'],atol=1e-7,rtol=0)
        if info['operator']!='cheyingtong':raise ValueError('operator mismatch')
        audit=root/'audit'/entry['group']
        model_audit=json.loads((audit/'model_audit.json').read_text())
        with np.load(audit/'trace.npz',allow_pickle=False)as source:
            env=entry['env']
            # Independent recording evidence establishes arrival-indexed controls.
            sub=source[f'ctrl_substeps_{env}']
            np.testing.assert_array_equal(sub,np.broadcast_to(arrays['targets'][1:,None,:],sub.shape))
            np.testing.assert_array_equal(source[f'qvel_{env}'][0],0.)
            np.testing.assert_array_equal(source[f'qpos_{env}'][0,:28],arrays['recorded_hand'][0])
            arrays['audit_initial_qpos']=source[f'qpos_{env}'][0].copy()
            arrays['target_reference_index']=source[f'target_index_{env}'].astype(np.int64)
            arrays['audit_first_qpos']=source[f'qpos_{env}'][:8].copy()
        env_options=extra['runtime_config']['environment']
        info.update(row_index=index,group=entry['group'],recorded_env=entry['env'],model_audit=model_audit,
                    environment=env_options,original_acceptance=extra['formal_export']['acceptance'],
                    frame_zero_integrated=False,control_schedule='targets[t] produces state[t] for t>=1;four equal substeps')
        np.savez_compressed(prepared/f'{index:03d}.npz',**arrays)
        dump(prepared/f'{index:03d}.json',info)
        rows.append(dict(row_index=index,uuid=info['uuid'],identity=info['identity'],group=info['group'],frames=info['frames']))
        print('PREPARED',index,info['identity'],info['frames'],flush=True)
    dump(a.output/'input_manifest.json',dict(dataset=str(a.dataset),dataset_version=1,rows=rows,
                manifest_sha256=PROFILE_SHA,source_checksums_sha256=sha(root/'checksums.json'),
                row_metadata_sha256=sha(root/'row_checkpoint_metadata.json'),
                schema_metadata={k.decode():v.decode()for k,v in dataset.schema.metadata.items()},
                prepared_at=datetime.now(timezone.utc).isoformat(),source_data_read_only=True))


def bin_entry(state, world, body, bin_body, mesh):
    vertices=Rotation.from_quat(state.xquat[world,bin_body][[1,2,3,0]]).apply(mesh)+state.xpos[world,bin_body]
    rim=vertices[vertices[:,2]>=vertices[:,2].max()-.004]
    lo,hi=rim[:,:2].min(0),rim[:,:2].max(0)
    center,radius=(lo+hi)/2,(hi-lo)/2
    pos=state.xpos[world,body]
    radial=float(np.linalg.norm((pos[:2]-center)/radius))
    return dict(inside_opening=radial<.8,below_rim=bool(pos[2]<vertices[:,2].max()),
                above_bin_bottom=bool(pos[2]>vertices[:,2].min()),normalized_radius=radial,
                height_relative_rim_m=float(pos[2]-vertices[:,2].max()),center_world=pos.tolist())


def run_group(a):
    if sha(a.dataset.parent/'asset_manifest.json')!=PROFILE_SHA:raise ValueError('asset manifest changed')
    os.environ['MANORL_ASSET_MANIFEST']=str(a.dataset.parent/'asset_manifest.json')
    from sim.manorl import assets
    assets.DEXSTREAM_ROOT=a.asset_root
    from sim.manorl.trajectory import ReferenceTrajectory,TrajectoryBatch
    from sim.manorl.contracts import TrajectoryIdentity
    from sim.manorl.environment import EnvironmentConfig,MujocoManoEnvironment
    from sim.manorl.observations import SOURCE_ALIGNED_COMPATIBILITY
    from sim.manorl.lance_v2 import corrected_contact_frames
    from sim.manorl.synthesis_acceptance import count_hand_object_contact_frames,evaluate_synthesis_acceptance,persisted_rotation_quaternion_xyzw
    manifest=json.loads((a.output/'input_manifest.json').read_text())
    infos=[];data=[];trajectories=[]
    for row in manifest['rows']:
        if row['group']!=a.group:continue
        index=row['row_index'];info=json.loads((a.output/'inputs'/f'{index:03d}.json').read_text())
        with np.load(a.output/'inputs'/f'{index:03d}.npz')as z:array={k:z[k]for k in z.files}
        n=info['frames'];ai=info['names'].index(info['active']);src=info['provenance']
        identity=TrajectoryIdentity(str(a.dataset),1,index,ai,info['uuid'],info['uuid'],info['identity'],0,n,
                                    info['movement']['start_frame'],info['movement']['end_frame'])
        trajectories.append(ReferenceTrajectory(identity,1,np.arange(n),array['time'],array['recorded_hand'],
            array['recorded_object_pos'][:,ai],array['recorded_object_pos'][:,ai],array['recorded_object_quat_xyzw'][:,ai],0.,
            reference_fps=120,control_fps=120,movement_start_step=info['movement']['start_frame'],
            movement_end_step=info['movement']['end_frame'],scene_object_types=tuple(info['names']),
            scene_object_initial_pos=array['recorded_object_pos'][0],scene_object_initial_quat_xyzw=array['recorded_object_quat_xyzw'][0]))
        infos.append(info);data.append(array)
    if not infos:raise ValueError('unknown group')
    nworld=len(infos);folder=a.output/a.group;folder.mkdir(exist_ok=False)
    config=EnvironmentConfig(num_envs=nworld,device='gpu',hand_side='right',reference_fps=120,control_fps=120,
         post_padding=0,compatibility=replace(SOURCE_ALIGNED_COMPATIBILITY,movement_pre_padding=120),residual_enabled=False,
         expected_contact_mode='five_fingertips',constraint_capacity=4096,contact_capacity=1024*nworld,unified_object_batch=True,
         warp_ccd_iterations=16,warp_ccd_contacts_per_world=256,warp_persistent_ccd_workspace=True,
         device_resident_controls=True,capture_transition_diagnostics=True,point_sampling_backend='numpy_per_env')
    e=MujocoManoEnvironment(TrajectoryBatch(tuple(trajectories)),config)
    m=e.model;mj=e.mujoco;names=list(e._unified_object_types)
    if assets.MANO_OPERATOR!='cheyingtong' or assets.asset_provenance()['asset_source_commit']!=ASSET_COMMIT:
        raise ValueError('wrong physical hand')
    expected_audit=infos[0]['model_audit']
    actual_joints=[mj.mj_id2name(m,mj.mjtObj.mjOBJ_JOINT,j)for j in range(m.njnt)]
    if names!=expected_audit['object_types'] or actual_joints!=expected_audit['joint_names']:
        raise ValueError('compiled scene or joint ordering differs from original model')
    if m.geom_bodyid.tolist()!=expected_audit['geom_bodyid']:
        raise ValueError('compiled geom/body binding differs from source')
    addresses={name:int(m.joint(name+'_free').qposadr[0])for name in names}
    if addresses!=expected_audit['qpos_addresses']:raise ValueError('object addresses differ')
    bodies={name:int(m.joint(name+'_free').bodyid[0])for name in names}
    geoms={name:np.flatnonzero((m.geom_bodyid==bodies[name])&((m.geom_contype!=0)|(m.geom_conaffinity!=0))).tolist()for name in names}
    kp=e.producer.keypoint_geom_ids_by_side['right'];wrist=int(e.producer.keypoint_body_ids[0])
    if list(kp)!=list(expected_audit['hand_geom_ids']):raise ValueError('hand contact geometry differs')
    qpos=np.asarray(e.data.qpos).copy();qvel=np.zeros_like(np.asarray(e.data.qvel));ctrl=np.zeros_like(np.asarray(e.data.ctrl))
    for world,(info,arrays)in enumerate(zip(infos,data)):
        qpos[world,:28]=arrays['recorded_hand'][0];ctrl[world]=arrays['targets'][0]
        for slot,name in enumerate(info['names']):
            adr=addresses[name];qpos[world,adr:adr+3]=arrays['recorded_object_pos'][0,slot]
            qpos[world,adr+3:adr+7]=arrays['recorded_object_quat_xyzw'][0,slot][[3,0,1,2]]
        np.testing.assert_allclose(qpos[world,:28],arrays['audit_initial_qpos'][:28],atol=1e-7,rtol=0)
        for name,adr in addresses.items():
            np.testing.assert_allclose(qpos[world,adr:adr+3],arrays['audit_initial_qpos'][adr:adr+3],atol=1e-7,rtol=0)
            old=arrays['audit_initial_qpos'][adr+3:adr+7]
            if abs(np.dot(qpos[world,adr+3:adr+7],old))<1-1e-6:raise ValueError('initial orientation mismatch')
    device=lambda value:e.jax.device_put(e.jp.asarray(value),e.device)
    e.data=e.data.replace(qpos=device(qpos),qvel=device(qvel),ctrl=device(ctrl),qacc_warmstart=device(np.zeros_like(qvel)),time=device(np.zeros(nworld)))
    e.data=e._forward_fn(e.data)
    outputs=[dict(qpos=[],qvel=[],ctrl=[],solver_object_pos=[],solver_object_quat_xyzw=[],physics_time=[],target_contact_force_N=[])for _ in infos]
    contacts=[[]for _ in infos];first_deviation=[None]*nworld;entry_seen=[False]*nworld;entries=[None]*nworld
    mesh=assets.object_collision_vertices('trash_bin').reshape(-1,3)if a.group=='01-egg'else None
    def capture(frame):
        state=e.producer.materialize_state(e.data);buffers=e.producer.materialize_contact_buffers(e.data,nworld)
        decoded=[[]for _ in infos]
        for name in names:
            frames=corrected_contact_frames(buffers=buffers,state=state,model=m,keypoint_geom_ids=kp,
                object_geom_ids=geoms[name],object_body_id=bodies[name],wrist_body_id=wrist,object_name=name)
            for w in range(nworld):decoded[w].extend(frames[w])
        for w,(info,arrays)in enumerate(zip(infos,data)):
            if frame>=info['frames']:continue
            out=outputs[w];out['qpos'].append(state.qpos[w].copy());out['qvel'].append(state.qvel[w].copy())
            out['ctrl'].append(np.asarray(e.data.ctrl)[w].copy())
            out['solver_object_pos'].append(state.xpos[w,[bodies[n]for n in names]].copy())
            out['solver_object_quat_xyzw'].append(state.xquat[w,[bodies[n]for n in names]][:,[1,2,3,0]].copy())
            out['physics_time'].append(float(np.asarray(e.data.time)[w]));contacts[w].append(decoded[w])
            target_entries=[x for x in decoded[w]if x['object_name']==info['active']]
            force=max((np.linalg.norm(np.asarray(x['total_force_world'],dtype=np.float32))for x in target_entries),default=0.)
            out['target_contact_force_N'].append(float(force))
            if frame:
                reference_index=int(arrays['target_reference_index'][frame-1])
                deviation=float(np.linalg.norm(state.xpos[w,bodies[info['active']]]-arrays['reference_object_pos'][reference_index]))
                if reference_index>=30 and deviation>.1 and first_deviation[w] is None:
                    first_deviation[w]=dict(frame=frame,time_s=frame/120,reference_index=reference_index,distance_m=deviation)
            if mesh is not None:
                event=bin_entry(state,w,bodies[info['active']],bodies['trash_bin'],mesh)
                if event['inside_opening'] and event['height_relative_rim_m']>=0:entry_seen[w]=True
                if entry_seen[w] and event['inside_opening'] and event['below_rim'] and event['above_bin_bottom'] and force<=.2 and entries[w]is None:
                    entries[w]=dict(frame=frame,time_s=frame/120,released=True,**event)
        return state
    capture(0)
    for frame in range(1,max(i['frames']for i in infos)):
        ctrl=np.stack([arrival_target(arrays['targets'],frame)if frame<info['frames']else arrays['targets'][-1]
                       for info,arrays in zip(infos,data)])
        e.data=e.data.replace(ctrl=device(ctrl))
        for _ in range(4):
            e.data=e._step_fn(e.data);e._check_warp_ccd_overflow()
        capture(frame)
        if frame%120==0:print('FRAME',a.group,frame,flush=True)
    results=[]
    from sim.manorl.local_contact_repair import MODEL_FIELDS,array_sha
    for w,(info,arrays)in enumerate(zip(infos,data)):
        n=info['frames'];trace={key:np.asarray(value)for key,value in outputs[w].items()}
        for key in ('qpos','qvel','ctrl','solver_object_pos'):
            if not np.isfinite(trace[key]).all():raise RuntimeError('nonfinite physical replay')
        np.testing.assert_array_equal(trace['ctrl'],arrays['targets'])
        if abs(trace['physics_time'][-1]-(n-1)/120)>.002:raise RuntimeError('integrated clock mismatch')
        idx=names.index(info['active']);oldidx=info['names'].index(info['active'])
        newpos=trace['solver_object_pos'][:,idx];newquat=trace['solver_object_quat_xyzw'][:,idx]
        pos_error=np.linalg.norm(newpos-arrays['recorded_object_pos'][:,oldidx],axis=1)
        angle_error=np.rad2deg((Rotation.from_quat(arrays['recorded_object_quat_xyzw'][:,oldidx]).inv()*Rotation.from_quat(newquat)).magnitude())
        frames=count_hand_object_contact_frames(contacts[w],target_object_name=info['active'])
        if mesh is not None:
            reached=entries[w] is not None
            before_deviation=reached and (first_deviation[w]is None or entries[w]['frame']<=first_deviation[w]['frame'])
            acceptance=dict(contract='user_egg_bin_entry_v1',accepted=bool(before_deviation),entry=entries[w],
                            hand_object_contact_frames=frames,failure_reasons=[]if before_deviation else['no_released_bin_entry_before_deviation'])
        else:
            acceptance=evaluate_synthesis_acceptance(trajectory_complete=first_deviation[w]is None,
                termination_reason_code=1 if first_deviation[w]is None else 2,
                simulated_final_object_quaternion_xyzw=persisted_rotation_quaternion_xyzw(newquat[-1]),
                reference_final_object_quaternion_xyzw=persisted_rotation_quaternion_xyzw(arrays['reference_object_quat_xyzw'][-1]),
                hand_object_contact_frames=frames).to_dict()
        # Keep qpos-derived object arrays for exact saved-state rendering; compare
        # source derived poses using solver_object_* at their original boundary.
        trace.update(scene_object_names=np.array(names),scene_object_qpos_addresses=np.array(list(addresses.values())),
                     source_frame=np.arange(n),time=np.arange(n)/120,desired=arrays['targets'],actual_wrist=trace['qpos'][:,:6],
                     actual_object_pos=np.stack([trace['qpos'][:,a:a+3]for a in addresses.values()],axis=1),
                     actual_object_quat_xyzw=np.stack([trace['qpos'][:,a+3:a+7][:,[1,2,3,0]]for a in addresses.values()],axis=1),
                     recorded_hand=arrays['recorded_hand'],recorded_object_pos=arrays['recorded_object_pos'],
                     recorded_object_quat_xyzw=arrays['recorded_object_quat_xyzw'],reference_object_pos=arrays['reference_object_pos'],
                     reference_object_quat_xyzw=arrays['reference_object_quat_xyzw'])
        path=folder/info['identity'];path.mkdir()
        np.savez_compressed(path/'trace.npz',**trace)
        dump(path/'native_contacts.json',contacts[w])
        runtime=dict(hand='cheyingtong',hand_side='right',asset_manifest_sha256=PROFILE_SHA,control_hz=120,physics_hz=480,
                     substeps=4,frame_zero_integrated=False,controller='saved absolute target, arrival-frame indexed',
                     policy_inference=False,object_forcing_after_reset=False,ccd_iterations=int(m.opt.ccd_iterations),
                     solver_cone=int(m.opt.cone),solver_impratio=float(m.opt.impratio),model_array_hashes={k:array_sha(getattr(m,k))for k in MODEL_FIELDS},
                     batch_size=nworld,source_batch_size=info['model_audit']['contact_capacity']//1024,
                     contact_boundary='native last480Hz forward sampled at120Hz, same as source',
                     ccd_overflow_guard=e.warp_ccd_metadata())
        result=dict(row_index=info['row_index'],uuid=info['uuid'],identity=info['identity'],group=a.group,frames=n,
                    acceptance=acceptance,first_deviation=first_deviation[w],
                    replay_recorded_position_error_cm=dict(final=float(pos_error[-1]*100),max=float(pos_error.max()*100),p95=float(np.quantile(pos_error,.95)*100)),
                    replay_recorded_rotation_error_deg=dict(final=float(angle_error[-1]),max=float(angle_error.max())),
                    hand_qpos_max_abs_error=float(abs(trace['qpos'][:,:28]-arrays['recorded_hand']).max()),
                    initial_audit_hand_max_abs_error=float(abs(trace['qpos'][0,:28]-arrays['audit_initial_qpos'][:28]).max()),
                    first8_audit_qpos_max_abs_error=float(abs(trace['qpos'][:8]-arrays['audit_first_qpos']).max()),
                    ctrl_exact=True,trace_sha256=sha(path/'trace.npz'),runtime=runtime)
        dump(path/'result.json',result);results.append(result)
        print('RESULT',info['identity'],acceptance['accepted'],acceptance.get('failure_reasons'),result['replay_recorded_position_error_cm'],flush=True)
    dump(folder/'summary.json',dict(group=a.group,rows=results,accepted=sum(r['acceptance']['accepted']for r in results)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--asset-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--prepare',action='store_true');p.add_argument('--group')
    a=p.parse_args()
    if a.prepare:prepare(a)
    elif a.group:run_group(a)
    else:p.error('select --prepare or --group')


if __name__=='__main__':
    main()
