"""Read-only exact800 campaign selection and native-force compact reconstruction."""
from pathlib import Path
import json
import numpy as np
from sim.manorl.u1_campaign import ACTIONS, SETTING, digest, file_sha, array_sha, child_uuid, verify_signed, plan, perturb

CONTRACT = 'u1_exact800_compact_native_v1'
IMPLEMENTATION = ('tools/run_u1_campaign.py', 'tools/u1_campaign_registry.py', 'sim/manorl/u1_campaign.py', 'tools/u1_interactive_session.py', 'sim/manorl/local_contact_repair.py', 'sim/manorl/mjx_sim.py')
BOUNDARY = 'Captured native last480Hz-substep contact wrenches/bases; frame0 native forward; CPU mj_kinematics only at saved qpos. Compact forces normal-only hand-to-object; full native wrenches retained separately.'

def require(condition, message):
    if not condition: raise ValueError(message)

def read_json(path):
    return json.loads(Path(path).read_text())

def checked_hashes(artifacts):
    for path, expected in artifacts.items():
        require(file_sha(path) == expected, 'artifact hash mismatch: '+str(path))

def select_records(rows, registry, planned, action):
    selected = {}; starts = {}; seen = set()
    for row in rows:
        status = row['status']
        require(status in ('started','selected','rejected','exhausted'), 'unknown ledger status')
        require(status != 'exhausted', 'exhausted slot')
        if status == 'started':
            require(row['uuid'] not in starts, 'duplicate start')
            starts[row['uuid']] = row
        if status == 'selected':
            uid = row['uuid']; slot = row['slot']
            require(slot not in selected and uid not in seen, 'duplicate selected slot/UUID')
            require(uid in starts and starts[uid]['slot'] == slot, 'missing matching start')
            c = starts[uid]['candidate']
            require(0 <= slot < 160 and c in planned['slots'][slot]['candidates'], 'candidate outside plan')
            require(uid == child_uuid(registry['digest'],action,planned['digest'],slot,c), 'semantic UUID mismatch')
            selected[slot] = dict(row, action=action, candidate=c); seen.add(uid)
    require(set(selected) == set(range(160)), 'exact160 selected slots required: '+action)
    latest = {r['uuid']:r['status'] for r in rows if 'uuid' in r}
    require(all(s in ('selected','rejected') for s in latest.values()), 'unfinished attempts')
    require(all(latest[r['uuid']] == 'selected' for r in selected.values()), 'selection superseded')
    return [selected[i] for i in range(160)]

def collect(registry_path, plan_path, attempts):
    r=read_json(registry_path); p=read_json(plan_path); verify_signed(r); verify_signed(p)
    require(r['setting_sha256']==SETTING and set(r['parents'])==set(ACTIONS), 'registry setting/actions')
    require(p == plan(r['digest'],p['seed'],p['reserves']), 'noncanonical plan')
    from tools.u1_campaign_registry import load_parent
    records=[]; audit=[]
    for action in ACTIONS:
        load_parent(registry_path,action)  # bundle hashes and runtime versions; no simulator
        parent=r['parents'][action]
        require(parent['setting_sha256']==SETTING and digest(parent['setting'])==SETTING, 'parent setting')
        require(action!='005' or ('no deep inversion' in parent['semantics'] and 'row847' in parent['semantics']), '005 semantics')
        require(action!='006' or ('release-v13' in parent['semantics'] and 'thumb-first' in parent['semantics']), '006 lineage')
        rows=[json.loads(line) for line in (Path(attempts)/(action+'.jsonl')).read_text().splitlines()]
        # Runner uses an absolute __file__ for itself, relative paths for the other modules.
        implementation={str(Path(f).resolve()) if i==0 else f:file_sha(f) for i,f in enumerate(IMPLEMENTATION)}
        identity=digest(dict(registry=r['digest'],plan=p['digest'],action=action,implementation=implementation))
        require(all(x['identity']==identity for x in rows), 'ledger runtime/implementation identity')
        for x in rows:
            if 'artifacts' in x: checked_hashes(x['artifacts'])
        for x in select_records(rows,r,p,action):
            folder=Path(attempts)/action/x['uuid']
            actual={str(f.resolve()) for f in folder.rglob('*') if f.is_file()}
            require(actual==set(x['artifacts']), 'ledger artifact coverage')
            reports=[]; controls=[]
            for phase in ('first','second'):
                d=folder/phase
                for name in ('result.json','trace.npz','target.npy','requested_target.npy','native_contacts.jsonl','contact_validation.npz','geometry_contacts.json'):
                    require(str((d/name).resolve()) in x['artifacts'], 'missing replay artifact '+name)
                result=read_json(d/'result.json'); reports.append(result)
                require(result['accepted'] is True and result['gates'] and all(v is True for v in result['gates'].values()), 'replay not accepted')
                require(result['contract']==parent['contract'] and result['semantics']==parent['semantics'] and result['delta']==x['candidate']['delta'], 'replay identity')
                target=np.load(d/'target.npy'); requested=np.load(d/'requested_target.npy'); controls.append(target)
                require(array_sha(target)==result['executed_target_sha256'] and array_sha(requested)==result['requested_target_sha256'], 'control hash')
                with np.load(d/'trace.npz') as z:
                    require(np.array_equal(z['ctrl'],target), 'trace/control parity')
            require(isinstance(reports[0]['pid'],int) and reports[0]['pid']>0 and reports[1]['pid']>0 and reports[0]['pid']!=reports[1]['pid'], 'independent PIDs required')
            require(np.array_equal(controls[0],controls[1]) and np.array_equal(controls[0],np.load(folder/'second/requested_target.npy')), 'frozen target changed')
            records.append(dict(x,folder=str(folder.resolve())))
        audit.extend(dict(action=action,**x) for x in rows if x['status']=='rejected')
    require(len(records)==800 and len({x['uuid'] for x in records})==800, 'exact800 unique UUIDs required')
    return r,p,records,audit

def native_frame(record, frame, ngeom):
    require(record['frame']==frame, 'native frame indexing')
    n=len(record['geom_pairs']); out={}
    for key,shape in [('geom_pairs',(n,2)),('force_contact_frame',(n,6)),('position',(n,3)),('contact_frame',(n,3,3)),('friction',(n,5)),('dimension',(n,)),('efc_address',(n,4)),('worldid',(n,))]:
        a=np.asarray(record[key]); a=a.reshape(shape) if n==0 and a.size==0 else a
        require(a.shape==shape and np.isfinite(a).all(), 'native dimensions: '+key); out[key]=a
    require(np.all(out['geom_pairs']==out['geom_pairs'].astype(int)) and np.all((out['geom_pairs']>=0)&(out['geom_pairs']<ngeom)), 'native geom IDs')
    require(np.all(out['worldid']==0) and np.all(out['dimension']==3) and np.all(out['efc_address']>=0), 'native world/condim/address')
    b=out['contact_frame']; require(np.allclose(b@b.transpose(0,2,1),np.eye(3),atol=2e-5,rtol=0), 'native basis not orthonormal')
    require(np.allclose(np.linalg.det(b),1,atol=2e-5,rtol=0), 'native basis handedness')
    return out

def world_force(wrench, basis, sign=1):
    return sign * np.asarray(basis).T @ np.asarray(wrench)[:3]

def compact_contacts(raw,m,d,names):
    from sim.manorl.contracts import KEYPOINT_NAMES
    hands={m.geom(name+'_collision').id:name for name in KEYPOINT_NAMES}
    bodies={m.body(name).id:name for name in names}
    objects={}
    for g in range(m.ngeom):
        b=int(m.geom_bodyid[g])
        while b and b not in bodies: b=int(m.body_parentid[b])
        if b in bodies: objects[g]=(b,bodies[b])
    wrist=m.body('palm').id; groups={}
    for i,(a,b) in enumerate(raw['geom_pairs'].astype(int)):
        if a in hands and b in objects: hg,og,sign=a,b,1
        elif b in hands and a in objects: hg,og,sign=b,a,-1
        else: continue
        ob,on=objects[og]; hb=int(m.geom_bodyid[hg]); jn=hands[hg]; key=(names.index(on),KEYPOINT_NAMES.index(jn))
        rotations={k:d.xmat[body].reshape(3,3) for k,body in [('wrist',wrist),('joint',hb),('object',ob)]}
        normal=sign*raw['force_contact_frame'][i,0]*raw['contact_frame'][i,0]; pos=raw['position'][i]
        pair=dict(force_normal=normal.tolist(),pos_world=pos.tolist())
        for k,body in [('wrist',wrist),('joint',hb),('object',ob)]: pair['pos_'+k]=(rotations[k].T@(pos-d.xpos[body])).tolist()
        if key not in groups: groups[key]=[dict(hand_name='right',joint_name=jn,object_name=on,contact_pairs=[]),rotations]
        groups[key][0]['contact_pairs'].append(pair)
    result=[]
    for key in sorted(groups):
        entry,rot=groups[key]; total=np.sum([p['force_normal'] for p in entry['contact_pairs']],axis=0)
        entry['total_force_world']=total.tolist()
        for k,R in rot.items(): entry['total_force_'+k]=(R.T@total).tolist()
        result.append(entry)
    return result

def schema():
    import pyarrow as pa
    from sim.manorl.lance_v2 import build_compact_schema
    s=build_compact_schema(control_fps=120,reference_fps=120)
    # Compact writer cannot retain full state/native evidence: explicit extension, never silent projection.
    for name,typ in [('physical',pa.struct([(k,pa.list_(pa.list_(pa.float64()))) for k in ('qpos','qvel','ctrl')])),('native_contacts',pa.list_(pa.string())),('lineage_json',pa.string()),('teacher_qpos',pa.list_(pa.list_(pa.float64()))),('reference_objects_json',pa.string())]: s=s.append(pa.field(name,typ))
    return s.with_metadata({**s.metadata, b'schema_version':CONTRACT.encode(), b'native_force_boundary':BOUNDARY.encode()})

def movement_record(parent, active, frames):
    """Resolve the one real manipulated-object interval without relabeling lineage."""
    source_meta=parent['metrics'].get('source_metadata',{})
    moves=source_meta.get('object_move')
    if not moves:
        moves=parent.get('source',{}).get('formal_info',{}).get('movement')
        moves=[moves] if moves else None
    if not moves:
        source=parent.get('source',{})
        if source.get('mode')=='full_row847_targets_mapped_to_current_bottle_initial_pose':
            binding=source['source']; dataset=Path(binding['dataset']); version=int(binding['version']); row=int(binding['row'])
        elif source.get('mode')=='raw':
            dataset=Path(source['dataset']); version=int(source['dataset_version']); row=int(source['row'])
        else: raise ValueError('missing movement lineage: '+parent['action'])
        import lance
        metadata=lance.dataset(str(dataset),version=version).take([row],columns=['trajectory_metadata']).to_pylist()[0]['trajectory_metadata']
        moves=metadata['trajectory_info']['object_move']
    require(isinstance(moves,list) and len(moves)==1,'exactly one movement interval required')
    move=dict(moves[0]); require(move['object_name']==active,'movement active object mismatch')
    start,end=int(move['start_frame']),int(move['end_frame']); require(0<=start<=end<frames,'movement frame range')
    return dict(object_name=active,start_frame=start,end_frame=end)

def make_row(registry_path, record, planned):
    import mujoco as mj
    from scipy.spatial.transform import Rotation as R
    from tools.u1_campaign_registry import load_parent
    from sim.manorl.contracts import KEYPOINT_NAMES
    from sim.manorl.environment import _FINGERTIP_NAMES, _FINGERTIP_LOCAL_OFFSETS
    from sim.manorl.mano_pose import right_urdf_trajectory_to_mano_48d
    from sim.manorl.lance_v2 import SYNTHETIC_LANCE_CONTRACT, FORCE_DIRECTION_CONTRACT
    r,p,I,m,base,teacher=load_parent(registry_path,record['action'])
    # Bind MANO conversion to the registry-pinned Cheyingtong assets, never defaults.
    import os
    from sim.manorl import assets
    source=r['parents']['006']['source']
    require(file_sha(source['asset_manifest'])==p['setting']['asset_manifest_sha256'], 'MANO asset manifest hash')
    os.environ['MANORL_ASSET_MANIFEST']=source['asset_manifest']
    assets.DEXSTREAM_ROOT=Path(source['asset_root'])
    assets.ASSET_MANIFEST=Path(source['asset_manifest'])
    require(read_json(assets.ASSET_MANIFEST)['hand_operator']=='cheyingtong','hand operator')
    assets._asset_manifest.cache_clear()
    from sim.manorl import mano_pose
    mano_pose._source_axes.cache_clear()
    folder=Path(record['folder'])/'second'; n=p['frames']
    with np.load(folder/'trace.npz') as z: physical={k:z[k] for k in ('qpos','qvel','ctrl')}
    for k,width in [('qpos',m.nq),('qvel',m.nv),('ctrl',28)]: require(physical[k].shape==(n,width) and np.isfinite(physical[k]).all(),'trace dimensions '+k)
    q=physical['qpos']; v=physical['qvel']; ctrl=physical['ctrl']
    _,q0=perturb(base,I.initial['qpos'],record['candidate']['delta'],p['C'])
    require(np.allclose(q[0],q0,atol=1e-7,rtol=0) and np.allclose(v[0],I.initial['qvel'],atol=1e-7,rtol=0),'initial state')
    require(np.array_equal(ctrl[p['C']-1:],base[p['C']-1:].astype(np.float32)), 'control suffix')
    lines=(folder/'native_contacts.jsonl').read_text().splitlines(); require(len(lines)==n,'incomplete native contacts')
    d=mj.MjData(m); contacts=[]; joints=[]
    ids=[m.body(name).id for name in KEYPOINT_NAMES]
    for f in range(n):
        d.qpos[:]=q[f]; mj.mj_kinematics(m,d)
        tips=[d.xpos[m.body(name).id]+d.xmat[m.body(name).id].reshape(3,3)@offset for name,offset in zip(_FINGERTIP_NAMES,_FINGERTIP_LOCAL_OFFSETS)]
        joints.append(np.concatenate([d.xpos[ids],tips],axis=0))
        contacts.append(compact_contacts(native_frame(json.loads(lines[f]),f,m.ngeom),m,d,p['names']))
    h=q[:,:28]; hand=dict(hand_name='right',mano_global_pos=h[:,:3].tolist(),mano_global_rot_aa=R.from_euler('XYZ',h[:,3:6]).as_rotvec().tolist(),mano_hand_pose=right_urdf_trajectory_to_mano_48d(h).tolist(),mano_joint_pos=np.asarray(joints).tolist(),urdf_dof=h.tolist(),urdf_dof_target=ctrl.tolist())
    objects=[]
    for name in p['names']:
        a=int(m.joint(name+'_free').qposadr[0]); objects.append(dict(pos=q[:,a:a+3].tolist(),rot_aa=R.from_quat(q[:,a+3:a+7][:,[1,2,3,0]]).as_rotvec().tolist()))
    active=I.metrics['source_metadata']['active_object'] if 'active_object' in I.metrics['source_metadata'] else p['names'][0]
    oi=p['names'].index(active)
    lineage=dict(action=record['action'],slot=record['slot'],candidate=record['candidate'],registry_digest=r['digest'],plan_digest=planned['digest'],setting_sha256=SETTING,parent=p,artifacts=record.get('artifacts',{}),native_force_boundary=BOUNDARY,reference_index_domain='resampled registry teacher120Hz, not original donor frame numbers')
    shape=p['source'].get('formal_info',{}).get('betas')
    # Shape is metadata, never infer donor hand identity. Read the pinned Cheyingtong profile.
    if shape is None:
        shape=r['parents']['003']['source']['formal_info']['betas']
    move=movement_record(p,active,n)
    return dict(index=dict(uuid=record['uuid'],seed_uuid=p['parent_uuid'],capMachine='native-U1',operator='cheyingtong',scene=','.join(p['names']),is_generated=True),trajectory_metadata=dict(data_fps=120,total_frames=n,gesture=record['action']+'-'+p['semantics'],hand_names=['right'],hand_slots=['right','left'],object_names=p['names'],mano_hand_shapes=[shape],trajectory_info=dict(object_move=[move])),timestamp=(np.arange(n)/120).tolist(),hands=[hand,dict(hand_name=None,**{k:[] for k in hand if k!='hand_name'})],objects=objects,contact=contacts,reference=dict(source_frame_index=list(range(n)),hand_urdf_dof=teacher['qpos'][:,:28].tolist(),object_pos=I.arrays['source_object_pos'][:,oi].tolist(),object_rot_aa=R.from_quat(I.arrays['source_object_quat_xyzw'][:,oi]).as_rotvec().tolist()),command_reference_index=list(range(1,n)),command_source_frame_index=list(range(1,n)),provenance=dict(contract=CONTRACT,source_contract=SYNTHETIC_LANCE_CONTRACT,force_contract=FORCE_DIRECTION_CONTRACT,reference_fps=120,control_fps=120,control_timestep_seconds=1/120,physics_fps=480,physics_timestep_seconds=1/480,physics_substeps_per_control=4,policy_mode='frozen_native_position',checkpoint_metadata_sha256=digest(lineage),warp_ccd_iterations=16,warp_ccd_contacts_per_world=256,seed=planned['seed'],episode_index=record['slot'],generation_attempt=record['candidate']['ordinal']+1,augmentation_identity=record['uuid']),physical={k:a.tolist() for k,a in physical.items()},native_contacts=lines,lineage_json=json.dumps(lineage,sort_keys=True),teacher_qpos=teacher['qpos'].tolist(),reference_objects_json=json.dumps({k:I.arrays[k].tolist() for k in ('scene_object_names','source_object_pos','source_object_quat_xyzw')},sort_keys=True))
