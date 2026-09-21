#!/usr/bin/env python3
"""Export/validate exact640 strict-C1 four-action U1 campaign."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

from sim.manorl.u1_campaign import array_sha,child_uuid,digest,file_sha,write_json
from tools.run_u1_largepose_campaign import (ACTIONS,VERSION,PREFIX_FRAMES,
    ROOT,SOURCE_REGISTRY,build_plan,make_target)
from tools.u1_campaign_registry import load_parent
from sim.manorl.u1_export import (native_frame,compact_contacts,
    schema as base_schema,require)

CONTRACT='u1_four_action_largepose640_c1_native_v1'
SETTING='c6db552f105d4de493880a52518772dbacad9496b105d2f4e43e87a907e692cd'

def schema():
 s=base_schema();md=dict(s.metadata or {});md[b'schema_version']=CONTRACT.encode();md[b'augmentation_contract']=VERSION.encode();return s.with_metadata(md)

def movement_record(parent,active,frames):
    source_meta=parent['metrics'].get('source_metadata',{});moves=source_meta.get('object_move')
    if not moves:
        movement=parent.get('source',{}).get('formal_info',{}).get('movement');moves=[movement] if movement else None
    if not moves:
        source=parent.get('source',{})
        require(source.get('mode')=='raw','missing movement lineage')
        import lance
        metadata=lance.dataset(str(source['dataset']),version=int(source['dataset_version'])).take(
            [int(source['row'])],columns=['trajectory_metadata']).to_pylist()[0]['trajectory_metadata']
        moves=metadata['trajectory_info']['object_move']
    active_moves=[dict(x)for x in moves if x.get('object_name')==active]
    require(len(active_moves)==1,'exactly one active-object movement interval required')
    move=active_moves[0];start,end=int(move['start_frame']),int(move['end_frame']);require(0<=start<=end<frames,'movement range')
    return dict(object_name=active,start_frame=start,end_frame=end)


def collect(attempts,plan_path):
 plan=json.loads(plan_path.read_text());require(plan['version']==VERSION and plan['prefix_frames']==PREFIX_FRAMES,'plan contract');slots=plan['slots'];require(slots==build_plan(),'saved plan mismatch')
 registry_sha=file_sha(SOURCE_REGISTRY);records=[];audit=[]
 for action in ACTIONS:
  rows=[json.loads(x)for x in(attempts/(action+'.jsonl')).read_text().splitlines()]
  require(all(x['status']in ('started','selected','rejected','exhausted')for x in rows),'ledger status')
  require(not any(x['status']=='exhausted'for x in rows),'exhausted '+action)
  expected_identity=digest(dict(version=VERSION,action=action,registry_sha=registry_sha,implementation={f:file_sha(ROOT/f)for f in ('sim/manorl/u1_largepose.py','tools/run_u1_largepose_campaign.py')}))
  require(all(x['identity']==expected_identity for x in rows),'ledger identity '+action)
  starts={};selected={};latest={}
  for x in rows:
   uid=x.get('uuid')
   if uid is not None:latest[uid]=x
   if x['status']=='started':require(uid not in starts,'duplicate start');starts[uid]=x
   elif x['status']=='selected':require(x['slot']not in selected,'duplicate slot');selected[x['slot']]=x
   elif x['status']=='rejected':audit.append(dict(action=action,**x))
  require(set(selected)==set(range(160)),'quota '+action)
  require(all(x['status']in ('selected','rejected')for x in latest.values()),'unfinished attempts '+action)
  r,p,I,m,base,teacher=load_parent(SOURCE_REGISTRY,action);require(p['setting_sha256']==SETTING,'setting')
  for slot in range(160):
   x=selected[slot];uid=x['uuid'];require(latest[uid]['status']=='selected','superseded selection');require(uid in starts,'missing start');candidate=starts[uid]['candidate'];require(candidate in slots[slot]['candidates'],'candidate outside plan')
   expected=child_uuid(dict(version=VERSION,registry=registry_sha),action,digest(slots[slot]),slot,candidate);require(uid==expected,'uuid')
   folder=attempts/action/uid
   actual={str(path.resolve())for path in folder.rglob('*')if path.is_file()};require(actual==set(x['artifacts']),'artifact coverage '+uid)
   for path,sha256 in x['artifacts'].items():require(file_sha(path)==sha256,'artifact hash '+path)
   reports=[];targets=[];expected_frames=p['frames']+PREFIX_FRAMES
   for phase in ('first','second'):
    d=folder/phase
    required={'result.json','trace.npz','target.npy','native_contacts.jsonl','contact_validation.npz','geometry_contacts.json'}
    require(all(str((d/name).resolve())in x['artifacts']for name in required),'missing replay artifact '+phase)
    rep=json.loads((d/'result.json').read_text());require(rep['accepted']and all(rep['gates'].values()),'failed replay');require(rep['contract']==p['contract']and rep['semantics']==p['semantics'],'replay lineage');require(rep['prefix_frames']==PREFIX_FRAMES,'prefix');require(len(rep.get('prefix_force_contacts',[]))==0,'prefix contact');reports.append(rep)
    target=np.load(d/'target.npy',allow_pickle=False);targets.append(target);require(target.shape==(expected_frames,28)and np.isfinite(target).all(),'target shape');require(array_sha(target)==rep['executed_target_sha256'],'executed target hash')
    with np.load(d/'trace.npz',allow_pickle=False)as z:
     require(z['qpos'].shape==(expected_frames,m.nq)and z['qvel'].shape==(expected_frames,m.nv),'trace shape');require(np.isfinite(z['qpos']).all()and np.isfinite(z['qvel']).all(),'trace finite');require(np.array_equal(z['ctrl'],target),'ctrl trace')
     q0=I.initial['qpos'].copy();q0[:6]+=np.asarray(candidate['delta']);require(np.allclose(z['qpos'][0],q0,atol=1e-7,rtol=0),'initial pose');require(np.allclose(z['qvel'][0],I.initial['qvel'],atol=1e-7,rtol=0),'initial velocity')
    lines=(d/'native_contacts.jsonl').read_text().splitlines();require(len(lines)==expected_frames,'native frames')
    for frame,line in enumerate(lines):native_frame(json.loads(line),frame,m.ngeom)
   require(all(isinstance(rep['pid'],int)and rep['pid']>0 for rep in reports)and reports[0]['pid']!=reports[1]['pid'],'pid');require(np.array_equal(targets[0],targets[1]),'frozen mismatch')
   requested=make_target(base,np.asarray(candidate['delta']),PREFIX_FRAMES,teacher['qpos'][:2,:28]);require(reports[0]['requested_target_sha256']==array_sha(requested),'first requested target hash');require(reports[1]['requested_target_sha256']==array_sha(targets[0]),'frozen second requested target hash');require(np.max(np.abs(targets[0].astype(float)-requested))<1e-6,'requested mismatch');require(targets[0][PREFIX_FRAMES:].tobytes()==base.astype(np.float32).tobytes(),'suffix')
   records.append(dict(action=action,slot=slot,uuid=uid,candidate=candidate,folder=str(folder)))
   if (slot+1)%20==0:print(json.dumps({'phase':'artifact_audit','action':action,'selected':slot+1}),flush=True)
  print(json.dumps({'phase':'audit','action':action,'selected':160}),flush=True)
 require(len(records)==640 and len({x['uuid']for x in records})==640,'exact640 unique UUIDs')
 return plan,records,audit

def make_row(record,plan):
 import os,mujoco as mj
 from scipy.spatial.transform import Rotation as R
 from sim.manorl.contracts import KEYPOINT_NAMES
 from sim.manorl.environment import _FINGERTIP_NAMES,_FINGERTIP_LOCAL_OFFSETS
 from sim.manorl.mano_pose import right_urdf_trajectory_to_mano_48d
 from sim.manorl.lance_v2 import SYNTHETIC_LANCE_CONTRACT,FORCE_DIRECTION_CONTRACT
 r,p,I,m,base,teacher=load_parent(SOURCE_REGISTRY,record['action']);folder=Path(record['folder'])/'second';n=p['frames']+PREFIX_FRAMES
 # Bind pinned Cheyingtong assets.
 from sim.manorl import assets
 source=r['parents']['006']['source'];os.environ['MANORL_ASSET_MANIFEST']=source['asset_manifest'];assets.DEXSTREAM_ROOT=Path(source['asset_root']);assets.ASSET_MANIFEST=Path(source['asset_manifest']);assets._asset_manifest.cache_clear()
 from sim.manorl import mano_pose;mano_pose._source_axes.cache_clear()
 with np.load(folder/'trace.npz')as z:physical={k:z[k]for k in('qpos','qvel','ctrl')}
 q,v,ctrl=physical['qpos'],physical['qvel'],physical['ctrl'];require(q.shape==(n,m.nq)and v.shape==(n,m.nv)and ctrl.shape==(n,28),'trace shape')
 lines=(folder/'native_contacts.jsonl').read_text().splitlines();require(len(lines)==n,'native count');d=mj.MjData(m);contacts=[];joints=[];ids=[m.body(name).id for name in KEYPOINT_NAMES]
 for f in range(n):
  d.qpos[:]=q[f];mj.mj_kinematics(m,d);tips=[d.xpos[m.body(name).id]+d.xmat[m.body(name).id].reshape(3,3)@off for name,off in zip(_FINGERTIP_NAMES,_FINGERTIP_LOCAL_OFFSETS)];joints.append(np.concatenate([d.xpos[ids],tips],0));contacts.append(compact_contacts(native_frame(json.loads(lines[f]),f,m.ngeom),m,d,p['names']))
 h=q[:,:28];hand=dict(hand_name='right',mano_global_pos=h[:,:3].tolist(),mano_global_rot_aa=R.from_euler('XYZ',h[:,3:6]).as_rotvec().tolist(),mano_hand_pose=right_urdf_trajectory_to_mano_48d(h).tolist(),mano_joint_pos=np.asarray(joints).tolist(),urdf_dof=h.tolist(),urdf_dof_target=ctrl.tolist());objects=[]
 for name in p['names']:
  a=int(m.joint(name+'_free').qposadr[0]);objects.append(dict(pos=q[:,a:a+3].tolist(),rot_aa=R.from_quat(q[:,a+3:a+7][:,[1,2,3,0]]).as_rotvec().tolist()))
 active=I.metrics['source_metadata']['active_object'];oi=p['names'].index(active);teacher_ext=np.concatenate([np.repeat(teacher['qpos'][:1],PREFIX_FRAMES,0),teacher['qpos']],0);source_pos=np.concatenate([np.repeat(I.arrays['source_object_pos'][:1],PREFIX_FRAMES,0),I.arrays['source_object_pos']],0);source_quat=np.concatenate([np.repeat(I.arrays['source_object_quat_xyzw'][:1],PREFIX_FRAMES,0),I.arrays['source_object_quat_xyzw']],0)
 move=movement_record(p,active,p['frames']);move=dict(move,start_frame=move['start_frame']+PREFIX_FRAMES,end_frame=move['end_frame']+PREFIX_FRAMES);mapping=[0]*PREFIX_FRAMES+list(range(p['frames']));lineage=dict(contract=CONTRACT,action=record['action'],slot=record['slot'],candidate=record['candidate'],setting_sha256=SETTING,parent_uuid=p['parent_uuid'],prefix_frames=PREFIX_FRAMES,join='physical_reference_discrete_c1',source_registry_sha256=file_sha(SOURCE_REGISTRY))
 shape=p['source'].get('formal_info',{}).get('betas')or r['parents']['003']['source']['formal_info']['betas'];empty=dict(hand_name=None,mano_global_pos=[],mano_global_rot_aa=[],mano_hand_pose=[],mano_joint_pos=[],urdf_dof=[],urdf_dof_target=[])
 return dict(index=dict(uuid=record['uuid'],seed_uuid=p['parent_uuid'],capMachine='native-U1',operator='cheyingtong',scene=','.join(p['names']),is_generated=True),trajectory_metadata=dict(data_fps=120,total_frames=n,gesture=record['action']+'-'+p['semantics'],hand_names=['right'],hand_slots=['right','left'],object_names=p['names'],mano_hand_shapes=[shape],trajectory_info=dict(object_move=[move])),timestamp=(np.arange(n)/120).tolist(),hands=[hand,empty],objects=objects,contact=contacts,reference=dict(source_frame_index=mapping,hand_urdf_dof=teacher_ext[:,:28].tolist(),object_pos=source_pos[:,oi].tolist(),object_rot_aa=R.from_quat(source_quat[:,oi]).as_rotvec().tolist()),command_reference_index=list(range(1,n)),command_source_frame_index=mapping[1:],provenance=dict(contract=CONTRACT,source_contract=SYNTHETIC_LANCE_CONTRACT,force_contract=FORCE_DIRECTION_CONTRACT,reference_fps=120,control_fps=120,control_timestep_seconds=1/120,physics_fps=480,physics_timestep_seconds=1/480,physics_substeps_per_control=4,policy_mode='frozen_native_position',checkpoint_metadata_sha256=digest(lineage),warp_ccd_iterations=16,warp_ccd_contacts_per_world=256,seed=plan['seed'],episode_index=record['slot'],generation_attempt=record['candidate']['ordinal']+1,augmentation_identity=record['uuid']),physical={k:x.tolist()for k,x in physical.items()},native_contacts=lines,lineage_json=json.dumps(lineage,sort_keys=True),teacher_qpos=teacher_ext.tolist(),reference_objects_json=json.dumps(dict(scene_object_names=I.arrays['scene_object_names'].tolist(),source_object_pos=source_pos.tolist(),source_object_quat_xyzw=source_quat.tolist()),sort_keys=True))

def export(output,attempts,plan_path):
 import lance,pyarrow as pa
 require(not output.exists(),'immutable output')
 print(json.dumps({'phase':'export','step':'audit_start'}),flush=True)
 plan,records,audit=collect(attempts,plan_path)
 output.mkdir(parents=True)
 write_json(output/'manifest.json',dict(contract=CONTRACT,rows=640,counts={a:160 for a in ACTIONS},setting_sha256=SETTING,plan_digest=plan['digest'],ordered_uuids=[x['uuid']for x in records]))
 write_json(output/'plan.json',plan);write_json(output/'rejections.json',audit);s=schema()
 def batches():
  for i,x in enumerate(records,1):
   yield pa.RecordBatch.from_pylist([make_row(x,plan)],schema=s)
   if i%20==0:print(json.dumps({'phase':'export','rows_built':i,'action':x['action']}),flush=True)
 lance.write_dataset(pa.RecordBatchReader.from_batches(s,batches()),str(output/'compact.lance'),mode='create')
 print(json.dumps({'phase':'export','step':'hash_start'}),flush=True)
 write_json(output/'sha256.json',{str(f.relative_to(output)):file_sha(f)for f in sorted(output.rglob('*'))if f.is_file()})
 print(json.dumps({'phase':'export','rows':640,'output':str(output)}),flush=True)
def validate(output,attempts,plan_path):
 import lance,pyarrow as pa
 print(json.dumps({'phase':'validate','step':'audit_start'}),flush=True)
 plan,records,audit=collect(attempts,plan_path);ds=lance.dataset(str(output/'compact.lance'));require(ds.count_rows()==640,'rows');require(ds.schema.equals(schema(),check_metadata=True),'schema')
 # Full lineage/trace/native readback equality, one row at a time.
 for i,x in enumerate(records,1):
  expected=pa.Table.from_pylist([make_row(x,plan)],schema=schema()).to_pylist()[0];actual=ds.take([i-1]).to_pylist()[0];require(actual==expected,'row '+str(i-1))
  if i%20==0:print(json.dumps({'phase':'validate','rows_read_back':i,'action':x['action']}),flush=True)
 report={'contract':CONTRACT,'validated':True,'rows':640,'counts':{a:160 for a in ACTIONS},'unique_uuids':len({x['uuid'] for x in records}),'setting_sha256':SETTING,'plan_digest':plan['digest'],'lance_version':ds.version,'checks':['exact action quotas','unique deterministic UUIDs','two accepted frozen-control replays with distinct PIDs','zero solved native prefix-force contacts','strict physical-reference discrete-C1 target','byte-exact parent control suffix','complete native contact frames','full source-bound Lance row equality']}
 write_json(output/'validation.json',report)
 print(json.dumps({'phase':'validate','rows':640,'validated':True,'report':str(output/'validation.json')}),flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=('audit','export','validate'));p.add_argument('--attempts',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path);a=p.parse_args()
 if a.mode=='audit':plan,records,audit=collect(a.attempts,a.plan);print(json.dumps({'rows':len(records),'rejections':len(audit),'unique':len({x['uuid']for x in records})}))
 elif a.mode=='export':export(a.output,a.attempts,a.plan)
 else:validate(a.output,a.attempts,a.plan)
if __name__=='__main__':main()
