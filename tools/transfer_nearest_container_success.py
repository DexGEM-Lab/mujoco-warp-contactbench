"""Condition the nearest verified contact primitive on a failed receiver scene."""
import argparse,copy,hashlib,json
from pathlib import Path
import numpy as np,lance
from scipy.spatial.transform import Rotation
from sim.manorl.trajectory import trajectory_from_lance_row,resample_reference_trajectory

p=argparse.ArgumentParser(__doc__);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--registry',type=Path,required=True);p.add_argument('--rows',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);ds=lance.dataset(a.dataset,version=5);registry=json.load(open(a.registry))['rows'];candidates={}
for e in registry:
 if e['status']=='accepted' and e['gesture'][:3] in ['005','006']:
  m=json.load(open(Path(e['run'])/'manifest.json'));t=np.load(Path(e['run'])/'trajectory.npz');candidates[e['row']]=(e,m,t)
items=[]
def smooth(time,begin,end):
 u=np.clip((time-begin)/(end-begin),0,1);return u*u*(3-2*u)
for i in [int(x) for x in a.rows.split(',')]:
 row=ds.take([i],columns=['index','trajectory_metadata','timestamp','hands','objects']).to_pylist()[0]
 base=resample_reference_trajectory(trajectory_from_lance_row(row,5,row_index=i,pre_padding=180,post_padding=500,hand_side='right'),reference_fps=100,control_fps=100)
 obj=base.scene_object_types[base.identity.object_index];ri=base.identity.object_index;rb=base.scene_object_types.index('bowl');initial=base.scene_object_initial_pos[ri].copy();bowl=base.scene_object_initial_pos[rb].copy();Rrec=Rotation.from_quat(base.scene_object_initial_quat_xyzw[ri]);best=None
 for donor_id,(e,m,t) in candidates.items():
  if m['patch']['active_object']!=obj:continue
  names=t['scene_object_names'].tolist();di=names.index(obj);db=names.index('bowl');adr=int(t['scene_object_qpos_addresses'][di]);badr=int(t['scene_object_qpos_addresses'][db]);p0=t['initial_qpos'][adr:adr+3];dR=Rotation.from_quat(t['initial_qpos'][adr+3:adr+7][[1,2,3,0]]);yaw=Rotation.from_euler('z',Rrec.as_euler('xyz')[2]-dR.as_euler('xyz')[2]);initial[2]=p0[2];delta_bowl=bowl-(initial+yaw.apply(t['initial_qpos'][badr:badr+3]-p0));delta_goal=base.object_pos[base.movement_end_step]-(initial+yaw.apply(t['scene_object_pos'][-1,di]-p0));score=np.linalg.norm(delta_bowl[:2])+np.linalg.norm(delta_goal[:2])+.1*abs(yaw.as_rotvec()[2])
  if best is None or score<best[0]:best=(score,donor_id,e,m,t,di,db,p0,dR,yaw,delta_bowl,delta_goal,initial.copy())
 score,donor_id,e,m,t,di,db,p0,dR,yaw,delta_bowl,delta_goal,initial=best
 ctrl=t['ctrl'].copy();N=len(ctrl);time=np.arange(N);controlRotation=Rotation.from_euler('XYZ',ctrl[:,3:6]);ctrl[:,:3]=initial+yaw.apply(ctrl[:,:3]-p0);ctrl[:,3:6]=np.unwrap((yaw*controlRotation).as_euler('XYZ'),axis=0)
 # Actual donor pour interval determines when recipient-bowl placement is active.
 op=t['scene_object_pos'][:,di];bp=t['scene_object_pos'][:,db];dist=np.linalg.norm(op[:,:2]-bp[:,:2],axis=1);near=np.flatnonzero(dist<dist.min()+.07);first=int(near[0]);last=int(near[-1]);source_end=int(m['movement_steps'][1]);end_index=np.flatnonzero(t['reference_indices']>=source_end)[0]
 begin=max(160,first-140);return_end=min(int(end_index)-20,last+140)
 if return_end<=last:return_end=last+80
 weight=smooth(time,begin,first)*(1-smooth(time,last,return_end))
 goalweight=smooth(time,last,max(last+1,end_index))
 delta_bowl[2]=0.;delta_goal[2]=0.
 ctrl[:,:3]+=weight[:,None]*delta_bowl+goalweight[:,None]*delta_goal
 initial_hand=t['initial_qpos'][:28].copy();initial_hand[:3]=initial+yaw.apply(initial_hand[:3]-p0);initial_hand[3:6]=(yaw*Rotation.from_euler('XYZ',initial_hand[3:6])).as_euler('XYZ')
 translation=initial-base.scene_object_initial_pos[ri]
 object_rotation=(Rrec.inv()*yaw*dR).as_rotvec()
 # Mapping follows donor source movement progress; receiver phase identity and
 # target endpoints remain explicit even though primitive duration is reused.
 donor_ref=t['reference_indices'];donor_start=int(m['movement_steps'][0]);donor_end=int(m['movement_steps'][1]);receiver_start=base.movement_start_step;receiver_end=base.movement_end_step;mapping=np.interp(donor_ref,[0,donor_start,donor_end,max(donor_end+1,int(donor_ref.max()))],[0,receiver_start,receiver_end,min(len(base.q_ref)-1,receiver_end+180)]).round().astype(np.int64)
 track=a.output/f'row{i}_commands_v4.npz';np.savez_compressed(track,ctrl=ctrl,reference_index=mapping)
 recipe={'schema':'direct_capture_repair.v1','name':f'row{i}_nearest_verified_contact_v4','active_object':obj,'source':{'dataset_name':a.dataset.name,'version':5,'row':i,'uuid':row['index']['uuid'],'frames':row['trajectory_metadata']['total_frames']},'solver':{'cone':'elliptic','impratio':100.},'edit':{'translation':translation.tolist(),'rotation':object_rotation.tolist()},'initial_hand':initial_hand.tolist(),'command_track':{'path':track.name,'sha256':hashlib.sha256(track.read_bytes()).hexdigest(),'frames':N},'transfer':{'donor_source_row':donor_id,'score':float(score),'yaw_alignment_rad':float(yaw.as_rotvec()[2]),'recipient_bowl_warp_m':delta_bowl.tolist(),'recipient_goal_warp_m':delta_goal.tolist(),'active_initial_tilt_corrected':True,'source_noncontact_replaced':True,'timing':'verified donor contact primitive; monotonic receiver phase mapping'},'scope':'Receiver-conditioned contact task repair, not original-timing motion denoising. Original static tilt is aligned to donor rest geometry; masses/mu/gains unchanged.'}
 path=a.output/f'row{i}_v4.json';path.write_text(json.dumps(recipe,indent=2)+'\n');items.append({'row':i,'patch':str(path.resolve()),'post_padding':500});print(i,'donor',donor_id,'score',score,'warp bowl',delta_bowl,'goal',delta_goal,flush=True)
(a.output/'inventory.json').write_text(json.dumps(items,indent=2)+'\n')
