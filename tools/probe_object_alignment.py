"""Compare explicit object/finger trajectory corrections with policy-free replay.

Candidates are supplied as JSON metre/radian corrections; no policy is fitted.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import lance
import numpy as np
from scipy.spatial.transform import Rotation
from replay_pose_edits import apply_replay_edits
from sim.manorl.trajectory import trajectory_from_lance_row, resample_reference_trajectory, TrajectoryBatch
from sim.manorl.environment import EnvironmentConfig, MujocoManoEnvironment
from sim.manorl.view_environment import SOURCE_ALIGNED_COMPATIBILITY, recommended_warp_contact_capacity

p=argparse.ArgumentParser(__doc__)
p.add_argument('--dataset',required=True); p.add_argument('--version',type=int,required=True)
p.add_argument('--row',type=int,required=True); p.add_argument('--candidates',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
p.add_argument('--static-hold-frame',type=int,default=None,help='Isolate a 3-second free-body held pose at this replay frame')
a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
raw=lance.dataset(a.dataset,version=a.version).take([a.row],columns=['index','trajectory_metadata','timestamp','hands','objects']).to_pylist()[0]
base=resample_reference_trajectory(trajectory_from_lance_row(raw,a.version,row_index=a.row,hand_side='right',pre_padding=180,post_padding=180),reference_fps=100,control_fps=100)
candidates=json.loads(a.candidates.read_text()); trs=[]; ai=base.identity.object_index
for c in candidates:
 tr=apply_replay_edits(base,c)
 if a.static_hold_frame is not None:
  frame=a.static_hold_frame; count=301; qs={s:np.repeat(q[frame:frame+1],count,axis=0) for s,q in tr.q_ref_by_side.items()}
  sp=tr.scene_object_initial_pos.copy(); sq=tr.scene_object_initial_quat_xyzw.copy(); sp[ai]=tr.object_pos[frame];sq[ai]=tr.object_quat_xyzw[frame]
  tr=replace(tr,source_indices=np.full(count,tr.source_indices[frame],dtype=np.int64),timestamps=np.arange(count)/100.,q_ref=qs['right'],q_ref_by_side=qs,object_pos_raw=np.repeat(tr.object_pos_raw[frame:frame+1],count,axis=0),object_pos=np.repeat(tr.object_pos[frame:frame+1],count,axis=0),object_quat_xyzw=np.repeat(tr.object_quat_xyzw[frame:frame+1],count,axis=0),scene_object_initial_pos=sp,scene_object_initial_quat_xyzw=sq,movement_start_step=0,movement_end_step=200)
 trs.append(tr)
base=trs[0]
n=len(trs); env=MujocoManoEnvironment(TrajectoryBatch(tuple(trs)),EnvironmentConfig(device='gpu',num_envs=n,residual_enabled=False,max_deviation_distance=1e6,reference_fps=100,control_fps=100,post_padding=180,hand_side='right',compatibility=replace(SOURCE_ALIGNED_COMPATIBILITY,movement_pre_padding=180),contact_capacity=recommended_warp_contact_capacity(n,('right',))))
body=env.model.body(base.scene_object_types[ai]).id; adr=env.model.joint(base.scene_object_types[ai]+'_free').qposadr[0]
qpos=[]; steps=[]; zero=np.zeros((n,28))
for step in range(len(base.q_ref)-1):
 _,_,done,_=env.step(zero)
 if step%5==0 or bool(np.any(done)):
  states=[env.host_data(i).qpos.copy() for i in range(n)]; qpos.append(np.stack(states)); steps.append(step)
 if step%100==0: print('STEP',step,'HEIGHTS',[round(float(v[adr+2]),3) for v in qpos[-1]],flush=True)
 if bool(np.any(done)): break
qpos=np.stack(qpos); pos=qpos[:,:,adr:adr+3]; quat=qpos[:,:,adr+3:adr+7]; steps=np.array(steps)
np.savez_compressed(a.output/'trace.npz',qpos=qpos,steps=steps,object_pos=pos,object_quat_wxyz=quat)
results=[]
for i,c in enumerate(candidates):
 z0=trs[i].object_pos[0,2]; tilt=np.degrees(np.arccos(np.clip(Rotation.from_quat(quat[:,i][:,[1,2,3,0]]).as_matrix()[:,2,2],-1,1)))
 tail=steps>=base.movement_end_step; movement=(steps>=base.movement_start_step)&(steps<=base.movement_end_step)
 result={**c,'max_lift_m':float(pos[:,i,2].max()-z0),'final_lift_m':float(pos[-1,i,2]-z0),'hold_min_lift_m':float((pos[tail,i,2]-z0).min()),'final_tilt_deg':float(tilt[-1]),'movement_rmse_m':float(np.sqrt(np.mean(np.sum((pos[movement,i]-trs[i].object_pos[steps[movement]])**2,axis=1))))}
 results.append(result)
print(json.dumps(results,indent=2),flush=True); (a.output/'results.json').write_text(json.dumps(results,indent=2))
