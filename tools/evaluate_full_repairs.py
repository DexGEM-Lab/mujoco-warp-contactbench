"""Evaluate saved actual object/hand motion against physical task outcomes."""
from __future__ import annotations
import argparse,json
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from sim.manorl.assets import compile_unified_model, object_collision_vertices

@lru_cache(maxsize=8)
def model_for(names):
    return compile_unified_model(object_types=names,object_collisions=True,physics_timestep=.0025)


def evaluate(directory, validation_destination=None):
    manifest=json.loads((directory/'manifest.json').read_text());result=json.loads((directory/'result.json').read_text());t=np.load(directory/'trajectory.npz',allow_pickle=False)
    patch=manifest['patch'];obj=patch['active_object'];names=tuple(t['scene_object_names'].tolist());idx=names.index(obj);adr=int(t['scene_object_qpos_addresses'][idx]);pos=t['scene_object_pos'][:,idx];rotation=Rotation.from_rotvec(t['scene_object_rot_aa'][:,idx]);up=rotation.apply([0,0,1]);tilt=np.degrees(np.arccos(np.clip(up[:,2],-1,1)));n=len(pos)
    refidx=t['reference_indices'];end=int(manifest['movement_steps'][1]);hold=refidx>=end
    if not np.any(hold):hold=np.arange(n)>=n-100
    lift=pos[:,2]-t['initial_qpos'][adr+2]
    mj,m=model_for(names);d=mj.MjData(m);ob=m.body(obj).id
    assert m.nq==t['qpos'].shape[1] and m.joint(obj+'_free').qposadr[0]==adr
    finalcontacts=[];samples=[]
    for frame in sorted(set([0,n-1,*range(0,n,20)])):
        d.qpos[:]=t['qpos'][frame];mj.mj_forward(m,d)
        if not np.allclose(d.xpos[ob],pos[frame],atol=2e-6):raise ValueError('model order does not reconstruct active pose')
        fingers=set();supports=set();depth=0.
        for c in d.contact[:d.ncon]:
            b1=int(m.geom_bodyid[c.geom1]);b2=int(m.geom_bodyid[c.geom2])
            if ob not in (b1,b2):continue
            other=b2 if b1==ob else b1;name=m.body(other).name
            if name in names or name=='world':supports.add(name)
            else:fingers.add(name.split('_')[0]);depth=min(depth,float(c.dist))
        sample={'frame':frame,'finger_count':len(fingers),'fingers':sorted(fingers),'support':sorted(supports),'penetration_m':-depth,'lift_m':float(lift[frame])};samples.append(sample)
        if frame==n-1:finalcontacts=sample
    action=manifest['source_gesture'].split('-')[0]
    drift=float(np.linalg.norm(pos[-1]-pos[max(0,n-101)]));final_error=float(np.linalg.norm(pos[-1]-t['reference_object_pos'][-1]));gates={'complete':bool(result['complete']),'finite':bool(np.all(np.isfinite(t['qpos']))),'lift':float(lift.max())>.065}
    hold_samples=[s for s in samples if hold[s['frame']]]
    outlet_evidence=None
    if action=='009':
        gates.update(held_above_support=float(lift[hold].min())>.10,
                     held_no_support=all(not s['support'] for s in hold_samples),
                     held_contact=all(s['finger_count']>=2 for s in hold_samples),
                     stable_drift=drift<.008,
                     held_final_tilt=float(tilt[-1])<15.)
    else:
        expected='cuboid1' if action=='007' else 'world'
        gates.update(placed_supported=expected in finalcontacts['support'],released=finalcontacts['finger_count']==0,
                     upright=float(tilt[-1])<15.,near_goal=final_error<.09,stable_drift=drift<.008)
        travel=float(np.max(np.linalg.norm(pos[:,:2]-pos[0,:2],axis=1)))
        expected_travel=float(np.max(np.linalg.norm(t['reference_object_pos'][:,:2]-t['reference_object_pos'][0,:2],axis=1)))
        gates['transport']=travel>max(.1,expected_travel*.65)
        if obj!='bowl':
            bowlidx=names.index('bowl');bowlpos=t['scene_object_pos'][:,bowlidx]
            verts=object_collision_vertices(obj)
            if obj=='pitcherbase':
                tip=verts[(verts[:,0]<-.065)&(verts[:,2]>.07)].mean(axis=0)
                tipped=tilt>40.
            else:
                tip=verts[verts[:,2]>.087].mean(axis=0)
                tipped=rotation.apply([0,0,1])[:,2]<0.
            outlet=pos+rotation.apply(tip)
            radial=np.linalg.norm(outlet[:,:2]-bowlpos[:,:2],axis=1)
            above=outlet[:,2]>bowlpos[:,2]+.030
            over=(radial<.075)&above&tipped
            longest=current=0
            for flag in over:
                current=current+1 if flag else 0;longest=max(longest,current)
            gates['outlet_over_bowl']=longest>=25
            outlet_evidence={'outlet_local':tip.tolist(),'opening_radius_used_m':.075,'longest_over_bowl_frames':int(longest),'total_over_bowl_frames':int(over.sum()),'min_outlet_radial_m':float(radial[above&tipped].min()) if np.any(above&tipped) else None,'semantic_boundary':'container outlet above bowl opening while tipped; no fluid model or delivered-volume claim'}
    gates={key:bool(value) for key,value in gates.items()}
    validation={'row':int(manifest['source']['row']),'uuid':manifest['source']['uuid'],'action':action,'object':obj,'status':'accepted' if all(gates.values()) else 'needs_fix','gates':gates,'frames':n,'max_lift_m':float(lift.max()),'held_min_lift_m':float(lift[hold].min()),'final_tilt_deg':float(tilt[-1]),'final_error_m':final_error,'last_second_drift_m':drift,'final_contacts':finalcontacts,'sampled_max_hand_penetration_m':max(s['penetration_m'] for s in samples),'contact_samples':samples,'outlet_evidence':outlet_evidence,'validation_scope':'sampled actual contact every20frames; full object height/pose/time arrays; no fluid simulation'}
    destination=Path(validation_destination) if validation_destination is not None else directory/'validation.json'
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(validation,indent=2)+'\n')
    return validation


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();out=[]
    for path in sorted(a.root.glob('row*')):
        if (path/'result.json').exists():
            v=evaluate(path);out.append({k:x for k,x in v.items() if k!='contact_samples'});print(v['row'],v['status'],{k:x for k,x in v['gates'].items() if not x},'tilt',round(v['final_tilt_deg'],2),'goalcm',round(v['final_error_m']*100,1),flush=True)
    (a.root/'validation_summary.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
