"""Select physically reexecuted normal streams and keep test tails separately."""
import argparse,copy,json
from pathlib import Path
import numpy as np
from sim.manorl.assets import compile_unified_model
from scipy.spatial.transform import Rotation

p=argparse.ArgumentParser(__doc__);p.add_argument('--baseline-registry',type=Path,required=True);p.add_argument('--run-roots',type=Path,nargs='+',required=True);p.add_argument('--patch-roots',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);baseline=json.load(open(a.baseline_registry));entries=copy.deepcopy(baseline['rows']);models={}
for e in entries:
    row=e['row'];candidates=[]
    for root in a.run_roots:
        run=root/f'row{row}'
        if not (run/'validation.json').exists():continue
        validation=json.load(open(run/'validation.json'))
        if validation['status']!='accepted':continue
        manifest=json.load(open(run/'manifest.json'));r=manifest['patch'];N=r['normal_timing']['normal_frames'];data=np.load(run/'trajectory.npz');names=tuple(data['scene_object_names'].tolist());idx=names.index(r['active_object'])
        if names not in models:models[names]=compile_unified_model(object_types=names,object_collisions=True,physics_timestep=.0025)
        mj,m=models[names];d=mj.MjData(m);d.qpos[:]=data['qpos'][N-1];mj.mj_forward(m,d);ob=m.body(r['active_object']).id;fingers=[];supports=[]
        for c in d.contact[:d.ncon]:
            b1=int(m.geom_bodyid[c.geom1]);b2=int(m.geom_bodyid[c.geom2])
            if ob not in (b1,b2):continue
            other=b2 if b1==ob else b1;name=m.body(other).name
            if name in names or name=='world':supports.append(name)
            else:fingers.append(name)
        up=Rotation.from_rotvec(data['scene_object_rot_aa'][N-1,idx]).apply([0,0,1]);tilt=float(np.degrees(np.arccos(np.clip(up[2],-1,1))))
        action=manifest['source_gesture'][:3]
        terminal_ok=(len({name.split('_')[0] for name in fingers})>=2 and not supports and tilt<15) if action=='009' else (len(fingers)==0 and ('cuboid1' if action=='007' else 'world')in supports and tilt<15)
        if not terminal_ok:continue
        candidates.append((r['grip_preload']['scale'],run,manifest,data,validation,{'finger_bodies':sorted(set(fingers)),'support_bodies':sorted(set(supports)),'tilt_deg':tilt}))
    if not candidates:
        e.update(status='normal_needs_fix');continue
    _,run,manifest,t,v,terminal=min(candidates,key=lambda x:x[0]);r=manifest['patch'];N=int(r['normal_timing']['normal_frames']);out=a.output/f'row{row}';out.mkdir(exist_ok=True)
    matching=[]
    for root in a.patch_roots:
        patch=root/f'row{row}_normal.json'
        if not patch.exists():continue
        normal=json.load(open(patch))
        if normal['grip_preload']==r['grip_preload'] and normal['normal_timing']==r['normal_timing']:matching.append((patch,normal))
    if not matching:raise ValueError(f'normal command asset absent row{row}')
    patch,normal=matching[0]
    with np.load(patch.parent/normal['command_track']['path']) as c:
        commanded=c['ctrl']
        if not np.allclose(commanded,t['ctrl'][:N],atol=1e-6):
            # final runtime clamping is allowed, but must be explicit.
            clipped=np.clip(commanded,m.jnt_range[:28,0],m.jnt_range[:28,1])
            if not np.allclose(clipped,t['ctrl'][:N],atol=1e-6):raise ValueError('normal prefix differs from physically executed controls')
    arrays={key:(t[key][:N] if t[key].ndim>0 and len(t[key])==len(t['qpos']) else t[key]) for key in t.files}
    np.savez_compressed(out/'trajectory.npz',**arrays)
    m2=copy.deepcopy(manifest);m2['patch']=normal;m2['audit_run']=str(run.resolve());m2['normal_output_frames']=N;m2['separate_stability_extension_frames']=len(t['qpos'])-N
    (out/'manifest.json').write_text(json.dumps(m2,indent=2)+'\n')
    pos=arrays['scene_object_pos'][:,idx];initial=arrays['initial_qpos'][int(arrays['scene_object_qpos_addresses'][idx])+2];result={'complete':True,'frames':N,'maximum_lift_m':float((pos[:,2]-initial).max()),'last_lift_m':float(pos[-1,2]-initial),'last_position_error_m':float(np.linalg.norm(pos[-1]-arrays['reference_object_pos'][-1])),'last_second_displacement_m':(pos[-1]-pos[max(0,N-101)]).tolist()}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    validation={'status':'accepted','normal_frames':N,'normal_terminal':terminal,'physical_prefix_reexecuted':True,'audit_tail_frames':len(t['qpos'])-N,'audit':v,'scope':'normal output is exact prefix of NEW shortened physical replay; appended constant-target stability test is separate and not included in normal time'}
    (out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n');e.update(status='accepted',run=str(out.resolve()),patch=str(patch.resolve()),validation=validation);print(row,'normal accepted',N/100,'grip',normal['grip_preload']['scale'],flush=True)
result={'source_dataset':baseline['source_dataset'],'source_version':5,'total_rows':59,'accepted_count':sum(e['status']=='accepted' for e in entries),'rows':entries};(a.output/'registry.json').write_text(json.dumps(result,indent=2)+'\n');print('NORMAL',result['accepted_count'],'/59','REMAINING',[e['row'] for e in entries if e['status']!='accepted'],flush=True)
