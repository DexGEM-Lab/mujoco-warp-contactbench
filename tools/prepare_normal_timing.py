"""Remove diagnostic command holds and optionally add contact-local pinch preload.

Every retained moving target is preserved. Physics must be rerun: the generated
normal stream is not a video edit. A separate audit track appends a held tail.
"""
import argparse,copy,hashlib,json
from pathlib import Path
import numpy as np
from evaluate_full_repairs import evaluate


def plateaus(commands, min_start=180, threshold=60, keep_count=30):
    equal=np.max(np.abs(np.diff(commands,axis=0)),axis=1)<=1e-8
    keep=np.ones(len(commands),dtype=bool);removed=[];i=0
    while i<len(equal):
        if not equal[i]:i+=1;continue
        first=i
        while i<len(equal) and equal[i]:i+=1
        stop=i+1;start=max(first,min_start)
        if stop-start>threshold:
            front=keep_count//2;back=keep_count-front
            keep[start+front:stop-back]=False
            removed.append({'original_start':start,'original_stop_exclusive':stop,'removed_frames':stop-start-keep_count})
    return np.flatnonzero(keep),removed


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--registry',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--rows',default='');p.add_argument('--grip-scale',type=float,default=0.);p.add_argument('--audit-tail',type=int,default=200);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);selected={int(x) for x in a.rows.split(',') if x};entries=json.load(open(a.registry))['rows'];inventory=[];summary=[]
    for entry in entries:
        row=entry['row']
        if selected and row not in selected:continue
        run=Path(entry['run']);m=json.load(open(run/'manifest.json'));t=np.load(run/'trajectory.npz',allow_pickle=False);recipe=copy.deepcopy(m['patch']);cmd=t['ctrl'].copy();ref=t['reference_indices'];N=len(cmd)
        validation=json.load(open(run/'validation.json')) if (run/'validation.json').exists() else evaluate(run,a.output/f'row{row}_original_validation.json')
        kept,removed=plateaus(cmd)
        grip=np.zeros_like(cmd)
        contact=validation['contact_samples'];start=next((s['frame'] for k,s in enumerate(contact[:-2]) if s['finger_count']>=2 and contact[k+1]['finger_count']>=2 and contact[k+2]['finger_count']>=2),int(m['movement_steps'][0]))+20
        if recipe['active_object']=='bowl':joints={9:.08,13:.08,14:.04}
        elif recipe['active_object']=='mayonnaisebottle':joints={7:.08,13:.08,17:.08,21:.08,25:.06}
        else:joints={9:.08,13:.08,21:.05,25:.05}
        indices=np.arange(N);u=np.clip((indices-start)/50.,0,1);weight=u*u*(3-2*u)
        action=m['source_gesture'][:3];release=None
        if action!='009':
            last=max((s['frame'] for s in contact if s['finger_count']>0),default=N-1);release=[max(start+50,last-100),max(start+100,last-30)]
            v=np.clip((indices-release[0])/(release[1]-release[0]),0,1);weight*=1-v*v*(3-2*v)
        for index,delta in joints.items():grip[:,index]=a.grip_scale*delta*weight
        normal=cmd[kept]+grip[kept];normal_ref=ref[kept];prefix=recipe.get('initial_hand',t['initial_qpos'][:28].tolist());recipe['initial_hand']=prefix
        recipe['normal_timing']={'baseline_run_name':run.name,'baseline_frames':N,'normal_frames':len(normal),'removed_constant_holds':removed,'moving_targets_unchanged_before_grip':True,'source_timing_exact':False,'normal_duration_s':len(normal)/100.}
        recipe['grip_preload']={'scale':a.grip_scale,'joint_index_offsets_rad':{str(k):v*a.grip_scale for k,v in joints.items()},'onset_in_baseline':start,'release_in_baseline':release,'mechanism':'additional position-target closure after sustained opposing contact; zero before contact/after release; actual actuator clamping unchanged'}
        def save(commands,mapping,tag):
            path=a.output/f'row{row}_{tag}_commands.npz';np.savez_compressed(path,ctrl=commands,reference_index=mapping)
            r=copy.deepcopy(recipe);r['name']=f'row{row}_normal_timing_grip{a.grip_scale}_{tag}';r['command_track']={'path':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'frames':len(commands)}
            r['audit_tail_frames']=a.audit_tail if tag=='audit' else 0
            patch=a.output/f'row{row}_{tag}.json';patch.write_text(json.dumps(r,indent=2)+'\n');return patch
        normal_patch=save(normal,normal_ref,'normal')
        audit=np.r_[normal,np.repeat(normal[-1:],a.audit_tail,axis=0)];audit_ref=np.r_[normal_ref,np.repeat(normal_ref[-1:],a.audit_tail)];audit_patch=save(audit,audit_ref,'audit')
        inventory.append({'row':row,'patch':str(audit_patch.resolve()),'normal_patch':str(normal_patch.resolve()),'post_padding':m['post_padding']})
        summary.append({'row':row,'original_s':N/100,'normal_s':len(normal)/100,'removed_s':(N-len(normal))/100,'hold_sections':removed,'grip_scale':a.grip_scale})
        print(summary[-1],flush=True)
    (a.output/'inventory.json').write_text(json.dumps(inventory,indent=2)+'\n');(a.output/'timing_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__':main()
