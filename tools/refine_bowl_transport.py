"""Use a measured stable receiver grasp to repair only carry/placement/release."""
import argparse,copy,hashlib,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from sim.manorl.assets import compile_unified_model

p=argparse.ArgumentParser(__doc__);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--rows',required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);inventory=[]
for row in [int(v) for v in a.rows.split(',')]:
    dpath=a.run_root/f'row{row}';t=np.load(dpath/'trajectory.npz');manifest=json.load(open(dpath/'manifest.json'));recipe=copy.deepcopy(manifest['patch']);active=t['scene_object_names'].tolist().index('bowl');action=manifest['source_gesture'][:3];target=t['reference_object_pos'][-1].copy()
    # The grasp remains stationary through donor's held phase. Select a measured
    # stable frame, before receiver-specific carry disturbances.
    anchor=650;command=t['ctrl'][anchor].copy();object_pos=t['scene_object_pos'][anchor,active];orientation=Rotation.from_rotvec(t['scene_object_rot_aa'][anchor,active]);up=orientation.apply([0,0,1]);cross=np.cross(up,[0,0,1]);angle=np.arccos(np.clip(up[2],-1,1));correction=Rotation.from_rotvec(cross/(np.linalg.norm(cross)+1e-12)*angle)
    # Tip/center relationship is preserved while transporting and leveling.
    goal=command.copy();goal[:3]=target+correction.apply(command[:3]-object_pos)
    goal[3:6]=command[3:6]+((correction*Rotation.from_euler('XYZ',command[3:6])).as_euler('XYZ')-command[3:6]+np.pi)%(2*np.pi)-np.pi
    rows=[q.copy() for q in t['ctrl'][:anchor+1]];mapping=t['reference_indices'][:anchor+1].tolist();end=int(manifest['movement_steps'][1]);source_anchor=mapping[-1]
    def append(first,last,count,mapping_end):
        origin=mapping[-1];delta=last-first;delta[3:6]=(delta[3:6]+np.pi)%(2*np.pi)-np.pi
        for u in np.linspace(0,1,count):
            rows.append(first+u*u*(3-2*u)*delta);mapping.append(int(round(origin+u*(mapping_end-origin))))
    # Move horizontally above the destination before lowering: no downward
    # scraping across cuboid edges while holding a source-specific bowl.
    over=goal.copy();over[2]+=max(0.,object_pos[2]-target[2])
    append(command,over,250,source_anchor+int((end-source_anchor)*.7))
    append(over,goal,200,end);append(goal,goal,100,end)
    opened=goal.copy()
    # A fully open rim grasp releases before wrist withdrawal. Targets are
    # canonical flexion coordinates, not per-capture finger data drift.
    for idx in [7,9,11,13,14,15,17,18,19,21,22,23,25,26,27]:opened[idx]=0.
    opened[2]-=.012
    append(goal,opened,150,end)
    direction=command[:3]-object_pos;direction[2]=0.;direction/=np.linalg.norm(direction)
    retreat=opened.copy();retreat[:3]+=direction*.15+[0,0,.08]
    append(opened,retreat,150,end);append(retreat,retreat,200,end)
    track=a.output/f'row{row}_commands_v2.npz';np.savez_compressed(track,ctrl=np.array(rows),reference_index=np.array(mapping,dtype=np.int64));recipe['name']=f'guangxue_bowl_row{row}_measured_carry_release_v2';recipe['command_track']={'path':track.name,'sha256':hashlib.sha256(track.read_bytes()).hexdigest(),'frames':len(rows)};recipe['transfer']['repair']={'measured_anchor_step':anchor,'prefix_preserved_commands':anchor+1,'cause':'first-pass carry drift or release hooking','method':'level measured grasp, horizontal carry then vertical lower, fully open fingers before withdrawal','retimed':True}
    path=a.output/f'row{row}_v2.json';path.write_text(json.dumps(recipe,indent=2)+'\n');inventory.append({'row':row,'patch':str(path.resolve()),'post_padding':500});print(row,'anchorheight',object_pos[2],'anchor tilt',np.degrees(angle),'goal',target,'frames',len(rows),flush=True)
(a.output/'inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
