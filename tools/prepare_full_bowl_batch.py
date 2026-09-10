"""Transfer validated contact primitives to all remaining bowl receivers.

Hold rows retain the receiver time/pose series. Transport rows use the proven
contact sequence, preserve the first receiver approach frames, and remap the
carry path to each receiver's own object trajectory; retiming is explicit.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import lance
import numpy as np
from scipy.spatial.transform import Rotation
from sim.manorl.trajectory import trajectory_from_lance_row, resample_reference_trajectory
from bowl_hold_template_transfer import _make_patch, _base


def base(row, index):
    return resample_reference_trajectory(trajectory_from_lance_row(row,5,row_index=index,
        hand_side='right',pre_padding=180,post_padding=500),reference_fps=100,control_fps=100)


def prepare(dataset_path,bundle,output):
    output.mkdir(parents=True,exist_ok=True)
    ds=lance.dataset(dataset_path,version=5)
    columns=['index','trajectory_metadata','timestamp','hands','objects']
    metadata=ds.to_table(columns=['index','trajectory_metadata']).to_pylist()
    # One read per row; no repeatedly decoding the same nested arrays.
    rows={i:ds.take([i],columns=columns).to_pylist()[0] for i in range(59)
          if i<19 or i==36 or i>=40}
    source49=base(rows[49],49)
    donor49=json.loads((bundle/'patches/guangxue_bowl09_row49_v3.json').read_text())
    inventory=[]
    for i in [52,53,56,57]:
        receiver=base(rows[i],i)
        recipe=_make_patch(donor49,source49,rows[i],receiver)
        recipe['source']['row']=i;recipe['name']=f'guangxue_bowl09_row{i}_contact_v1'
        path=output/f'row{i}_v1.json';path.write_text(json.dumps(recipe,indent=2)+'\n')
        inventory.append({'row':i,'action':'09','patch':str(path.resolve()),'post_padding':500})
    donor_specs={
        '03':('bowl-stove-to-table','guangxue_bowl03_row0_v1.json',0,1385,1685),
        '07':('bowl-table-to-stove','guangxue_bowl07_row36_v1.json',36,1385,1685),
    }
    for action,(key,patchname,donor_row,carry_start,carry_end) in donor_specs.items():
        donor_recipe=json.loads((bundle/'patches'/patchname).read_text())
        donor=np.load(bundle/'recordings'/key/'trajectory.npz',allow_pickle=False)
        donor_commands=np.load(bundle/'patches'/donor_recipe['command_track']['path'],allow_pickle=False)['ctrl']
        donor_initial=donor['initial_qpos']
        donor_bowl_address=int(donor['scene_object_qpos_addresses'][donor['scene_object_names'].tolist().index('bowl')])
        donor_initial_pos=donor_initial[donor_bowl_address:donor_bowl_address+3]
        donor_initial_rot=Rotation.from_quat(donor_initial[donor_bowl_address+3:donor_bowl_address+7][[1,2,3,0]])
        donor_base=base(rows[donor_row],donor_row)
        donor_goal=donor_base.object_pos[donor_base.movement_end_step]+donor_recipe['edit']['translation']
        source_ids=[i for i,r in enumerate(metadata) if r['index']['gesture'].split('-')[0]==str(int(action)).zfill(3) and i!=donor_row]
        for i in source_ids:
            receiver=base(rows[i],i)
            initial_pos=receiver.object_pos[0].copy()+np.asarray(donor_recipe['edit']['translation'])
            # Table pickup has no support beneath the bowl; align its lower surface
            # using the donor's physically settled table height, not capture noise.
            if action=='07':initial_pos[2]=donor_initial_pos[2]
            offset=initial_pos-donor_initial_pos
            translation=initial_pos-receiver.object_pos[0]
            goal=receiver.object_pos[receiver.movement_end_step]+translation
            commands=donor_commands.copy();commands[:,:3]+=offset
            # Keep contact acquisition; transport the held object to receiver goal.
            # Receiver X/Y path is monotonic-phase resampled, donor vertical lift
            # clearance retained. At completion both goal contributions are exact.
            start=int(receiver.movement_start_step);end=int(receiver.movement_end_step)
            sourcepath=receiver.object_pos[start:end+1]
            grid=np.linspace(0,len(sourcepath)-1,carry_end-carry_start)
            rawpath=np.column_stack([np.interp(grid,np.arange(len(sourcepath)),sourcepath[:,j]) for j in range(3)])
            endpoint_offset=goal-(donor_goal+offset)
            progress=np.linspace(0.,1.,carry_end-carry_start);smooth=progress*progress*(3.-2.*progress)
            change=np.zeros((len(commands),3))
            change[carry_start:carry_end]=smooth[:,None]*endpoint_offset
            change[carry_end:]=endpoint_offset
            # Retain each source's lateral route (deviation from endpoint segment),
            # smoothed near boundaries to preserve the donor's contact stability.
            straight=sourcepath[0]+progress[:,None]*(sourcepath[-1]-sourcepath[0])
            path_deviation=rawpath-straight
            change[carry_start:carry_end,:2]+=(np.sin(np.pi*progress)**2)[:,None]*path_deviation[:,:2]
            commands[:,:3]+=change
            # Non-contact prefix is source motion. A bounded approach blend ends
            # before donor grasp closes; nothing after this prefix is feedback.
            prefix_end=60;blend_end=140
            for k in range(blend_end+1):
                src=receiver.q_ref[min(k,len(receiver.q_ref)-1)]
                if k<=prefix_end:commands[k]=src
                else:
                    u=(k-prefix_end)/(blend_end-prefix_end);s=u*u*(3-2*u)
                    delta=commands[k]-src;delta[3:6]=(delta[3:6]+np.pi)%(2*np.pi)-np.pi
                    commands[k]=src+s*delta
            # Dense command-frame -> receiver source-reference correspondence.
            mapping=np.empty(len(commands),dtype=np.int64)
            mapping[:180]=np.arange(180)
            mapping[180:carry_start]=np.rint(np.linspace(start,start+(end-start)*.3,carry_start-180)).astype(int)
            mapping[carry_start:carry_end]=np.rint(np.linspace(start+(end-start)*.3,end,carry_end-carry_start)).astype(int)
            mapping[carry_end:]=end
            track=output/f'row{i}_commands_v1.npz'
            np.savez_compressed(track,ctrl=commands,reference_index=mapping)
            recipe={
                'schema':'direct_capture_repair.v1','name':f'guangxue_bowl{action}_row{i}_contact_carry_v1','active_object':'bowl',
                'source':{'dataset_name':dataset_path.name,'version':5,'row':i,'uuid':rows[i]['index']['uuid'],'frames':rows[i]['trajectory_metadata']['total_frames']},
                'solver':donor_recipe['solver'],
                'edit':{'translation':translation.tolist(),'rotation':(Rotation.from_quat(receiver.object_quat_xyzw[0]).inv()*donor_initial_rot).as_rotvec().tolist()},
                'command_track':{'path':track.name,'sha256':hashlib.sha256(track.read_bytes()).hexdigest(),'frames':len(commands)},
                'transfer':{'donor_row':donor_row,'prefix_preserved_through_step':prefix_end,'approach_blend_steps':[prefix_end,blend_end],
                    'carry_steps':[carry_start,carry_end],'receiver_source_movement_steps':[start,end],
                    'retimed':True,'source_goal_world':goal.tolist(),'receiver_lateral_path_applied':True},
                'scope':'Contact-template candidate. Source noncontact prefix preserved, grasp/carry/level/release primitive transferred to receiver initial scene and receiver lateral path/end goal. Explicit retiming. Requires physical validation.'}
            patch=output/f'row{i}_v1.json';patch.write_text(json.dumps(recipe,indent=2)+'\n')
            inventory.append({'row':i,'action':action,'patch':str(patch.resolve()),'post_padding':500})
            print('prepared',i,action,flush=True)
    inventory.sort(key=lambda x:x['row'])
    (output/'inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
    return inventory


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--donor-bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.dataset,a.donor_bundle,a.output)),flush=True)
if __name__=='__main__':main()
