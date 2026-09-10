"""Evidence-specific final release and collision-clearance corrections."""
import copy,hashlib,json
from pathlib import Path
import numpy as np
from sim.manorl.assets import compile_unified_model

work=Path(__file__).resolve().parents[1];root=work/'outputs/full_repair';out=work/'patches/full_containers_v5';out.mkdir(parents=True,exist_ok=True);inventory=[]
for row,version,anchor in [(26,3,650),(29,4,650)]:
 run=root/f'containers_pass{version}/row{row}';t=np.load(run/'trajectory.npz');manifest=json.load(open(run/'manifest.json'));recipe=copy.deepcopy(manifest['patch']);q=t['qpos'][anchor].copy();cmd=t['ctrl'][anchor].copy();names=t['scene_object_names'].tolist();idx=names.index('mayonnaisebottle');pos=t['scene_object_pos'][anchor,idx];mj,m=compile_unified_model(object_types=names,object_collisions=True,physics_timestep=.0025);d=mj.MjData(m);d.qpos[:]=q;mj.mj_forward(m,d)
 opened=q[:28].copy();opened[:6]+=d.qfrc_bias[:6]/m.actuator_gainprm[:6,0]
 for j in [7,9,11,13,14,15,17,18,19,21,22,23,25,26,27]:opened[j]=0.
 opened[6:]=np.clip(opened[6:],m.jnt_range[6:28,0],m.jnt_range[6:28,1]);direction=q[:3]-pos;direction[2]=0.;direction/=np.linalg.norm(direction);far=opened.copy();far[:3]+=direction*.18+[0,0,.07]
 commands=list(t['ctrl'][:anchor+1]);mapping=t['reference_indices'][:anchor+1].tolist();end=int(manifest['movement_steps'][1])
 for first,last,n in [(cmd,cmd,80),(cmd,opened,150),(opened,opened,80),(opened,far,180),(far,far,200)]:
  delta=last-first;delta[3:6]=(delta[3:6]+np.pi)%(2*np.pi)-np.pi;begin=mapping[-1]
  for u in np.linspace(0,1,n):commands.append(first+u*u*(3-2*u)*delta);mapping.append(int(round(begin+u*(end-begin))))
 track=out/f'row{row}_commands_v5.npz';np.savez_compressed(track,ctrl=np.array(commands),reference_index=np.array(mapping,dtype=np.int64));recipe['command_track']={'path':track.name,'sha256':hashlib.sha256(track.read_bytes()).hexdigest(),'frames':len(commands)};recipe['name']=f'row{row}_supported_hand_unload_release_v5';recipe['late_repair']={'prefix_end':anchor,'mechanism':'supported bottle was driven away by continuing loaded wrist target; unload at actual hand pose, fully open then retreat','retimed':True};path=out/f'row{row}_v5.json';path.write_text(json.dumps(recipe,indent=2));inventory.append({'row':row,'patch':str(path.resolve()),'post_padding':500})
row=38;run=root/'containers_pass4/row38';t=np.load(run/'trajectory.npz');manifest=json.load(open(run/'manifest.json'));recipe=copy.deepcopy(manifest['patch']);ctrl=t['ctrl'].copy();time=np.arange(len(ctrl))
def smooth(begin,end):
 u=np.clip((time-begin)/(end-begin),0,1);return u*u*(3-2*u)
height=.10*smooth(270,370)*(1-smooth(550,670));ctrl[:,2]+=height
track=out/'row38_commands_v5.npz';np.savez_compressed(track,ctrl=ctrl,reference_index=t['reference_indices']);recipe['command_track']={'path':track.name,'sha256':hashlib.sha256(track.read_bytes()).hexdigest(),'frames':len(ctrl)};recipe['name']='row38_early_pour_clearance_v5';recipe['late_repair']={'mechanism':'avoid pitcher-recipient bowl collision before it perturbs grasp','clearance_z_m':.10,'window':[270,370,550,670]};path=out/'row38_v5.json';path.write_text(json.dumps(recipe,indent=2));inventory.append({'row':38,'patch':str(path.resolve()),'post_padding':500});(out/'inventory.json').write_text(json.dumps(inventory,indent=2));print('prepared',inventory,flush=True)
