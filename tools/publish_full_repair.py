"""Publish a complete UUID-covered repair bundle; refuses partial acceptance."""
import argparse,copy,hashlib,json,shutil,sys
from pathlib import Path
import numpy as np
import lance,pyarrow as pa

p=argparse.ArgumentParser(__doc__);p.add_argument('--registry',type=Path,required=True);p.add_argument('--first-bundle',type=Path,required=True);p.add_argument('--media',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--command-roots',type=Path,nargs='+',required=True);a=p.parse_args()
reg=json.load(open(a.registry));entries=reg['rows'];assert reg['total_rows']==59
if len(entries)!=59 or len({e['uuid'] for e in entries})!=59 or any(e['status']!='accepted' for e in entries):raise ValueError('all59source UUIDs must pass before publication')
source=lance.dataset(reg['source_dataset'],version=reg['source_version']);identities=source.to_table(columns=['index']).to_pylist()
for e in entries:
 if identities[e['row']]['index']['uuid']!=e['uuid']:raise ValueError('source UUID mismatch')
sys.path.insert(0,str(a.first_bundle/'tools'));from export_repaired_motion import export_motion
# Check every required media asset and command before creating publication paths.
assets={}
for e in entries:
 media=a.media/f'row{e["row"]:02d}'
 for name in ['preview.jpg','replay.mp4']:
  if not (media/name).is_file():raise FileNotFoundError(media/name)
 recipe=json.load(open(Path(e['run'])/'manifest.json'))['patch']
 if 'command_track' in recipe:
  r=recipe['command_track'];matches=[]
  for root in a.command_roots:
   for path in root.rglob(Path(r['path']).name):
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256']:matches.append(path)
  if not matches:raise FileNotFoundError(f'no hash-matched command asset for row{e["row"]}')
  assets[e['row']]=matches[0]
a.output.mkdir(parents=True,exist_ok=False)
for n in ['recordings','patches','videos','previews','tools']:(a.output/n).mkdir()
for name in ['replay_repaired_capture.py','replay_pose_edits.py','export_repaired_motion.py']:shutil.copyfile(a.first_bundle/'tools'/name,a.output/'tools'/name)
columns=[];catalog=[]
for e in entries:
 row=e['row'];key=f'row{row:02d}';run=Path(e['run']);out=a.output/'recordings'/key;out.mkdir();manifest=json.load(open(run/'manifest.json'));recipe=copy.deepcopy(manifest['patch']);original_recipe=copy.deepcopy(recipe)
 if row in assets:
  filename=key+'_commands.npz';shutil.copyfile(assets[row],a.output/'patches'/filename);recipe['command_track']['path']=filename
 (a.output/'patches'/(key+'.json')).write_text(json.dumps(recipe,indent=2)+'\n')
 manifest['patch']=recipe;manifest['original_recipe_sha256']=hashlib.sha256(json.dumps(original_recipe,sort_keys=True).encode()).hexdigest();manifest['publication_note']='Command asset relative path normalized only; source, edit and hashed bytes unchanged.'
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 for n in ['trajectory.npz','result.json']:shutil.copyfile(run/n,out/n)
 v=copy.deepcopy(e['validation']);(out/'validation.json').write_text(json.dumps(v,indent=2)+'\n')
 export_motion(out,out/'motion.lance');table=lance.dataset(out/'motion.lance',version=1).to_table();columns.append(table.replace_schema_metadata({b'contract':b'direct_repaired_physical_motion.v1',b'coverage':b'all59source_v5_rows'}))
 media=a.media/key;shutil.copyfile(media/'replay.mp4',a.output/'videos'/(key+'.mp4'));shutil.copyfile(media/'preview.jpg',a.output/'previews'/(key+'.jpg'))
 t=np.load(out/'trajectory.npz',allow_pickle=False);record=table.to_pylist()[0]
 for i,object_state in enumerate(record['objects']):
  np.testing.assert_array_equal(object_state['pos'],t['scene_object_pos'][:,i]);np.testing.assert_array_equal(object_state['rot_aa'],t['scene_object_rot_aa'][:,i])
 np.testing.assert_array_equal(record['hands'][0]['command_target_dof'],t['ctrl'])
 catalog.append({'row':row,'source_uuid':e['uuid'],'gesture':e['gesture'],'status':'accepted','patch':'patches/'+key+'.json','recording':'recordings/'+key,'video':'videos/'+key+'.mp4','preview':'previews/'+key+'.jpg','post_padding':manifest['post_padding'],'frames':len(t['ctrl']),'validation':v})
 print('published',row,flush=True)
merged=pa.concat_tables(columns);lance.write_dataset(merged,a.output/'repaired_all59.lance',mode='create');ds=lance.dataset(a.output/'repaired_all59.lance',version=1)
if ds.count_rows()!=59:raise RuntimeError('aggregate row count changed')
check=ds.to_table(columns=['index','trajectory_metadata']).to_pylist()
assert [r['index']['source_row_index'] for r in check]==list(range(59))
assert [r['index']['seed_uuid'] for r in check]==[e['uuid'] for e in entries]
(a.output/'catalog.json').write_text(json.dumps({'schema':'direct_capture_repair_bundle.v1','source_dataset':reg['source_dataset'],'source_version':5,'source_rows_total':59,'repaired_rows':59,'dataset':'repaired_all59.lance','generated_motion_contract':'direct_repaired_physical_motion.v1','solver':{'cone':'elliptic','impratio':100.},'samples':catalog,'frames_total':sum(x['frames'] for x in catalog)},indent=2,ensure_ascii=False)+'\n')
print('COMPLETE',a.output,'rows',ds.count_rows(),flush=True)
