#!/usr/bin/env python3
"""Build full per-trajectory nine-object visualization layout from offsets."""
from pathlib import Path
import argparse,json
import numpy as np
from tools.compute_u1_exact800_background_offsets import CANON,ALL_NAMES,HAND_TARGET,OBJ_TARGET

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--lance',type=Path,required=True);p.add_argument('--version',type=int,default=1);p.add_argument('--offsets',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();import lance
 ds=lance.dataset(str(a.lance),version=a.version);off=json.loads(a.offsets.read_text());entries=[];index=0
 for batch in ds.to_batches(batch_size=32,columns=['index','trajectory_metadata','objects']):
  for row in batch.to_pylist():
   meta=row['trajectory_metadata'];active=list(meta['object_names']);uuid=row['index']['uuid'];offsets=off.get('per_row_offsets',{}).get(uuid,{});omit=off.get('per_row_omit',{}).get(uuid,[]);objects={}
   for name,obj in zip(active,row['objects']):
    objects[name]={'role':'active','pos':[round(float(x),7)for x in obj['pos'][0]],'rot_aa':[round(float(x),7)for x in obj['rot_aa'][0]],'pose_source':'lance_frame0'}
   for name in ALL_NAMES:
    if name in active or name in omit:continue
    delta=offsets.get(name);pos=np.asarray(CANON[name],float)
    if delta is not None:pos[:2]+=np.asarray(delta,float)
    objects[name]={'role':'background','pos':[round(float(x),7)for x in pos],'offset_applied':delta,'pose_source':'canonical_layout_plus_offset'if delta else'canonical_layout'}
   entries.append({'row_index':index,'uuid':uuid,'seed_uuid':row['index']['seed_uuid'],'action':meta['gesture'].split('-')[0],'gesture':meta['gesture'],'active_objects':active,'objects':objects});index+=1
 if index!=ds.count_rows():raise ValueError('row count changed during layout build')
 payload={'contract':'manorl.per-trajectory-visual-layout.v2','description':'Visualization-only nine-object layout; active poses from exact800 Lance frame0, static backgrounds from canonical kitchen layout plus approximate clearance offsets. Backgrounds are outside recorded physics.','lance':str(a.lance),'lance_version':a.version,'total_trajectories':index,'canonical_positions':CANON,'hand_clearance_m':HAND_TARGET,'object_clearance_m':OBJ_TARGET,'per_row_omit':off.get('per_row_omit',{}),'trajectories':entries};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n');print(json.dumps({'rows':index,'output':str(a.output)}))
if __name__=='__main__':main()
