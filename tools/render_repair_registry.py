"""Render accepted saved motions; physics is not resimulated or altered."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from sim.manorl.assets import compile_unified_model,COLLISION_GEOM_GROUP

p=argparse.ArgumentParser(__doc__);p.add_argument('--registry',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--videos',action='store_true');p.add_argument('--rows',default='');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
selected={int(x) for x in a.rows.split(',') if x};items=json.load(open(a.registry))['rows'];models={};renderers={};font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
for row in items:
 if row['status']!='accepted' or selected and row['row'] not in selected:continue
 out=a.output/f'row{row["row"]:02d}';out.mkdir(exist_ok=True)
 manifest=json.load(open(Path(row['run'])/'manifest.json'));source=str(Path(row['run']).resolve())
 if (out/'source.json').exists() and json.load(open(out/'source.json'))['run']==source and (out/'preview.jpg').exists() and (not a.videos or (out/'replay.mp4').exists()):continue
 t=np.load(Path(row['run'])/'trajectory.npz');names=tuple(t['scene_object_names'].tolist());active=names.index(manifest['patch']['active_object']);pos=t['scene_object_pos'][:,active];action=manifest['source_gesture'][:3]
 if names not in models:
  mj,m=compile_unified_model(object_types=names,object_collisions=True,physics_timestep=.0025,visual_meshes=True);models[names]=(mj,m);renderers[names]=mj.Renderer(m,width=640,height=480)
 mj,m=models[names];d=mj.MjData(m);ren=renderers[names];assert m.nq==t['qpos'].shape[1];camera=mj.MjvCamera();xyz=t['scene_object_pos'].reshape(-1,3);low=xyz.min(0);high=xyz.max(0);camera.lookat[:]=(low+high)/2+[0,0,.035];camera.distance=max(.6,float(np.linalg.norm(high-low)*1.35)+.35);camera.azimuth=105;camera.elevation=-27;option=mj.MjvOption();option.geomgroup[COLLISION_GEOM_GROUP]=0
 def draw_frame(i):
  d.qpos[:]=t['qpos'][i];mj.mj_forward(m,d);ren.update_scene(d,camera=camera,scene_option=option);image=Image.new('RGB',(640,510),'white');image.paste(Image.fromarray(ren.render()),(0,30));ImageDraw.Draw(image).text((8,5),f'Row {row["row"]} | {i/100:.2f}s | source {t["source_frame"][i]}',font=font,fill='black');return image
 n=len(pos);peak=int(pos[:,2].argmax());move=np.linalg.norm(pos[:,:2]-pos[0,:2],axis=1);furthest=int(move.argmax());stages=sorted([0,max(0,peak-70),peak,furthest,int(n*.85),n-1])
 sheet=Image.new('RGB',(1280,1530))
 for k,i in enumerate(stages):sheet.paste(draw_frame(i),((k%2)*640,(k//2)*510))
 sheet.save(out/'preview.jpg',quality=88)
 if a.videos:
  import imageio.v2 as imageio
  with imageio.get_writer(out/'replay.mp4',fps=25,codec='libx264',quality=7,macro_block_size=2) as writer:
   for i in range(0,n,4):writer.append_data(np.asarray(draw_frame(i)))
 (out/'source.json').write_text(json.dumps({'run':source,'row':row['row'],'source_uuid':row['uuid'],'frames':n},indent=2));print('rendered',row['row'],'video',a.videos,flush=True)
for r in renderers.values():r.close()
