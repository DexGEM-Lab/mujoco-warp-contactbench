"""Check published actual values, temporal separation, IDs, and video contracts."""
import argparse,hashlib,json,subprocess
from pathlib import Path
import lance,numpy as np
p=argparse.ArgumentParser(__doc__);p.add_argument('--bundle',type=Path,required=True);a=p.parse_args();b=a.bundle;catalog=json.load(open(b/'catalog.json'));ds=lance.dataset(b/catalog['dataset'],version=1);source=lance.dataset(catalog['source_dataset'],version=5);identities=source.to_table(columns=['index']).to_pylist();assert ds.count_rows()==59;frames=0
for c in catalog['samples']:
 i=c['row'];r=ds.take([i]).to_pylist()[0];assert r['index']['source_row_index']==i;assert r['index']['seed_uuid']==identities[i]['index']['uuid']==c['source_uuid'];N=c['frames'];frames+=N
 trace=np.load(b/c['recording']/'trajectory.npz',allow_pickle=False);audit=np.load(b/c['stability_test']/'trajectory.npz',allow_pickle=False)
 assert len(trace['qpos'])==N and len(audit['qpos'])==N+200
 for k in ['qpos','ctrl','scene_object_pos','scene_object_rot_aa']:
  np.testing.assert_array_equal(trace[k],audit[k][:N])
 np.testing.assert_allclose(audit['ctrl'][N:],np.repeat(audit['ctrl'][N-1:N],200,axis=0),rtol=0,atol=0)
 for j,state in enumerate(r['objects']):
  np.testing.assert_array_equal(state['pos'],trace['scene_object_pos'][:,j]);np.testing.assert_array_equal(state['rot_aa'],trace['scene_object_rot_aa'][:,j])
 np.testing.assert_array_equal(r['hands'][0]['command_target_dof'],trace['ctrl']);np.testing.assert_array_equal(r['hands'][0]['urdf_dof'],trace['qpos'][:,:28])
 patch=json.load(open(b/c['patch']));record=patch['command_track'];asset=(b/c['patch']).parent/record['path'];assert hashlib.sha256(asset.read_bytes()).hexdigest()==record['sha256'];assert record['frames']==N and patch['audit_tail_frames']==0
 video=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,nb_frames,r_frame_rate,duration','-of','json',str(b/c['video'])],text=True))['streams'][0]
 assert int(video['width'])==640 and int(video['height'])==510
 assert abs(float(video['duration'])-N/100.)<=.041
 assert int(video['nb_frames'])==int(np.ceil(N/4))
 assert c['validation']['status']=='accepted' and c['validation']['normal_frames']==N
 print('verified',i,N,flush=True)
assert frames==catalog['frames_total']==65000
report={'status':'verified','source_version':5,'source_rows':59,'output_rows':59,'normal_frames':frames,'video_count':59,'normal_equals_new_physics_prefix':True,'audit_extension_frames_each':200,'export_values_equal_recordings':True,'source_uuids_exact':True,'video_durations_match_normal_motion':True};(b/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
