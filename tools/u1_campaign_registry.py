"""Freeze the approved local parent loaders into portable model/input bundles.
Ignored historical loaders are used only at build time; their source hashes are
recorded. Runtime loads the immutable bundle, never an interactive workspace.
"""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json
import sys
import uuid
import numpy as np
from sim.manorl.u1_campaign import ACTIONS, SETTING, digest, file_sha, array_sha, write_json, verify_signed

def build(root, output):
    import mujoco as mj
    from sim.manorl.local_contact_repair import MODEL_FIELDS
    from sim.manorl.start_augmentation import scene_contacts
    from tools.u1_placement_workspace import assert_u1
    root = Path(root).resolve(); output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    scratch = root/'outputs/four_action_single_v1'; sys.path.insert(0, str(scratch))
    from run_formal import load_formal
    from run_teacher import load_teacher
    from run_reference import load, setting
    bowl = json.loads((scratch/'correct_bowl_single_candidates.json').read_text())
    parents = {}
    for action in ACTIONS:
        if action in ('003','007','009'):
            I,m,b,t,source,adrs = load_formal('formal_005') if action=='003' else load_teacher({'007':'B_row200','009':'A_row050'}[action])
            meta = bowl['actions'][action]
            target_path = scratch/meta['target']; evidence_path=scratch/meta['validation']; trace_path=scratch/meta['representative_trace']
            parent_uuid=meta['candidate_uuid']; semantics={'003':'bowl stove-to-world placement','007':'bowl placement on cuboid1','009':'bowl sustained unsupported hold'}[action]
        elif action=='005':
            I,m,b,t,source,adrs = load_teacher('A_row030')
            z=np.load(scratch/'live005_row847_full_U1/candidate.npz'); tq=z['teacher_qpos']; t={'qpos':tq}
            I=SimpleNamespace(**vars(I)); I.frames=len(tq); I.initial={'qpos':tq[0].copy(),'qvel':np.zeros(m.nv)}
            I.metrics={'source_metadata':dict(I.metrics['source_metadata'],source_gesture='005-mayonnaisebottle-pick-place-alternate')}
            I.arrays=dict(I.arrays, source_object_pos=np.stack([tq[:,a:a+3] for a in adrs],1),source_object_quat_xyzw=np.stack([tq[:,a+3:a+7][:,[1,2,3,0]] for a in adrs],1))
            source=json.loads((scratch/'live005_row847_full_U1/derivation.json').read_text())
            target_path=scratch/'action005_row847_full_U1/independent01/target.npy'; evidence_path=target_path.parent/'report.json'; trace_path=target_path.parent/'trace.npz'
            parent_uuid=str(uuid.uuid5(uuid.NAMESPACE_URL,'U1-row847-alternate-pick-place:'+file_sha(target_path)))
            semantics='alternate strong-pinch pick/place; action04 donor row847; no deep inversion'
        else:
            I,m,b,t,source,adrs=load(82)
            target_path=root/'outputs/row82_release_repair_v13/target.npy'; evidence_path=target_path.parent/'validation.json'
            meta=json.loads(evidence_path.read_text()); trace_path=target_path.parent/meta['representative_trace']; parent_uuid=meta['candidate_uuid']
            semantics='pitcher pour/return; supported yaw registration; thumb-first reverse-path release-v13'
        target=np.load(target_path); evidence=json.loads(evidence_path.read_text())
        contract=assert_u1(m,I); fingerprint=setting(m,I.arrays['scene_object_names'].tolist())
        if digest(fingerprint)!=SETTING: raise ValueError('U1 setting fingerprint mismatch')
        expected=evidence.get('target_sha256',evidence.get('frozen_target_sha256'))
        if expected != array_sha(target): raise ValueError('accepted evidence target hash mismatch')
        accepted=(evidence.get('physical',{}).get('physical_pose_pass') is True or str(evidence.get('status','')).startswith('local_U1_two_fresh_release'))
        if not accepted: raise ValueError('missing explicit final parent acceptance')
        tr=np.load(trace_path); names=I.arrays['scene_object_names'].tolist()
        first=next((f for f,(q,v) in enumerate(zip(tr['qpos'],tr['qvel'])) if scene_contacts(m,q,v,names)),None)
        C=None if first is None else first-12
        if C is None or C<3: raise ValueError(f'{action}: insufficient precontact window: {first}')
        folder=output/action; folder.mkdir()
        np.save(folder/'target.npy',target)
        mj.mj_saveModel(m,str(folder/'model.mjb'),None)
        arrays={k:np.asarray(v) for k,v in I.arrays.items() if isinstance(v,np.ndarray)}
        np.savez_compressed(folder/'input.npz',**arrays,qpos0=I.initial['qpos'],qvel0=I.initial['qvel'],teacher=t['qpos'],**{'native_'+k:np.asarray(I.manifest['native'][k]) for k in MODEL_FIELDS})
        write_json(folder/'evidence.json',evidence)
        parents[action]=dict(action=action, parent_uuid=parent_uuid, semantics=semantics, source=source,
            final_registry_status='accepted_for_named_semantics', stale_derivation_acceptance_superseded=True,
            frames=I.frames, row_id=I.row_id, metrics=I.metrics, names=names, C=C, first_scene_contact=first,
            contract=contract, setting_sha256=SETTING, setting=fingerprint, initial_qpos_sha256=array_sha(I.initial['qpos']),initial_qvel_sha256=array_sha(I.initial['qvel']),
            target_array_sha256=array_sha(target), source_target=str(target_path), source_trace_sha256=file_sha(trace_path),
            files={name:file_sha(folder/name) for name in ('model.mjb','input.npz','target.npy','evidence.json')})
    import importlib.metadata
    runtime={name:importlib.metadata.version(name) for name in ('mujoco','warp-lang','numpy')}
    payload=dict(schema='u1-parent-registry-v1',setting_sha256=SETTING,parents=parents,runtime=runtime,
        builder_sources={p.name:file_sha(p) for p in (scratch/'run_formal.py',scratch/'run_teacher.py',scratch/'run_reference.py',scratch/'launch_005_row847_full.py',scratch/'validate_005_row847.py')})
    write_json(output/'registry.json',dict(payload,digest=digest(payload)))
    return output/'registry.json'

def load_parent(registry_path, action):
    import mujoco as mj
    from sim.manorl.local_contact_repair import MODEL_FIELDS
    registry_path=Path(registry_path); r=json.loads(registry_path.read_text()); verify_signed(r)
    if r['setting_sha256']!=SETTING: raise ValueError('wrong U1 setting')
    p=r['parents'][action]; folder=registry_path.parent/action
    if p['final_registry_status']!='accepted_for_named_semantics': raise ValueError('parent not accepted')
    for name,h in p['files'].items():
        if file_sha(folder/name)!=h: raise ValueError('parent bundle file changed: '+name)
    import importlib.metadata
    for name,version in r['runtime'].items():
        if importlib.metadata.version(name)!=version: raise ValueError('runtime version changed: '+name)
    m=mj.MjModel.from_binary_path(str(folder/'model.mjb')); z=np.load(folder/'input.npz')
    I=SimpleNamespace(row_id=p['row_id'],frames=p['frames'],hz=120,metrics=p['metrics'],
        arrays={k:z[k] for k in z.files if not k.startswith('native_') and k not in ('qpos0','qvel0','teacher')},
        initial={'qpos':z['qpos0'],'qvel':z['qvel0']},manifest={'native':{k:z['native_'+k] for k in MODEL_FIELDS}})
    return r,p,I,m,np.load(folder/'target.npy'),{'qpos':z['teacher']}
