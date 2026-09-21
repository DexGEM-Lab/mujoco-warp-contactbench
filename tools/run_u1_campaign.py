"""Native single-world U1 campaign. Production requires --execute-production.
No dataset publication. Registry, plans, failures and traces live on caller storage.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import numpy as np
from sim.manorl.u1_campaign import (ACTIONS, VERSION, Ledger, array_sha, child_uuid,
    digest, file_sha, plan, perturb, verify_signed, write_json)
from tools.u1_campaign_registry import build, load_parent

class Runtime:
    def __init__(self, registry, action):
        from tools.u1_interactive_session import Session
        self.r,self.p,self.I,self.m,self.base,self.teacher=load_parent(registry,action)
        self.s=Session(self.I,self.m,self.base,self.teacher,self.p)
        self.action=action

    def replay(self, delta, folder, frozen=None):
        from sim.manorl.mjx_sim import command_target
        from sim.manorl.start_augmentation import scene_contacts, unsupported_intervals
        from sim.manorl.local_contact_repair import evaluate
        folder=Path(folder); folder.mkdir(parents=True,exist_ok=False)
        target,q0=perturb(self.base,self.I.initial['qpos'],delta,self.p['C'])
        if frozen is not None: target=np.load(frozen)
        np.save(folder/'requested_target.npy',target)
        s=self.s; m=self.m; s.restore(s.initial); s.target=target.copy()
        # Every mutable native buffer restored, then only initial hand pose changes.
        s.data.qpos.assign(q0[None].astype(np.float32))
        s.data.ctrl.assign(command_target(target[0],q0[:28],*m.jnt_range[:28].T)[None].astype(np.float32))
        s.mw.forward(s.wm,s.data); s.wp.synchronize()
        names=self.p['names']; adrs=[int(m.joint(n+'_free').qposadr[0]) for n in names]
        q=[]; v=[]; controls=[]; precontact=[]; gaps=[]; radii=[]; heights=[]; pair_force=[]
        native_path=folder/'native_contacts.jsonl'; high={'contacts':0,'constraints':0}
        with native_path.open('w') as evidence:
            for f in range(self.I.frames):
                if f:
                    # Production uses explicit native substeps so capacity guards
                    # observe every480Hz state, not only the120Hz arrival boundary.
                    s.frame += 1
                    control=command_target(target[f],s.data.qpos.numpy()[0,:28],*m.jnt_range[:28].T)
                    s.data.ctrl.assign(control[None].astype(np.float32))
                    for substep in range(4):
                        s.mw.step(s.wm,s.data); s.wp.synchronize()
                        nsub=int(s.data.nacon.numpy().ravel()[0]); efcsub=int(np.max(s.data.nefc.numpy()))
                        high['contacts']=max(high['contacts'],nsub); high['constraints']=max(high['constraints'],efcsub)
                        if nsub>=1024 or efcsub>=4096:
                            raise ValueError(f'native capacity reached at frame{f} substep{substep}')
                pose=s.data.qpos.numpy()[0].copy(); vel=s.data.qvel.numpy()[0].copy()
                if not np.isfinite(pose).all() or not np.isfinite(vel).all(): raise ValueError('nonfinite state')
                q.append(pose); v.append(vel); controls.append(s.data.ctrl.numpy()[0].copy())
                n=int(s.data.nacon.numpy().ravel()[0]); nefc=int(np.max(s.data.nefc.numpy()))
                high['contacts']=max(high['contacts'],n); high['constraints']=max(high['constraints'],nefc)
                if n>=1024 or nefc>=4096: raise ValueError('native capacity reached')
                pairs=s.data.contact.geom.numpy()[:n]
                ids=s.wp.array(np.arange(n,dtype=np.int32),dtype=s.wp.int32)
                forces=s.wp.zeros(n,dtype=s.wp.spatial_vector)
                if n: s.mw.contact_force(s.wm,s.data,ids,False,forces)
                F=forces.numpy()
                evidence.write(json.dumps(dict(frame=f,geom_pairs=pairs.tolist(),force_contact_frame=F.tolist(),position=s.data.contact.pos.numpy()[:n].tolist()))+'\n')
                if f<self.p['C']-1 and scene_contacts(m,pose,vel,names): precontact.append(f)
                if self.action=='006':
                    d=s.cpu; d.qpos[:]=pose; d.qvel[:]=vel; s.mj.mj_forward(m,d)
                    tg=m.geom('thumb_ip_collision').id; ig=m.geom('index_dip_collision').id
                    gaps.append(float(s.mj.mj_geomDistance(m,d,tg,ig,.15,np.zeros(6))))
                    ob=m.body('pitcherbase').id; bb=m.body('bowl').id
                    sp=d.xpos[ob]+d.xmat[ob].reshape(3,3)@np.array([-.081061,-.0049836,.08246349])
                    radii.append(float(np.linalg.norm(sp[:2]-d.xpos[bb,:2]))); heights.append(float(sp[2]-d.xpos[bb,2]))
                    mask=((pairs[:,0]==tg)&(pairs[:,1]==ig))|((pairs[:,0]==ig)&(pairs[:,1]==tg))
                    pair_force.append(float(F[mask,0].sum()))
        q=np.asarray(q); v=np.asarray(v); controls=np.asarray(controls)
        np.save(folder/'target.npy',controls) # actual executed native controls, not desired commands
        np.savez_compressed(folder/'trace.npz',qpos=q,qvel=v,ctrl=controls)
        trace=dict(qpos=q,qvel=v,desired=target,actual_object_pos=np.stack([q[:,a:a+3] for a in adrs],1),actual_object_quat_xyzw=np.stack([q[:,a+3:a+7][:,[1,2,3,0]] for a in adrs],1))
        physical,diag=evaluate(m,self.I,trace); a=diag['arrays']; gates=dict(physical['gates'])
        error=controls.astype(float)-target; error[:,3:6]=np.arctan2(np.sin(error[:,3:6]),np.cos(error[:,3:6]))
        gates.update(no_premerge_scene_contact=not precontact, no_uncontrolled_flight=not unsupported_intervals(a,120),
            controls_canonical=float(np.max(np.abs(error)))<1e-6,
            initial_pose=bool(np.allclose(q[0],q0,atol=1e-7,rtol=0)),
            initial_velocity=bool(np.allclose(v[0],self.I.initial['qvel'],atol=1e-7,rtol=0)),
            executed_suffix_exact=controls[self.p['C']-1:].tobytes()==self.base[self.p['C']-1:].astype(np.float32).tobytes())
        if self.action=='009':
            gates['terminal_100ms_unsupported_multifinger']=bool(np.all(a['finger_ray_count'][-12:]>=2) and not np.any(a['has_support'][-12:]))
        if self.action=='006':
            f=np.arange(len(q)); pour=(f>=560)&(f<=840)&(a['world_tilt_deg']>=70)
            da=int(m.joint('pitcherbase_free').dofadr[0])
            gates.update(pour_present=bool(np.any(pour)), final_source_orientation_under15deg=physical['object_orientation_final_deg']<15,
                pour_spout_proxy_inside_bowl=bool(np.any(pour) and np.all(np.array(radii)[pour]<.077101396)),
                pour_above_bowl=bool(np.any(pour) and np.all(np.array(heights)[pour]>.025)),
                distal_surfaces_touch_during_pour=bool(np.any(pour) and np.mean(np.array(gaps)[pour]<.0005)>.95),
                native_tip_pair_force_during_pour=bool(np.any(pour & (f%12==0)) and np.all(np.array(pair_force)[pour & (f%12==0)]>.1)),
                settled_final_object=bool(np.linalg.norm(v[-1,da:da+6])<.01),
                release_upright=bool(np.max(a['world_tilt_deg'][1130:])<10))
            a.update(tip_pair_gap_m=np.array(gaps),spout_proxy_radius_m=np.array(radii),spout_proxy_height_m=np.array(heights),tip_pair_normal_force_N=np.array(pair_force))
        np.savez_compressed(folder/'contact_validation.npz',**a)
        write_json(folder/'geometry_contacts.json',diag['contacts'])
        report=dict(accepted=all(gates.values()),gates=gates,physical=physical,premerge_contact_frames=precontact,
            high_water=high, high_water_sampling='every480Hz substep', contract=s.contract,semantics=self.p['semantics'],delta=delta,
            native_force_boundary='last480Hz substep, contact-frame wrench; positions solver contact coordinates',
            sampled_flight_limit='three120Hz samples; subframe flight unmeasured',pid=os.getpid(),
            requested_target_sha256=array_sha(target),executed_target_sha256=array_sha(controls))
        write_json(folder/'result.json',report)
        return report

def execute(a):
    p=json.loads(a.plan.read_text()); verify_signed(p)
    r=json.loads(a.registry.read_text()); verify_signed(r)
    if p['registry_digest']!=r['digest']: raise ValueError('plan registry mismatch')
    if p != plan(r['digest'],p['seed'],p['reserves']): raise ValueError('noncanonical plan')
    a.staging.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (a.staging/(a.action+'.lock')).open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        identity=digest(dict(registry=r['digest'],plan=p['digest'],action=a.action,implementation={str(path):file_sha(path) for path in (Path(__file__),Path('tools/u1_campaign_registry.py'),Path('sim/manorl/u1_campaign.py'),Path('tools/u1_interactive_session.py'),Path('sim/manorl/local_contact_repair.py'),Path('sim/manorl/mjx_sim.py'))}))
        ledger=Ledger(a.staging/(a.action+'.jsonl'),identity)
        if a.mode=='status':
            print(json.dumps(dict(selected=len(ledger.selected()),quota=160,exhausted=sum(x['status']=='exhausted' for x in ledger.rows)))); return
        if a.mode=='run-action' and not a.execute_production: raise ValueError('full run requires --execute-production')
        runtime=Runtime(a.registry,a.action)
        if a.mode=='pilot':
            # Exactly two diagnostic executions; never counted toward production quota.
            for label,delta in [('zero',[0.]*6),('extreme',p['slots'][-1]['candidates'][0]['delta'])]:
                folder=a.staging/a.action/label
                if folder.exists(): raise ValueError('pilot already attempted; do not reroll')
                try: report=runtime.replay(delta,folder)
                except Exception as e:
                    folder.mkdir(parents=True,exist_ok=True); write_json(folder/'failure.json',dict(error=repr(e))); raise
                print(label,json.dumps(report),flush=True)
            return
        if not a.execute_production: raise ValueError('full run requires --execute-production')
        for slot in p['slots']:
            sid=slot['slot']
            if sid in ledger.selected(): continue
            for candidate in slot['candidates']:
                uid=child_uuid(r['digest'],a.action,p['digest'],sid,candidate)
                if any(row.get('uuid')==uid for row in ledger.rows): continue
                folder=a.staging/a.action/uid
                ledger.append(uuid=uid,slot=sid,candidate=candidate,status='started')
                try:
                    first=runtime.replay(candidate['delta'],folder/'first')
                    if not first['accepted']: raise ValueError('first-pass gates failed')
                    command=[sys.executable,'-m','tools.run_u1_campaign','second-pass','--registry',str(a.registry.resolve()),'--action',a.action,'--staging',str(folder/'second'),'--delta',json.dumps(candidate['delta']),'--frozen',str(folder/'first/target.npy')]
                    with (folder/'second-process.log').open('w') as log: subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
                    second=json.loads((folder/'second/result.json').read_text())
                    if not second['accepted'] or second['pid']==first['pid']: raise ValueError('independent gates failed')
                    artifacts={str(path.resolve()):file_sha(path) for path in folder.rglob('*') if path.is_file()}
                    ledger.append(uuid=uid,slot=sid,status='selected',artifacts=artifacts); break
                except Exception as e:
                    ledger.append(uuid=uid,slot=sid,status='rejected',reason=repr(e))
            if sid not in ledger.selected():
                if not any(row['status']=='exhausted' and row['slot']==sid for row in ledger.rows): ledger.append(slot=sid,status='exhausted')
        print(json.dumps(dict(selected=len(ledger.selected()),quota=160)),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['build-registry','plan','pilot','run-action','status','second-pass'])
    p.add_argument('--root',type=Path,default=Path.cwd()); p.add_argument('--registry',type=Path)
    p.add_argument('--plan',type=Path); p.add_argument('--staging',type=Path,required=True)
    p.add_argument('--action',choices=ACTIONS); p.add_argument('--execute-production',action='store_true')
    p.add_argument('--delta'); p.add_argument('--frozen',type=Path)
    a=p.parse_args()
    if a.mode=='build-registry': print(build(a.root,a.staging))
    elif a.mode=='plan':
        r=json.loads(a.registry.read_text()); verify_signed(r); a.staging.mkdir(parents=True,exist_ok=False); write_json(a.staging/'plan.json',plan(r['digest']))
    elif a.mode=='second-pass':
        result=Runtime(a.registry,a.action).replay(json.loads(a.delta),a.staging,a.frozen)
        if not result['accepted']: sys.exit(2)
    else: execute(a)

if __name__=='__main__': main()
