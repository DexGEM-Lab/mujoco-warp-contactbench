"""Single row130 acquisition solve and frozen MJX-Warp discriminator."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import mujoco as mj
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
from tools.fit_action005_reference_contacts import smooth, bearing_topology

BASE = Path('outputs/action005_direct120_v1/baseline')
FINGERS = ('thumb', 'index', 'middle', 'ring', 'pinky')


def construct(root, m):
    reference = np.load(BASE/'reference.npy')
    target = np.load(BASE/'target.npy')
    original = target.copy()
    # The baseline supported object pose is geometry evidence only, never runtime state.
    supported = np.load(BASE/'trace.npz')['qpos'][200]
    q = reference[200].copy()
    q[28:] = supported[28:]
    seed = q[6:28].copy()
    d = mj.MjData(m)
    ob = m.body('mayonnaisebottle').id
    objects = [i for i in range(m.ngeom) if m.geom_bodyid[i] == ob]
    distal = [m.geom(f'{f}_{"ip" if f=="thumb" else "dip"}_collision').id for f in FINGERS]

    def geometry(x):
        d.qpos[:] = q; d.qpos[6:28] = x
        mj.mj_forward(m, d)
        rot = d.xmat[ob].reshape(3, 3)
        gaps=[]; points=[]; normals=[]
        for g in distal:
            choices=[]
            for o in objects:
                witness=np.zeros(6)
                gap=float(mj.mj_geomDistance(m,d,g,o,1.,witness))
                if abs(np.linalg.norm(witness[3:]-witness[:3])-abs(gap)) < 1e-5 and np.linalg.norm(witness[3:]-witness[:3]) > 1e-8:
                    normal=(witness[3:]-witness[:3])/np.linalg.norm(witness[3:]-witness[:3])
                    choices.append((gap,witness[3:].copy(),normal))
                for c in d.contact:
                    if set(map(int,c.geom)) == {g,o}:
                        normal=np.asarray(c.frame).reshape(3,3)[0]*(1 if int(c.geom[0])==g else -1)
                        choices.append((float(c.dist),np.asarray(c.pos).copy(),normal))
            if not choices:
                raise RuntimeError(f'No valid surface witness for {FINGERS[len(gaps)]}')
            gap,p,n=min(choices,key=lambda item:item[0])
            gaps.append(gap);points.append(rot.T@(p-d.xpos[ob]));normals.append(rot.T@n)
        return np.asarray(gaps),np.asarray(points),np.asarray(normals)

    _,initial_points,_=geometry(seed)
    side=initial_points[1:,:2].mean(axis=0);side/=np.linalg.norm(side)
    axial=initial_points[1:,2].copy()
    # Preserve source index-to-pinky axial ordering, not a donor's contact patches.
    for i in range(1,4): axial[i]=min(axial[i],axial[i-1]-.012)
    def residual(x):
        gaps,points,normals=geometry(x)
        radial=points[:,:2]/np.maximum(np.linalg.norm(points[:,:2],axis=1)[:,None],1e-8)
        return np.r_[ (gaps+.0004)*1000,
                     (points[1:,2]-axial)*250,
                     np.maximum(.5-radial[1:]@side,0)*10,
                     max(0.,radial[0]@side+.3)*10,
                     (x-seed)*.04 ]
    lo,hi=m.jnt_range[6:28].T
    fit=least_squares(residual,np.clip(seed,lo+1e-7,hi-1e-7),bounds=(lo,hi),max_nfev=240,ftol=1e-8,xtol=1e-8,gtol=1e-8)
    gaps,points,normals=geometry(fit.x)
    # One offline solve only. Geometry quality is reported; physics adjudicates it.
    f=np.arange(len(target))
    nonthumb=smooth((f-140)/50)*smooth((340-f)/40)
    thumb=smooth((f-205)/18)*smooth((340-f)/40)
    corrected=reference[:,:28].copy()
    corrected[:,6:28]+=fit.x-seed
    corrected[190:225,12:28]=fit.x[6:]
    target[:,12:28]+=nonthumb[:,None]*(corrected[:,12:28]-reference[:,12:28])
    # Hold the source's earlier open thumb through the four-finger phase.
    opening=smooth((f-140)/30)*smooth((223-f)/18)
    target[:,6:12]+=opening[:,None]*(reference[160,6:12]-reference[:,6:12])
    target[:,6:12]+=thumb[:,None]*(fit.x[:6]-seed[:6])
    target=np.clip(target,*m.actuator_ctrlrange.T)
    assert np.array_equal(target[:,:6],original[:,:6])
    assert np.array_equal(target[340:],original[340:])
    assert np.isfinite(target).all()
    np.save(root/'target.npy',target)
    np.save(root/'fit_qpos.npy',np.r_[q[:6],fit.x,q[28:]])
    report=dict(solve_count=1,frame=200,wrist_translation_m=0.,wrist_rotation_deg=0.,
                success=bool(fit.success),nfev=fit.nfev,gaps_m=gaps.tolist(),points_object=points.tolist(),
                normals_object=normals.tolist(),finger_delta_rad=(fit.x-seed).tolist(),
                nonthumb_phase=[140,190,225],thumb_phase=[205,223],correction_fade=[300,340],
                target_sha256=hashlib.sha256(target.tobytes()).hexdigest(),
                baseline_target_sha256=hashlib.sha256(original.tobytes()).hexdigest(),
                shallow_geometry=bool(np.max(abs(gaps+.0004))<.001),
                reference_wrist_exact=True,reference_tail_exact_from=340)
    (root/'fit.json').write_text(json.dumps(report,indent=2)+'\n')
    print('FIT',json.dumps(report),flush=True)
    return target


def local_contacts(contacts, position, rotation):
    return [dict(c, object_local_position=(rotation.T@(np.asarray(c['position'])-position)).tolist(),
                 object_local_normal=(rotation.T@(np.asarray(c['basis'])[0]*c['object_sign'])).tolist()) for c in contacts]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--restart',action='store_true');parser.add_argument('--phase',choices=['focused','full','repeat'],default='focused');args=parser.parse_args()
    sys.path.insert(0,str(Path('outputs/four_action_single_v1').resolve()))
    from run_reference import load,setting
    from tools.u1_interactive_session import Session
    from tools.search_action005_critical_grip import observe,U1
    from sim.manorl.local_contact_repair import evaluate
    inp,m,b,t,provenance,adrs=load(130)
    common=setting(m,inp.arrays['scene_object_names'].tolist())
    digest=hashlib.sha256(json.dumps(common,sort_keys=True,separators=(',',':')).encode()).hexdigest();assert digest==U1
    root=args.output
    if args.phase=='focused' and not args.restart:root.mkdir(exist_ok=False);target=construct(root,m)
    elif args.restart:
        assert args.phase=='focused'
        target=np.load(root/'target.npy')
        prior=json.loads((root/'interrupted/manifest.json').read_text())
        assert hashlib.sha256(target.tobytes()).hexdigest()==prior['target_sha256']
        assert digest==prior['setting_sha256']
        assert np.array_equal(m.body_mass,mj.MjModel.from_binary_path(str(BASE/'model.mjb')).body_mass)
    else:
        assert json.loads((root/'focused/result.json').read_text())['accepted']
        if args.phase=='repeat':assert json.loads((root/'full/result.json').read_text())['accepted']
        target=np.load(root/'target.npy')
    out=root/args.phase;out.mkdir(exist_ok=False)
    manifest=dict(source=provenance,setting_sha256=digest,setting=common,pid=os.getpid(),
                  backend='MJX-Warp cuda:0',post_frame0_state_writes=False,applied_forces=False,
                  target_sha256=hashlib.sha256(target.tobytes()).hexdigest(),phase=args.phase,
                  geometry_only_cpu=True,force_sampling_hz=120,capacity_sampling_hz=480,
                  boundary='thumb>1N before all nonthumb>0.2N, acquisition missing at205, lift missing at250, opposed ring+pinky missing after lift')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    s=Session(inp,m,target,t,provenance);d=s.cpu;hb=s.hand_body;ob=s.object_body
    hr=d.xmat[hb].reshape(3,3);baseline=(hr.T@(d.xpos[ob]-d.xpos[hb]),hr.T@d.xmat[ob].reshape(3,3),inp.initial['qpos'].copy());previous=baseline[:2]
    poses=[s.data.qpos.numpy()[0].copy()];vel=[s.data.qvel.numpy()[0].copy()];controls=[s.data.ctrl.numpy()[0].copy()]
    adr=int(m.joint('mayonnaisebottle_free').qposadr[0]);first={f:None for f in FINGERS};high=dict(nacon=0,ncollision=0,nefc=0);capacities=[];rows=[];failure=None
    end=300 if args.phase=='focused' else inp.frames-1
    with gzip.open(out/'telemetry.jsonl.gz','wt') as stream:
        for frame in range(1,end+1):
            s.frame=frame;s.data.ctrl.assign(target[frame][None].astype(np.float32))
            for sub in range(4):
                s.mw.step(s.wm,s.data);s.wp.synchronize();values=[]
                for field,limit in [('nacon',1024),('ncollision',256),('nefc',4096)]:
                    value=int(getattr(s.data,field).numpy().max());values.append(value);high[field]=max(high[field],value)
                    if value>=limit:failure=dict(frame=frame,substep=sub,reason='capacity '+field)
                capacities.append([frame,sub,*values])
                assert not np.any(s.data.xfrc_applied.numpy()) and not np.any(s.data.qfrc_applied.numpy())
                if failure:break
            row,previous,q,v,ctrl=observe(s,baseline,previous,target[frame],frame)
            poses.append(q);vel.append(v);controls.append(ctrl)
            row['lift_m']=float(q[adr+2]-poses[0][adr+2]);row['actual_tilt_deg']=float(np.rad2deg(np.arccos(np.clip(R.from_quat(q[adr+3:adr+7],scalar_first=True).as_matrix()[2,2],-1,1))))
            row['contacts']=local_contacts(row['contacts'],q[adr:adr+3],R.from_quat(q[adr+3:adr+7],scalar_first=True).as_matrix())
            row['topology']=bearing_topology(row['contacts'])
            for f in FINGERS:
                if first[f] is None and row['bearing_normal_N'][f]>(1. if f=='thumb' else .2):first[f]=frame
            if not np.isfinite(q).all() or not np.isfinite(v).all() or not np.isfinite(ctrl).all():failure=dict(frame=frame,reason='nonfinite')
            if row['applied_control_error']>1e-6:failure=dict(frame=frame,reason='noncanonical control')
            if not failure and first['thumb'] is not None and any(first[f] is None or first[f]>=first['thumb'] for f in FINGERS[1:]):failure=dict(frame=frame,reason='thumb high loading before four-finger acquisition')
            if not failure and frame==205 and any(first[f] is None for f in FINGERS[1:]):failure=dict(frame=frame,reason='four-finger acquisition absent at phase boundary205')
            if not failure and frame==250 and row['lift_m']<=.02:failure=dict(frame=frame,reason='lift <=2cm at early-lift boundary250')
            if not failure and 250<=frame<=300 and (any(row['bearing_normal_N'][f]<=.2 for f in FINGERS) or row['thumb_opposition_cosine']<.3):failure=dict(frame=frame,reason='opposed five-finger bearing lost during early lift/tilt')
            rows.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
            if frame%30==0 or failure:print('FRAME',frame,'lift',row['lift_m'],'bearing',row['bearing_normal_N'],'failure',failure,flush=True)
            if failure:break
    poses=np.asarray(poses);vel=np.asarray(vel);controls=np.asarray(controls)
    np.savez_compressed(out/'trace.npz',qpos=poses,qvel=vel,ctrl=controls,capacities=capacities)
    physical=None
    if not failure and args.phase!='focused':
        tr=dict(qpos=poses,qvel=vel,desired=t['qpos'][:,:28],actual_object_pos=np.stack([poses[:,a:a+3] for a in adrs],axis=1),actual_object_quat_xyzw=np.stack([poses[:,a+3:a+7][:,[1,2,3,0]] for a in adrs],axis=1))
        physical,_=evaluate(m,inp,tr)
        if not physical['physical_pose_pass'] or max(r['actual_tilt_deg'] for r in rows)<110:failure=dict(frame=frame,reason='full action gates failed')
    result=dict(accepted=failure is None,failure=failure,frames=len(poses),first_bearing_frames=first,
                max_lift_m=float((poses[:,adr+2]-poses[0,adr+2]).max()),peak_actual_tilt_deg=max(r['actual_tilt_deg'] for r in rows),
                capacity_high_water=high,finite=bool(np.isfinite(poses).all() and np.isfinite(controls).all()),
                ctrl_target_max_error=float(abs(controls-target[:len(controls)]).max()),physical=physical,
                last_bearing_N=rows[-1]['bearing_normal_N'],last_topology=rows[-1]['topology'],pid=os.getpid(),
                full_pass_count=0 if args.phase=='focused' or failure else (2 if args.phase=='repeat' else 1))
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print('RESULT',json.dumps(result),flush=True)

if __name__=='__main__':main()
