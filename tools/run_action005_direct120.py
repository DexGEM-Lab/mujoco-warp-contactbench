"""One frozen row130 nominal MJX-Warp U1 baseline; no runtime corrections."""
import gzip
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R


def main():
    import mujoco as mj
    sys.path.insert(0, str(Path('outputs/four_action_single_v1').resolve()))
    from run_reference import load, build, setting
    from tools.u1_interactive_session import Session
    from tools.search_action005_critical_grip import observe, U1
    from sim.manorl.local_contact_repair import evaluate
    root = Path(sys.argv[1]); out = root/'baseline'; out.mkdir(exist_ok=False)
    selection = json.loads((root/'selection.json').read_text()); assert selection['selected_row'] == 130
    inp, m, b, t, provenance, adrs = load(130)
    common = setting(m, inp.arrays['scene_object_names'].tolist())
    digest = hashlib.sha256(json.dumps(common, sort_keys=True, separators=(',', ':')).encode()).hexdigest(); assert digest == U1
    target, recipe = build(inp, m, b, t)
    assert np.isfinite(target).all()
    np.save(out/'target.npy', target); np.save(out/'reference.npy', t['qpos']); mj.mj_saveModel(m,str(out/'model.mjb'))
    manifest = dict(source=provenance, recipe=recipe, setting_sha256=digest, setting=common,
                    backend='mujoco.mjx.third_party.mujoco_warp', device='cuda:0', post_frame0_state_writes=False,
                    telemetry='capacities every480Hz substep; native force/basis120Hz last-substep; CPU pose FK only')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    s = Session(inp,m,target,t,provenance); d = s.cpu; hb=s.hand_body; ob=s.object_body
    hr=d.xmat[hb].reshape(3,3); baseline=(hr.T@(d.xpos[ob]-d.xpos[hb]),hr.T@d.xmat[ob].reshape(3,3),inp.initial['qpos'].copy()); previous=baseline[:2]
    poses=[s.data.qpos.numpy()[0].copy()]; velocities=[s.data.qvel.numpy()[0].copy()]; controls=[s.data.ctrl.numpy()[0].copy()]
    capacity=[]; rows=[]; adr=int(m.joint('mayonnaisebottle_free').qposadr[0]); high={'nacon':0,'ncollision':0,'nefc':0}
    with gzip.open(out/'telemetry.jsonl.gz','wt') as stream:
        for frame in range(1,inp.frames):
            s.frame=frame; s.data.ctrl.assign(target[frame][None].astype(np.float32))
            for sub in range(4):
                s.mw.step(s.wm,s.data); s.wp.synchronize(); values=[]
                for field,limit in [('nacon',1024),('ncollision',256),('nefc',4096)]:
                    value=int(getattr(s.data,field).numpy().max()); values.append(value); high[field]=max(high[field],value)
                    if value>=limit: raise RuntimeError('capacity '+field)
                capacity.append([frame,sub,*values])
                assert not np.any(s.data.xfrc_applied.numpy()) and not np.any(s.data.qfrc_applied.numpy())
            row,previous,q,v,ctrl=observe(s,baseline,previous,target[frame],frame)
            poses.append(q); velocities.append(v); controls.append(ctrl)
            ref=t['qpos'][frame]; row['reference_position_error_cm']=float(np.linalg.norm(q[adr:adr+3]-ref[adr:adr+3])*100)
            row['actual_tilt_deg']=float(np.rad2deg(np.arccos(np.clip(R.from_quat(q[adr+3:adr+7],scalar_first=True).as_matrix()[2,2],-1,1))))
            row['reference_tilt_deg']=float(np.rad2deg(np.arccos(np.clip(R.from_quat(ref[adr+3:adr+7],scalar_first=True).as_matrix()[2,2],-1,1))))
            rows.append(row); stream.write(json.dumps(row)+'\n')
            if frame%60==0: print('FRAME',frame,'position_error_cm',row['reference_position_error_cm'],'bearing',row['bearing_normal_N'],flush=True)
    poses=np.asarray(poses); velocities=np.asarray(velocities); controls=np.asarray(controls)
    np.savez_compressed(out/'trace.npz',qpos=poses,qvel=velocities,ctrl=controls,capacities=capacity)
    tr=dict(qpos=poses,qvel=velocities,desired=t['qpos'][:,:28],actual_object_pos=np.stack([poses[:,a:a+3] for a in adrs],axis=1),actual_object_quat_xyzw=np.stack([poses[:,a+3:a+7][:,[1,2,3,0]] for a in adrs],axis=1))
    physical,diag=evaluate(m,inp,tr)
    lift_ref=t['qpos'][:,adr+2]-t['qpos'][0,adr+2]; lift=poses[:,adr+2]-poses[0,adr+2]
    first=next((r for r in rows if lift_ref[r['frame']]>.02 and lift[r['frame']]<.5*lift_ref[r['frame']]),None)
    first_bearing={f:next((r['frame'] for r in rows if r['bearing_normal_N'][f]>(1. if f=='thumb' else .2)),None) for f in ('thumb','index','middle','ring','pinky')}
    result=dict(physical=physical,accepted=False,complete_pass_count=0,first_lift_divergence=first,first_bearing_frames=first_bearing,capacity_high_water=high,
                finite=bool(np.isfinite(poses).all() and np.isfinite(controls).all()),ctrl_target_max_error=float(abs(controls-target).max()),
                reference_peak_frame=int(np.argmax([r['reference_tilt_deg'] for r in rows]))+1,
                actual_peak_tilt_deg=max(r['actual_tilt_deg'] for r in rows),baseline_full_frames=len(poses),repair_run=False)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n'); print('RESULT',json.dumps(physical),flush=True)

if __name__=='__main__': main()
