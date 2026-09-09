"""Record one policy-free replay, source poses and contact geometry for object repair.

Imports the existing replay adapter and viewer; does not train or load a policy.
A requested object transform modifies initialization and reference together.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import lance
import numpy as np
from scipy.spatial.transform import Rotation

from sim.manorl.assets import COLLISION_GEOM_GROUP
from sim.manorl.trajectory import trajectory_from_lance_row, resample_reference_trajectory
from sim.manorl.environment import EnvironmentConfig, MujocoManoEnvironment
from sim.manorl.view_environment import _compile_native_viewer_model, _mirror_native_viewer_data, SOURCE_ALIGNED_COMPATIBILITY


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--dataset', required=True)
    p.add_argument('--version', type=int, required=True)
    p.add_argument('--row', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--delta-pos', type=float, nargs=3, default=[0., 0., 0.])
    p.add_argument('--delta-rot', type=float, nargs=3, default=[0., 0., 0.], help='Local object rotation vector, radians')
    p.add_argument('--render', action='store_true')
    p.add_argument('--pre-padding', type=int, default=180)
    p.add_argument('--post-padding', type=int, default=180)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    ds = lance.dataset(a.dataset, version=a.version)
    raw = ds.take([a.row], columns=['index','trajectory_metadata','timestamp','hands','objects']).to_pylist()[0]
    tr = trajectory_from_lance_row(raw, ds.version, row_index=a.row, hand_side='right', pre_padding=a.pre_padding, post_padding=a.post_padding)
    tr = resample_reference_trajectory(tr, reference_fps=100, control_fps=100)
    delta = np.asarray(a.delta_pos)
    active = tr.identity.object_index
    spos = tr.scene_object_initial_pos.copy(); spos[active] += delta
    sq = tr.scene_object_initial_quat_xyzw.copy()
    oq = (Rotation.from_quat(tr.object_quat_xyzw) * Rotation.from_rotvec(a.delta_rot)).as_quat()
    sq[active] = oq[0]
    tr = replace(tr, object_pos_raw=tr.object_pos_raw+delta, object_pos=tr.object_pos+delta,
                 object_quat_xyzw=oq, scene_object_initial_pos=spos, scene_object_initial_quat_xyzw=sq)
    env = MujocoManoEnvironment(tr, EnvironmentConfig(device='gpu', num_envs=1, residual_enabled=False,
         compatibility=replace(SOURCE_ALIGNED_COMPATIBILITY, movement_pre_padding=a.pre_padding),
         max_deviation_distance=1e6, reference_fps=100, control_fps=100, post_padding=a.post_padding, hand_side='right'))
    mj, vm = _compile_native_viewer_model(env)
    vd = mj.MjData(vm); rd = mj.MjData(vm)
    names = tr.scene_object_types
    objname = names[active]
    object_joints = [vm.joint(n+'_free').qposadr[0] for n in names]
    objbody = vm.body(objname).id
    print('IDENTITY',raw['index'], 'N',len(tr.q_ref), 'BODY_NAMES',[vm.body(i).name for i in range(vm.nbody)], flush=True)
    T = len(tr.q_ref)
    qpos=[]; qvel=[]; ctrl=[]; contacts=[]; positions=[]; rots=[]; refs=[]; body_pos=[]
    renderer = mj.Renderer(vm, height=480, width=640) if a.render else None
    cam=mj.MjvCamera(); cam.lookat[:]=tr.object_pos[0]+[0,0,.1]; cam.distance=.7; cam.azimuth=100; cam.elevation=-35
    opt=mj.MjvOption(); opt.geomgroup[COLLISION_GEOM_GROUP]=0
    lo=int(tr.movement_start_step); hi=int(tr.movement_end_step)
    snap=set([max(0,lo-35),lo,min(hi,lo+30),min(hi,lo+70),min(hi,lo+130),hi,min(T-2,hi+80)])
    panels=[]
    zeros=np.zeros((1,28))
    for call in range(T-1):
        _,_,done,_=env.step(zeros)
        state=env.host_data(0)
        _mirror_native_viewer_data(mj,vm,vd,state)
        ref=min(int(env.progress[0])-1,T-1)
        qpos.append(vd.qpos.copy()); qvel.append(vd.qvel.copy()); ctrl.append(vd.ctrl.copy())
        positions.append(vd.xpos[objbody].copy()); rots.append(vd.xquat[objbody].copy()); refs.append(ref); body_pos.append(vd.xpos.copy())
        cc=[]
        # host_data mirrors coordinates, not a valid native contact buffer.
        # mj_forward above recomputes geometric contacts at the recorded state.
        for c in vd.contact[:vd.ncon]:
            b1=int(vm.geom_bodyid[c.geom1]); b2=int(vm.geom_bodyid[c.geom2])
            if objbody in (b1,b2):
                cc.append({'bodies':[vm.body(b1).name,vm.body(b2).name], 'dist':float(c.dist),'pos':c.pos.tolist(),'kind':'native_recomputed_geometry'})
        contacts.append(cc)
        if call % 100 == 0 or bool(done[0]):
            print('STEP',call,'ref',ref,'object',positions[-1].round(4).tolist(),'reference',tr.object_pos[ref].round(4).tolist(),'contacts',[x['bodies'] for x in cc],flush=True)
        if renderer and ref in snap:
            from PIL import Image,ImageDraw
            rd.qpos[:]=vd.qpos
            rd.qpos[:28]=tr.q_ref[ref]
            for i,adr in enumerate(object_joints):
                src=int(tr.source_indices[ref])
                rd.qpos[adr:adr+3]=np.asarray(raw['objects'][i]['pos'][src])+[0,0,tr.object_z_shift]
                quat=Rotation.from_rotvec(raw['objects'][i]['rot_aa'][src]).as_quat()
                rd.qpos[adr+3:adr+7]=quat[[3,0,1,2]]
            adr=object_joints[active]
            rd.qpos[adr:adr+3]=tr.object_pos[ref]; rd.qpos[adr+3:adr+7]=tr.object_quat_xyzw[ref][[3,0,1,2]]
            mj.mj_forward(vm,rd)
            renderer.update_scene(rd,camera=cam,scene_option=opt); left=Image.fromarray(renderer.render())
            renderer.update_scene(vd,camera=cam,scene_option=opt); right=Image.fromarray(renderer.render())
            panel=Image.new('RGB',(1280,510),'white'); panel.paste(left,(0,30)); panel.paste(right,(640,30))
            draw=ImageDraw.Draw(panel); draw.text((12,8),f'Source object+hand | row {a.row} raw frame {tr.source_indices[ref]}',fill='black'); draw.text((652,8),'Free-body MJX replay | hand commands unchanged',fill='black')
            panel.save(a.output/f'frame_{ref:04d}.png'); panels.append(panel); snap.remove(ref)
        if bool(done[0]): break
    np.savez_compressed(a.output/'trace.npz', qpos=qpos,qvel=qvel,ctrl=ctrl,object_pos=positions,object_quat_wxyz=rots,reference_index=refs,source_indices=tr.source_indices,reference_object_pos=tr.object_pos,reference_object_quat_xyzw=tr.object_quat_xyzw,reference_hand=tr.q_ref,body_pos=body_pos,body_names=np.array([vm.body(i).name for i in range(vm.nbody)]),scene_initial_pos=tr.scene_object_initial_pos)
    (a.output/'contacts.json').write_text(json.dumps(contacts))
    pos=np.array(positions); refs=np.array(refs); err=np.linalg.norm(pos-tr.object_pos[refs],axis=1)
    info={'source_dataset':a.dataset,'version':a.version,'row':a.row,'index':raw['index'],'delta_pos':a.delta_pos,'delta_rot':a.delta_rot,'source_start':tr.identity.source_start,'movement_steps':[lo,hi],'object_z_shift':tr.object_z_shift,'initial_height_m':float(tr.object_pos[0,2]),'max_lift_m':float(pos[:,2].max()-tr.object_pos[0,2]),'last_lift_m':float(pos[-1,2]-tr.object_pos[0,2]),'position_rmse_m':float(np.sqrt(np.mean(err**2))),'final_error_m':float(err[-1]),'hand_commands_modified':False,'object_pose_forced_after_reset':False}
    (a.output/'summary.json').write_text(json.dumps(info,indent=2,ensure_ascii=False)); print(json.dumps(info,ensure_ascii=False),flush=True)
    if renderer:
        renderer.close()
        if panels:
            sheet=Image.new('RGB',(1280,510*len(panels)))
            for i,panel in enumerate(panels): sheet.paste(panel,(0,i*510))
            sheet.resize((960,int(sheet.height*.75))).save(a.output/'comparison.jpg')

if __name__=='__main__': main()
