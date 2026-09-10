"""Explicit, smooth data-level edits for a source-reference replay (no policy)."""
from dataclasses import replace
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from sim.manorl.contracts import JOINT_NAMES


def apply_replay_edits(base, edit):
    dp=np.asarray(edit.get('translation',[0,0,0]),dtype=float)
    dr=Rotation.from_rotvec(edit.get('rotation',[0,0,0]))
    if dp.shape != (3,) or not np.all(np.isfinite(dp)):
        raise ValueError('translation must contain three finite metre offsets')
    ai=base.identity.object_index
    sp=base.scene_object_initial_pos.copy(); sp[ai]+=dp
    oq=(Rotation.from_quat(base.object_quat_xyzw)*dr).as_quat()
    sq=base.scene_object_initial_quat_xyzw.copy(); sq[ai]=oq[0]
    qmap={side:q.copy() for side,q in base.q_ref_by_side.items()}
    offsets=edit.get('hand_offsets',{})
    targets=edit.get('finger_targets',{})
    grasp=edit.get('object_relative_grasp')
    wrist=edit.get('wrist_offset',[0,0,0,0,0,0])
    wrist=np.asarray(wrist,dtype=float)
    if wrist.shape != (6,) or not np.all(np.isfinite(wrist)):
        raise ValueError('wrist_offset must contain finite XYZ metres and Euler radians')
    if offsets or targets or grasp or np.any(wrist):
        lo=base.movement_start_step
        begin=lo+int(edit.get('close_start_offset',-30))
        end=lo+int(edit.get('close_end_offset',10))
        if end<=begin: raise ValueError('closure ramp must have positive duration')
        u=np.clip((np.arange(len(base.q_ref))-begin)/(end-begin),0.,1.)
        weight=u*u*(3.-2.*u)
        if edit.get('release_at_end',False):
            release_start=int(edit.get('release_start_step',base.movement_end_step-20))
            release_end=int(edit.get('release_end_step',base.movement_end_step+50))
            if release_end<=release_start: raise ValueError('release end must follow start')
            u=np.clip((np.arange(len(base.q_ref))-release_start)/(release_end-release_start),0.,1.)
            weight*=1.-u*u*(3.-2.*u)
        qmap['right'][:,:6]+=weight[:,None]*wrist[None,:]
        if grasp:
            object_rotation=Rotation.from_quat(oq)
            desired_pos=base.object_pos+dp+object_rotation.apply(np.asarray(grasp['translation']))
            desired_rot=object_rotation*Rotation.from_quat(grasp['rotation_xyzw'])
            desired_angles=np.unwrap(desired_rot.as_euler('XYZ'),axis=0)
            angle_delta=(desired_angles-qmap['right'][:,3:6]+np.pi)%(2*np.pi)-np.pi
            approach_begin=lo+int(edit.get('grasp_approach_start_offset',-130))
            approach_end=lo+int(edit.get('grasp_approach_end_offset',-35))
            if approach_end<=approach_begin: raise ValueError('grasp approach ramp must have positive duration')
            u=np.clip((np.arange(len(base.q_ref))-approach_begin)/(approach_end-approach_begin),0.,1.)
            blend=u*u*(3.-2.*u)
            if edit.get('release_at_end',False):
                wrist_release_start=int(edit.get('wrist_release_start_step',release_start))
                wrist_release_end=int(edit.get('wrist_release_end_step',release_end))
                if wrist_release_end<=wrist_release_start: raise ValueError('wrist release end must follow start')
                release_u=np.clip((np.arange(len(base.q_ref))-wrist_release_start)/(wrist_release_end-wrist_release_start),0.,1.)
                blend*=1.-release_u*release_u*(3.-2.*release_u)
            qmap['right'][:,:3]=(1-blend[:,None])*qmap['right'][:,:3]+blend[:,None]*desired_pos
            qmap['right'][:,3:6]+=blend[:,None]*angle_delta
        for joint,value in targets.items():
            idx=JOINT_NAMES.index(joint)
            if idx<6 or not np.isfinite(value): raise ValueError('finger_targets requires finite finger angles')
            qmap['right'][:,idx]=(1.-weight)*qmap['right'][:,idx]+weight*float(value)
        for joint,delta in offsets.items():
            idx=JOINT_NAMES.index(joint)
            if idx<6: raise ValueError('this edit contract is finger-only; wrist path must be preserved')
            if not np.isfinite(delta): raise ValueError('joint offsets must be finite radians')
            qmap['right'][:,idx]+=float(delta)*weight
    if 'acquisition' in edit:
        acquisition=edit['acquisition']
        start,pregrasp,arrive,closed=[int(acquisition[k]) for k in ('start_step','pregrasp_step','arrive_step','closed_step')]
        if not 0<=start<pregrasp<arrive<closed<len(base.q_ref):
            raise ValueError('acquisition requires ordered in-range start/pregrasp/arrive/closed steps')
        initial=qmap['right'][start].copy()
        final=qmap['right'][closed].copy()
        open_pose=final.copy()
        for joint,value in acquisition['open_offsets'].items():
            index=JOINT_NAMES.index(joint)
            if index<6 or not np.isfinite(value): raise ValueError('open offsets must name finite finger angles')
            open_pose[index]+=float(value)
        clearance=np.asarray(acquisition['clearance_object_local'],dtype=float)
        if clearance.shape!=(3,) or not np.all(np.isfinite(clearance)): raise ValueError('clearance must be a finite XYZ vector')
        far_pose=open_pose.copy()
        far_pose[:3]+=Rotation.from_quat(oq[base.movement_start_step]).apply(clearance)
        for i in range(start,closed):
            if i<pregrasp:
                a,b,begin,end=initial,far_pose,start,pregrasp
            elif i<arrive:
                a,b,begin,end=far_pose,open_pose,pregrasp,arrive
            else:
                a,b,begin,end=open_pose,qmap['right'][i].copy(),arrive,closed
            u=(i-begin)/(end-begin);blend=u*u*(3.-2.*u)
            delta=b-a;delta[3:6]=(delta[3:6]+np.pi)%(2*np.pi)-np.pi
            qmap['right'][i]=a+blend*delta
    for stage in edit.get('stages',[]):
        start=int(stage['start_step']);end=int(stage['end_step'])
        if end<=start: raise ValueError('stage ramp must have positive duration')
        u=np.clip((np.arange(len(base.q_ref))-start)/(end-start),0.,1.)
        blend=u*u*(3.-2.*u)
        if 'wrist_translation_world' in stage:
            displacement=np.asarray(stage['wrist_translation_world'],dtype=float)
            if displacement.shape!=(3,) or not np.all(np.isfinite(displacement)):
                raise ValueError('wrist translation must be a finite XYZ vector')
            qmap['right'][:,:3]+=blend[:,None]*displacement
        if 'wrist_rotation_world' in stage:
            rv=np.asarray(stage['wrist_rotation_world'],dtype=float)
            pivot_delta=np.asarray(stage['pivot_from_object'],dtype=float)
            if rv.shape!=(3,) or pivot_delta.shape!=(3,) or not np.all(np.isfinite(np.r_[rv,pivot_delta])):
                raise ValueError('wrist rotation and pivot must be finite XYZ vectors')
            correction=Rotation.from_rotvec(blend[:,None]*rv[None,:])
            pivot=base.object_pos+dp+pivot_delta
            qmap['right'][:,:3]=pivot+correction.apply(qmap['right'][:,:3]-pivot)
            old_angles=qmap['right'][:,3:6].copy()
            angles=(correction*Rotation.from_euler('XYZ',old_angles)).as_euler('XYZ')
            qmap['right'][:,3:6]=old_angles+(angles-old_angles+np.pi)%(2*np.pi)-np.pi
        for joint,value in stage.get('hand_offsets',{}).items():
            index=JOINT_NAMES.index(joint)
            if index<6 or not np.isfinite(value): raise ValueError('stage offsets must be finite finger angles')
            qmap['right'][:,index]+=blend*float(value)
    if 'wrist_delta_keyframes' in edit:
        keyframes=edit['wrist_delta_keyframes']
        steps=np.asarray([k['step'] for k in keyframes],dtype=float)
        values=np.asarray([k['delta'] for k in keyframes],dtype=float)
        if len(steps)<2 or values.shape!=(len(steps),6) or not np.all(np.isfinite(values)) or not np.all(np.isfinite(steps)) or np.any(np.diff(steps)<=0):
            raise ValueError('wrist_delta_keyframes requires increasing steps and finite 6D deltas')
        time=np.arange(len(base.q_ref))
        for j in range(6):
            qmap['right'][:,j]+=np.interp(time,steps,values[:,j],left=0.,right=0.)
    result=replace(base,object_pos_raw=base.object_pos_raw+dp,object_pos=base.object_pos+dp,
                   object_quat_xyzw=oq,scene_object_initial_pos=sp,scene_object_initial_quat_xyzw=sq,
                   q_ref=qmap['right'],q_ref_by_side=qmap)
    scale=float(edit.get('time_scale',1.))
    if not np.isfinite(scale) or scale<1: raise ValueError('time_scale must be finite and at least one')
    if scale!=1.:
        old=np.arange(len(result.q_ref),dtype=float)
        query=np.linspace(0.,old[-1],int(np.ceil(old[-1]*scale))+1)
        def interp(values):
            return np.column_stack([np.interp(query,old,values[:,j]) for j in range(values.shape[1])])
        qs={s:interp(q) for s,q in result.q_ref_by_side.items()}
        quats=Slerp(old,Rotation.from_quat(result.object_quat_xyzw))(query).as_quat()
        result=replace(result,q_ref=qs['right'],q_ref_by_side=qs,object_pos=interp(result.object_pos),object_pos_raw=interp(result.object_pos_raw),object_quat_xyzw=quats,
            source_indices=np.floor(np.interp(query,old,result.source_indices)+1e-10).astype(np.int64),
            timestamps=result.timestamps[0]+np.arange(len(query))/float(result.control_fps),
            movement_start_step=int(round(result.movement_start_step*scale)),movement_end_step=int(round(result.movement_end_step*scale)))
    return result
