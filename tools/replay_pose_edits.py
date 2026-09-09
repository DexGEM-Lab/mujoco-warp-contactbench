"""Explicit, smooth data-level edits for a source-reference replay (no policy)."""
from dataclasses import replace
import numpy as np
from scipy.spatial.transform import Rotation
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
    wrist=edit.get('wrist_offset',[0,0,0,0,0,0])
    wrist=np.asarray(wrist,dtype=float)
    if wrist.shape != (6,) or not np.all(np.isfinite(wrist)):
        raise ValueError('wrist_offset must contain finite XYZ metres and Euler radians')
    if offsets or targets or np.any(wrist):
        lo=base.movement_start_step
        begin=lo+int(edit.get('close_start_offset',-30))
        end=lo+int(edit.get('close_end_offset',10))
        if end<=begin: raise ValueError('closure ramp must have positive duration')
        u=np.clip((np.arange(len(base.q_ref))-begin)/(end-begin),0.,1.)
        weight=u*u*(3.-2.*u)
        if edit.get('release_at_end',False):
            u=np.clip((np.arange(len(base.q_ref))-(base.movement_end_step-20))/70.,0.,1.)
            weight*=1.-u*u*(3.-2.*u)
        qmap['right'][:,:6]+=weight[:,None]*wrist[None,:]
        for joint,value in targets.items():
            idx=JOINT_NAMES.index(joint)
            if idx<6 or not np.isfinite(value): raise ValueError('finger_targets requires finite finger angles')
            qmap['right'][:,idx]=(1.-weight)*qmap['right'][:,idx]+weight*float(value)
        for joint,delta in offsets.items():
            idx=JOINT_NAMES.index(joint)
            if idx<6: raise ValueError('this edit contract is finger-only; wrist path must be preserved')
            if not np.isfinite(delta): raise ValueError('joint offsets must be finite radians')
            qmap['right'][:,idx]+=float(delta)*weight
    return replace(base,object_pos_raw=base.object_pos_raw+dp,object_pos=base.object_pos+dp,
                   object_quat_xyzw=oq,scene_object_initial_pos=sp,scene_object_initial_quat_xyzw=sq,
                   q_ref=qmap['right'],q_ref_by_side=qmap)
