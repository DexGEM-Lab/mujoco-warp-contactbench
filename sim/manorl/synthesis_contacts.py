"""Vectorized replay/visual contact coordinates from actual solved Warp buffers."""
from collections import defaultdict

import numpy as np
from scipy.spatial.transform import Rotation

from sim.manorl.contracts import KEYPOINT_NAMES


def corrected_contact_frames_vectorized(*, buffers, state, model, keypoint_geom_ids,
        object_geom_ids, object_body_id, wrist_body_id, object_name, hand_name='right'):
    """Same contract/order as lance_v2.corrected_contact_frames, without per-contact
    scipy constructors and repeated three-vector validation/transform calls.
    No native MuJoCo forward or contact-force reconstruction is performed.
    """
    batch=len(state.qpos)
    if len(keypoint_geom_ids)!=len(KEYPOINT_NAMES):
        raise ValueError('v2 contact export requires one geom per MANO keypoint')
    object_ids=set(map(int,object_geom_ids))
    if not object_ids or object_ids & set(keypoint_geom_ids):
        raise ValueError('v2 contact export requires disjoint hand/object geoms')
    if buffers.position.shape!=(buffers.capacity,3):
        raise ValueError('v2 contact export requires raw MJX-Warp contact positions')
    def rotations(wxyz):
        q=np.asarray(wxyz,dtype=np.float64)
        if q.shape[-1:]!=(4,) or not np.isfinite(q).all():
            raise ValueError('body quaternion must contain finite WXYZ values')
        return Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix()
    wrist_rot=rotations(state.xquat[:,wrist_body_id])
    object_rot=rotations(state.xquat[:,object_body_id])
    geom=np.asarray(buffers.geom[:buffers.count],dtype=np.int64)
    ngeom=len(model.geom_bodyid)
    lookup=np.full(ngeom,-1,dtype=np.int64)
    for i,g in enumerate(keypoint_geom_ids):lookup[int(g)]=i
    objects=np.zeros(ngeom,dtype=bool);objects[list(object_ids)]=True
    valid=((geom>=0)&(geom<ngeom)).all(axis=1)
    bounded=np.clip(geom,0,ngeom-1)
    first_hand=lookup[bounded[:,0]]>=0;second_hand=lookup[bounded[:,1]]>=0
    forward=valid & first_hand & objects[bounded[:,1]] & ~second_hand
    reverse=valid & second_hand & objects[bounded[:,0]] & ~first_hand
    selected=np.flatnonzero(forward|reverse)
    if not len(selected):return [[] for _ in range(batch)]
    world=np.asarray(buffers.world[selected],dtype=np.int64)
    if ((world<0)|(world>=batch)).any():raise RuntimeError('contact references invalid world')
    if (np.asarray(buffers.dimension[selected])!=3).any():
        raise RuntimeError('v2 contact export requires condim=3 solved contacts')
    address=np.asarray(buffers.addresses[selected],dtype=np.int64)
    if ((address<0)|(address>=np.asarray(buffers.nefc)[world,None])).any():
        raise RuntimeError('contact has invalid solved-force addresses')
    hand_geom=np.where(forward[selected],geom[selected,0],geom[selected,1])
    keypoint=lookup[hand_geom];body=np.asarray(model.geom_bodyid)[hand_geom]
    joint_rot=rotations(state.xquat[world,body])
    magnitude=np.asarray(buffers.constraint_force[world[:,None],address],dtype=np.float64).sum(axis=1)
    sign=np.where(forward[selected],1.,-1.)
    normal=sign[:,None]*magnitude[:,None]*np.asarray(buffers.frame[selected,0],dtype=np.float64)
    pos=np.asarray(buffers.position[selected],dtype=np.float64)
    fields={'force_normal':normal,'pos_world':pos,
        'pos_wrist':np.einsum('nji,nj->ni',wrist_rot[world],pos-state.xpos[world,wrist_body_id]),
        'pos_joint':np.einsum('nji,nj->ni',joint_rot,pos-state.xpos[world,body]),
        'pos_object':np.einsum('nji,nj->ni',object_rot[world],pos-state.xpos[world,object_body_id])}
    if not all(np.isfinite(v).all() for v in fields.values()):
        raise ValueError('v2 contact export requires finite 3-vectors')
    lists={name:value.tolist() for name,value in fields.items()}
    grouped=[defaultdict(list) for _ in range(batch)]
    for i,(w,k) in enumerate(zip(world,keypoint)):
        grouped[int(w)][int(k)].append(i)
    output=[]
    for w,groups in enumerate(grouped):
        entries=[]
        for k,indices in sorted(groups.items()):
            force=normal[indices].sum(axis=0)
            if not np.isfinite(force).all():raise ValueError('non-finite grouped force')
            entries.append({'hand_name':hand_name,'joint_name':KEYPOINT_NAMES[k],'object_name':object_name,
                'total_force_world':force.tolist(),'total_force_wrist':(wrist_rot[w].T@force).tolist(),
                'total_force_joint':(joint_rot[indices[0]].T@force).tolist(),
                'total_force_object':(object_rot[w].T@force).tolist(),
                'contact_pairs':[{name:value[i] for name,value in lists.items()} for i in indices]})
        output.append(entries)
    return output
