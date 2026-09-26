from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from sim.manorl.environment import MaterializedContactBuffers, MaterializedState
from sim.manorl.lance_v2 import corrected_contact_frames
from sim.manorl.synthesis_contacts import corrected_contact_frames_vectorized


def fixture(seed):
    rng=np.random.default_rng(seed);batch=5;bodies=20;capacity=220;count=190
    xyz=rng.normal(size=(batch,bodies,3))
    xyzw=Rotation.random(batch*bodies,random_state=rng).as_quat().reshape(batch,bodies,4)
    state=MaterializedState(qpos=np.zeros((batch,35)),qvel=np.zeros((batch,34)),xpos=xyz,
        xquat=xyzw[...,[3,0,1,2]],keypoints=xyz[:,:16],keypoint_quats=xyzw[:,:16],fingertips=xyz[:,:5])
    geom=np.column_stack([rng.integers(0,16,size=capacity),rng.integers(16,19,size=capacity)])
    geom[::2]=geom[::2,::-1];geom[::9]=[18,19]  # unrelated floor/object contacts
    buffers=MaterializedContactBuffers(count=count,capacity=capacity,geom=geom,
        position=rng.normal(size=(capacity,3)),world=rng.integers(0,batch,size=capacity),
        dimension=np.full(capacity,3),addresses=rng.integers(0,128,size=(capacity,4)),
        nefc=np.full(batch,128),friction=np.zeros((capacity,5)),
        frame=Rotation.random(capacity,random_state=rng).as_matrix(),constraint_force=rng.uniform(0,3,size=(batch,128)),
        raw_metadata={},host_metadata={})
    return dict(buffers=buffers,state=state,model=SimpleNamespace(geom_bodyid=np.arange(bodies)),
                keypoint_geom_ids=tuple(range(16)),object_geom_ids=(16,17),object_body_id=16,wrist_body_id=0,object_name='cube2')


@pytest.mark.parametrize('seed',[0,1,42])
def test_vectorized_contacts_preserve_scalar_values_frames_signs_and_order(seed):
    kwargs=fixture(seed);reference=corrected_contact_frames(**kwargs);actual=corrected_contact_frames_vectorized(**kwargs)
    assert len(reference)==len(actual)
    for before,after in zip(reference,actual):
        assert len(before)==len(after)
        for b,a in zip(before,after):
            for name in ['hand_name','joint_name','object_name']:assert b[name]==a[name]
            for name in ['total_force_world','total_force_wrist','total_force_joint','total_force_object']:
                np.testing.assert_allclose(a[name],b[name],rtol=2e-13,atol=2e-13)
            assert len(b['contact_pairs'])==len(a['contact_pairs'])
            for bp,ap in zip(b['contact_pairs'],a['contact_pairs']):
                for name in bp:np.testing.assert_allclose(ap[name],bp[name],rtol=2e-13,atol=2e-13)


def test_vectorized_contacts_reject_invalid_solved_addresses():
    kwargs=fixture(5);kwargs['buffers'].geom[1]=[0,16];kwargs['buffers'].addresses[1,0]=-1
    with pytest.raises(RuntimeError,match='addresses'):corrected_contact_frames_vectorized(**kwargs)


def test_vectorized_contacts_empty_frames_preserve_world_count():
    from dataclasses import replace
    kwargs=fixture(5);kwargs['buffers']=replace(kwargs['buffers'],count=0)
    assert corrected_contact_frames_vectorized(**kwargs)==[[],[],[],[],[]]
