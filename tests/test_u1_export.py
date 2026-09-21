import json
from pathlib import Path
import numpy as np
import pytest
from sim.manorl.u1_campaign import plan, child_uuid
from sim.manorl.u1_export import select_records, checked_hashes, native_frame, world_force, make_row, schema


def ledger():
    r={'digest':'registry'}; p=plan(r['digest']); rows=[]
    for slot in p['slots']:
        c=slot['candidates'][0]; uid=child_uuid(r['digest'],'003',p['digest'],slot['slot'],c)
        rows.extend([dict(status='started',uuid=uid,slot=slot['slot'],candidate=c),dict(status='selected',uuid=uid,slot=slot['slot'],artifacts={})])
    return r,p,rows


def test_exact_quota_and_uuid():
    r,p,rows=ledger(); assert len(select_records(rows,r,p,'003'))==160
    for bad in [rows[:-2],rows+[rows[-1]],rows+[dict(status='exhausted',slot=0)]]:
        with pytest.raises(ValueError): select_records(bad,r,p,'003')
    rows[-1]['uuid']='wrong'
    with pytest.raises(ValueError): select_records(rows,r,p,'003')


def test_hash_fail_closed(tmp_path):
    f=tmp_path/'x';f.write_text('a')
    with pytest.raises(ValueError): checked_hashes({str(f):'0'*64})


def test_native_transform_and_sign():
    b=np.array([[0,1,0],[-1,0,0],[0,0,1]])
    assert np.array_equal(world_force([2,3,4,0,0,0],b),[-3,2,4])
    assert np.array_equal(world_force([2,3,4,0,0,0],b,-1),[3,-2,-4])
    raw=dict(frame=0,geom_pairs=[[0,1]],force_contact_frame=[[2,0,0,0,0,0]],position=[[0,0,0]],contact_frame=[b.tolist()],friction=[[1]*5],dimension=[3],efc_address=[[0,1,2,3]],worldid=[0])
    native_frame(raw,0,2)
    for key in ('contact_frame','force_contact_frame'):
        bad=dict(raw);bad[key]=[]
        with pytest.raises(ValueError): native_frame(bad,0,2)
    with pytest.raises(ValueError): native_frame(dict(raw,worldid=[1]),0,2)


def test_real_smoke_row(tmp_path):
    root=Path('outputs/u1_contact_capture_smoke/003/zero')
    registry=Path('outputs/u1_5x160_registry_v3/registry.json')
    if not root.exists() or not registry.exists(): pytest.skip('local captured smoke evidence unavailable')
    # Exercise real capture without treating a pilot as selected production.
    import shutil
    import lance
    import pyarrow as pa
    from sim.manorl.u1_export import read_json
    shutil.copytree(root,tmp_path/'second')
    r=read_json(registry); p=plan(r['digest']); c=dict(ordinal=0,delta=[0.]*6)
    record=dict(action='003',slot=0,candidate=c,uuid=child_uuid(r['digest'],'003',p['digest'],0,c),folder=str(tmp_path))
    row=make_row(registry,record,p)
    assert len(row['contact'])==641
    assert len(row['command_reference_index'])==640
    assert row['hands'][1]['hand_name'] is None
    assert np.array_equal(row['physical']['qpos'],np.load(root/'trace.npz')['qpos'])
    table=pa.Table.from_pylist([row],schema=schema())
    lance.write_dataset(table,str(tmp_path/'smoke.lance'))
    assert lance.dataset(str(tmp_path/'smoke.lance')).to_table().to_pylist()==table.to_pylist()


def test_contact_filter_sign_coordinates():
    import mujoco as mj
    from sim.manorl.u1_export import compact_contacts
    from sim.manorl.contracts import KEYPOINT_NAMES
    from types import SimpleNamespace
    names=list(KEYPOINT_NAMES)+['object']
    class Model:
        ngeom=len(names)
        geom_bodyid=np.arange(len(names))+1
        body_parentid=np.zeros(len(names)+1,dtype=int)
        def geom(self,name): return SimpleNamespace(id=names.index(name.removesuffix('_collision')))
        def body(self,name): return SimpleNamespace(id=names.index(name)+1)
    m=Model(); d=SimpleNamespace(xpos=np.zeros((len(names)+1,3)),xmat=np.tile(np.eye(3).reshape(9),(len(names)+1,1)))
    d.xpos[m.body('object').id]=[1,0,0]
    raw=dict(geom_pairs=np.array([[0,16],[16,0],[0,1]]),force_contact_frame=np.array([[2,0,0,0,0,0],[3,0,0,0,0,0],[99,0,0,0,0,0]]),contact_frame=np.tile(np.eye(3),(3,1,1)),position=np.zeros((3,3)))
    out=compact_contacts(raw,m,d,['object'])
    assert len(out)==1 and len(out[0]['contact_pairs'])==2
    assert out[0]['total_force_world']==[-1,0,0]
    assert out[0]['contact_pairs'][0]['pos_object']==[-1,0,0]
