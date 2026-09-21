import json
import numpy as np
import pytest
from sim.manorl.u1_campaign import plan, perturb, child_uuid, Ledger, digest, verify_signed, file_sha

def test_balanced_unique_plan():
    p=plan('parent'); verify_signed(p)
    assert len(p['slots'])==160 and p==plan('parent')
    assert len({tuple(c['delta']) for s in p['slots'] for c in s['candidates']})==2560
    for s in p['slots']:
        assert len(s['candidates'])==16
        for c in s['candidates']:
            assert np.isclose(np.linalg.norm(c['delta'][:3])*1000,s['radius_mm'])
            assert np.array_equal(np.sign(c['delta'][:3]),s['octant'])
            assert np.count_nonzero(c['delta'][3:])==1

@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_taper_suffix_dtype(dtype):
    base=np.arange(560,dtype=dtype).reshape(20,28)/100
    q=np.arange(42,dtype=float); delta=np.array([.01,-.01,.01,.005,0,0])
    out,initial=perturb(base,q,delta,10)
    assert out.dtype==base.dtype and out[9:].tobytes()==base[9:].tobytes()
    assert out[:,6:].tobytes()==base[:,6:].tobytes()
    np.testing.assert_allclose(initial[:6],q[:6]+delta)
    assert initial[6:].tobytes()==q[6:].tobytes()
    np.testing.assert_allclose(out[0,:6],base[0,:6]+delta,atol=1e-8)
    with pytest.raises(ValueError): perturb(base,q,delta,2)

def test_hash_contract_and_uuid():
    p=plan('r'); verify_signed(p); p['seed']+=1
    with pytest.raises(ValueError): verify_signed(p)
    c={'ordinal':0,'delta':[1,2,3,4,5,6]}
    assert child_uuid('r','003','p',0,c)==child_uuid('r','003','p',0,c)
    assert child_uuid('r','003','p',0,c)!=child_uuid('r','005','p',0,c)

def test_resume_rejection_and_artifact_binding(tmp_path):
    path=tmp_path/'ledger'; ledger=Ledger(path,'identity')
    ledger.append(uuid='a',slot=0,status='started')
    resumed=Ledger(path,'identity'); assert resumed.rows[-1]['status']=='rejected'
    with pytest.raises(ValueError): Ledger(path,'different')
    evidence=tmp_path/'trace'; evidence.write_bytes(b'real')
    resumed.append(uuid='b',slot=0,status='selected',artifacts={str(evidence):file_sha(evidence)})
    assert len(Ledger(path,'identity').selected())==1
    evidence.write_bytes(b'changed')
    with pytest.raises(ValueError): Ledger(path,'identity')

def test_selected_slot_cannot_duplicate(tmp_path):
    path=tmp_path/'ledger'; ledger=Ledger(path,'x')
    ledger.append(uuid='a',slot=0,status='selected',artifacts={})
    ledger.append(uuid='b',slot=0,status='selected',artifacts={})
    with pytest.raises(ValueError): Ledger(path,'x')

def test_registry_rejects_changed_status(tmp_path):
    pytest.importorskip('mujoco')
    from tools.u1_campaign_registry import load_parent
    payload={'setting_sha256':'incorrect','parents':{}}
    payload['digest']=digest(payload)
    path=tmp_path/'registry.json'; path.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match='setting'): load_parent(path,'003')
    payload['parents']['003']={'final_registry_status':'accepted:false'}
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match='digest'): load_parent(path,'003')
