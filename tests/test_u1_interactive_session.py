import numpy as np
import pytest
from tools.u1_interactive_session import parse_command, window_offset, Session


def test_commands():
    assert parse_command({'action':'step'},100)['frames']==1
    for m in ({'action':'force'},{'action':'step','frames':True},{'action':'goto','frame':100},
              {'action':'offset','joint':28,'value':0,'start':1,'end':90}):
        with pytest.raises(ValueError):parse_command(m,100)


def test_ramp():
    w=window_offset(100,10,90,20)
    assert np.all(w[:11]==0) and np.all(w[90:]==0)
    assert w[20]==.5 and w[30]==1 and w[70]==1 and w[80]==.5
    np.testing.assert_array_equal(w[10:91],w[10:91][::-1])


def test_complete_workspace_restore(monkeypatch):
    import tools.u1_interactive_session as module
    class Buffer:
        def __init__(self,v):self.v=np.asarray(v).copy()
        def numpy(self):return self.v
        def assign(self,v):self.v=v.copy()
    from types import SimpleNamespace
    s=Session.__new__(Session)
    s.wp=SimpleNamespace(synchronize=lambda:None)
    s.data={k:Buffer([i]) for i,k in enumerate(('qpos','qvel','ctrl','qacc_warmstart','time','contact.cache'))}
    monkeypatch.setattr(module,'device_arrays',lambda data:data.items())
    s.frame=1030;s.target=np.ones((1100,28));s.edits=[dict(joint=6,value=.1)]
    cp=s.checkpoint()
    for b in s.data.values():b.assign(np.array([-1]))
    s.frame=1090;s.target[:]=3;s.edits[0]['value']=9
    s.restore(cp)
    assert s.frame==1030 and s.edits==[dict(joint=6,value=.1)]
    np.testing.assert_array_equal(s.target,np.ones((1100,28)))
    for k,b in s.data.items():np.testing.assert_array_equal(b.numpy(),cp['buffers'][k])
