import numpy as np
from tools.repair_action005_direct120 import local_contacts, bearing_topology


def test_first_bearing_contact_local_coordinates():
    contact=dict(finger='index',position=[1.,3.,3.],basis=np.eye(3).tolist(),object_sign=-1,local_wrench=[.3,0,0,0,0,0])
    contacts=local_contacts([contact],np.array([1.,2.,3.]),np.eye(3))
    topology=bearing_topology(contacts)
    assert topology['bearing_points_m']['index']==[0.,1.,0.]
    assert topology['bearing_normals']['index']==[-1.,0.,0.]
    assert topology['missing_fingers']==['thumb','middle','ring','pinky']
