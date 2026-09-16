import unittest
from types import SimpleNamespace
import numpy as np
from sim.manorl.local_contact_repair import ReplayInput
from sim.manorl.start_augmentation import augment, unsupported_intervals


class StartAugmentationTest(unittest.TestCase):
    def setUp(self):
        self.parent = np.arange(100*28, dtype=float).reshape(100,28)/10000
        self.inp = ReplayInput('test',120,{'desired':self.parent},
            {'qpos':np.arange(42,dtype=float), 'qvel':np.arange(40,dtype=float),
             'donor_preload':np.arange(22,dtype=float)}, {}, {}, {})

    def test_real_initialization_and_invariants(self):
        shifted, target, env, info = augment(self.inp,self.parent,[.1,-.04,0],48)
        np.testing.assert_array_equal(shifted.initial['qpos'][:3], self.inp.initial['qpos'][:3]+[.1,-.04,0])
        np.testing.assert_array_equal(shifted.initial['qpos'][3:],self.inp.initial['qpos'][3:])
        for key in ('qvel','donor_preload'):
            np.testing.assert_array_equal(shifted.initial[key],self.inp.initial[key])
        np.testing.assert_array_equal(target[:,3:],self.parent[:,3:])
        self.assertEqual(target[47:].tobytes(),self.parent[47:].tobytes())
        np.testing.assert_array_equal(np.gradient(target,1/120,axis=0)[48],np.gradient(self.parent,1/120,axis=0)[48])
        self.assertTrue(np.all(np.diff(env)<=1e-12))
        self.assertEqual(env[0],1)
        # Analytic quintic endpoint first/second derivatives are zero.
        for u in (0.,1.):
            self.assertEqual(-30*u*u+60*u**3-30*u**4,0)
            self.assertEqual(-60*u+180*u*u-120*u**3,0)
        self.assertAlmostEqual(info['taper_duration_s'],47/120)
        self.assertAlmostEqual(info['peak_residual_speed_m_s'],1.875*np.linalg.norm([.1,-.04,0])/(47/120))

    def test_zero_is_byte_exact(self):
        shifted, target, _, _ = augment(self.inp,self.parent,[0,0,0],48)
        self.assertEqual(target.tobytes(),self.parent.tobytes())
        self.assertEqual(shifted.initial['qpos'].tobytes(),self.inp.initial['qpos'].tobytes())

    def test_invalid_inputs(self):
        for C,delta in ((1,[0,0,0]),(99,[0,0,0]),(48,[np.nan,0,0]),(48,[0,0])):
            with self.assertRaises(ValueError): augment(self.inp,self.parent,delta,C)

    def test_unsupported_threshold(self):
        def check(n):
            return unsupported_intervals(dict(height_m=np.full(n,.021), has_support=np.zeros(n,bool),finger_ray_count=np.zeros(n,int)),120)
        self.assertFalse(check(2))
        self.assertEqual(check(3),[dict(start_frame=0,end_frame_exclusive=3,duration_s=.025)])

    def test_failure_serialization(self):
        import tempfile,json
        from pathlib import Path
        from unittest.mock import patch
        from tools.pilot_start_augmentation import one
        with tempfile.TemporaryDirectory() as temp:
            a=SimpleNamespace(output=Path(temp),row='A_row035',variant='zero',bundle=Path(temp))
            with patch('tools.pilot_start_augmentation.reconstruct',side_effect=ValueError('bad source')):
                with self.assertRaises(ValueError): one(a)
            failure=json.loads((Path(temp)/'A_row035/zero/failure.json').read_text())
            self.assertEqual(failure['message'],'bad source')


if __name__=='__main__': unittest.main()
