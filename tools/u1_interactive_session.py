"""Native U1 target editing. Diagnostic branches never become accepted replays."""
from copy import deepcopy
from pathlib import Path
import json
import numpy as np


def integer(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'expected integer {low}..{high}')
    return value


def parse_command(message, frames):
    if not isinstance(message, dict):
        raise ValueError('command must be an object')
    m = deepcopy(message)
    action = m.get('action')
    if action not in ('play', 'pause', 'step', 'save', 'restore', 'reset', 'goto', 'offset', 'export'):
        raise ValueError('unknown action')
    if action == 'step':
        m['frames'] = integer(m.get('frames', 1), 1, frames)
    if action == 'goto':
        m['frame'] = integer(m['frame'], 0, frames-1)
    if action == 'offset':
        m['joint'] = integer(m['joint'], 0, 27)
        m['value'] = float(m['value'])
        if not np.isfinite(m['value']):
            raise ValueError('offset must be finite, metres/radians')
        m['start'] = integer(m['start'], 0, frames-2)
        m['end'] = integer(m['end'], m['start']+1, frames-1)
        m['ramp'] = integer(m.get('ramp', 12), 1, (m['end']-m['start'])//2)
    return m


def window_offset(n, start, end, ramp):
    """Zero at both inclusive endpoints; smooth ramp in/out, constant interior."""
    f = np.arange(n)
    u = np.clip(np.minimum((f-start)/ramp, (end-f)/ramp), 0, 1)
    return u*u*(3-2*u)


def device_arrays(obj, prefix=''):
    """Walk native Warp Data including nested Contact/Constraint buffers."""
    import warp as wp
    for key, value in vars(obj).items():
        path = prefix + key
        if isinstance(value, wp.array):
            yield path, value
        elif hasattr(value, '__dict__') and value.__class__.__module__.startswith('mujoco'):
            yield from device_arrays(value, path+'.')


class Session:
    def __init__(self, inp, model, target, teacher, provenance=None):
        import mujoco as mj
        import warp as wp
        from mujoco.mjx.third_party import mujoco_warp as mw
        from tools.u1_placement_workspace import assert_u1, DIRECT_WARP_CAPACITY
        from sim.manorl.mjx_sim import command_target
        self.contract = assert_u1(model, inp)
        self.mj, self.wp, self.mw = mj, wp, mw
        self.model, self.inp = model, inp
        self.base = np.asarray(target, dtype=float).copy()
        if self.base.shape != (inp.frames, 28) or not np.isfinite(self.base).all():
            raise ValueError('target must be finite [frames,28]')
        self.target = self.base.copy()
        self.edits = []
        self.provenance = deepcopy(provenance or {})
        self.teacher = teacher
        if np.shape(teacher['qpos']) != (inp.frames, model.nq) or not np.isfinite(teacher['qpos']).all():
            raise ValueError('teacher must be finite [frames,nq]')
        self.frame = 0
        self.names = inp.arrays['scene_object_names'].tolist()
        self.active = inp.metrics['source_metadata']['active_object']
        self.object_body = model.body(self.active).id
        # Wrist root of right hand, identified from the first actuated joint.
        self.hand_body = int(model.jnt_bodyid[5])
        self.cpu = mj.MjData(model)
        self.cpu.qpos[:] = inp.initial['qpos']
        self.cpu.qvel[:] = inp.initial['qvel']
        self.cpu.ctrl[:] = command_target(self.target[0], self.cpu.qpos[:28], *model.jnt_range[:28].T)
        mj.mj_forward(model, self.cpu)
        wp.init(); wp.set_device('cuda:0')
        self.wm = mw.put_model(model)
        self.data = mw.put_data(model, self.cpu, nworld=1, **DIRECT_WARP_CAPACITY)
        initial = self.checkpoint()
        self._step4()
        with wp.ScopedCapture() as capture:
            self._step4()
        self.graph = capture.graph
        self.restore(initial)
        self.initial = self.checkpoint()

    def _step4(self):
        for _ in range(4):
            self.mw.step(self.wm, self.data)

    def checkpoint(self):
        self.wp.synchronize()
        return dict(buffers={k:v.numpy().copy() for k,v in device_arrays(self.data)},
                    frame=self.frame, target=self.target.copy(), edits=deepcopy(self.edits))

    def restore(self, snapshot):
        buffers = dict(device_arrays(self.data))
        if buffers.keys() != snapshot['buffers'].keys():
            raise ValueError('checkpoint buffer schema mismatch')
        for key, value in snapshot['buffers'].items():
            buffers[key].assign(value)
        self.frame = snapshot['frame']
        self.target = snapshot['target'].copy()
        self.edits = deepcopy(snapshot['edits'])
        self.wp.synchronize()

    def save_checkpoint(self, folder):
        folder = Path(folder); folder.mkdir(parents=True, exist_ok=False)
        snapshot = self.checkpoint()
        np.savez_compressed(folder/'state.npz', **snapshot['buffers'], target=snapshot['target'])
        (folder/'workspace.json').write_text(json.dumps(dict(frame=self.frame, edits=self.edits,
            buffers=list(snapshot['buffers']), source=self.provenance, contract=self.contract, diagnostic_only=True), indent=2))
        return snapshot

    def offset(self, m):
        m = parse_command(m, len(self.target))
        if m['start'] < self.frame:
            raise ValueError('edit begins before live cursor; goto earlier checkpoint first')
        proposed = self.target.copy()
        proposed[:,m['joint']] += m['value'] * window_offset(len(proposed), m['start'],m['end'],m['ramp'])
        lo, hi = self.model.jnt_range[m['joint']]
        # Raw references may already exceed limits outside the edit window. Those
        # values are resolved by command_target, and must not reject an unrelated
        # local edit. Reject newly introduced/worsened violations anywhere.
        old_values=self.target[:,m['joint']]
        new_values=proposed[:,m['joint']]
        old_violation=np.maximum(np.maximum(lo-old_values,old_values-hi),0)
        new_violation=np.maximum(np.maximum(lo-new_values,new_values-hi),0)
        if np.any(new_violation > old_violation + 1e-12):
            raise ValueError('edit introduces or worsens joint-limit violations; reduce offset')
        self.target = proposed
        self.edits.append(m)

    def step(self):
        from sim.manorl.mjx_sim import command_target
        if self.frame >= len(self.target)-1:
            return False
        self.frame += 1
        control = command_target(self.target[self.frame], self.data.qpos.numpy()[0,:28], *self.model.jnt_range[:28].T)
        self.data.ctrl.assign(control[None].astype(np.float32))
        self.wp.capture_launch(self.graph)
        return True

    def reset(self, clear=False):
        target, edits = self.target.copy(), deepcopy(self.edits)
        self.restore(self.initial)
        if not clear:
            self.target, self.edits = target, edits

    def state(self):
        mj, d, m = self.mj, self.cpu, self.model
        d.qpos[:] = self.data.qpos.numpy()[0]
        d.qvel[:] = self.data.qvel.numpy()[0]
        d.ctrl[:] = self.data.ctrl.numpy()[0]
        mj.mj_forward(m,d)
        contacts = []
        for c in d.contact[:d.ncon]:
            b1,b2 = int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2])
            if self.object_body not in (b1,b2):
                continue
            other = m.body(b2 if b1 == self.object_body else b1).name
            contacts.append(dict(link=other, position=c.pos.tolist(), distance=float(c.dist)))
        links = sorted({c['link'] for c in contacts if c['link'] not in self.names+['world']})
        hR = d.xmat[self.hand_body].reshape(3,3)
        oR = d.xmat[self.object_body].reshape(3,3)
        relative = dict(position=(hR.T@(d.xpos[self.object_body]-d.xpos[self.hand_body])).tolist(),
                        rotation=(hR.T@oR).tolist())
        td = mj.MjData(m); td.qpos[:] = self.teacher['qpos'][self.frame]; mj.mj_forward(m,td)
        teacher_contacts=[]
        for c in td.contact[:td.ncon]:
            b1,b2 = int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2])
            if self.object_body in (b1,b2):
                teacher_contacts.append(dict(link=m.body(b2 if b1==self.object_body else b1).name,
                                             position=c.pos.tolist(), distance=float(c.dist)))
        tR=td.xmat[self.hand_body].reshape(3,3)
        return dict(teacher_contacts=teacher_contacts,
            teacher_hand_links=sorted({c['link'] for c in teacher_contacts if c['link'] not in self.names+['world']}),
            frame=self.frame, target_cursor=self.frame, physics_seconds=float(self.data.time.numpy()[0]),
            qpos=d.qpos.tolist(), current_28d=d.qpos[:28].tolist(), desired_28d=self.target[self.frame].tolist(),
            applied_ctrl=d.ctrl.tolist(), contacts=contacts, hand_links=links, contact_count=len(contacts),
            hand_contact_count=sum(c['link'] in links for c in contacts), ray_count=len({x.split('_')[0] for x in links}),
            contact_boundary='CPU geometry witness at live GPU qpos; ray_count = distinct contacting hand prefixes',
            object_position=d.xpos[self.object_body].tolist(), object_quaternion_wxyz=d.xquat[self.object_body].tolist(),
            object_tilt_deg=float(np.rad2deg(np.arccos(np.clip(oR[2,2],-1,1)))), hand_object_transform=relative,
            teacher_hand_object_transform=dict(position=(tR.T@(td.xpos[self.object_body]-td.xpos[self.hand_body])).tolist(),
                rotation=(tR.T@td.xmat[self.object_body].reshape(3,3)).tolist()), edits=self.edits)

    def export(self, folder):
        folder=Path(folder); folder.mkdir(parents=True,exist_ok=False)
        np.save(folder/'target.npy',self.target)
        (folder/'provenance.json').write_text(json.dumps(dict(contract=self.contract, source=self.provenance, edits=self.edits,
            accepted=False, status='Requires full frame0 U1 validation and independent replay'),indent=2))


def validate_restore(session, folder, steps=12):
    saved=session.checkpoint()
    def continuation():
        values=[]
        for _ in range(steps):
            if not session.step(): break
            values.append(np.r_[session.data.qpos.numpy().ravel(),session.data.qvel.numpy().ravel()])
        return np.array(values)
    a=continuation(); session.restore(saved)
    for key, value in device_arrays(session.data):
        np.testing.assert_array_equal(value.numpy(), saved['buffers'][key])
    b=continuation(); session.restore(saved)
    delta=np.abs(a-b)
    qerror=float(delta[:,:session.model.nq].max()) if len(a) else 0.
    verror=float(delta[:,session.model.nq:].max()) if len(a) else 0.
    # GPU parallel constraint reductions need not be bitwise deterministic.
    # Enforce sub-microradian/metre pose and 1e-4 velocity continuation bounds.
    report=dict(frame=session.frame, frames=len(a), max_qpos_absolute_error=qerror,
                max_qvel_absolute_error=verror, qpos_tolerance=1e-6, qvel_tolerance=1e-4,
                complete_buffer_count=len(saved['buffers']), passed=qerror<=1e-6 and verror<=1e-4)
    (Path(folder)/'restore_validation.json').write_text(json.dumps(report,indent=2))
    if not report['passed']: raise AssertionError(report)
    return report
