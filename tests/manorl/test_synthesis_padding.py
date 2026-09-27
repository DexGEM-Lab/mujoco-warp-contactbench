"""Behavior tests for parked padding of short synthesis waves.

Mechanism under test (paired CCD-overflow attribution evidence
bowl-ccd-overflow-20260928, condition B): a short wave parks its extra
physical worlds with a constant joint-limit-valid hand reference and a
stationary object reference plus an EMPTY policy window (disable == enable),
instead of duplicating the last real reference (condition A, which saturated
CCD capacity). Real claim worlds keep production semantics unchanged and are
the only worlds recorded, terminated or credited.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from sim.manorl.contracts import TrajectoryIdentity
from sim.manorl.trajectory import ReferenceTrajectory
from sim.manorl.synthesis_runtime import (
    BatchSynthesisRuntime,
    _stationary_reference,
    padding_policy_windows,
)


def trajectory(*, frames=240, prefix_frames=60, identity='cubeA_01_001',
               scene=('cubeA', 'bowlB'), moving=True, scene_pos=None):
    n = frames
    q = np.zeros((n, 28)); q[:, :3] = [0.1, -0.2, 0.4]
    if moving:
        q[:, 3] = np.linspace(0, .3, n)
    if scene_pos is None:
        scene_pos = np.tile([0., 0., 0.03], (len(scene), 1))
    scene_pos = np.asarray(scene_pos, dtype=float)
    active_object = identity.rsplit('_', 2)[0]
    active_pos = scene_pos[scene.index(active_object)]
    pos = np.tile(active_pos, (n, 1)); quat = np.tile([0., 0., 0., 1.], (n, 1))
    return ReferenceTrajectory(
        identity=TrajectoryIdentity(dataset_path='/fixture/raw.lance', dataset_version=1,
            row_index=0, object_index=0, uuid='01234567-0123-4567-89ab-0123456789ab',
            file_uuid='raw', identity=identity, source_start=0, source_stop=n,
            movement_start_raw=80, movement_end_raw=170),
        dataset_version=1, source_indices=np.arange(n), timestamps=np.arange(n) / 120,
        q_ref=q, object_pos_raw=pos, object_pos=pos.copy(), object_quat_xyzw=quat,
        object_z_shift=0., reference_fps=120, control_fps=120,
        movement_start_step=max(2, n // 3), movement_end_step=2 * (n // 3),
        augmentation_prefix_frames=prefix_frames, augmentation_base_early_phase_steps=80,
        scene_object_types=tuple(scene),
        scene_object_initial_pos=scene_pos,
        scene_object_initial_quat_xyzw=np.tile([0., 0., 0., 1.], (len(scene), 1)))


def _with_object_at(template, pos):
    """Keep the tracked-object reference stationary at its scene slot pose."""
    n = len(template.q_ref)
    return replace(template, object_pos=np.tile(pos, (n, 1)), object_pos_raw=np.tile(pos, (n, 1)))


def parked(trajectory_):
    """Constant-reference trajectory shape produced by build_parked_filler."""
    q = np.repeat(np.asarray(trajectory_.q_ref)[:1], len(trajectory_.q_ref), axis=0)
    return replace(trajectory_, q_ref=q, q_ref_by_side={'right': q},
                   object_pos=np.repeat(trajectory_.object_pos[:1], len(q), axis=0))


# ---------------------------------------------------------------------------
# 1. padding_policy_windows + install_wave composition
# ---------------------------------------------------------------------------

def test_padding_policy_windows_real_and_filler_windows():
    enable = np.array([90, 100, 80, 70], dtype=np.int64)
    lengths = np.array([200, 210, 190, 180], dtype=np.int64)
    disable = padding_policy_windows(enable, lengths, claim_worlds=2)
    np.testing.assert_array_equal(disable[:2], lengths[:2] + 1)   # operational no-op vs None
    np.testing.assert_array_equal(disable[2:], enable[2:])        # EMPTY windows
    np.testing.assert_array_equal(enable, [90, 100, 80, 70])      # enable never modified
    with pytest.raises(ValueError):
        padding_policy_windows(enable, lengths, claim_worlds=0)
    with pytest.raises(ValueError):
        padding_policy_windows(enable, lengths, claim_worlds=4)
    with pytest.raises(ValueError):
        padding_policy_windows(enable, lengths[:2], claim_worlds=1)


class RecordingRuntime:
    """Fake runtime capturing install/park calls; no physics involved."""

    def __init__(self, filler):
        self.filler = filler
        self.calls = []
        self.template = None
        self.padding_evidence = {'fingers': 'clipped_zero', 'raise_m': .05,
                                 'template_length': 240, 'external_hand_contacts': 0,
                                 'baseline_hand_self_contacts': 3}

    def build_parked_filler(self, template):
        self.template = template
        return self.filler

    def install(self, trajectories, seeds, **kwargs):
        self.calls.append((list(trajectories), list(seeds), kwargs))


def test_install_wave_full_batch_unchanged(capsys):
    from tools.run_manorl_synthesis_campaign import install_wave
    reals = [trajectory(identity=f'cubeA_{i:02d}_001') for i in range(4)]
    samples = [{'seed': 11 + i, 'slot': i} for i in range(4)]
    runtime = RecordingRuntime(None)
    out_traj, out_samples = install_wave(runtime, reals, samples, 4)
    assert len(runtime.calls) == 1
    traj, seeds, kwargs = runtime.calls[0]
    assert kwargs == {} or kwargs.get('claim_worlds') is None  # no padding path taken
    assert runtime.template is None                              # no park attempted
    assert out_traj == reals and all(a is b for a, b in zip(out_traj, reals))
    assert out_samples == samples and seeds == [11, 12, 13, 14]
    assert 'batch_parked_padding' not in capsys.readouterr().out


def test_install_wave_short_batch_preserves_real_inputs_bit_for_bit(capsys):
    from tools.run_manorl_synthesis_campaign import install_wave
    batch = 256
    reals = [trajectory(identity=f'cubeA_{i % 100:02d}_001') for i in range(11)]
    samples = [{'seed': 1000 + i, 'slot': i, 'radius_stratum_m': [i, i + 1]} for i in range(11)]
    before = [(t.q_ref.copy(), t.object_pos.copy(), t.object_pos_raw.copy(),
               t.object_quat_xyzw.copy(), t.source_indices.copy(), t.timestamps.copy(),
               t.scene_object_initial_pos.copy()) for t in reals]
    filler = parked(trajectory())
    runtime = RecordingRuntime(filler)
    out_traj, out_samples = install_wave(runtime, reals, samples, batch)
    traj, seeds, kwargs = runtime.calls[0]
    assert kwargs == {'claim_worlds': 11}
    assert len(traj) == batch and len(seeds) == batch and len(out_samples) == batch
    # Real worlds are the same objects, bit-for-bit unchanged.
    for i, real in enumerate(reals):
        assert traj[i] is real and out_traj[i] is real
        q, pos, pos_raw, quat, src, ts, scene = before[i]
        np.testing.assert_array_equal(real.q_ref, q)
        np.testing.assert_array_equal(real.object_pos, pos)
        np.testing.assert_array_equal(real.object_pos_raw, pos_raw)
        np.testing.assert_array_equal(real.object_quat_xyzw, quat)
        np.testing.assert_array_equal(real.source_indices, src)
        np.testing.assert_array_equal(real.timestamps, ts)
        np.testing.assert_array_equal(real.scene_object_initial_pos, scene)
        assert out_samples[i] is samples[i] and traj is not reals
    # Every extra physical world holds the identical parked filler; filler seeds
    # and samples repeat the last real's, exactly like the pre-padding contract.
    for i in range(11, batch):
        assert traj[i] is filler and out_traj[i] is filler
        assert seeds[i] == 1010 and out_samples[i] is samples[10]
    emitted = capsys.readouterr().out
    assert 'batch_parked_padding' in emitted and '"padding_worlds": 245' in emitted


# ---------------------------------------------------------------------------
# 2. Stepper window semantics with the REAL _CheckpointPolicyStepper
# ---------------------------------------------------------------------------

def _stepper_fixture(num_envs=1):
    import torch
    from sim.manorl.view_environment import _CheckpointPolicyStepper

    class Physical:
        action_dim = 28
        trajectory_steps = np.zeros(num_envs, dtype=np.int64)
        config = SimpleNamespace(num_envs=num_envs)

    physical = Physical()

    class Wrapped:
        def step(self, actions):
            self.actions = actions.clone()
            physical.trajectory_steps += 1
            return (torch.zeros((num_envs, 2)), torch.zeros(num_envs),
                    torch.zeros(num_envs, dtype=torch.bool),
                    torch.zeros(num_envs, dtype=torch.bool), {})

    class Runtime:
        def __init__(self):
            self.env = Wrapped()
            self.gymnasium_env = SimpleNamespace(environment=physical)
            self.calls = 0

        def deterministic_actions(self, observations):
            self.calls += 1
            return torch.ones((num_envs, 28))

    runtime = Runtime()
    observations = torch.zeros((num_envs, 2))
    stepper = _CheckpointPolicyStepper(runtime, observations,
                                       policy_enable_steps=np.full(num_envs, 3, dtype=np.int64),
                                       policy_disable_steps=np.full(num_envs, 3, dtype=np.int64))
    return stepper, runtime


def test_empty_filler_window_zero_residual_every_step():
    stepper, runtime = _stepper_fixture()
    for _ in range(6):
        stepper.step()
    assert runtime.calls == 0                       # policy never invoked
    np.testing.assert_array_equal(runtime.env.actions.numpy(), 0.0)  # zero residual


def test_real_disable_length_plus_one_operationally_identical_to_none():
    import torch
    from sim.manorl.view_environment import _CheckpointPolicyStepper

    length = 4
    def run(disable):
        class Physical:
            action_dim = 28
            trajectory_steps = np.zeros(1, dtype=np.int64)
            config = SimpleNamespace(num_envs=1)

        physical = Physical()

        class Wrapped:
            def step(self, actions):
                self.actions = actions.clone(); physical.trajectory_steps += 1
                return (torch.zeros((1, 2)), torch.zeros(1), torch.zeros(1, dtype=torch.bool),
                        torch.zeros(1, dtype=torch.bool), {})

        class Runtime:
            def __init__(self):
                self.env = Wrapped(); self.calls = 0
                self.gymnasium_env = SimpleNamespace(environment=physical)

            def deterministic_actions(self, observations):
                self.calls += 1; return torch.ones((1, 28))

        runtime = Runtime()
        stepper = _CheckpointPolicyStepper(
            runtime, torch.zeros((1, 2)), policy_enable_steps=np.asarray([2]),
            policy_disable_steps=None if disable is None else np.asarray([disable]))
        trace = []
        for _ in range(length + 1):  # episode lifetime: steps 0..length inclusive
            stepper.step()
            trace.append((runtime.calls, float(runtime.env.actions.numpy().sum())))
        return trace

    assert run(None) == run(length + 1)


# ---------------------------------------------------------------------------
# Fake-environment install: short-wave masks and full-wave reset
# ---------------------------------------------------------------------------

class FakeData:
    def __init__(self):
        self.qacc_warmstart = np.zeros(3)
        self.time = np.zeros(3)

    def replace(self, **kwargs):
        updated = FakeData()
        updated.__dict__.update(self.__dict__)
        updated.__dict__.update(kwargs)
        return updated


def _install_runtime(trajectories):
    """BatchSynthesisRuntime shell with a minimal fake environment."""
    runtime = object.__new__(BatchSynthesisRuntime)
    runtime.object_name = trajectories[0].identity.identity.rsplit('_', 2)[0]
    runtime.scene_names = tuple(trajectories[0].scene_object_types)
    runtime.batch_size = len(trajectories)
    stepper = SimpleNamespace(
        _runtime=SimpleNamespace(env=SimpleNamespace(reset=lambda: (object(), None))),
        _observations=None, _pending_done='stale', _policy_enable_steps=None,
        _policy_disable_steps=np.full(len(trajectories), 999, dtype=np.int64))
    env = SimpleNamespace(
        trajectory=None, trajectories=None, action_ids=None, expected_keypoint_ids=(),
        expected_contact_mask=np.zeros((len(trajectories), 4)),
        config=SimpleNamespace(residual_joint_mode='all', expected_contact_mode='strict', contact_capacity=4096,
                               constraint_capacity=4096, warp_ccd_contacts_per_world=16),
        data=FakeData(), jax=SimpleNamespace(
            device_put=lambda value, device: value,
            zeros_like=lambda value: np.zeros_like(value)),
        device=object(), jp=SimpleNamespace(zeros_like=lambda v: np.zeros_like(v)),
        _warp_ccd_overflow_guard_available=False, _reset_data_fn=None)
    def _build_reference_tables():
        env.trajectories = tuple(env.trajectories)
        env.trajectory_lengths = np.asarray([len(t.q_ref) for t in env.trajectories], dtype=np.int64)
        env.early_phase_lengths = np.asarray(
            [t.augmentation_prefix_frames + t.augmentation_base_early_phase_steps
             for t in env.trajectories], dtype=np.int64)
        env.reference_q_model = np.stack([t.q_ref for t in env.trajectories])
    env._build_reference_tables = _build_reference_tables
    env._initial_qpos = lambda: np.zeros((len(trajectories), 42))
    env.reseed_point_templates_per_env = lambda seeds: setattr(env, 'seeds', np.asarray(seeds))
    runtime.environment = env
    runtime.stepper = stepper
    return runtime, env


def test_install_short_wave_sets_masks_then_full_wave_clears_them(monkeypatch):
    import sim.manorl.environment as environment_module
    monkeypatch.setattr(environment_module, '_expected_keypoint_ids', lambda *a, **k: ())
    monkeypatch.setattr(environment_module, '_build_masked_reset_data_fn', lambda **k: None)
    reals = [trajectory(identity=f'cubeA_{i:02d}_001', frames=100, prefix_frames=5) for i in range(3)]
    fillers = [parked(reals[-1]) for _ in range(2)]
    runtime, env = _install_runtime(reals + fillers)
    runtime.install(reals + fillers, list(range(5)), claim_worlds=3)
    enable = np.asarray([t.augmentation_prefix_frames + 80 for t in reals + fillers])
    np.testing.assert_array_equal(runtime.stepper._policy_enable_steps, enable)
    np.testing.assert_array_equal(runtime.stepper._policy_disable_steps[:3],
                                  env.trajectory_lengths[:3] + 1)
    np.testing.assert_array_equal(runtime.stepper._policy_disable_steps[3:], enable[3:])
    assert runtime.stepper._pending_done is None
    # Next wave is a FULL batch: the stale per-world disable mask must clear.
    full = [trajectory(identity=f'cubeA_{i:02d}_001') for i in range(5)]
    runtime.install(full, list(range(5)))
    assert runtime.stepper._policy_disable_steps is None
    np.testing.assert_array_equal(runtime.stepper._policy_enable_steps,
                                  env.early_phase_lengths)


def test_install_refuses_active_duplicate_padding(monkeypatch):
    import sim.manorl.environment as environment_module
    monkeypatch.setattr(environment_module, '_expected_keypoint_ids', lambda *a, **k: ())
    monkeypatch.setattr(environment_module, '_build_masked_reset_data_fn', lambda **k: None)
    reals = [trajectory(identity='cubeA_01_001'), trajectory(identity='cubeA_02_001')]
    runtime, env = _install_runtime(reals + reals)  # trailing copies are NOT parked
    with pytest.raises(ValueError, match='constant parked reference'):
        runtime.install(reals + reals, [1, 2, 3, 4], claim_worlds=2)


def test_stationary_reference_detector():
    real = trajectory()
    assert not _stationary_reference(real)
    assert _stationary_reference(parked(real))
    scene = real.scene_object_initial_pos.copy(); scene[0] = [0., 0., .1]
    varying = np.tile([[0., 0., .1], [0., 0., .12]], (len(real.q_ref) // 2, 1))
    moved = replace(parked(real), object_pos=varying, scene_object_initial_pos=scene)
    assert not _stationary_reference(moved)
    quat = parked(real).object_quat_xyzw.copy()
    quat[1:, [2, 3]] = [1., 0.]
    rotating = replace(parked(real), object_quat_xyzw=quat)
    assert not _stationary_reference(rotating)


# ---------------------------------------------------------------------------
# 3. Parked-filler geometry against a native mujoco model
# ---------------------------------------------------------------------------

HAND_XML = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="3 3 .1"/>
    <body name="hand">
      <joint type="slide" axis="1 0 0" range="-1 1"/>
      <joint type="slide" axis="0 1 0" range="-1 1"/>
      <joint type="slide" axis="0 0 1" range="0 1"/>
      <joint type="hinge" axis="1 0 0" range="-1 1"/>
      <joint type="hinge" axis="0 1 0" range="-1 1"/>
      <joint type="hinge" axis="0 0 1" range="-1 1"/>
      <geom name="hand" type="sphere" size=".04"/>
      {fingers}
    </body>
    <body name="cubeA"><freejoint name="cubeA_joint"/><geom name="cubeA_geom" type="sphere" size=".02" pos="{a_pos}"/></body>
    <body name="bowlB"><freejoint name="bowlB_joint"/><geom name="bowlB_geom" type="sphere" size=".02" pos="{b_pos}"/></body>
  </worldbody>
</mujoco>
"""

FINGERS = ''.join(
    '<body pos="0 0 0"><joint type="hinge" range="-1 1"/>'
    '<inertial pos="0 0 0" mass=".01" diaginertia=".001 .001 .001"/>'
    '<geom type="sphere" size=".04"/></body>' for _ in range(22))


def _geometry_runtime(xml):
    import mujoco
    model = mujoco.MjModel.from_xml_string(xml)
    hand_geom = int(model.geom('hand').id)
    runtime = object.__new__(BatchSynthesisRuntime)
    runtime.environment = SimpleNamespace(
        model=model, mujoco=mujoco,
        joint_lower=model.jnt_range[:28, 0].astype(float),
        joint_upper=model.jnt_range[:28, 1].astype(float))
    runtime.scene_names = ('cubeA', 'bowlB')
    runtime.object_qpos = {
        name: int(model.jnt_qposadr[int(mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f'{name}_joint'))])
        for name in runtime.scene_names}
    hand_body = int(model.body('hand').id)
    hand_bodies = set()
    for body in range(model.nbody):
        ancestor = body
        while ancestor:
            if ancestor == hand_body:
                hand_bodies.add(body)
                break
            ancestor = int(model.body_parentid[ancestor])
    # Palm and all 22 child finger geoms natively overlap; descendants belong
    # to the hand even though their immediate body id differs from the palm's.
    runtime.hand_geoms = {int(geom) for geom in range(model.ngeom)
                          if int(model.geom_bodyid[geom]) in hand_bodies}
    runtime.padding_evidence = {}
    return runtime, model


def _bodies_qposadr(model, mujoco):
    addresses = {}
    for name in ('cubeA', 'bowlB'):
        joint = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f'{name}_joint'))
        addresses[name] = int(model.jnt_qposadr[joint])
    return addresses


def test_parked_filler_geometry_allows_self_overlap_and_preserves_scalars():
    import mujoco
    xml = HAND_XML.format(fingers=FINGERS, a_pos='0 0 0', b_pos='0 0 0')
    runtime, model = _geometry_runtime(xml)
    template = trajectory(scene_pos=[[2, 0, 0.5], [-2, 0, 0.5]])
    lower, upper = runtime.environment.joint_lower, runtime.environment.joint_upper
    filler = runtime.build_parked_filler(template)
    # Constant, joint-limit-valid neutral park: clipped-zero fingers, raised wrist.
    T = len(template.q_ref)
    np.testing.assert_array_equal(filler.q_ref, np.repeat(filler.q_ref[:1], T, axis=0))
    np.testing.assert_array_equal(filler.object_pos, np.repeat(template.object_pos[:1], T, axis=0))
    np.testing.assert_array_equal(filler.object_pos_raw, np.repeat(template.object_pos_raw[:1], T, axis=0))
    np.testing.assert_array_equal(filler.object_quat_xyzw, np.repeat(template.object_quat_xyzw[:1], T, axis=0))
    assert filler.q_ref[0, 6:].max() == 0.0 and filler.q_ref[0, 6:].min() == 0.0
    assert np.all(filler.q_ref[0] >= lower - 1e-12) and np.all(filler.q_ref[0] <= upper + 1e-12)
    assert filler.q_ref[0, 2] > template.q_ref[0, 2]          # wrist raised within limits
    # Every non-replaced scalar/clock/phase field is the template's own object.
    assert filler.identity is template.identity
    for field in ('source_indices', 'timestamps', 'scene_object_initial_pos',
                  'scene_object_initial_quat_xyzw', 'movement_start_step', 'movement_end_step',
                  'augmentation_prefix_frames', 'augmentation_base_early_phase_steps',
                  'reference_fps', 'control_fps', 'scene_object_types'):
        assert getattr(filler, field) is getattr(template, field)
    # Native same-hand overlaps exist and are allowed; zero external contacts.
    data = mujoco.MjData(model)
    data.qpos[:28] = filler.q_ref[0]
    mujoco.mj_kinematics(model, data); mujoco.mj_comPos(model, data); mujoco.mj_collision(model, data)
    hand_geoms = runtime.hand_geoms
    self_pairs = external = 0
    for contact in data.contact[:data.ncon]:
        a, b = int(contact.geom1), int(contact.geom2)
        if (a in hand_geoms) and (b in hand_geoms):
            self_pairs += 1
        elif (a in hand_geoms) != (b in hand_geoms):
            external += 1
    assert self_pairs > 0 and external == 0
    assert runtime.padding_evidence['external_hand_contacts'] == 0
    assert runtime.padding_evidence['baseline_hand_self_contacts'] == self_pairs
    assert runtime.padding_evidence['template_length'] == T


def test_parked_filler_uses_actual_scene_addresses_not_index_zero():
    # cubeA (scene slot 0) is far away; ONLY bowlB (slot 1, its own qpos
    # address) overlaps the raised hand. A blind index-0 check would pass.
    import mujoco
    template = trajectory(scene_pos=[[2, 0, 0.5], [0.1, -0.2, 0.85]])
    xml = HAND_XML.format(fingers=FINGERS, a_pos='0 0 0', b_pos='0 0 0')
    runtime, model = _geometry_runtime(xml)
    addresses = _bodies_qposadr(model, mujoco)
    assert addresses['bowlB'] > addresses['cubeA'] >= 28
    # Move ONLY the bowlB scene-initial slot onto the parked hand; the geoms'
    # default positions are irrelevant because the filler places each object at
    # its scene slot pose at its own qpos address.
    scene_pos = np.array([[2, 0, 0.5], [0.1, -0.2, 0.85]])
    template = replace(template, scene_object_initial_pos=scene_pos)
    with pytest.raises(RuntimeError, match='external hand contacts'):
        runtime.build_parked_filler(template)
    assert runtime.padding_evidence == {}   # no partial evidence on refusal
    # A valid tracked-object reference remains equal to its active scene slot;
    # the stationary filler preserves both sides of that required invariant.
    template = trajectory(scene_pos=[[.4, .4, .4], [2, 0, 0.6]])
    filler = runtime.build_parked_filler(template)
    np.testing.assert_array_equal(filler.object_pos[0], template.object_pos[0])
    np.testing.assert_array_equal(filler.object_pos[0], filler.scene_object_initial_pos[0])
    np.testing.assert_array_equal(filler.scene_object_initial_pos, template.scene_object_initial_pos)


def test_infeasible_park_fails_loudly():
    # A world-fixed external shell around the whole hand workspace makes every
    # joint-limit-valid posture collide: the filler must fail, not pad.
    xml = HAND_XML.format(fingers=FINGERS, a_pos='2 0 0.5', b_pos='-2 0 0.5')
    xml = xml.replace('<geom name="floor"', '<geom name="cage" type="box" size=".5 .5 .6" pos="0 0 .5"/>'
                                          '<geom name="floor"')
    runtime, _ = _geometry_runtime(xml)
    runtime.hand_geoms = {int(g) for g in range(runtime.environment.model.ngeom)
                          if int(runtime.environment.model.geom_bodyid[g])
                          == int(runtime.environment.model.body('hand').id)}
    with pytest.raises(RuntimeError, match='no joint-limit-valid parked posture'):
        runtime.build_parked_filler(trajectory())


# ---------------------------------------------------------------------------
# 4. collect(): claim-only accounting; collection stops with real claims
# ---------------------------------------------------------------------------

class FakeGate:
    def __init__(self, accepted=True):
        self.accepted = accepted

    def to_dict(self):
        return {'accepted': self.accepted, 'contract': 'fake-gate'}


def _collect_runtime(worlds, lengths, reset_steps):
    """Fake runtime around BatchSynthesisRuntime.collect with fake physics."""
    runtime = object.__new__(BatchSynthesisRuntime)
    n = len(worlds)
    runtime.scene_names = ('cubeA', 'bowlB')
    runtime.object_bodies = {'cubeA': 1, 'bowlB': 2}
    runtime.capacity_evidence = {'max_contacts': 0, 'max_constraints': 0}
    runtime.padding_evidence = {}
    physical = SimpleNamespace(
        mano_dof_pos=np.zeros((n, 28)), hand_keypoint_positions=np.zeros((n, 21, 3)),
        fingertip_positions=np.zeros((n, 5, 3)), object_position=np.zeros((n, 3)),
        object_orientation_xyzw=np.zeros((n, 4)))
    state = SimpleNamespace(xpos=np.zeros((n, 4, 3)), xquat=np.zeros((n, 4, 4)))
    step_counter = {'value': 0}
    termination = SimpleNamespace(
        reset=np.zeros(n, dtype=bool), reason_code=np.zeros(n, dtype=int),
        success=np.zeros(n, dtype=bool))

    def termination_update():
        termination.reset[:] = False
        for world, step in reset_steps.items():
            if step_counter['value'] == step:
                termination.reset[world] = True
                termination.reason_code[world] = 1
                termination.success[world] = True

    class Stepper:
        def __init__(self):
            self.steps = 0

        def step(self):
            self.steps += 1
            step_counter['value'] += 1
            termination_update()

    env = SimpleNamespace(
        trajectory_lengths=np.asarray(lengths, dtype=np.int64),
        reference_q_model=np.zeros((n, max(lengths), 28)),
        data=SimpleNamespace(ctrl=np.zeros((n, 28))),
        last_physical=physical, last_reward=SimpleNamespace(total=np.zeros(n)),
        last_termination=termination, reference_source_indices=np.zeros((n, max(lengths)), dtype=int),
        _target_indices=lambda: np.zeros(n, dtype=int))
    runtime.environment = env
    runtime.stepper = Stepper()
    contacts = [[] for _ in range(n)]
    floor = np.zeros(n, dtype=bool); floor[2:] = True       # filler worlds "collide"
    obj = np.zeros(n, dtype=bool); obj[3:] = True
    runtime._contacts = lambda: (contacts, floor.copy(), obj.copy(), state)
    return runtime, env


def test_collect_returns_only_real_claims_and_stops_when_they_finish(monkeypatch):
    import tools.export_manorl_synthetic_lance as export_module
    evaluated = []
    monkeypatch.setattr(export_module, '_evaluate_collected_candidate',
                        lambda **kwargs: evaluated.append(kwargs) or FakeGate())
    worlds = [trajectory(identity=f'cubeA_{i:02d}_001') for i in range(2)] + \
             [parked(trajectory()) for _ in range(3)]
    # Filler worlds (12 steps) outlive every real claim (5 and 8).
    runtime, env = _collect_runtime(worlds, [5, 8, 12, 12, 12], {0: 3, 1: 5})
    claims = [{'claim': i, 'slot': i} for i in range(2)]
    samples = [{'seed': 7, 'slot': 0}, {'seed': 8, 'slot': 1}] + [{'seed': 8, 'slot': 1}] * 3
    results, phase = runtime.collect(worlds, samples, claims)
    # Only real claims produce results; fillers never reach the gate.
    assert len(results) == 2 and len(evaluated) == 2
    assert [r['claim']['claim'] for r in results] == [0, 1]
    assert all(r['diagnostic']['accepted'] for r in results)
    assert all(r['diagnostic']['prefix_failure'] is None for r in results)
    # Filler collisions (floor/obj True for worlds >= 2) never became failures.
    assert [r['diagnostic']['prefix_failure'] for r in results] == [None, None]
    # Collection stopped when the LAST REAL claim finished, not at filler length.
    assert phase['steps'] == 5 and runtime.stepper.steps == 5
    assert phase['worlds'] == 5 and phase['base_accepted'] == 2
    # The fake stepper increments before checking reset_steps: reset at 3/5
    # yields three/five executed steps, plus the initial stored frame.
    assert len(results[0]['states']['urdf_dof']) == 4
    assert len(results[1]['states']['urdf_dof']) == 6
    assert all(len(r['contacts']) >= 4 for r in results)
    # No store was created for a filler world even indirectly.
    assert all(r['states'] is not None for r in results)


def test_collect_rejects_empty_or_overlong_claims(monkeypatch):
    import tools.export_manorl_synthetic_lance as export_module
    monkeypatch.setattr(export_module, '_evaluate_collected_candidate', lambda **k: FakeGate())
    worlds = [trajectory(), parked(trajectory())]
    runtime, _ = _collect_runtime(worlds, [5, 6], {})
    with pytest.raises(ValueError):
        runtime.collect(worlds, [{}], [])
    with pytest.raises(ValueError):
        runtime.collect(worlds, [{}], [{}, {}, {}])
