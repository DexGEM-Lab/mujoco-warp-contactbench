"""Batching preserves per-world scalar PID and prescribed sampling boundaries."""
import numpy as np
import pytest


def test_batched_pid_matches_scalar_cpu_kernel():
    import warp as wp
    from sim.manorl.local_contact_dynamics import wrist_targets
    from sim.manorl.start_batch_dynamics import batch_wrist_targets
    wp.init()
    device = 'cpu'
    rng = np.random.default_rng(7)
    q = rng.normal(0, .2, (3, 28)).astype(np.float32)
    v = rng.normal(0, .1, (3, 28)).astype(np.float32)
    target = rng.normal(0, .2, (3, 6)).astype(np.float32)
    target[2, 4] += 2*np.pi
    speed = rng.normal(0, .2, (3, 6)).astype(np.float32)
    integral = rng.normal(0, .05, (3, 6)).astype(np.float32)
    kp = wp.array(np.full(6, 100, np.float32), device=device)
    kv = wp.array(np.arange(1, 7, dtype=np.float32), device=device)
    qc, vc = wp.array(q, device=device), wp.array(v, device=device)
    ctrl = wp.zeros((3, 28), dtype=wp.float32, device=device)
    acc = wp.array(integral, device=device)
    wp.launch(batch_wrist_targets, dim=(3, 6), inputs=[qc, vc, ctrl, wp.array(target, device=device),
              wp.array(speed, device=device), kp, kv, acc, 1.8, 1/480], device=device)
    for world in range(3):
        cs = wp.zeros((1, 28), dtype=wp.float32, device=device)
        ins = wp.array(integral[world], device=device)
        wp.launch(wrist_targets, dim=6, inputs=[wp.array(q[world:world+1], device=device),
                  wp.array(v[world:world+1], device=device), cs, wp.array(target[world], device=device),
                  wp.array(speed[world], device=device), kp, kv, ins, 1.8, 1/480], device=device)
        np.testing.assert_array_equal(ctrl.numpy()[world], cs.numpy()[0])
        np.testing.assert_array_equal(acc.numpy()[world], ins.numpy())


def test_merge_guard_is_physical_time():
    from tools.run_start_augmentation import merge_frame, RADII
    from types import SimpleNamespace
    inp = SimpleNamespace(frames=253, metrics={'source_metadata': {'object_move': [{'start_frame': 58}]}})
    counts = np.zeros(253, int); counts[60:] = 2
    C, first = merge_frame(inp, {'finger_ray_count': counts}, .1)
    assert (C, first) == (48, 60)
    assert RADII == (.03, .05, .075, .1)
    with pytest.raises(ValueError): merge_frame(inp, {'finger_ray_count': counts}, np.nan)
    with pytest.raises(ValueError): merge_frame(inp, {'finger_ray_count': np.zeros(253)}, .1)
