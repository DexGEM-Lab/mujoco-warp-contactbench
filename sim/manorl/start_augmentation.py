"""Position-only starts on reconstructed repaired commands; no merge-state reset."""
from __future__ import annotations

from dataclasses import replace
import numpy as np


def augment(inp, parent, delta, merge_frame):
    delta = np.asarray(delta, dtype=float)
    C = merge_frame
    if delta.shape != (3,) or not np.isfinite(delta).all():
        raise ValueError("offset must be a finite XYZ vector")
    if not isinstance(C, int) or not 3 <= C < len(parent) - 1:
        raise ValueError("merge requires an interior central-difference frame")
    u = np.clip(np.arange(len(parent)) / (C - 1), 0, 1)
    envelope = 1 - 10*u**3 + 15*u**4 - 6*u**5
    envelope[C-1:] = 0
    desired = parent.copy()
    desired[:C-1, :3] += envelope[:C-1, None] * delta
    initial = {k: v.copy() for k, v in inp.initial.items()}
    initial['qpos'][:3] += delta
    shifted = replace(inp, initial=initial)
    assert desired[C-1:].tobytes() == parent[C-1:].tobytes()
    np.testing.assert_array_equal(np.gradient(desired, 1/inp.hz, axis=0)[C],
                                  np.gradient(parent, 1/inp.hz, axis=0)[C])
    duration = (C-1)/inp.hz
    info = dict(delta_xyz_m=delta.tolist(), merge_frame=C, residual_zero_from_frame=C-1,
                taper_duration_s=duration, peak_residual_speed_m_s=float(1.875*np.linalg.norm(delta)/duration),
                peak_residual_acceleration_m_s2=float(10/np.sqrt(3)*np.linalg.norm(delta)/duration**2),
                suffix_byte_exact=True, central_difference_velocity_exact=True)
    return shifted, desired, envelope, info


def scene_contacts(model, qpos, qvel, names):
    """Native geometry only: self-contact is not a hand/scene collision."""
    import mujoco as mj
    data = mj.MjData(model)
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mj.mj_forward(model, data)
    scene = {0, *(model.body(name).id for name in names)}
    contacts = []
    for c in data.contact[:data.ncon]:
        b1, b2 = (int(model.geom_bodyid[g]) for g in (c.geom1, c.geom2))
        if (b1 in scene) == (b2 in scene):
            continue
        contacts.append(dict(geom1=int(c.geom1), geom2=int(c.geom2),
                             body1=model.body(b1).name, body2=model.body(b2).name,
                             distance_m=float(c.dist)))
    return contacts


def unsupported_intervals(arrays, hz):
    mask = (arrays['height_m'] > .02) & ~arrays['has_support'] & (arrays['finger_ray_count'] == 0)
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    # Conservative sampled occupancy: three unsupported 120Hz samples =25ms.
    return [dict(start_frame=int(a), end_frame_exclusive=int(b), duration_s=(b-a)/hz)
            for a, b in zip(edges[::2], edges[1::2]) if (b-a)/hz >= .025-1e-12]
