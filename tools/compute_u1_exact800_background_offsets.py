#!/usr/bin/env python3
"""Joint, mesh-accurate background-placement solver for a caller-selected exact800 Lance.

For every row in the selected compact Lance:
  - active objects keep their own recorded trajectory (never touched)
  - every OTHER (background) canonical object is checked against the row's
    hand+keypoint sweep AND against every other background object, using
    real (voxel-downsampled) collision meshes, not a bounding-circle proxy
  - objects needing a nudge are solved in most-deficient-first order so a
    later object's search always respects earlier objects' chosen position
  - a cheap bounding-circle broad phase prunes obviously-safe candidates
    before the expensive real-mesh check

Parallelized across rows (one process per CPU, each rebuilds the small
downsampled KD-trees locally -- cheap, <1s -- and solves its own row slice
independently since rows never interact with each other).

Output: one JSON with per-action stats and a flat per_row_offsets table
keyed by row UUID (only rows that need >=1 nudge are present).
Visualization-only: nothing here touches the delivered Lance.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

os.environ.setdefault(
    'MANORL_ASSET_MANIFEST',
    '/mnt/nas-222-project/mocap_v2/lance_datasets/manorl_rl_formal_cheyingtong_noaug_20260917_115140/asset_manifest.json',
)

LANCE = os.environ.get('MANORL_LAYOUT_LANCE', '/MISSING/compact.lance')
VERSION = int(os.environ.get('MANORL_LAYOUT_VERSION', '1'))
OUTPUT = os.environ.get('MANORL_LAYOUT_OUTPUT', 'outputs/u1_exact800_background_offsets.json')

# Revised 2026-09-18: egg_cup relocated to a real-data-adjacent spot near the trash
# bin (verified clear of both the hand's own resting-position sweep in 005/006 and
# the bowl's carry sweep in 003-007, by >=8cm and >=31cm respectively). The egg-zone
# cluster (rack/ellipsoid/cylinder7) is shifted together (dx=-0.10, dy=-0.06) to move
# it out of the densest part of the bowl-carry corridor -- a pure translation cannot
# clear 100% of the bowl's cross-action sweep (that union covers most of the table;
# verified via an AABB check), so the remaining touch points in 003/005 for
# egg_stick_rack/egg_ellipsoid are left to the per-row solver below.
# Revised 2026-09-18 (second pass): egg_ellipsoid is only ever active in action 001
# (crack the egg, drop it in the trash). Every OTHER action happens narratively
# *after* 001 in the same task sequence, so whenever the egg is a background object
# it should already be sitting inside the trash bin -- not still resting by the rack.
# It is grouped with trash_bin (moves together, and deliberately not clearance-checked
# against it since containment is the intended look), separate from the
# rack+stick, which really do return to the egg station after action 002.
CANON = {
    # Position derived from the bin's own mesh floor (world z = trash_bin z + local
    # min-z), not guessed: an egg dropped in settles near the bottom, not floating near
    # the rim. See tools/find_trash_floor_position.py for the derivation.
    'egg_ellipsoid': [0.4877, -0.2876, 0.0234],
    'egg_stick_rack': [-0.294, -0.329, 0.022],
    'cylinder7': [-0.315, -0.400, 0.122],
    'egg_cup': [0.55, -0.05, 0.068],
    'trash_bin': [0.476, -0.304, 0.076],
    'bowl': [-0.131, 0.132, 0.050],        # on-stove default; only used as background for 001/002
    'cuboid1': [-0.157, 0.225, 0.016],
    'mayonnaisebottle': [0.0423, 0.2666, 0.1015],
    'pitcherbase': [0.3265, 0.2888, 0.1352],
}
ALL_NAMES = list(CANON)

# Objects that naturally rest together in the real captures (rack+stick returning to
# their station, bowl resting on its stove, the disposed egg sitting inside the bin)
# must always move as one rigid unit -- offsetting one member alone to dodge a
# conflict would visibly separate it from its partner.
GROUPS = {
    'egg_station': ['egg_stick_rack', 'cylinder7'],
    'trash': ['egg_ellipsoid', 'trash_bin'],
    'stove': ['bowl', 'cuboid1'],
}

# The hand's collision capsules extend up to ~3cm beyond joint keypoint centres
# (palm_collision radius = 2.92cm; thumb 1.07-1.60cm). A 2cm margin proved
# visually insufficient: frame 816 of the 004 row showed the palm surface
# clipping the trash bin even with a 2.32cm keypoint clearance. 4cm gives
# the ~3cm capsule radius plus ~1cm visible air gap.
HAND_TARGET = 0.04
OBJ_TARGET = 0.02
VOXEL = 0.005
N_DIRS = 16
MAGS = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15]

# Active-object sweep: coarser voxel/stride since this augments the intruder point
# set (accuracy matters less than for the fixed hand+keypoint trace).
ACTIVE_VOXEL = 0.01
ACTIVE_STRIDE = 4

ASSET_ROOT = '/home/jay/dexrobot/FromSSH/manoRL_mujoco/assets/dexstream_digital_assets'


def voxel_downsample(pts: np.ndarray, voxel: float = VOXEL) -> np.ndarray:
    keys = np.floor(pts / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return pts[idx]


_WORKER_STATE = {}


def _worker_init():
    """Runs once per worker process: build the small per-object KD-trees locally."""
    from pathlib import Path as _Path

    from sim.manorl import assets as _assets
    from scipy.spatial import cKDTree as _cKDTree

    _assets.DEXSTREAM_ROOT = _Path(ASSET_ROOT)
    canon = {k: np.array(v) for k, v in CANON.items()}
    verts_local = {k: voxel_downsample(_assets.object_collision_vertices(k)) for k in ALL_NAMES}
    trees_canon = {k: _cKDTree(verts_local[k] + canon[k]) for k in ALL_NAMES}
    radii = {
        k: float(np.linalg.norm((_assets.object_collision_vertices(k).max(0)
                                  - _assets.object_collision_vertices(k).min(0))[:2]) / 2)
        for k in ALL_NAMES
    }
    dirs = [np.array([np.cos(a), np.sin(a)]) for a in np.linspace(0, 2 * np.pi, N_DIRS, endpoint=False)]
    active_local = {k: voxel_downsample(_assets.object_collision_vertices(k), ACTIVE_VOXEL) for k in ALL_NAMES}
    _WORKER_STATE.update(canon=canon, verts_local=verts_local, trees_canon=trees_canon,
                          radii=radii, dirs=dirs, cKDTree=_cKDTree, active_local=active_local)


def _solve_row(row: dict) -> dict:
    canon = _WORKER_STATE['canon']
    verts_local = _WORKER_STATE['verts_local']
    trees_canon = _WORKER_STATE['trees_canon']
    radii = _WORKER_STATE['radii']
    dirs = _WORKER_STATE['dirs']

    gesture = row['trajectory_metadata']['gesture']
    action = gesture.split('-')[0]
    active = set(row['trajectory_metadata']['object_names'])
    bg_names = [n for n in ALL_NAMES if n not in active]
    uuid = row['index']['uuid']

    hand = np.asarray(row['hands'][0]['urdf_dof'], dtype=float)[:, :3]
    kp = np.asarray(row['hands'][0]['mano_joint_pos'], dtype=float)
    pts = np.concatenate([hand[:, None, :], kp.reshape(kp.shape[0], -1, 3)], axis=1).reshape(-1, 3)

    # The active object has its own real, rotating mesh -- not just a point at the
    # wrist. Sweep it (strided, coarser voxel) and add it to the intruder set so a
    # background object must clear the carried object's body too, not just the hand.
    from scipy.spatial.transform import Rotation as _Rot
    active_local = _WORKER_STATE['active_local']
    for oi, oname in enumerate(row['trajectory_metadata']['object_names']):
        if oname not in active_local:
            continue
        opos = np.asarray(row['objects'][oi]['pos'], dtype=float)[::ACTIVE_STRIDE]
        orot = np.asarray(row['objects'][oi]['rot_aa'], dtype=float)[::ACTIVE_STRIDE]
        Rm = _Rot.from_rotvec(orot).as_matrix()
        swept = np.einsum('fij,pj->fpi', Rm, active_local[oname]) + opos[:, None, :]
        pts = np.concatenate([pts, swept.reshape(-1, 3)], axis=0)

    # Build placement units: a whole group when every one of its members is
    # background for this row (so they must move together, rigidly); otherwise each
    # remaining background object is its own singleton unit (this also covers a
    # group member that is background while its partner(s) are the row's own active
    # object -- e.g. egg_ellipsoid alone in action 002 -- where there is no shared
    # rigid position to preserve).
    units: list[tuple[str, list[str]]] = []
    handled: set[str] = set()
    for gname, members in GROUPS.items():
        if all(m in bg_names for m in members):
            units.append((gname, members))
            handled.update(members)
    for n in bg_names:
        if n not in handled:
            units.append((n, [n]))

    deficient = []
    for uname, members in units:
        dmin_unit = min(float(trees_canon[m].query(pts, k=1)[0].min()) for m in members)
        if dmin_unit < HAND_TARGET:
            deficient.append((dmin_unit, uname, members))

    result = {'action': action, 'uuid': uuid, 'offsets': {}, 'unresolved': []}
    if not deficient:
        return result
    deficient.sort(key=lambda x: x[0])

    resolved_pos = {obj: canon[obj].copy() for obj in bg_names}
    for orig_dmin, uname, members in deficient:
        others = [(on, ms) for on, ms in units if on != uname]
        found = None
        for mag in MAGS:
            for dvec in dirs:
                delta = dvec * mag
                # every member of this unit must individually clear the hand/active sweep
                ok_hand = True
                for m in members:
                    shifted = pts.copy()
                    shifted[:, 0] -= delta[0]
                    shifted[:, 1] -= delta[1]
                    dmin_h, _ = trees_canon[m].query(shifted, k=1)
                    if float(dmin_h.min()) < HAND_TARGET:
                        ok_hand = False
                        break
                if not ok_hand:
                    continue
                # every member must also clear every OTHER unit's (already resolved) members
                ok = True
                for m in members:
                    cand = canon[m][:2] + delta
                    for on, oms in others:
                        for om in oms:
                            center_gap = float(np.linalg.norm(cand - resolved_pos[om][:2]))
                            if center_gap >= radii[m] + radii[om] + OBJ_TARGET:
                                continue
                            om_delta = resolved_pos[om][:2] - canon[om][:2]
                            cand_verts = verts_local[m] + np.array([cand[0], cand[1], canon[m][2]])
                            shifted_verts = cand_verts.copy()
                            shifted_verts[:, 0] -= om_delta[0]
                            shifted_verts[:, 1] -= om_delta[1]
                            dmin_o, _ = trees_canon[om].query(shifted_verts, k=1)
                            if float(dmin_o.min()) < OBJ_TARGET:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                if ok:
                    found = (mag, dvec)
                    break
            if found:
                break
        if found is None:
            result['unresolved'].append([uname, orig_dmin])
            continue
        mag, dvec = found
        delta = dvec * mag
        for m in members:
            resolved_pos[m][:2] = canon[m][:2] + delta
            result['offsets'][m] = [round(float(delta[0]), 4), round(float(delta[1]), 4)]

    return result


def _row_generator(lance_path: str, version: int):
    """Stream rows via small Arrow batches so the (slow, single-threaded)
    Arrow-to-Python conversion overlaps with the worker pool instead of
    blocking everything up front as one giant to_pylist() call would."""
    import lance

    d = lance.dataset(lance_path, version=version)
    for batch in d.to_batches(batch_size=32, columns=['hands', 'objects', 'trajectory_metadata', 'index']):
        yield from batch.to_pylist()


def main() -> None:
    t_start = time.time()
    import lance

    total_rows = lance.dataset(LANCE, version=VERSION).count_rows()
    print(f'dataset has {total_rows} rows, streaming in batches of 32', flush=True)

    n_workers = max(1, min(26, mp.cpu_count() - 2))
    print(f'using {n_workers} worker processes', flush=True)

    results = []
    t0 = time.time()
    with mp.Pool(n_workers, initializer=_worker_init) as pool:
        for i, res in enumerate(pool.imap_unordered(_solve_row, _row_generator(LANCE, VERSION), chunksize=8)):
            results.append(res)
            if (i + 1) % 200 == 0:
                print(f'... {i+1}/{total_rows} rows solved, {time.time()-t0:.1f}s elapsed', flush=True)
    print(f'TOTAL rows solved: {len(results)}, elapsed {time.time()-t0:.1f}s', flush=True)

    by_action = defaultdict(lambda: {'rows': 0, 'need': 0, 'pairs': 0, 'unresolved': [], 'mags': []})
    per_row_offsets: dict[str, dict[str, list[float]]] = {}
    for res in results:
        stat = by_action[res['action']]
        stat['rows'] += 1
        if res['offsets']:
            per_row_offsets[res['uuid']] = res['offsets']
            stat['need'] += 1
            stat['pairs'] += len(res['offsets'])
            for delta in res['offsets'].values():
                stat['mags'].append(float(np.linalg.norm(delta)))
        for obj, dmin in res['unresolved']:
            stat['unresolved'].append([res['uuid'], obj, dmin])

    report = {}
    for action, stat in sorted(by_action.items()):
        report[action] = {
            'rows': stat['rows'],
            'rows_needing_offset': stat['need'],
            'object_pairs_offset': stat['pairs'],
            'unresolved_pairs': len(stat['unresolved']),
            'unresolved_detail': stat['unresolved'],
            'offset_mag_cm': {
                'min': round(min(stat['mags']) * 100, 2) if stat['mags'] else None,
                'median': round(float(np.median(stat['mags'])) * 100, 2) if stat['mags'] else None,
                'max': round(max(stat['mags']) * 100, 2) if stat['mags'] else None,
            },
        }
        print(action, report[action], flush=True)

    out = {
        'lance': LANCE,
        'lance_version': VERSION,
        'hand_target_m': HAND_TARGET,
        'object_target_m': OBJ_TARGET,
        'canonical': CANON,
        'report_by_action': report,
        'per_row_offsets': per_row_offsets,
    }
    out_path = Path(OUTPUT); out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1))
    print('saved', out_path, flush=True)
    print(f'grand total elapsed {time.time()-t_start:.1f}s', flush=True)


if __name__ == '__main__':
    main()
