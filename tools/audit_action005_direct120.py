"""Read-only, every-frame source-state geometry audit for the ten native 005 captures.

No integration or targets are generated here. Distances and contact order are
geometric, not evidence of bearing force. Output directories must be new.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R

ROWS = (62, 69, 124, 125, 126, 127, 128, 129, 130, 131)
FINGERS = ('thumb', 'index', 'middle', 'ring', 'pinky')
DATA = Path('/mnt/nas-222-project/mocap_v2/lance_datasets/dexgem_vla_demo_cma2lance_20260915_030735.lance')
ASSETS = Path('/home/jay/dexrobot/FromSSH/manoRL_mujoco/assets/dexstream_digital_assets')
MANIFEST = Path('/home/jay/dexrobot/FromSSH/manoRL_mujoco/outputs/cheyingtong120_start_augmentation_20260917/seeds/asset_manifest.json')


def motion_metrics(pos, rotations, bowl, timestamps):
    tilt = np.rad2deg(np.arccos(np.clip(rotations.as_matrix()[:, 2, 2], -1, 1)))
    peak = int(np.argmax(tilt))
    moved = np.flatnonzero(np.linalg.norm(pos-pos[0], axis=1) > .008)
    remaining = np.flatnonzero(np.linalg.norm(pos-pos[-1], axis=1) > .008)
    lift = np.flatnonzero(pos[:, 2]-pos[0, 2] > .02)
    high = np.flatnonzero(tilt > 110)
    upright = np.flatnonzero((np.arange(len(pos)) > peak) & (tilt < 15))
    dt = np.diff(timestamps)
    return dict(frames=len(pos), timestamp_strict=bool(np.all(dt > 0)),
                timestamp_dt_range_s=[float(dt.min()), float(dt.max())], mean_hz=float(1/dt.mean()),
                max_position_step_mm=float(np.linalg.norm(np.diff(pos, axis=0), axis=1).max()*1000),
                max_rotation_step_deg=float(np.rad2deg((rotations[:-1].inv()*rotations[1:]).magnitude()).max()),
                movement_interval=[int(moved[0]), int(remaining[-1]+1)] if len(moved) and len(remaining) else None,
                lift_interval=[int(lift[0]), int(lift[-1])] if len(lift) else None,
                max_lift_cm=float((pos[:, 2]-pos[0, 2]).max()*100),
                peak_frame=peak, max_tilt_deg=float(tilt[peak]),
                max_orientation_change_deg=float(np.rad2deg((rotations[0].inv()*rotations).magnitude()).max()),
                deep_interval=[int(high[0]), int(high[-1])] if len(high) else None,
                return_upright_frame=int(upright[0]) if len(upright) else None,
                final_tilt_deg=float(tilt[-1]), final_position_from_start_cm=float(np.linalg.norm(pos[-1]-pos[0])*100),
                final_height_delta_cm=float((pos[-1, 2]-pos[0, 2])*100),
                peak_bottle_bowl_xy_cm=float(np.linalg.norm(pos[peak, :2]-bowl[peak, :2])*100),
                peak_bottle_above_bowl_cm=float((pos[peak, 2]-bowl[peak, 2])*100))


def geometry(mj, model, qpos):
    d = mj.MjData(model); ob = model.body('mayonnaisebottle').id; hb = model.body('palm').id
    objects = [g for g in range(model.ngeom) if model.geom_bodyid[g] == ob]
    geoms = [g for g in range(model.ngeom) if model.body(int(model.geom_bodyid[g])).name.split('_')[0] in (*FINGERS, 'palm')]
    names = [model.geom(g).name for g in geoms]
    gaps = []; points = []; normals = []; hand_normals = []; relative_p = []; relative_r = []
    for q in qpos:
        d.qpos[:] = q; mj.mj_forward(model, d)
        rot = d.xmat[ob].reshape(3, 3); hr = d.xmat[hb].reshape(3, 3)
        gd, gp, gn, hn = [], [], [], []
        for g in geoms:
            options = []
            for o in objects:
                segment = np.zeros(6); distance = mj.mj_geomDistance(model, d, g, o, 2., segment)
                options.append((distance, segment))
            distance, segment = min(options, key=lambda x: x[0])
            normal = (segment[3:]-segment[:3]) / (distance if abs(distance) > 1e-12 else 1e-12)
            normal /= max(np.linalg.norm(normal), 1e-12)
            gd.append(distance); gp.append(rot.T@(segment[3:]-d.xpos[ob])); gn.append(rot.T@normal)
            hn.append(d.xmat[int(model.geom_bodyid[g])].reshape(3, 3).T@normal)
        gaps.append(gd); points.append(gp); normals.append(gn); hand_normals.append(hn)
        relative_p.append(hr.T@(d.xpos[ob]-d.xpos[hb])); relative_r.append(hr.T@rot)
    return names, dict(gap_m=np.asarray(gaps), object_points_m=np.asarray(points), object_normals=np.asarray(normals),
                       hand_local_normals=np.asarray(hand_normals), relative_position_m=np.asarray(relative_p), relative_rotation=np.asarray(relative_r))


def summarize_geometry(names, arrays, metrics):
    gaps = arrays['gap_m']; points = arrays['object_points_m']; normals = arrays['object_normals']
    distal = [names.index(f+'_ip_collision' if f == 'thumb' else f+'_dip_collision') for f in FINGERS]
    groups = [[i for i, n in enumerate(names) if n.startswith(f+'_')] for f in FINGERS]
    finger_gaps = np.stack([gaps[:, ids].min(axis=1) for ids in groups], axis=1)
    start, end = metrics['lift_interval'] or [0, len(gaps)-1]
    anchor = start
    rp = arrays['relative_position_m']; rr = R.from_matrix(arrays['relative_rotation'])
    selected = sorted(set([0, max(0, start-24), start, (start+metrics['peak_frame'])//2, metrics['peak_frame'], end, len(gaps)-1]))
    snapshots = []
    for k in selected:
        ids = [group[int(np.argmin(gaps[k, group]))] for group in groups]
        pp = points[k, ids]; nn = normals[k, ids]
        snapshots.append(dict(frame=k, closest_links=[names[i] for i in ids], closest_gaps_mm=(gaps[k, ids]*1000).tolist(),
                              distal_gaps_mm=(gaps[k, distal]*1000).tolist(), object_points_mm=(pp*1000).tolist(),
                              object_normals=nn.tolist(), thumb_opposition_cosines=(-nn[1:]@nn[0]).tolist(),
                              nonthumb_axial_z_mm=(pp[1:, 2]*1000).tolist(),
                              nonthumb_order_descending=bool(np.all(np.diff(pp[1:, 2]) < 0)),
                              index_pinky_span_mm=float((pp[1, 2]-pp[4, 2])*1000),
                              max_hand_penetration_mm=float(max(0, -gaps[k].min())*1000)))
    return dict(evidence_kind='Provisional CPU source-qpos distance geometry; collision verification required; zero-distance witnesses may be invalid; no source force claim',
                unverified_distance_touch_frame={f: int(np.flatnonzero(finger_gaps[:, i] <= 0)[0]) if np.any(finger_gaps[:, i] <= 0) else None for i, f in enumerate(FINGERS)},
                airborne_four_nonthumb_near_fraction=float(np.mean(np.all(finger_gaps[start:end+1, 1:] < .003, axis=1))),
                airborne_thumb_near_fraction=float(np.mean(finger_gaps[start:end+1, 0] < .003)),
                airborne_distal_median_gap_mm=(1000*np.median(gaps[start:end+1, distal], axis=0)).tolist(),
                max_hand_penetration_mm=float(max(0, -gaps.min())*1000),
                airborne_relative_translation_max_cm=float(np.linalg.norm(rp[start:end+1]-rp[anchor], axis=1).max()*100),
                airborne_relative_rotation_max_deg=float(np.rad2deg((rr[anchor].inv()*rr[start:end+1]).magnitude()).max()),
                relative_position_max_step_mm=float(np.linalg.norm(np.diff(rp, axis=0), axis=1).max()*1000),
                final_min_hand_gap_mm=float(gaps[-1].min()*1000), snapshots=snapshots)


def main():
    import lance
    import mujoco as mj
    from tools.u1_raw_source import load_raw_source
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    ds = lance.dataset(str(DATA), version=4)
    found = tuple(i for i, r in enumerate(ds.to_table(columns=['index']).to_pylist()) if r['index']['gesture'].startswith('005') and not r['index']['is_generated'])
    assert found == ROWS, found
    reports = []
    for number in ROWS:
        row = ds.take([number]).to_pylist()[0]
        inp, m, base, teacher, provenance = load_raw_source(DATA, 4, number, ASSETS, MANIFEST)
        q = teacher['qpos']; adr = int(m.joint('mayonnaisebottle_free').qposadr[0]); ba = int(m.joint('bowl_free').qposadr[0])
        metrics = motion_metrics(q[:, adr:adr+3], R.from_quat(q[:, adr+3:adr+7], scalar_first=True), q[:, ba:ba+3], np.asarray(row['timestamp']))
        names, arrays = geometry(mj, m, q)
        report = dict(row=number, provenance=provenance, source_contact_is_null=row['contact'] is None,
                      hand_names=row['trajectory_metadata']['hand_names'], right_slot=row['trajectory_metadata']['hand_names'].index('right'),
                      metadata_hz=row['trajectory_metadata']['data_fps'], declared_movement=row['trajectory_metadata']['trajectory_info'],
                      metrics=metrics, geometry=summarize_geometry(names, arrays, metrics),
                      max_wrist_position_step_mm=float(np.linalg.norm(np.diff(base[:, :3], axis=0), axis=1).max()*1000),
                      max_finger_step_rad=float(np.abs(np.diff(base[:, 6:], axis=0)).max()))
        np.savez_compressed(args.output/f'row{number}_geometry.npz', qpos=q, geom_names=names, **arrays)
        (args.output/f'row{number}.json').write_text(json.dumps(report, indent=2)+'\n'); reports.append(report)
        print('AUDIT', number, json.dumps(metrics), json.dumps({k:v for k,v in report['geometry'].items() if k != 'snapshots'}), flush=True)
    (args.output/'audit.json').write_text(json.dumps(dict(dataset=str(DATA), version=4, rows=reports,
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2)+'\n')

if __name__ == '__main__': main()
