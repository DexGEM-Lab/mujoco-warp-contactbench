"""Verify cached source geometry using CPU collision contacts, never forces."""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
from tools.audit_action005_direct120 import ROWS, FINGERS, DATA, ASSETS, MANIFEST


def valid_witness(distance, segment):
    segment = np.asarray(segment)
    length = np.linalg.norm(segment[3:]-segment[:3])
    return bool(np.isfinite(segment).all() and np.isfinite(distance) and length > 1e-10 and abs(length-abs(distance)) < 1e-7)


def longest(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return int(np.max(np.flatnonzero(edges == -1)-np.flatnonzero(edges == 1), initial=0))


def main():
    import mujoco as mj
    from tools.u1_raw_source import load_raw_source
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--audit', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    summaries = []
    for number in ROWS:
        source = json.loads((args.audit/f'row{number}.json').read_text()); archive = np.load(args.audit/f'row{number}_geometry.npz')
        inp, m, _, teacher, _ = load_raw_source(DATA, 4, number, ASSETS, MANIFEST)
        assert np.array_equal(archive['qpos'], teacher['qpos'])
        q = archive['qpos']; d = mj.MjData(m); ob = m.body('mayonnaisebottle').id; hb = m.body('palm').id
        geoms = [i for i in range(m.ngeom) if m.body(int(m.geom_bodyid[i])).name.split('_')[0] in (*FINGERS, 'palm')]
        objects = [i for i in range(m.ngeom) if m.geom_bodyid[i] == ob]
        contact_gaps = np.full((len(q), 5), 2.); near_gaps = contact_gaps.copy(); invalid = np.zeros((len(q), 5), dtype=int)
        depths = np.zeros(len(q)); sample = []; start, end = source['metrics']['lift_interval']; peak = source['metrics']['peak_frame']
        with gzip.open(args.output/f'row{number}_contacts.jsonl.gz', 'wt') as f:
            for k, pose in enumerate(q):
                d.qpos[:] = pose; mj.mj_forward(m, d); rot = d.xmat[ob].reshape(3, 3); contacts = []
                for c in d.contact:
                    g1, g2 = map(int, c.geom)
                    hand = g1 if g1 in geoms and g2 in objects else g2 if g2 in geoms and g1 in objects else None
                    if hand is None: continue
                    link = m.body(int(m.geom_bodyid[hand])).name; finger = link.split('_')[0]
                    sign = 1 if g1 == hand else -1
                    if finger in FINGERS: contact_gaps[k, FINGERS.index(finger)] = min(contact_gaps[k, FINGERS.index(finger)], c.dist)
                    depths[k] = max(depths[k], -c.dist)
                    contacts.append(dict(geom_ids=[g1,g2], geom_names=[m.geom(g1).name,m.geom(g2).name], link=link, finger=finger,
                        distance_m=float(c.dist), position=c.pos.tolist(), frame=c.frame.reshape(3,3).tolist(),
                        object_local_point=(rot.T@(c.pos-d.xpos[ob])).tolist(), object_local_normal=(rot.T@(sign*c.frame[:3])).tolist()))
                for g in geoms:
                    finger = m.body(int(m.geom_bodyid[g])).name.split('_')[0]
                    if finger not in FINGERS: continue
                    fi = FINGERS.index(finger)
                    for o in objects:
                        segment = np.zeros(6); distance = mj.mj_geomDistance(m,d,g,o,2.,segment)
                        if valid_witness(distance, segment): near_gaps[k,fi] = min(near_gaps[k,fi],distance)
                        else: invalid[k,fi] += 1
                row = dict(frame=k, contacts=contacts, valid_witness_min_gap_mm=(near_gaps[k]*1000).tolist(), invalid_witness_count=invalid[k].tolist())
                f.write(json.dumps(row)+'\n')
                if k in (start, peak, end): sample.append(row)
        air = slice(start,end+1); touches = contact_gaps <= 0; five = np.all(touches,axis=1); plausible = five & (depths <= .002)
        deep = source['metrics']['deep_interval']; deepmask = np.zeros(len(q),bool)
        if deep: deepmask[deep[0]:deep[1]+1] = True
        summary = dict(row=number, uuid=source['provenance']['index']['uuid'], metrics=source['metrics'],
            geometry_semantics='CPU collision at source qpos; contact force and bearing order unknown',
            first_collision_frame={f:int(np.flatnonzero(touches[:,i])[0]) if touches[:,i].any() else None for i,f in enumerate(FINGERS)},
            airborne_touch_fraction=dict(zip(FINGERS,np.mean(touches[air],axis=0).tolist())),
            airborne_five_contact_fraction=float(np.mean(five[air])), airborne_plausible_five_fraction=float(np.mean(plausible[air])),
            longest_plausible_five_frames=longest(plausible[air]), airborne_max_penetration_mm=float(depths[air].max()*1000),
            deep_max_penetration_mm=float(depths[deepmask].max()*1000) if deep else None,
            deep_five_contact_fraction=float(np.mean(five[deepmask])) if deep else None,
            valid_witness_airborne_median_gap_mm=(np.median(near_gaps[air],axis=0)*1000).tolist(),
            invalid_witness_count=int(invalid.sum()), snapshots=sample,
            relative_translation_cm=source['geometry']['airborne_relative_translation_max_cm'],
            relative_rotation_deg=source['geometry']['airborne_relative_rotation_max_deg'])
        summaries.append(summary); print('VERIFIED',number,summary['airborne_plausible_five_fraction'],summary['airborne_max_penetration_mm'],summary['valid_witness_airborne_median_gap_mm'],flush=True)
    (args.output/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')

if __name__ == '__main__': main()
