#!/usr/bin/env python3
"""Measure saved first-pass bowl-hold replays from actual free-body qpos."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import lance
import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--run-root',type=Path,required=True)
    return p.parse_args()


def hand_contact_metrics(model,qpos,active_adr):
    data=mj.MjData(model)
    # Bowl collision pieces are child bodies of bowl_free, so qpos-address
    # equality is insufficient. The explicit geom prefix is the stable asset
    # contract used to select active-object contacts.
    counts,supports=[],[]
    finger_bodies={'thumb_cmc','thumb_mcp','thumb_ip','index_mcp','index_pip','index_dip','middle_mcp','middle_pip','middle_dip','ring_mcp','ring_pip','ring_dip','pinky_mcp','pinky_pip','pinky_dip'}
    for pose in qpos:
        data.qpos[:]=pose;mj.mj_forward(model,data);fingers,scene=set(),set()
        for contact in data.contact[:data.ncon]:
            g1,g2=int(contact.geom1),int(contact.geom2);n1,n2=model.geom(g1).name or '',model.geom(g2).name or ''
            other=g2 if n1.startswith('bowl_') else g1 if n2.startswith('bowl_') else None
            if other is None: continue
            body=model.body(model.geom_bodyid[other]).name or ''
            if body in finger_bodies: fingers.add(body.split('_')[0])
            else: scene.add(model.geom(other).name or body)
        counts.append(len(fingers));supports.append(sorted(scene))
    return np.asarray(counts),supports


def base_hand(dataset,manifest):
    from sim.manorl.trajectory import trajectory_from_lance_row,resample_reference_trajectory
    source=manifest['source'];row=dataset.take([int(source['row'])],columns=['index','trajectory_metadata','timestamp','hands','objects']).to_pylist()[0]
    return resample_reference_trajectory(trajectory_from_lance_row(row,dataset.version,row_index=source['row'],hand_side='right',pre_padding=180,post_padding=int(manifest['post_padding'])),reference_fps=100,control_fps=100)


def analyze_one(dataset,directory):
    manifest=json.loads((directory/'manifest.json').read_text());patch=manifest['patch'];saved=np.load(directory/'trajectory.npz',allow_pickle=False)
    qpos=saved['qpos'];names=[str(n) for n in saved['scene_object_names']];active_index=names.index(patch['active_object']);active_adr=int(saved['scene_object_qpos_addresses'][active_index])
    pos=saved['scene_object_pos'][:,active_index];rot=Rotation.from_rotvec(saved['scene_object_rot_aa'][:,active_index]);lift=pos[:,2]-float(saved['initial_qpos'][active_adr+2])
    movement_end=int(manifest['movement_steps'][1]);hold=np.arange(len(qpos))>=movement_end;prefix=int(patch['transfer']['timing']['prefix_preserved_through_step']);indices=saved['reference_indices']
    base=base_hand(dataset,manifest);mask=indices<=prefix;prefix_error=float(np.max(np.abs(saved['reference_hand'][mask]-base.q_ref[indices][mask])))
    from sim.manorl.assets import compile_unified_model
    _,model=compile_unified_model(object_types=tuple(sorted(names)),object_collisions=True,physics_timestep=.0025)
    contacts,supports=hand_contact_metrics(model,qpos,active_adr);up=rot.apply(np.array([0.,0.,1.]));tilt=np.degrees(np.arccos(np.clip(up[:,2],-1.,1.)));last_second=pos[-1]-pos[max(0,len(pos)-101)]
    source=manifest['source'];result={'row':int(source['row']),'uuid':source['uuid'],'complete':bool(json.loads((directory/'result.json').read_text())['complete']),'frames':int(len(qpos)),'maximum_lift_m':float(lift.max()),'held_lift_minimum_m':float(lift[hold].min()),'held_lift_final_m':float(lift[-1]),'held_final_tilt_deg':float(tilt[-1]),'held_max_tilt_deg':float(tilt[hold].max()),'last_second_drift_m':last_second.tolist(),'last_second_drift_norm_m':float(np.linalg.norm(last_second)),'actual_contact_recomputed_from_saved_qpos':{'acquisition_max_distinct_finger_rays':int(contacts[:movement_end].max()),'hold_min_distinct_finger_rays':int(contacts[hold].min()),'hold_median_distinct_finger_rays':float(np.median(contacts[hold])),'hold_has_support_contact':bool(any(supports[i] for i in np.flatnonzero(hold)))},'prefix_preservation':{'through_reference_step':prefix,'max_abs_hand_target_error':prefix_error,'passed_exactly':bool(prefix_error<=1e-12)},'edit_interval_reference_steps':[prefix+1,int(patch['transfer']['timing']['late_leveling_steps'][1])]}
    (directory/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    a=parse_args();dataset=lance.dataset(a.dataset,version=5);summary=[analyze_one(dataset,d) for d in sorted(a.run_root.glob('row*')) if (d/'trajectory.npz').exists()];(a.run_root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
