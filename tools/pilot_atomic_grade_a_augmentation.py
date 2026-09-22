#!/usr/bin/env python3
"""27 fixed action002 current-A parents: approach generation plus independent replay."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import uuid

import numpy as np
from scipy.spatial.transform import Rotation

from tools.replay_atomic_benchmark_pilot import (
    configure_modules, decode_row, run_physics, sha256, validate_gpu_binding,
)
from sim.manorl.approach_prefix import _discrete_c1_prefix

CONTRACT = "manorl.current-grade-a-action002-prefix-pilot.v1"
PREFIX = 120
PARENT_RESULT_SHA = "2ef18456c7ba2179bea6df34c7092e8f874a8e22577c03c1375048d37adda9b8"


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def array_sha(x):
    x = np.ascontiguousarray(x)
    return hashlib.sha256(x.dtype.str.encode() + str(x.shape).encode() + x.tobytes()).hexdigest()


def plan_deltas():
    # One fixed candidate per parent: nine at each radius, no reserves.
    result = []
    for i in range(27):
        radius = (0.05, 0.10, 0.15)[i % 3]
        sector = i % 8
        azimuth = sector * np.pi / 4
        elevation = np.pi / 6
        rotation = (i // 3 + 2*(i % 3)) % 6
        delta = np.zeros(6)
        delta[:3] = radius * np.array([np.cos(elevation)*np.cos(azimuth),
                                      np.cos(elevation)*np.sin(azimuth), np.sin(elevation)])
        delta[3 + rotation // 2] = (1 if rotation % 2 == 0 else -1) * np.pi / 6
        result.append({"radius_m": radius, "sector": sector, "rotation_axis": rotation//2,
                       "rotation_sign": 1 if rotation % 2 == 0 else -1, "delta": delta.tolist()})
    return result


def prefix_target(parent_target, physical_teacher, delta):
    base = np.asarray(parent_target, dtype=np.float32)
    teacher = np.asarray(physical_teacher, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    if base.ndim != 2 or base.shape[1] != 28 or teacher.shape != base.shape:
        raise ValueError("parent control and physical teacher must be [N,28]")
    if not np.isfinite(base).all() or not np.isfinite(teacher).all() or delta.shape != (6,):
        raise ValueError("nonfinite parent or invalid delta")
    np.testing.assert_array_equal(base[0], teacher[0].astype(np.float32))
    start = teacher[0].copy()
    start[:6] += delta
    prefix = np.repeat(base[:1].astype(np.float64), PREFIX, axis=0)
    prefix[:, :3] = _discrete_c1_prefix(start[:3], teacher[:2, :3], frames=PREFIX,
                                       dt=1/120, vertical_arc_height=0.04)
    prefix[:, 3:6] = _discrete_c1_prefix(start[3:6], teacher[:2, 3:6], frames=PREFIX, dt=1/120)
    child = np.concatenate([prefix.astype(np.float32), base])
    np.testing.assert_array_equal(child[PREFIX:], base)
    np.testing.assert_array_equal(child[:PREFIX, 6:], np.repeat(base[:1, 6:], PREFIX, axis=0))
    reference_residual = (child[PREFIX, :6].astype(float) - child[PREFIX-1, :6]
                          - (teacher[1, :6] - teacher[0, :6]))
    control_residual = (child[PREFIX, :6].astype(float) - child[PREFIX-1, :6]
                        - (base[1, :6].astype(float) - base[0, :6]))
    if np.max(np.abs(reference_residual)) > 3e-7:
        raise ValueError("physical-reference splice mismatch")
    return child, start.astype(np.float32), {
        "reference_splice_residual": reference_residual.tolist(),
        "control_splice_residual": control_residual.tolist(),
        "teacher_source": "current_parent_recorded_physical_qpos",
        "reference_C1": True, "control_C1_exact": bool(np.all(control_residual == 0)),
        "parent_suffix_sha256": array_sha(base), "child_suffix_sha256": array_sha(child[PREFIX:]),
    }


def hold_extend(x):
    return np.concatenate([np.repeat(x[:1], PREFIX, axis=0), x])


def prepare(a):
    import lance
    if a.output.exists():
        raise FileExistsError(a.output)
    if sha256(a.parent_summary) != PARENT_RESULT_SHA:
        raise ValueError("parent Grade result changed")
    parents = []
    for root in a.grade_roots:
        for p in root.glob("shard*/action002/rows/row*.json"):
            row = json.loads(p.read_text())
            if row["grade"] == "A":
                parents.append((row, p))
    parents.sort(key=lambda x: x[0]["row_index"])
    if len(parents) != 27 or len({p[0]["uuid"] for p in parents}) != 27:
        raise ValueError("expected27 unique current action002 A parents")
    ds = lance.dataset(str(a.dataset), version=1)
    if ds.count_rows() != 534:
        raise ValueError("parent dataset changed")
    columns = [n for n in ds.schema.names if n != "contact"]
    a.output.mkdir(parents=True)
    candidates = []
    for ordinal, ((grade, source), sampling) in enumerate(zip(parents, plan_deltas(), strict=True)):
        row = ds.take([grade["row_index"]], columns=columns).to_pylist()[0]
        if row["index"]["uuid"] != grade["uuid"]:
            raise ValueError("parent row identity changed")
        item = decode_row(grade["row_index"], row)
        base = np.asarray(row["hands"][0]["urdf_dof_target"], dtype=np.float32)
        teacher = np.asarray(row["hands"][0]["urdf_dof"], dtype=np.float32)
        target, initial, splice = prefix_target(base, teacher, sampling["delta"])
        signature = {"contract": CONTRACT, "parent": grade["uuid"], "sampling": sampling,
                     "target_sha": array_sha(target), "physics": "u1-table/headroom"}
        child_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, json.dumps(signature, sort_keys=True)))
        folder = a.output / f"candidate{ordinal:02d}"
        folder.mkdir()
        # Original row excludes only contacts: new contacts must come from physics.
        dump(folder / "parent.json", row)
        np.save(folder / "target.npy", target, allow_pickle=False)
        np.save(folder / "initial_hand.npy", initial, allow_pickle=False)
        record = {"ordinal": ordinal, "uuid": child_uuid, "parent_uuid": grade["uuid"],
                  "parent_row": grade["row_index"], "parent_frames": item["frames"],
                  "frames": len(target), "source_grade_error_m": grade["max_target_position_error_m"],
                  "grade_record": str(source), "grade_record_sha256": sha256(source),
                  "sampling": sampling, "splice": splice,
                  "files": {name: sha256(folder/name) for name in ("parent.json", "target.npy", "initial_hand.npy")}}
        dump(folder / "candidate.json", record)
        candidates.append(record)
    batches = []
    for start in range(0, 27, 5):
        real = list(range(start, min(start+5, 27)))
        padded = real + [real[i % len(real)] for i in range(5-len(real))]
        batches.append({"batch": len(batches), "real_count": len(real), "ordinals": padded})
    manifest = {"contract": CONTRACT, "parent_dataset": str(a.dataset), "dataset_version": 1,
                "parent_summary_sha256": PARENT_RESULT_SHA, "candidate_count": 27,
                "prefix_frames": PREFIX, "reserves": 0, "candidates": candidates, "batches": batches,
                "acceptance": {"prefix_hand_scene_normal_force_max_N": 0.2,
                               "parent_position_error_max_m_exclusive": 0.03,
                               "repeat_position_error_max_m_exclusive": 0.03,
                               "post_prefix_target_contact_frames_min": 101,
                               "final_parent_rotation_max_deg": 35,
                               "scope": "parent-fidelity/repeatability pilot, not an exhaustive stirring classifier"}}
    dump(a.output / "plan.json", manifest)
    print(json.dumps({"plan": str(a.output/"plan.json"), "candidates": 27, "batches": batches,
                      "plan_sha256": sha256(a.output/"plan.json")}))


def load_child(root, ordinal):
    folder = root/f"candidate{ordinal:02d}"
    meta = json.loads((folder/"candidate.json").read_text())
    for name, expected in meta["files"].items():
        if sha256(folder/name) != expected:
            raise ValueError("frozen candidate changed: " + name)
    row = json.loads((folder/"parent.json").read_text())
    item = decode_row(meta["parent_row"], row)
    item["hand_recorded"] = hold_extend(item["hand_recorded"])
    item["hand_recorded"][0] = np.load(folder/"initial_hand.npy", allow_pickle=False)
    item["commands"] = np.load(folder/"target.npy", allow_pickle=False)
    item["object_recorded_pos"] = {k: hold_extend(v) for k,v in item["object_recorded_pos"].items()}
    item["object_recorded_quat_wxyz"] = {k: hold_extend(v) for k,v in item["object_recorded_quat_wxyz"].items()}
    item.update(frames=meta["frames"], uuid=meta["uuid"], seed_uuid=meta["parent_uuid"])
    item["movement"] = {**item["movement"], "start_frame": item["movement"]["start_frame"]+PREFIX,
                        "end_frame": item["movement"]["end_frame"]+PREFIX}
    return meta, row, item


def native_pair_forces(buffers):
    n = buffers.count
    world = buffers.world[:n]
    address = buffers.addresses[:n]
    if np.any(world < 0) or np.any(world >= len(buffers.nefc)):
        raise ValueError("invalid native contact world")
    valid = (buffers.dimension[:n] == 3) & np.all(address >= 0, axis=1)
    valid &= np.all(address < buffers.nefc[world, None], axis=1)
    pyramid = np.zeros((n, 4))
    pyramid[valid] = buffers.constraint_force[world[valid, None], address[valid]]
    return pyramid, valid


class Observer:
    """Read-only solved-state/contact recording; never forwards or steps physics."""
    def __init__(self, items):
        self.items = items
        self.states = [[] for _ in items]
        self.contacts = [[] for _ in items]
        self.joints = [[] for _ in items]
        self.raw = [[] for _ in items]
        self.prefix_max = np.zeros(len(items))
        self.target_frames = np.zeros(len(items), dtype=int)
        self.layout = None

    def _forces(self, e):
        buffers = e.producer.materialize_contact_buffers(e.data, len(self.items))
        pyramid, valid = native_pair_forces(buffers)
        hand = np.asarray(e.producer.keypoint_geom_ids_by_side['right'], dtype=int)
        ishand = np.isin(buffers.geom[:buffers.count], hand)
        hand_scene = ishand[:,0] ^ ishand[:,1]
        if np.any(hand_scene & ~valid):
            raise ValueError("hand-scene force has no valid native solver address")
        maxima = np.zeros(len(self.items))
        for w in range(len(self.items)):
            values = np.abs(pyramid.sum(axis=1)[hand_scene & (buffers.world[:buffers.count] == w)])
            maxima[w] = values.max(initial=0)
        return buffers, pyramid, valid, maxima

    def substep(self, e, frame, substep):
        if frame < PREFIX:
            self.prefix_max = np.maximum(self.prefix_max, self._forces(e)[3])

    def frame(self, e, state, frame):
        from sim.manorl.lance_v2 import corrected_contact_frames
        m = e.model
        names = tuple(e._unified_object_types)
        if self.layout is None:
            self.layout = {"objects": list(names), "addresses": {n:int(m.joint(n+'_free').qposadr[0]) for n in names},
                           "geom_names": [m.geom(i).name for i in range(m.ngeom)],
                           "geom_bodyid": np.asarray(m.geom_bodyid).tolist()}
            self.fk = e.mujoco.MjData(m)
            from sim.manorl.environment import _FINGERTIP_NAMES, _FINGERTIP_LOCAL_OFFSETS
            self.tip_bodies = [int(m.body(n).id) for n in _FINGERTIP_NAMES]
            self.tip_offsets = _FINGERTIP_LOCAL_OFFSETS.copy()
            initial = np.stack([x['hand_recorded'][0] for x in self.items])
            np.testing.assert_allclose(state.qpos[:,:28], initial, atol=1e-7, rtol=0)
            if np.any(initial[:,:6] < m.jnt_range[:6,0]) or np.any(initial[:,:6] > m.jnt_range[:6,1]):
                raise ValueError('perturbed initial wrist outside compiled limits')
        buffers,pyramid,valid,maxima = self._forces(e)
        if frame < PREFIX:
            self.prefix_max = np.maximum(self.prefix_max, maxima)
        compact = [[] for _ in self.items]
        kp = e.producer.keypoint_geom_ids_by_side['right']
        wrist = int(e.producer.keypoint_body_ids[0])
        for name in names:
            body = int(m.body(name).id)
            geoms = np.flatnonzero((m.geom_bodyid == body) & ((m.geom_contype != 0) | (m.geom_conaffinity != 0)))
            decoded = corrected_contact_frames(buffers=buffers, state=state, model=m,
                keypoint_geom_ids=kp, object_geom_ids=geoms, object_body_id=body,
                wrist_body_id=wrist, object_name=name)
            for w in range(len(self.items)):
                compact[w].extend(decoded[w])
        control = np.asarray(e.data.ctrl)
        for w,item in enumerate(self.items):
            np.testing.assert_array_equal(control[w], item['commands'][min(frame,item['frames']-1)].astype(control.dtype))
            self.states[w].append((state.qpos[w].copy(),state.qvel[w].copy(),control[w].copy()))
            # FK mirror only: no CPU integration, no writes to the simulated world.
            self.fk.qpos[:] = state.qpos[w]
            e.mujoco.mj_kinematics(m, self.fk)
            tips = [self.fk.xpos[b] + self.fk.xmat[b].reshape(3, 3) @ offset
                    for b,offset in zip(self.tip_bodies,self.tip_offsets,strict=True)]
            self.joints[w].append(np.concatenate([self.fk.xpos[e.producer.keypoint_body_ids],tips]))
            self.contacts[w].append(compact[w])
            if frame >= PREFIX and any(
                c['object_name'] == item['target'] and any(
                    np.linalg.norm(pair['force_normal']) > .2 for pair in c['contact_pairs']
                ) for c in compact[w]
            ):
                self.target_frames[w] += 1
            ids = np.flatnonzero(buffers.world[:buffers.count] == w)
            self.raw[w].append({
                'geom': buffers.geom[ids].copy(), 'position': buffers.position[ids].copy(),
                'contact_frame': buffers.frame[ids].copy(), 'friction': buffers.friction[ids].copy(),
                'dimension': buffers.dimension[ids].copy(), 'efc_address': buffers.addresses[ids].copy(),
                'pyramid_force': pyramid[ids].copy(), 'solver_address_valid': valid[ids].copy(),
            })

    def save(self, folder, world):
        folder.mkdir()
        states = self.states[world]
        np.savez_compressed(folder/'physical_trace.npz', qpos=np.stack([x[0] for x in states]),
                            qvel=np.stack([x[1] for x in states]), ctrl=np.stack([x[2] for x in states]),
                            mano_joint_pos=np.stack(self.joints[world]))
        raw = self.raw[world]
        lengths = [len(x['geom']) for x in raw]
        np.savez_compressed(folder/'native_contacts.npz', frame_offsets=np.r_[0,np.cumsum(lengths)],
                            **{k:np.concatenate([x[k] for x in raw]) for k in raw[0]})
        dump(folder/'compact_contacts.json', self.contacts[world])
        dump(folder/'model_layout.json', self.layout)


def run(a):
    validate_gpu_binding(a.gpu)
    if a.output.exists():
        raise FileExistsError(a.output)
    plan = json.loads(a.plan.read_text())
    if plan['contract'] != CONTRACT:
        raise ValueError('pilot contract mismatch')
    batch = plan['batches'][a.batch]
    loaded = [load_child(a.plan.parent,i) for i in batch['ordinals']]
    items = [x[2] for x in loaded]
    _native,_visual,consumer,assets,contracts = configure_modules(a)
    observer = Observer(items)
    a.output.mkdir(parents=True)
    dump(a.output/'status.json', {'state':'running','pid':os.getpid(),'batch':a.batch,'pass':a.pass_name})
    outputs,physics = run_physics(dataset_path=a.dataset,dataset_version=1,decoded=items,
        consumer_visual=consumer,assets=assets,contracts=contracts,decorative_scene_spec=a.scene,
        physics_profile='u1-table',observer=observer)
    reports=[]
    for w,(meta,row,item) in enumerate(loaded):
        folder=a.output/f'world{w}'
        observer.save(folder,w)
        out=outputs[w];target=item['target']
        final_angle=float(np.degrees((Rotation.from_quat(item['object_recorded_quat_wxyz'][target][-1][[1,2,3,0]]).inv()*Rotation.from_quat(out['object_simulated_quat_wxyz'][target][-1][[1,2,3,0]])).magnitude()))
        report={'candidate':meta['ordinal'],'uuid':meta['uuid'],'parent_uuid':meta['parent_uuid'],
                'parent_row':meta['parent_row'],'world':w,'padding':w>=batch['real_count'],
                'prefix_max_normal_force_N':float(observer.prefix_max[w]),
                'post_prefix_target_contact_frames':int(observer.target_frames[w]),
                'parent_max_error_m':float(out['metrics']['max_target_position_error_m']),
                'parent_final_rotation_error_deg':final_angle,
                'controls_exact':True,'pid':os.getpid(),
                'files':{p.name:sha256(p) for p in folder.iterdir() if p.is_file()}}
        report['gates']={'prefix_force':report['prefix_max_normal_force_N']<=.2,
                         'parent_fidelity_A':report['parent_max_error_m']<.03,
                         'target_contact_duration':report['post_prefix_target_contact_frames']>=101,
                         'parent_final_rotation':final_angle<=35,'controls_exact':True}
        report['passed']=all(report['gates'].values())
        dump(folder/'report.json',report);reports.append(report)
    result={'state':'complete','contract':CONTRACT,'batch':a.batch,'pass':a.pass_name,
            'pid':os.getpid(),'plan_sha256':sha256(a.plan),'manorl_commit':a.manorl_commit,
            'physics':physics,'real_count':batch['real_count'],'reports':reports}
    dump(a.output/'manifest.json',result);dump(a.output/'status.json',{'state':'complete','pid':os.getpid()})
    print(json.dumps({'batch':a.batch,'pass':a.pass_name,'passed':sum(r['passed']for r in reports[:batch['real_count']]),
                      'reports':[{k:r[k] for k in ('candidate','parent_row','gates','parent_max_error_m','prefix_max_normal_force_N')}for r in reports[:batch['real_count']]]}),flush=True)


def collect(a):
    import lance
    import pyarrow as pa
    if a.asset_manifest is None or a.asset_root is None:
        raise ValueError('collect requires explicit --asset-manifest and --asset-root for native hand conversion')
    os.environ['MANORL_ASSET_MANIFEST'] = str(a.asset_manifest.resolve(strict=True))
    from sim.manorl import assets
    assets.DEXSTREAM_ROOT = a.asset_root.resolve(strict=True)
    from sim.manorl.mano_pose import right_urdf_trajectory_to_mano_48d
    if a.output.exists():raise FileExistsError(a.output)
    plan=json.loads(a.plan.read_text());candidate_rows=[];reports=[]
    for batch in plan['batches']:
        b=batch['batch'];root=a.run_roots[0] if b<4 else a.run_roots[1]
        first=root/f'batch{b}/generation';second=root/f'batch{b}/verification'
        for folder in (first,second):
            guard=json.loads(Path(str(folder)+'.log.guard.json').read_text())
            if guard['state']!='complete' or guard['overflow'] is not None:raise ValueError('guard failed')
        fm=json.loads((first/'manifest.json').read_text());sm=json.loads((second/'manifest.json').read_text())
        if fm['pid']<=0 or sm['pid']<=0 or fm['pid']==sm['pid']:raise ValueError('replays not independent')
        if fm['plan_sha256']!=sha256(a.plan) or sm['plan_sha256']!=sha256(a.plan):raise ValueError('plan mismatch')
        if fm['manorl_commit'] != sm['manorl_commit'] or fm['physics']['runtime'] != sm['physics']['runtime']:
            raise ValueError('independent replay physics identity differs')
        for w in range(batch['real_count']):
            ordinal=batch['ordinals'][w];meta,row,item=load_child(a.plan.parent,ordinal)
            fr,sr=fm['reports'][w],sm['reports'][w]
            if fr['uuid'] != meta['uuid'] or sr['uuid'] != meta['uuid']:
                raise ValueError('candidate identity differs across replay passes')
            for folder,report in ((first,fr),(second,sr)):
                for name,h in report['files'].items():
                    if sha256(folder/f'world{w}'/name)!=h:raise ValueError('changed replay artifact')
            layout=json.loads((first/f'world{w}/model_layout.json').read_text());adr=layout['addresses'][item['target']]
            with np.load(first/f'world{w}/physical_trace.npz')as z:q1=z['qpos'].copy();ctrl=z['ctrl'].copy()
            with np.load(second/f'world{w}/physical_trace.npz')as z:q2=z['qpos'].copy();ctrl2=z['ctrl'].copy()
            np.testing.assert_array_equal(ctrl,ctrl2);np.testing.assert_array_equal(ctrl,item['commands'])
            repeat=float(np.linalg.norm(q2[:,adr:adr+3]-q1[:,adr:adr+3],axis=1).max())
            passed=fr['passed'] and sr['passed'] and repeat<.03
            record={'candidate':ordinal,'uuid':meta['uuid'],'parent_row':meta['parent_row'],
                    'radius_m':meta['sampling']['radius_m'],'first':fr,'second':sr,
                    'repeat_max_error_m':repeat,'repeat_grade':'A'if repeat<.03 else'B'if repeat<.08 else'C',
                    'accepted':bool(passed),'generation':str(first/f'world{w}'),'verification':str(second/f'world{w}')}
            reports.append(record)
            new=copy.deepcopy(row);n=len(ctrl);h=new['hands'][0];qh=q1[:,:28]
            h['urdf_dof']=qh.tolist();h['urdf_dof_target']=ctrl.tolist();h['mano_global_pos']=qh[:,:3].tolist()
            h['mano_global_rot_aa']=Rotation.from_euler('XYZ',qh[:,3:6]).as_rotvec().tolist()
            h['mano_hand_pose']=right_urdf_trajectory_to_mano_48d(qh).tolist()
            # Joint positions are reconstructed only for accepted rows below from physical qpos.
            new['index'].update(uuid=meta['uuid'],seed_uuid=meta['parent_uuid'],is_generated=True)
            new['trajectory_metadata']['total_frames']=n
            new['trajectory_metadata']['trajectory_info']['object_move']=[item['movement']]
            new['timestamp']=(np.arange(n)/120).tolist()
            new['objects']=[{'pos':q1[:,layout['addresses'][name]:layout['addresses'][name]+3].tolist(),
                             'rot_aa':Rotation.from_quat(q1[:,layout['addresses'][name]+3:layout['addresses'][name]+7][:,[1,2,3,0]]).as_rotvec().tolist()}
                            for name in new['trajectory_metadata']['object_names']]
            new['contact']=json.loads((first/f'world{w}/compact_contacts.json').read_text())
            teacher=np.asarray(row['hands'][0]['urdf_dof']);target_i=row['trajectory_metadata']['object_names'].index(item['target'])
            new['reference']={'source_frame_index':[0]*PREFIX+list(range(meta['parent_frames'])),
                'hand_urdf_dof':hold_extend(teacher).tolist(),
                'object_pos':hold_extend(np.asarray(row['objects'][target_i]['pos'])).tolist(),
                'object_rot_aa':hold_extend(np.asarray(row['objects'][target_i]['rot_aa'])).tolist()}
            new['command_reference_index']=list(range(1,n));new['command_source_frame_index']=new['reference']['source_frame_index'][1:]
            new['provenance'].update(policy_mode='current_grade_a_parent_approach_pilot',augmentation_identity=meta['uuid'],
                dataset_path=plan['parent_dataset'],dataset_version=1,row_index=meta['parent_row'],
                warp_ccd_contacts_per_world=2048,episode_index=ordinal,generation_attempt=1,
                software_commit=fm['manorl_commit'],checkpoint_metadata_sha256=hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest())
            candidate_rows.append(new)
    if len(reports)!=27 or len({r['uuid']for r in reports})!=27:raise ValueError('incomplete pilot')
    a.output.mkdir(parents=True)
    # Native FK keypoints saved by the observer, not padded ancestor positions.
    for row,record in zip(candidate_rows,reports,strict=True):
        with np.load(Path(record['generation'])/'physical_trace.npz')as z:
            row['hands'][0]['mano_joint_pos']=z['mano_joint_pos'].tolist()
    accepted=[row for row,record in zip(candidate_rows,reports,strict=True) if record['accepted']]
    replay_a=[row for row,record in zip(candidate_rows,reports,strict=True) if record['repeat_grade']=='A']
    schema=lance.dataset(str(a.dataset),version=1).schema
    for name,rows in [('all_candidates.lance',candidate_rows),('replay_a_candidates.lance',replay_a),('accepted.lance',accepted)]:
        if not rows:continue
        lance.write_dataset(pa.Table.from_pylist(rows,schema=schema),str(a.output/name))
        readback=lance.dataset(str(a.output/name))
        if readback.count_rows()!=len(rows):raise ValueError('export row count mismatch')
        # Compare full row readback at the actual persisted Arrow precision.
        expected=pa.Table.from_pylist(rows,schema=schema)
        actual=readback.to_table()
        if not actual.equals(expected):raise ValueError('export fields changed during Lance readback')
    result={'contract':CONTRACT,'candidate_count':27,'accepted_count':len(accepted),
            'candidate_dataset':'all_candidates.lance',
            'replay_a_candidate_dataset':'replay_a_candidates.lance'if replay_a else None,
            'candidate_usage':'diagnostic only; replay-A alone does not satisfy the frozen parent-fidelity gates',
            'repeat_grade_counts':dict(Counter(r['repeat_grade']for r in reports)),
            'by_radius':{str(r):{'tried':9,'accepted':sum(x['accepted']for x in reports if x['radius_m']==r)}for r in (.05,.10,.15)},
            'plan_sha256':sha256(a.plan),'reports':reports,'accepted_dataset':'accepted.lance'if accepted else None}
    dump(a.output/'summary.json',result)
    print(json.dumps({k:v for k,v in result.items()if k!='reports'},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','run','collect']);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--parent-summary',type=Path)
    p.add_argument('--grade-roots',type=Path,nargs=2);p.add_argument('--plan',type=Path)
    p.add_argument('--run-roots',type=Path,nargs=2);p.add_argument('--batch',type=int)
    p.add_argument('--pass-name',choices=['generation','verification']);p.add_argument('--gpu',type=int)
    for key in ('asset-manifest','asset-root','manorl-root','client-root','benchmark-root','scene'):
        p.add_argument('--'+key,type=Path)
    p.add_argument('--manorl-commit')
    a=p.parse_args();{'prepare':prepare,'run':run,'collect':collect}[a.mode](a)


if __name__=='__main__':main()
