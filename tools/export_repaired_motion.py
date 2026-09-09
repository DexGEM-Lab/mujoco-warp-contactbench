"""Export an actual repaired physics trajectory, preserving its command track.

The export is a generated-motion record, not an unmodified human capture.
Objects are measured from the free-body rollout. Reproduction uses the original
capture plus the adjacent patch; replaying measured hand poses is a different
experiment from replaying actuator targets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import uuid

import lance
import numpy as np
import pyarrow as pa


def export_motion(run: Path, output: Path):
    manifest = json.loads((run/'manifest.json').read_text())
    result = json.loads((run/'result.json').read_text())
    if not result['complete']:
        raise ValueError('cannot publish an interrupted trajectory')
    if output.exists():
        raise FileExistsError(output)
    trace = np.load(run/'trajectory.npz', allow_pickle=False)
    frames = len(trace['qpos'])
    recipe = manifest['patch']
    patch_digest = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    identifier = str(uuid.uuid5(uuid.NAMESPACE_URL, recipe['source']['uuid']+patch_digest))
    active = recipe['active_object']
    names = trace['scene_object_names'].tolist()
    if active not in names:
        raise ValueError('patch active object absent from trace')
    row = {
        'index': {'uuid': identifier, 'seed_uuid': recipe['source']['uuid'],
                  'is_generated': True, 'scene': ','.join(names),
                  'gesture': manifest['source_gesture'],
                  'source_row_index': recipe['source']['row']},
        'trajectory_metadata': {'data_fps': 100, 'total_frames': frames,
                                'hand_names': ['right'], 'object_names': names},
        'timestamp': trace['time_seconds'].tolist(),
        'hands': [{'urdf_dof': trace['qpos'][:,:28].tolist(),
                   'command_target_dof': trace['ctrl'].tolist()}],
        'objects': [{'pos': trace['scene_object_pos'][:,i].tolist(),
                     'rot_aa': trace['scene_object_rot_aa'][:,i].tolist()}
                    for i in range(len(names))],
        'source_frame': trace['source_frame'].tolist(),
        'repair_provenance_json': json.dumps(manifest, ensure_ascii=False),
    }
    vec3 = pa.list_(pa.float64(), 3)
    vec28 = pa.list_(pa.float64(), 28)
    schema = pa.schema([
        ('index', pa.struct([('uuid',pa.string()),('seed_uuid',pa.string()),
                             ('is_generated',pa.bool_()),('scene',pa.string()),
                             ('gesture',pa.string()),('source_row_index',pa.int64())])),
        ('trajectory_metadata',pa.struct([('data_fps',pa.int64()),('total_frames',pa.int64()),
            ('hand_names',pa.list_(pa.string())),('object_names',pa.list_(pa.string()))])),
        ('timestamp',pa.list_(pa.float64())),
        ('hands',pa.list_(pa.struct([('urdf_dof',pa.list_(vec28)),
                                     ('command_target_dof',pa.list_(vec28))]))),
        ('objects',pa.list_(pa.struct([('pos',pa.list_(vec3)),('rot_aa',pa.list_(vec3))]))),
        ('source_frame',pa.list_(pa.int64())),('repair_provenance_json',pa.string()),
    ], metadata={b'contract':b'direct_repaired_physical_motion.v1',
                 b'hand_urdf_dof_semantics':b'measured post-step joints; command_target_dof is the applied actuator target',
                 b'object_pose_semantics':b'measured free-body post-step poses in simulation world frame',
                 b'patch_sha256':patch_digest.encode()})
    lance.write_dataset(pa.Table.from_pylist([row],schema=schema),output,mode='create')
    loaded = lance.dataset(output,version=1).take([0]).to_pylist()[0]
    if not np.array_equal(np.asarray(loaded['hands'][0]['command_target_dof']),trace['ctrl']):
        raise RuntimeError('command track changed during export')
    for i in range(len(names)):
        if not np.array_equal(np.asarray(loaded['objects'][i]['pos']),trace['scene_object_pos'][:,i]):
            raise RuntimeError('physical object path changed during export')
    return {'output':str(output),'rows':1,'frames':frames,'objects':names,'patch_sha256':patch_digest}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(export_motion(args.run,args.output)))


if __name__=='__main__':
    main()
