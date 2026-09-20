"""Read-only raw Lance adapter for the arrival-indexed native U1 workspace."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np


def load_raw_source(dataset, version, row_number, asset_root, asset_manifest):
    import lance
    from scipy.spatial.transform import Rotation
    from tools.replay_capture_no_policy import COLUMNS, activate_hand_profile, source_arrays

    if type(version) is not int or version < 1 or type(row_number) is not int or row_number < 0:
        raise ValueError('explicit positive version and nonnegative positional row required')
    ds = lance.dataset(str(dataset), version=version)
    if int(ds.version) != version:
        raise ValueError('dataset version mismatch')
    row = ds.take([row_number], columns=COLUMNS).to_pylist()[0]
    manifest = activate_hand_profile(row['index']['operator'], asset_root, asset_manifest)
    source = source_arrays(row, manifest, 120)
    from sim.manorl import assets
    from sim.manorl.local_contact_repair import MODEL_FIELDS
    from tools.u1_placement_workspace import assert_u1

    names = tuple(sorted(source['names']))
    _, model = assets.compile_unified_model(object_types=names, object_collisions=True,
                                            hand_side='right', physics_timestep=1/480)
    if model.nu != 28 or model.nq != 28 + 7*len(names) or any(
            model.body(i).name.startswith('left_') for i in range(model.nbody)):
        raise ValueError('raw workspace requires exactly one right 28-DoF hand')
    target = source['commands'].copy()
    teacher = np.zeros((len(target), model.nq))
    teacher[:, :28] = target
    for name in names:
        slot = source['names'].index(name)
        adr = int(model.joint(name+'_free').qposadr[0])
        teacher[:, adr:adr+3] = source['positions'][:, slot]
        teacher[:, adr+3:adr+7] = Rotation.from_rotvec(source['rotvecs'][:, slot]).as_quat(scalar_first=True)
    provenance = dict(mode='raw', dataset=str(Path(dataset).resolve()), dataset_version=version,
                      row=row_number, index=row['index'], source_ground_shift_m=source['shift'],
                      source_mean_fps=source['actual_fps'], hand_side='right',
                      asset_root=str(Path(asset_root).resolve()), asset_manifest=str(Path(asset_manifest).resolve()),
                      asset_provenance=assets.asset_provenance(), teacher_role='display-only after frame0',
                      target_role='recorded right-hand qpos plus common grounding shift',
                      frame_indexing='frame0 prestep; target[k] advances frame k-1 to k')
    inp = SimpleNamespace(frames=len(target), hz=120, arrays={'scene_object_names':np.asarray(names)},
                          metrics={'source_metadata':{'active_object':source['active']}},
                          initial={'qpos':teacher[0].copy(), 'qvel':np.zeros(model.nv)},
                          manifest={'native':{field:np.asarray(getattr(model, field)).copy() for field in MODEL_FIELDS}})
    assert_u1(model, inp)
    return inp, model, target, {'qpos':teacher}, provenance
