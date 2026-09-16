"""Render measured local-repair states with the selected physical hand skin.

Runtime paths/profile manifests are supplied by a JSON job file, never inferred
from this worktree (whose asset submodule may deliberately be empty).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PIN = '778614d09e917deffed0bff3f357aa237efa762d'
WIDTH, HEIGHT, HEADER = 1280, 640, 80


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def bind(job, profile):
    primary = Path(__file__).resolve().parents[1]
    manifest = Path(job['manifests'][profile]).resolve()
    assert not any(k.startswith('sim.manorl') for k in sys.modules)
    os.environ['MANORL_ASSET_MANIFEST'] = str(manifest)
    sys.path.insert(0, str(primary))
    from sim.manorl import assets
    assert Path(assets.__file__).resolve() == primary / 'sim/manorl/assets.py'
    assert assets.ASSET_MANIFEST == manifest
    assert assets.MANO_OPERATOR == profile
    assert assets.asset_provenance()['asset_source_commit'] == PIN
    assets.DEXSTREAM_ROOT = Path(job['primary']) / 'assets/dexstream_digital_assets'
    assert digest(manifest)==job['manifest_sha256']
    assert assets.DEXSTREAM_ROOT.is_dir()
    assert assets.hand_urdf_path('right').is_file()
    assert assets._hand_skin_path('right').is_file()
    return assets


def mapping(t, model, assets):
    names = t['scene_object_names'].tolist()
    assert len(set(names)) == len(names)
    assert model.nq == t['qpos'].shape[1] == 28 + 7 * len(names)
    joints = [model.joint(i).name for i in range(28)]
    assert tuple(joints) == assets.hand_joint_names('right')
    assert np.array_equal(model.jnt_qposadr[:28], np.arange(28))
    addresses = {name: int(model.joint(name + '_free').qposadr[0]) for name in names}
    assert list(addresses.values()) == t['scene_object_qpos_addresses'].tolist()
    # These independently saved measurements discriminate measured from desired qpos.
    if 'actual_hand' in t:
        np.testing.assert_allclose(t['qpos'][:, :28], t['actual_hand'], atol=1e-7)
    if 'actual_wrist' in t:
        np.testing.assert_allclose(t['qpos'][:, :6], t['actual_wrist'], atol=1e-7)
    for slot, adr in enumerate(addresses.values()):
        np.testing.assert_allclose(t['qpos'][:, adr:adr+3], t['actual_object_pos'][:, slot], atol=1e-7)
    return {'hand_joint_names': joints, 'object_qpos_addresses': addresses}


def kinematics(mj, model, data, qpos):
    data.qpos[:] = qpos
    mj.mj_kinematics(model, data)
    mj.mj_comPos(model, data)
    mj.mj_camlight(model, data)
    assert data.time == 0 and data.ncon == 0
    assert np.array_equal(data.qpos, qpos)


def texture_evidence(model, assets, names):
    records = []
    manifest = assets._asset_manifest()
    for name in names:
        body = model.body(name).id
        ids = [i for i in range(model.ngeom) if model.geom_bodyid[i] == body and model.geom_group[i] == assets.VISUAL_GEOM_GROUP]
        assert ids, name
        links = []
        for gid in ids:
            mid = int(model.geom_matid[gid])
            tids = model.mat_texid[mid].tolist() if mid >= 0 else []
            links.append({'geom': model.geom(gid).name, 'material_id': mid, 'texture_ids': tids})
        if name in ('pitcherbase', 'mayonnaisebottle'):
            assert any(any(i >= 0 for i in link['texture_ids']) for link in links), name
        records.append({'object': name, 'links': links, 'native_materials': manifest['objects'][name].get('materials')})
    return records


def render(row, job, assets, preview_only):
    import imageio.v2 as imageio
    path = Path(row['trace'])
    before = digest(path)
    t = np.load(path, allow_pickle=False)
    qpos = t['qpos']
    assert np.isfinite(qpos).all()
    assert row['source_hz'] in (100,120) and job['physics_hz']==4*row['source_hz']
    assert digest(path)==row['trace_sha256']
    names = t['scene_object_names'].tolist()
    mj, model = assets.compile_unified_model(object_types=names, object_collisions=False, visual_meshes=True)
    bindings = mapping(t, model, assets)
    assert model.nskin == 1 and model.nlight == 0
    model.vis.headlight.ambient[:] = .4
    model.vis.headlight.diffuse[:] = .65
    np.testing.assert_allclose(model.vis.headlight.ambient, [.4]*3)
    np.testing.assert_allclose(model.vis.headlight.diffuse, [.65]*3)
    model.vis.global_.offwidth = WIDTH
    model.vis.global_.offheight = HEIGHT
    data = mj.MjData(model)
    renderer = mj.Renderer(model, width=WIDTH, height=HEIGHT)
    camera = mj.MjvCamera()
    camera.azimuth, camera.elevation = 110, -30
    camera.distance = 1
    option = mj.MjvOption()
    option.geomgroup[assets.COLLISION_GEOM_GROUP] = 0
    indices = np.arange(0, len(qpos), 4)
    picks = indices[np.linspace(0, len(indices)-1, 6).astype(int)]
    # Bounds use oriented compiled geom boxes AND the actual deformed skin at
    # every displayed state, not desired wrist positions or root-point padding.
    gids = np.where((model.geom_group == assets.VISUAL_GEOM_GROUP) & (model.geom_type != mj.mjtGeom.mjGEOM_PLANE))[0]
    corners = np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
    local = model.geom_aabb[gids, :3][:,None,:] + corners[None,:,:] * model.geom_aabb[gids,3:][:,None,:]
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for i in indices:
        kinematics(mj, model, data, qpos[i])
        renderer.update_scene(data, camera=camera, scene_option=option)
        geom = np.einsum('gij,gkj->gki', data.geom_xmat[gids].reshape(-1,3,3), local) + data.geom_xpos[gids,None,:]
        points = np.concatenate([geom.reshape(-1,3), renderer.scene.skinvert.reshape(-1,3)])
        lo, hi = np.minimum(lo, points.min(0)), np.maximum(hi, points.max(0))
    camera.lookat[:] = (lo+hi)/2
    # Fit the route box in camera coordinates rather than using a diagonal
    # sphere, which wastes most of a widescreen viewport on empty floor.
    renderer.update_scene(data, camera=camera, scene_option=option)
    forward = np.asarray(renderer.scene.camera[0].forward, dtype=float)
    up = np.asarray(renderer.scene.camera[0].up, dtype=float)
    right = np.cross(forward, up)
    box = corners * (hi-lo)/2
    tan_y = np.tan(np.deg2rad(model.vis.global_.fovy)/2)
    required = np.maximum(np.abs(box @ up)/tan_y,
                          np.abs(box @ right)/(tan_y*WIDTH/HEIGHT)) - box @ forward
    camera.distance = float(required.max() * 1.10)
    out = Path(job['output']) / row['id']
    out.mkdir(parents=True, exist_ok=False)
    font = ImageFont.truetype('DejaVuSans.ttf', 21)
    small = ImageFont.truetype('DejaVuSans.ttf', 19)
    def frame(i):
        kinematics(mj, model, data, qpos[i])
        renderer.update_scene(data, camera=camera, scene_option=option)
        image = Image.new('RGB', (WIDTH, HEIGHT+HEADER), 'white')
        image.paste(Image.fromarray(renderer.render().copy()), (0, HEADER))
        d = ImageDraw.Draw(image)
        d.text((14,5), f"{row['id']} | {row['profile']}/right | time {i / row['source_hz']:.2f} s", font=font, fill='black')
        d.text((14,31), row['label'], font=small, fill='#163e61')
        d.text((14,55), f"{row['status']} | control {row['source_hz']} Hz / physics {job['physics_hz']} Hz", font=small, fill='black')
        return image
    thumbnails = []
    for i in picks:
        image = frame(i)
        thumbnails.append(image.resize((640,360)))
    board = Image.new('RGB',(1280,1080),'white')
    for slot, image in enumerate(thumbnails):
        board.paste(image, ((slot%2)*640,(slot//2)*360))
    board.save(out/'storyboard.jpg',quality=90)
    frame(int(picks[2])).resize((960,540)).save(out/'preview.jpg',quality=92)
    if not preview_only:
        with imageio.get_writer(out/'render.mp4', fps=row['source_hz']/4, codec='libx264', quality=7, macro_block_size=2) as writer:
            for i in indices:
                writer.append_data(np.asarray(frame(int(i))))
    renderer.close()
    assert digest(path) == before
    result = {**row, 'render_complete': not preview_only, 'trace_sha256_before': before, 'trace_sha256_after': digest(path),
              'qpos_sha256': hashlib.sha256(qpos.tobytes()).hexdigest(), 'current_asset': assets.asset_provenance(),
              'builder': assets.__file__, 'asset_root': str(assets.DEXSTREAM_ROOT), 'manifest_path': str(assets.ASSET_MANIFEST),
              'hand_urdf': str(assets.hand_urdf_path('right')), 'native_skin': str(assets._hand_skin_path('right')), 'skin_count': model.nskin,
              'bindings': bindings, 'textures': texture_evidence(model,assets,names),
              'lights': {'requested_ambient':[.4]*3,'requested_diffuse':[.65]*3,'actual_ambient':model.vis.headlight.ambient.tolist(),'actual_diffuse':model.vis.headlight.diffuse.tolist(),'extra_lights':model.nlight},
              'camera': {'lookat':camera.lookat.tolist(),'distance':camera.distance,'azimuth':camera.azimuth,'elevation':camera.elevation,'geometry_skin_bounds':[lo.tolist(),hi.tolist()]},
              'state_indices':indices.tolist(),'source_frame_indices':t['source_frame'][indices].tolist(),'fps':row['source_hz']/4,
              'physics_steps':0,'contact_solves':0,'acceptance_changed':False,
              'paths':{k:str(out/v) for k,v in [('video','render.mp4'),('preview','preview.jpg'),('storyboard','storyboard.jpg'),('manifest','manifest.json')]}}
    dump(out/'manifest.json',result)
    print(row['id'], 'preview' if preview_only else 'complete', flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--job',required=True)
    p.add_argument('--profile',required=True)
    p.add_argument('--only',nargs='*')
    p.add_argument('--preview-only',action='store_true')
    a=p.parse_args()
    job=json.loads(Path(a.job).read_text())
    assets=bind(job,a.profile)
    for row in job['rows']:
        if row['profile']==a.profile and (not a.only or row['id'] in a.only):
            render(row,job,assets,a.preview_only)

if __name__=='__main__':
    main()
