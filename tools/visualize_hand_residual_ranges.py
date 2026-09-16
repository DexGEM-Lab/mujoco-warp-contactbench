"""Render per-joint residual envelopes as overlaid MANO hands (kinematics only)."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from dataclasses import asdict
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from sim.manorl.abi import ResidualActionConfig
from sim.manorl.contracts import JOINT_NAMES_28

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
FINGER_ZH = dict(zip(FINGERS, ("拇指", "食指", "中指", "无名指", "小指")))
MOTION_ZH = {"cmc_abd": "CMC 外展", "cmc_flex": "CMC 屈曲", "cmc_twist": "CMC 扭转",
             "mcp_abd": "MCP 外展", "mcp_flex": "MCP 屈曲", "ip": "IP 屈曲",
             "pip": "PIP 屈曲", "dip": "DIP 屈曲"}
BG = np.array([17, 24, 34], dtype=np.uint8)
CYAN = (61, 209, 224)
ORANGE = (255, 190, 89)
WIDTH, PANEL_H = 1280, 520


def residual_bounds(reference, lower, upper, config=ResidualActionConfig()):
    """Limiting reachable offsets from zero for fixed reference and |action|<=1."""
    if not 0 <= config.gamma_joints < 1:
        raise ValueError("visualized decay must be in [0, 1)")
    scale = np.asarray(config.joint_scale) * config.joint_scale_multiplier
    cap = np.asarray(config.max_joint_offset) * config.joint_max_offset_multiplier
    envelope = np.minimum(cap, scale / (1 - config.gamma_joints))
    reference, lower, upper = map(np.asarray, (reference, lower, upper))
    if reference.shape != (22,) or np.any(reference < lower) or np.any(reference > upper):
        raise ValueError("reference must be22 valid finger joint angles")
    return np.maximum(-envelope, lower - reference), np.minimum(envelope, upper - reference), envelope


def sweep(progress: float, low: float, high: float) -> float:
    """Slow visual interpolation, with pauses at zero and both extrema."""
    keys = [(0, 0), (.08, 0), (.28, low), (.38, low), (.72, high), (.82, high), (.96, 0), (1, 0)]
    for (t0, x0), (t1, x1) in zip(keys, keys[1:]):
        if t0 <= progress <= t1:
            a = (progress - t0) / (t1 - t0)
            return x0 + (x1 - x0) * (a * a * (3 - 2 * a))
    raise ValueError("progress must be in [0, 1]")


def joint_label(name: str):
    _, finger, motion = name.split("_", 2)
    return f"{FINGER_ZH[finger]} · {MOTION_ZH[motion]}"


def make_model(operator, output):
    import mujoco
    from sim.manorl import assets
    from tools.generate_manorl_asset_manifest import generate

    manifest = generate(assets.DEXSTREAM_ROOT, assets.REPOSITORY_ROOT, hand_operator=operator)
    profile = output / "asset_manifest.json"
    profile.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    assets.MANO_OPERATOR = operator
    assets.ASSET_MANIFEST = profile
    assets._asset_manifest.cache_clear()
    root = ET.fromstring(assets.build_scene_xml(object_type="cube1", hand_side="right", visual_meshes=True))
    # This is a rendering fixture: remove the demonstration cube/floor; no physics step.
    world = root.find("worldbody")
    for node in list(world):
        if node.tag == "geom" or (node.tag == "body" and node.get("name") == "cube1"):
            world.remove(node)
    for geom in root.iter("geom"):
        geom.set("contype", "0"); geom.set("conaffinity", "0")
    asset = root.find("asset")
    for texture in list(asset):
        if texture.tag == "texture" and texture.get("type") == "skybox":
            asset.remove(texture)
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    headlight = visual.find("headlight")
    if headlight is None:
        headlight = ET.SubElement(visual, "headlight")
    # Ambient and diffuse inherit the shared ManoRL scene defaults.
    headlight.set("specular", "0.12 0.12 0.12")
    glob = visual.find("global")
    if glob is None:
        glob = ET.SubElement(visual, "global")
    glob.set("offwidth", "640"); glob.set("offheight", str(PANEL_H))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    assert model.nq == 28 and model.nu == 28 and model.nskin == 1
    actual_names = tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(28))
    assert actual_names == JOINT_NAMES_28
    return mujoco, model, assets.asset_provenance()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--operator", default="cheyingtong")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seconds-per-joint", type=float, default=4.)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    if args.fps < 1 or args.seconds_per_joint < 2:
        parser.error("fps must be positive; seconds-per-joint must be >=2")
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    mj, model, provenance = make_model(args.operator, out)
    reference = np.zeros(28)
    reference[6:12] = [-.3, -.4, 0, .45, 0, .25]
    reference[12:] = [0, .35, .4, .25] * 4
    reference = np.clip(reference, model.jnt_range[:, 0], model.jnt_range[:, 1])
    config = ResidualActionConfig()
    low, high, envelope = residual_bounds(reference[6:], model.jnt_range[6:, 0], model.jnt_range[6:, 1], config)
    data = mj.MjData(model)
    option = mj.MjvOption(); option.geomgroup[3] = 0
    renderer = mj.Renderer(model, width=640, height=PANEL_H)
    renderer.scene.flags[mj.mjtRndFlag.mjRND_SHADOW] = 0
    data.qpos[:] = reference; mj.mj_forward(model, data)
    # Asset hand extends mainly along local -X. Fixed cameras never track the moving finger.
    center = np.mean(data.xpos[model.body_geomnum > 0], axis=0)
    cameras = []
    for azimuth, elevation in [(90, -75), (270, 20)]:
        camera = mj.MjvCamera(); mj.mjv_defaultCamera(camera)
        camera.lookat[:] = center; camera.distance = .30
        camera.azimuth = azimuth; camera.elevation = elevation
        cameras.append(camera)
    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if not font_path.exists():
        raise FileNotFoundError("Install Noto Sans CJK fonts for Chinese labels")
    fonts = {size: ImageFont.truetype(str(font_path), size) for size in (18, 22, 26, 34)}

    def render(q, camera, color, joint=None):
        data.qpos[:] = q; mj.mj_forward(model, data)
        model.skin_rgba[0] = (*color, 1.)
        renderer.disable_depth_rendering()
        renderer.update_scene(data, camera=camera, scene_option=option)
        marker_pixel = None
        if joint is not None:
            glcam = renderer.scene.camera[0]
            eye = (renderer.scene.camera[0].pos + renderer.scene.camera[1].pos) * .5
            forward = np.asarray(glcam.forward); up = np.asarray(glcam.up)
            right = np.cross(forward, up)
            relative = data.xanchor[joint] - eye
            distance = np.dot(relative, forward)
            focal = PANEL_H / (2 * math.tan(math.radians(float(model.vis.global_.fovy)) / 2))
            marker_pixel = (320 + focal * np.dot(relative, right) / distance,
                            PANEL_H / 2 - focal * np.dot(relative, up) / distance)
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        depth = renderer.render().copy()
        renderer.disable_depth_rendering()
        mask = depth < 1.
        return rgb, mask, marker_pixel

    refs = [render(reference, c, (.65, .68, .72)) for c in cameras]
    def frame_for(j, delta):
        q = reference.copy(); q[j] += delta
        assert np.count_nonzero(np.abs(q-reference)>1e-12) <= 1
        assert np.all(q >= model.jnt_range[:, 0]-1e-10) and np.all(q <= model.jnt_range[:, 1]+1e-10)
        canvas = Image.new("RGB", (WIDTH, 760), tuple(BG))
        for view, (camera, (ref, rm, _)) in enumerate(zip(cameras, refs)):
            rgb, mask, marker_pixel = render(q, camera, (.14, .73, .83), joint=j)
            panel = np.broadcast_to(BG, rgb.shape).copy()
            panel[rm] = (.42*ref[rm] + .58*BG).astype(np.uint8)
            # Visible pale outline makes an unmoving reference readable through overlap.
            im = Image.fromarray((rm*255).astype(np.uint8))
            edge = np.asarray(im.filter(ImageFilter.MaxFilter(3))) > np.asarray(im)
            panel[edge] = [110, 124, 140]
            panel[mask] = (.90*rgb[mask] + .10*panel[mask]).astype(np.uint8)
            canvas.paste(Image.fromarray(panel), (640*view, 125))
            if marker_pixel is not None:
                px, py = marker_pixel; px += 640*view; py += 125
                marker_draw = ImageDraw.Draw(canvas)
                marker_draw.ellipse((px-7,py-7,px+7,py+7),outline=ORANGE,width=2)
                marker_draw.ellipse((px-2,py-2,px+2,py+2),fill=ORANGE)
        draw = ImageDraw.Draw(canvas)
        draw.text((24, 12), f"{j-5:02d} / 22   {joint_label(JOINT_NAMES_28[j])}", font=fonts[34], fill="white")
        draw.text((26, 64), f"DoF {j}   |   偏移 {delta:+.3f} rad  /  {math.degrees(delta):+.1f}°", font=fonts[26], fill=CYAN)
        draw.text((790, 21), "灰色：固定参考    青色：变化后的手", font=fonts[22], fill=(182,195,206))
        draw.text((790, 58), "橙点：当前关节   ·   幅度未放大", font=fonts[22], fill=ORANGE)
        for x, text in [(25, "正视"), (665, "侧面")]:
            draw.text((x, 126), text, font=fonts[22], fill=(192,207,220))
        draw.line((640,125,640,640),fill=(53,66,80),width=1)
        i=j-6
        draw.text((25, 650), f"本姿态可用偏移：{low[i]:+.3f} ~ {high[i]:+.3f} rad   ({math.degrees(low[i]):+.1f}° ~ {math.degrees(high[i]):+.1f}°)",font=fonts[22],fill="white")
        draw.text((25, 690), "固定略弯曲参考姿态 · 一次只变一个关节 · 运动学范围演示，非训练回放",font=fonts[18],fill=(151,169,184))
        x0,x1,y=935,1210,670;draw.line((x0,y,x1,y),fill=(90,105,122),width=3)
        for v in (low[i],0,high[i]):
            x=x0+(v-low[i])/(high[i]-low[i])*(x1-x0);draw.line((x,y-6,x,y+6),fill=(185,197,210),width=2)
        x=x0+(delta-low[i])/(high[i]-low[i])*(x1-x0);draw.ellipse((x-6,y-6,x+6,y+6),fill=CYAN)
        return canvas

    records = []
    for j in range(6,28):
        records.append(dict(dof=j,name=JOINT_NAMES_28[j],label=joint_label(JOINT_NAMES_28[j]),
            reference_rad=float(reference[j]),negative_delta_rad=float(low[j-6]),positive_delta_rad=float(high[j-6]),
            effective_envelope_rad=float(envelope[j-6]),joint_limit_rad=model.jnt_range[j].tolist()))
    manifest = dict(kind="kinematic_residual_envelope_not_policy_rollout",operator=args.operator,
        asset=provenance,reference_qpos=reference.tolist(),reference_pose="fixed_relaxed_illustrative",
        residual_config=asdict(config),frames_per_joint=round(args.fps*args.seconds_per_joint),
        fps=args.fps,joints=records)
    (out/'ranges.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    try:
        frame_for(14,high[8]).save(out/'preview.png')
        if args.preview:
            print('PREVIEW',out/'preview.png',flush=True); return
        import imageio.v2 as imageio
        frames_per_joint=round(args.fps*args.seconds_per_joint)
        with imageio.get_writer(out/'hand_residual_ranges.mp4',fps=args.fps,codec='libx264',quality=8,macro_block_size=1) as writer:
            for j in range(6,28):
                for k in range(frames_per_joint):
                    delta=sweep(k/(frames_per_joint-1),low[j-6],high[j-6])
                    writer.append_data(np.asarray(frame_for(j,delta)))
                frame_for(j,low[j-6]).save(out/f'dof_{j:02d}_negative.png')
                frame_for(j,high[j-6]).save(out/f'dof_{j:02d}_positive.png')
                print('RENDERED',j,JOINT_NAMES_28[j],flush=True)
        # A short animation for inline preview; full resolution remains in MP4.
        gif=[]
        for k in range(48):
            delta=sweep(k/47,low[8],high[8]);gif.append(frame_for(14,delta).resize((768,456)))
        gif[0].save(out/'preview.gif',save_all=True,append_images=gif[1:],duration=84,loop=0)
        links=''.join(f'<li><a href="hand_residual_ranges.mp4#t={(j-6)*args.seconds_per_joint:.1f}">{joint_label(JOINT_NAMES_28[j])}</a> <a href="dof_{j:02d}_negative.png">负端</a> / <a href="dof_{j:02d}_positive.png">正端</a></li>' for j in range(6,28))
        (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>右手22关节残差范围</title><style>body{background:#111822;color:#e5edf5;font:18px sans-serif;max-width:1280px;margin:24px auto}a{color:#3dd1e0}video{width:100%}ul{columns:3}</style><h1>右手22关节：实际能调整多少</h1><p>灰色固定参考，青色逐关节变化；考虑0.9衰减累积与关节限位。固定示例姿态，不是训练或接触仿真。</p><video controls loop poster="preview.png" src="hand_residual_ranges.mp4"></video><ul>'+links+'</ul><p><a href="ranges.json">精确角度与资产版本</a></p>',encoding='utf-8')
        print('DONE',out,flush=True)
    finally:
        renderer.close()


if __name__ == '__main__':
    main()
