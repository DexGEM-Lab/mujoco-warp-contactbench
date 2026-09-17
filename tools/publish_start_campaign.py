#!/usr/bin/env python3
"""Stage accepted physical starts, complete candidate ledger and a video gallery."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('campaign','plan','bundle','videos','validation','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    summary=json.loads((a.campaign/'summary.json').read_text())
    status=json.loads((a.campaign/'status.json').read_text())
    validation=json.loads(a.validation.read_text())
    plan_document=json.loads(a.plan.read_text())
    preflight_rejections=sum(len(s['geometric_rejections']) for p in plan_document['parents'] for s in p['slots'])
    if json.loads((a.campaign/'job.json').read_text())['plan_sha256'] != digest(a.plan):
        raise ValueError('publication plan differs from executed plan')
    if status['state']!='complete' or summary['attempted']!=1344 or validation['checked']!=1344:
        raise ValueError('campaign is not completely validated')
    records=summary['results'];accepted=[r for r in records if r['accepted']]
    assert len({r['variant_id']for r in records})==1344 and len(accepted)==summary['accepted']
    a.output.mkdir(parents=True,exist_ok=False)
    rows=[];catalog=[];categories=defaultdict(list);radii=defaultdict(list);parents=defaultdict(list);failures=Counter()
    labels={'003':'电磁炉→桌面搬碗','007':'桌面→电磁炉搬碗','005':'酱料瓶','006':'水壶','009':'举碗保持'}
    for r in records:
        physical=r['physical'];plan=r['plan'];row=r['row_id'];slot=plan['slot']
        reason=physical['failed_gates']+[k for k,v in r['gates'].items()if not v and k!='physical_pose']
        failures.update(reason)
        record=dict(id=r['variant_id'],parent_id=row,slot=slot,category=physical['category'],accepted=r['accepted'],
                    radius_cm=plan['radius_m']*100,dx_cm=plan['delta_xyz_m'][0]*100,dy_cm=plan['delta_xyz_m'][1]*100,
                    dz_cm=plan['delta_xyz_m'][2]*100,octant=plan['octant'],merge_frame=plan['merge_frame'],
                    max_lift_cm=physical['max_lift_m']*100,final_position_cm=physical['object_position_final_cm'],
                    full_rotation_error_deg=physical['object_orientation_final_deg'],world_tilt_deg=physical['final_world_tilt_deg'],
                    failed_gates=';'.join(reason),trace_sha256=r['trace_sha256'])
        rows.append(record);categories[physical['category'][:3]].append(record);radii[record['radius_cm']].append(record);parents[row].append(record)
        source=a.campaign/row/slot
        if r['accepted']:
            target=a.output/'accepted'/row/slot;target.mkdir(parents=True)
            for filename in ('trace.npz','result.json','metrics.json','provenance.json','contact_validation.npz'):
                shutil.copy2(source/filename,target/filename)
            assert digest(target/'trace.npz')==r['trace_sha256']
            catalog.append(dict(id=r['variant_id'],parent_id=row,slot=slot,trace=str((target/'trace.npz').relative_to(a.output)),
                                trace_sha256=r['trace_sha256'],result=str((target/'result.json').relative_to(a.output)),
                                delta_xyz_m=plan['delta_xyz_m'],radius_m=plan['radius_m'],merge_frame=plan['merge_frame']))
        else:
            target=a.output/'rejected'/row/slot;target.mkdir(parents=True)
            shutil.copy2(source/'result.json',target/'result.json')
    with (a.output/'candidates.csv').open('w',encoding='utf-8-sig',newline='')as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (a.output/'candidates.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    (a.output/'accepted_ids.txt').write_text('\n'.join(r['id']for r in catalog)+'\n')
    (a.output/'rejected_ids.txt').write_text('\n'.join(r['id']for r in rows if not r['accepted'])+'\n')
    (a.output/'catalog.json').write_text(json.dumps(dict(schema='manorl.physical-start-augmentation.v1',hand='cheyingtong',control_hz=120,
                                  physics_hz=480,parent_count=42,attempted=1344,accepted=len(accepted),rows=catalog),indent=2)+'\n')
    for path in (a.plan,a.validation):shutil.copy2(path,a.output/path.name)
    shutil.copy2(a.campaign/'code_pin.json',a.output/'execution_code_pin.json')
    shutil.copy2(a.bundle/'asset_manifest.json',a.output/'asset_manifest.json')
    seeds=a.output/'seeds';seeds.mkdir()
    for name in ('inputs100','recipes','trajectories'):
        shutil.copytree(a.bundle/name,seeds/name)
    for name in ('comparison.json','accepted_full_pose_ids.txt','asset_manifest.json'):
        shutil.copy2(a.bundle/name,seeds/name)
    shutil.copytree(a.videos,a.output/'videos')
    media=[]
    for video in sorted((a.output/'videos').glob('*/render.mp4')):
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name,width,height,avg_frame_rate','-of','json',str(video)]))['streams'][0]
        assert probe['codec_name']=='h264' and probe['avg_frame_rate']=='30/1'
        subprocess.run(['ffmpeg','-v','error','-i',str(video),'-f','null','-'],check=True,stdout=subprocess.DEVNULL)
        media.append(dict(path=str(video.relative_to(a.output)),**probe))
    (a.output/'media_validation.json').write_text(json.dumps(media,indent=2)+'\n')
    count=lambda group:sum(r['accepted']for r in group)
    lines=['# Cheyingtong120Hz手起点扩增','',f'**1344个候选，{len(accepted)}条通过，{1344-len(accepted)}条剔除；通过率{len(accepted)/1344:.1%}。**',
           '', '新增数据不含原42条种子。所有候选都从扰动后的真实手部初始XYZ开始，以连续动力学完成接近、抓取和原任务；不是场景平移或旧状态拼接。',
           '', '## 分布','', '|动作|尝试|通过|','|---|---:|---:|']
    for key,group in categories.items():lines.append(f'|{labels[key]}|{len(group)}|{count(group)}|')
    lines+=['','|起点偏移距离|尝试|通过|','|---|---:|---:|']
    for radius,group in sorted(radii.items()):lines.append(f'|{radius:g}厘米|{len(group)}|{count(group)}|')
    lines+=['','## 扩增与验收方式','',
            f"- 42条严格成功种子，每条3/5/7.5/10cm四档×8个三维方向区间，共32个候选。随机种子{plan_document['seed']}。",
            f'- 每个固定半径/方向区间至多尝试16个初始几何位置；{preflight_rejections}次不合理几何方向被记录并在同区间重新选取。物理失败后不补抽、不挑重跑成功。',
            '- 只改变手部起始XYZ和接近段腕部XYZ目标；接触前约0.1秒恢复原修复目标，后段目标与速度衔接保持一致。手指修复/预载不变，物体初始状态不变。',
            '- 控制120Hz、物理480Hz，每帧4子步，帧0未积分，N状态/N−1控制区间；不改变质量、摩擦、增益、重力、摩擦锥或impratio。',
            '- 每条独立检查原物理＋姿态判据、接近衔接前无场景接触、抬升超过2cm时无持续≥25ms的无支撑且无手接触片段。接触来自120Hz几何重建，不代表480Hz力闭合证明。',
            f'- {len(accepted)}表示这次实际回放通过的轨迹数，不保证所有命令无限次重放都成功。已知边缘接触可能对数值扰动敏感。',
            '', '失败集中在少数种子，而不是随偏移半径单调增加：A020只有6/32通过，B200只有7/32通过。失败样例视频与逐条原因一并保留，未混入成功数据。',
            '', '## 文件','',
            '- `accepted/<parent>/<slot>/trace.npz`：实际手/物体状态、父轨迹目标、扰动目标、手指修正及480Hz子步控制。',
            '- `catalog.json`、`accepted_ids.txt`：严格通过的独立变体及来源、偏移、哈希。',
            '- `candidates.csv` / `candidates.json`：全部1344条的成功/失败与参数。',
            '- `rejected/`：83条失败判据；原失败动力学轨迹保留在实验目录。',
            '- `seeds/`：原始修复输入和配方副本，用于复现；B035不在成功种子清单。',
            '- `index.html`：五类成功动作及两条拒绝样例的视频。',
            '', '## 复现物理扩增','', '使用项目集成后的代码、项目Python环境及已物化的DexStream资产：','', '```bash',
            'PYTHONPATH=. python -m tools.run_start_augmentation \\',
            '  --bundle "$DELIVERY/seeds" --asset-root "$ASSET_ROOT" \\',
            '  --plan "$DELIVERY/start_plan_v1.json" --output outputs/new_start_replay --batch-size 32',
            '```', '', '单独诊断可以增加`--rows A_row035 --slot r4_o4`。模型和配方哈希不符会失败，绝不自动改用别的手型。',
            '', '## 各种子产出','', '|种子|通过/尝试|','|---|---:|']
    for name,group in parents.items():lines.append(f'|{name}|{count(group)}/{len(group)}|')
    lines+=['','瓶和壶仅验证刚体运动及落位，不模拟或声明液体输送成功。生成的实际轨迹、原人体采集和控制器目标有各自语义，不能混为同一种输入。']
    (a.output/'README.md').write_text('\n'.join(lines)+'\n')
    cards=[]
    for folder in sorted((a.output/'videos').iterdir()):
        if not folder.is_dir():continue
        manifest=json.loads((folder/'manifest.json').read_text());name=folder.name
        accepted_video=manifest['status'].startswith('Accepted')
        label='通过'if accepted_video else'拒绝样例'
        cards.append(f'<article><h2>{html.escape(name)} · {label}</h2><p>{html.escape(manifest["label"])}</p><video controls preload="metadata" poster="videos/{name}/preview.jpg" src="videos/{name}/render.mp4"></video><p><a href="videos/{name}/storyboard.jpg">动作分镜</a> · <a href="videos/{name}/render.mp4" download>视频</a></p></article>')
    table=''.join(f'<tr><td>{labels[k]}</td><td>{len(v)}</td><td>{count(v)}</td></tr>'for k,v in categories.items())
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Cheyingtong120Hz · 起点扩增</title><style>body{{font-family:system-ui,sans-serif;background:#f3f5f8;color:#172c40;margin:0}}main{{max-width:1400px;margin:auto;padding:24px}}header,article{{background:white;border-radius:12px;padding:20px;margin-bottom:18px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,500px),1fr));gap:20px}}video{{width:100%;background:#233c55;border-radius:6px}}h1{{font-size:28px}}h2{{font-size:19px}}p{{line-height:1.6}}td,th{{padding:5px 20px 5px 0;text-align:left}}a{{color:#075eac}}</style><main><header><h1>Cheyingtong右手 · 120Hz起点扩增</h1><p><strong>1344个独立物理候选 → {len(accepted)}条通过；{1344-len(accepted)}条剔除。</strong></p><p>42个成功种子，每条4档距离×8方向；只变手起始XYZ，物体不动，原抓取目标和物理参数保持。帧0未积分，控制120Hz／物理480Hz。</p><table><tr><th>动作</th><th>尝试</th><th>通过</th></tr>{table}</table><p><a href="README.md">说明与复现</a> · <a href="candidates.csv">全部候选CSV</a> · <a href="catalog.json">成功轨迹目录</a> · <a href="accepted_ids.txt">成功ID</a></p><p>示例读取新模拟实际状态。标为拒绝的样例不在成功数据中。</p></header><section class="grid">{''.join(cards)}</section></main></html>'''
    (a.output/'index.html').write_text(page)
    report=dict(attempted=1344,accepted=len(accepted),rejected=1344-len(accepted),parents=42,
                by_category={labels[k]:dict(attempted=len(v),accepted=count(v))for k,v in categories.items()},
                by_radius_cm={str(k):dict(attempted=len(v),accepted=count(v))for k,v in radii.items()},
                overlapping_failure_gate_counts=dict(failures),media=len(media),data_status='physical NPZ complete; optional training export attached separately')
    (a.output/'delivery_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    main()
