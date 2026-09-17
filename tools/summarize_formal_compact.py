#!/usr/bin/env python3
"""Summarize new formal compact physics, not the source dataset's old labels."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path


def summarize(root):
    original=json.loads((root/'input_manifest.json').read_text())
    rows=[]
    for group in ('01-egg','02-stir','03-bowl','04-pour'):
        summary=json.loads((root/group/'summary.json').read_text())
        rows.extend(summary['rows'])
    rows.sort(key=lambda r:r['row_index'])
    if len(rows)!=28 or [x['row_index']for x in rows]!=list(range(28)):
        raise ValueError('all28 independent row results are required')
    labels={'01-egg':'鸡蛋入桶','02-stir':'搅拌','03-bowl':'拿碗','04-pour':'倒杯'}
    groups=defaultdict(list)
    for row in rows:groups[row['group']].append(row)
    result=dict(dataset=original['dataset'],dataset_version=1,replay_hand='cheyingtong/right',control_hz=120,physics_hz=480,
                mode='recorded actuator targets, no policy inference',attempted=28,
                accepted=sum(r['acceptance']['accepted']for r in rows),
                by_group={labels[k]:dict(attempted=len(v),accepted=sum(r['acceptance']['accepted']for r in v))for k,v in groups.items()},rows=rows)
    (root/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    table=[]
    for r in rows:
        table.append(dict(identity=r['identity'],row_index=r['row_index'],action=labels[r['group']],
                          passed=r['acceptance']['accepted'],frames=r['frames'],
                          contact_frames=r['acceptance']['hand_object_contact_frames'],
                          source_reference_final_xyz_mean_deg=r['acceptance'].get('final_rotation_xyz_mean_error_deg'),
                          recorded_position_final_cm=r['replay_recorded_position_error_cm']['final'],
                          recorded_position_max_cm=r['replay_recorded_position_error_cm']['max'],
                          recorded_full_rotation_final_deg=r['replay_recorded_rotation_error_deg']['final'],
                          first_deviation_frame=None if r['first_deviation']is None else r['first_deviation']['frame'],
                          failure_reasons=';'.join(r['acceptance']['failure_reasons'])))
    with (root/'comparison.csv').open('w',encoding='utf-8-sig',newline='')as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    lines=['# Formal compact：Cheyingtong120Hz目标回放','',f"**28条全部回放，{result['accepted']}条通过原任务门限。**",'',
           '输入为用户指定compact.lance版本1，使用其保存的执行器目标，120Hz控制／480Hz物理。未重新调用策略、未修复调参，物体只在初始帧设置，随后完全由动力学驱动。',
           '', '|动作|条数|通过|','|---|---:|---:|']
    for group,count in result['by_group'].items():lines.append(f"|{group}|{count['attempted']}|{count['accepted']}|")
    lines+=['','## 如何理解','',
            '- 非鸡蛋：未触发原0.1m物体偏差条件，末端intrinsicXYZ逐轴绕回误差均值≤35°，原生手—目标法向力>0.2N的帧数>100。',
            '- 鸡蛋：按原规则检查释放后的鸡蛋中心实际穿入桶口，不套普通末端姿态门限，也不延长原记录时长。',
            '- 下表的位置/旋转差是新回放与文件所存实际运动的差，不是相对于人类参考的任务误差。两种指标分开保存。',
            '- 每条初始化为文件实际帧0；target[t]应用四个物理子步生成帧t。所有控制目标与文件逐项相同。',
            '- 原文件是策略闭环推理的结果；本轮执行的是已录制控制序列。原文件合格不保证开放回放逐次成功。',
            '- 使用来源原生CCD16/256每世界及全部场景物体；本轮批量仅包含已导出的行，与原生成批量大小不同，接触数值可以有微小差异。',
            '', '|轨迹|原门限|与记录终点位置差/cm|全程最大位置差/cm|与记录终点旋转差/°|手—目标接触帧|失败原因|',
            '|---|---|---:|---:|---:|---:|---|']
    for r in table:
        lines.append(f"|{r['identity']}|{'通过'if r['passed']else'未通过'}|{r['recorded_position_final_cm']:.3f}|{r['recorded_position_max_cm']:.3f}|{r['recorded_full_rotation_final_deg']:.2f}|{r['contact_frames']}|{r['failure_reasons']or'无'}|")
    lines+=['','原始Lance与配套审计数据保持不变；新的实际状态/原生接触和逐条JSON保存在各动作子目录。倒杯不代表验证流体输送。']
    (root/'README.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    result=summarize(a.root);print(json.dumps({k:v for k,v in result.items()if k!='rows'},ensure_ascii=False,indent=2))
