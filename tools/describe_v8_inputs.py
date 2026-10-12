#!/usr/bin/env python3
"""Generate the source-ordered v8 observation and goal-snapshot dictionaries."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.manorl.autonomy_contracts import (
    CRITIC_ONLY_FIELDS_VOXMANI453,
    RAW_OBSERVATION_FIELDS_VOXMANI453,
)
from sim.manorl.autonomy_v4 import ANTIWINDUP_ERROR, DOF_RATE
from sim.manorl.contracts import JOINT_NAMES_28, KEYPOINT_NAMES
from sim.manorl.voxmani453.config import PAST_FRAMES, SCENE_SIGMA, T_MAX

AXES = ("x", "y", "z")
QUAT = ("w", "x", "y", "z")
ROT6 = ("R00", "R10", "R20", "R01", "R11", "R21")
JOINT_LABELS = (
    "腕部x平移",
    "腕部y平移",
    "腕部z平移",
    "腕部x旋转",
    "腕部y旋转",
    "腕部z旋转",
    "拇指CMC外展",
    "拇指CMC屈曲",
    "拇指CMC扭转",
    "拇指MCP屈曲",
    "拇指MCP外展",
    "拇指IP屈曲",
    "食指MCP外展",
    "食指MCP屈曲",
    "食指PIP屈曲",
    "食指DIP屈曲",
    "中指MCP外展",
    "中指MCP屈曲",
    "中指PIP屈曲",
    "中指DIP屈曲",
    "无名指MCP外展",
    "无名指MCP屈曲",
    "无名指PIP屈曲",
    "无名指DIP屈曲",
    "小指MCP外展",
    "小指MCP屈曲",
    "小指PIP屈曲",
    "小指DIP屈曲",
)
BONE_LABELS = (
    "掌部",
    "食指近端",
    "食指中段",
    "食指远端",
    "中指近端",
    "中指中段",
    "中指远端",
    "无名指近端",
    "无名指中段",
    "无名指远端",
    "小指近端",
    "小指中段",
    "小指远端",
    "拇指根部",
    "拇指中段",
    "拇指远端",
)


@dataclass(frozen=True)
class Dimension:
    name: str
    formula: str
    unit: str
    purpose: str


def xyz(name: str, formula: str, unit: str, purpose: str) -> list[Dimension]:
    return [
        Dimension(f"{name}.{axis}", formula.replace("{a}", axis), unit, purpose)
        for axis in AXES
    ]


def pose_dimensions(name: str, purpose: str) -> list[Dimension]:
    return xyz(
        f"{name}.position",
        f"{name}.position.{{a}} / {SCENE_SIGMA:g}",
        "m / 0.2m",
        purpose,
    ) + [
        Dimension(
            f"{name}.quat.{component}",
            f"{name}.quat_wxyz[{i}]",
            "无量纲",
            f"{purpose}；四元数四个分量共同表示朝向",
        )
        for i, component in enumerate(QUAT)
    ]


def observation_fields() -> dict[str, list[Dimension]]:
    fields = {
        "obj_part_pose": pose_dimensions("object", "重建当前物体表面点和当前场景特征"),
        "obj_past_pose": [
            d
            for lag in PAST_FRAMES
            for d in pose_dimensions(
                f"object_at_t_minus_{lag}", "重建实测历史物体几何，提供运动趋势"
            )
        ],
        "obj_past_pad": [
            Dimension(
                f"history_missing_lag_{lag}",
                f"history_count < {lag}",
                "0或1",
                "1表示历史不足，姿态由当前姿态填充；作为点特征标志输入，并非硬注意力屏蔽",
            )
            for lag in PAST_FRAMES
        ],
        "hand_body_pose": [
            d
            for bone, label in zip(KEYPOINT_NAMES, BONE_LABELS)
            for d in pose_dimensions(bone, f"放置{label}表面点，保留手部局部几何")
        ],
        "finger_q": [],
        "finger_qd": [],
        "bone_contact": [],
        "obj_force_obs": [],
        "wrist_twist": xyz(
            "palm_origin_velocity",
            "palm_v_com.{a} + (palm_w × (palm_origin-palm_com)).{a}",
            "m/s",
            "掌部原点线速度，用于运动状态编码",
        )
        + xyz(
            "palm_angular_velocity",
            "palm_w.{a} / 10",
            "(rad/s) / 10",
            "掌部角速度，用于运动状态编码",
        ),
        "obj_twist": xyz(
            "object_com_velocity",
            "object_v_com.{a}",
            "m/s",
            "物体质心线速度，用于运动状态编码",
        )
        + xyz(
            "object_angular_velocity",
            "object_w.{a} / 10",
            "(rad/s) / 10",
            "物体角速度，用于运动状态编码",
        ),
        "mass_obs": [
            Dimension(
                "mass_channel",
                "0",
                "无量纲",
                "当前固定质量下恒0；不提供跨质量差异，力归一化仍读取真实质量",
            )
        ],
        "prev_action": [],
        "priv_13": [],
        "support_ratio_obs": [
            Dimension(
                "signed_hand_vertical_support",
                "tanh(sum_bone(force_on_object.z) / weight)",
                "无量纲，范围(-1,1)",
                "手对物体的向上净接触力相对重量；不是支撑百分比，也不含桌面接触力",
            )
        ],
        "contact_bone_count_obs": [
            Dimension(
                "contact_bone_fraction",
                "count(paired_count > 0) / 16",
                "0到1",
                "有求解手物接触的骨骼占比；不是接触点总数或承载骨骼比例",
            )
        ],
        "servo_error": [],
        "critic_keypoint_error": [],
        "terminal_hand_goal": xyz(
            "final_palm_displacement",
            "(final_reference_palm.{a}-actual_palm.{a}) / 0.2",
            "m / 0.2m",
            "给Actor和Critic提供最终掌部目标方向与距离",
        )
        + [
            Dimension(
                f"terminal_relative_rotation.{component}",
                f"R(inverse(current_palm) * final_palm).{component}",
                "无量纲",
                "最终掌部相对当前掌部的旋转；前两列按列展开，不是六个角度",
            )
            for component in ROT6
        ]
        + [
            Dimension(
                "reference_end_reached",
                "index >= max(last_index, 1)",
                "0或1",
                "参考索引是否达到终点；不是实际任务成功标志",
            )
        ],
    }
    for local, (name, label) in enumerate(zip(JOINT_NAMES_28[6:], JOINT_LABELS[6:])):
        fields["finger_q"].append(
            Dimension(
                name, f"q_raw[{local + 6}]", "rad", f"{label}实际角度，送入骨架传播"
            )
        )
        fields["finger_qd"].append(
            Dimension(
                name,
                f"qdot_normalized[{local + 6}] * {DOF_RATE[local + 6]:g} / 10",
                "(rad/s) / 10",
                f"{label}实际角速度除以10，送入骨架传播",
            )
        )
    for bone, label in zip(KEYPOINT_NAMES, BONE_LABELS):
        fields["bone_contact"] += xyz(
            bone,
            f"tanh(-force_on_object[{bone}].{{a}} / 10)",
            "无量纲",
            f"物体对{label}的反力，经tanh压缩；送入骨架传播",
        )
        fields["critic_keypoint_error"] += xyz(
            bone,
            f"(actual_origin[{bone}].{{a}}-reference_origin[{bone}].{{a}}) / 0.2",
            "m / 0.2m",
            f"{label}实际位置相对当前参考位置的误差；仅用于价值估计",
        )
    fields["obj_force_obs"] += xyz(
        "gravity",
        "mass * gravity_world.{a} / weight",
        "无量纲",
        "重力方向；与接触力分开，普通竖直重力下为[0,0,-1]",
    )
    for bone, label in zip(KEYPOINT_NAMES, BONE_LABELS):
        fields["obj_force_obs"] += xyz(
            f"force_on_object_from_{bone}",
            f"force_on_object[{bone}].{{a}} / weight",
            "N / weight(N)",
            f"{label}作用于物体的世界坐标接触力，给两塔低维状态分支",
        )
    fields["obj_force_obs"] += xyz(
        "hand_torque_on_object",
        "sum_bone(torque_about_object_com).{a} / torque_scale",
        "Nm / torque_scale(Nm)",
        "手对物体质心的净接触力矩",
    )
    fields["obj_force_obs"] += xyz(
        "non_hand_contact_force",
        "(object_all_contact_force-sum_bone(hand_force)).{a} / weight",
        "N / weight(N)",
        "桌面等非手接触施加于物体的净力；此项不含重力",
    )
    fields["obj_force_obs"] += xyz(
        "non_hand_contact_torque",
        "(object_all_contact_torque-sum_bone(hand_torque)).{a} / torque_scale",
        "Nm / torque_scale(Nm)",
        "桌面等非手接触对物体质心的净力矩",
    )
    for i, (name, label) in enumerate(zip(JOINT_NAMES_28, JOINT_LABELS)):
        fields["prev_action"].append(
            Dimension(
                name,
                f"clip(previous_raw_action[{i}], -1, 1)",
                "无量纲",
                f"{label}上一物理步施加的归一化速率指令；不等于实际位移或原始Normal样本",
            )
        )
        fields["servo_error"].append(
            Dimension(
                name,
                f"(held_command[{i}]-q_raw[{i}]) / {ANTIWINDUP_ERROR[i]:g}",
                "m/m" if i < 3 else "rad/rad",
                f"{label}持久伺服目标与实际位置差；提供执行滞后/加载信息，除数为antiwindup尺度",
            )
        )
    fields["priv_13"] = (
        xyz(
            "object_reference_position_error",
            "(actual_object.{a}-current_reference_object.{a}) / 0.2",
            "m / 0.2m",
            "当前物体位置追踪误差；仅Critic",
        )
        + [
            Dimension(
                "object_reference_rotation_error",
                "shortest_quaternion_angle(actual_object, reference_object) / pi",
                "rad / pi",
                "物体朝向误差大小，无方向；仅Critic",
            )
        ]
        + xyz(
            "object_reference_velocity_error",
            "actual_object_v_com.{a}-reference_object_v_com.{a}",
            "m/s",
            "当前物体质心速度追踪误差；仅Critic",
        )
        + [
            Dimension(
                "hand_position_rms_error",
                "RMS(actual_16_bone_positions-reference_positions) / 0.1",
                "m / 0.1m",
                "16块骨骼位置误差的均方根；仅Critic",
            ),
            Dimension(
                "finger_angle_rms_error",
                "RMS(actual_22_finger_angles-reference_feasible_angles)",
                "rad",
                "22个手指角度误差的均方根；仅Critic",
            ),
            Dimension(
                "reference_progress",
                "now / end",
                "无量纲",
                "参考时间进度；仅Critic，不代表任务完成率",
            ),
            Dimension(
                "reference_remaining_time",
                f"(end-now) / {T_MAX:g}",
                "s / T_MAX",
                "剩余参考时间；仅Critic",
            ),
            Dimension(
                "constant_one", "1", "无量纲", "当前实现中的常量占位通道；仅Critic"
            ),
            Dimension(
                "reference_remaining_short_window",
                f"min(end-now, control_timestep) / {T_MAX:g}",
                "s / T_MAX",
                "最多一个控制步的剩余时间窗口；大部分轨迹中恒定，末端变化；仅Critic",
            ),
        ]
    )
    return fields


def observation_rows() -> list[dict]:
    fields = observation_fields()
    if set(fields) != {name for name, _ in RAW_OBSERVATION_FIELDS_VOXMANI453}:
        raise ValueError("dimension descriptions do not match observation fields")
    rows = []
    for field, width in RAW_OBSERVATION_FIELDS_VOXMANI453:
        dims = fields[field]
        if len(dims) != width:
            raise ValueError(f"{field}: expected {width} descriptions, got {len(dims)}")
        for d in dims:
            rows.append(
                dict(
                    index=len(rows),
                    field=field,
                    component=d.name,
                    formula=d.formula,
                    unit=d.unit,
                    purpose=d.purpose,
                    consumer=(
                        "Critic"
                        if field in CRITIC_ONLY_FIELDS_VOXMANI453
                        else "Actor + Critic"
                    ),
                )
            )
    return rows


def goal_rows() -> list[dict]:
    object_dims = (
        xyz(
            "object_goal_position",
            "reference_object.position.{a}",
            "m",
            "目标物体位置，直接送入目标记录投影",
        )
        + [
            Dimension(
                f"object_goal_rotation.{c}",
                f"R(reference_object).{c}",
                "无量纲",
                "目标物体旋转矩阵前两列",
            )
            for c in ROT6
        ]
        + [
            Dimension(
                "is_endpoint",
                "target_frame == last_frame",
                "0或1",
                "区分最终目标与中间目标",
            )
        ]
        + [
            d
            for point in (0, 16, 32, 48)
            for d in xyz(
                f"object_goal_point_{point}",
                f"(reference_object.position + R(reference_object) * local_point[{point}]).{{a}}",
                "m",
                "4个物体目标几何点之一，表明目标几何位置",
            )
        ]
    )
    hand_dims = (
        xyz("palm_goal_position", "reference_palm.position.{a}", "m", "目标掌部位置")
        + [
            Dimension(
                f"palm_goal_rotation.{c}",
                f"R(reference_palm).{c}",
                "无量纲",
                "目标掌部旋转矩阵前两列",
            )
            for c in ROT6
        ]
        + [
            Dimension(name, f"reference_q_feasible[{i + 6}]", "rad", f"{label}目标角度")
            for i, (name, label) in enumerate(zip(JOINT_NAMES_28[6:], JOINT_LABELS[6:]))
        ]
        + [
            d
            for bone, label in zip(KEYPOINT_NAMES, BONE_LABELS)
            for d in xyz(
                f"goal_{bone}",
                f"reference_bone_origin_world[{bone}].{{a}}",
                "m",
                f"{label}目标位置",
            )
        ]
    )
    if (len(object_dims), len(hand_dims)) != (22, 79):
        raise ValueError("goal raw dimensions must be 22 and 79")
    rows = []

    def add(stream, slot, d):
        rows.append(
            dict(
                index=len(rows),
                stream=stream,
                slot=slot,
                component=d.name,
                formula=d.formula,
                unit=d.unit,
                purpose=d.purpose,
            )
        )

    for stream, dims, kind in (("object", object_dims, 0), ("hand", hand_dims, 2)):
        for slot in range(5):
            for d in dims:
                add(stream, slot, d)
        for metadata, formula, unit, purpose in (
            (
                "mask",
                "target_valid",
                "0或1",
                "有效记录标志，控制目标注意力；无效raw也置零",
            ),
            (
                "dt",
                "(target_frame-current_frame) * control_timestep",
                "s",
                "目标相对当前时刻；编码为dt/T_MAX与8组sin/cos，共17维",
            ),
            ("frame_id", "target_frame", "帧号", "保存来源，完全不进入网络编码"),
            ("type_id", str(kind), "类别编号", "目标类型嵌入：物体0，手部2"),
        ):
            for slot in range(5):
                add(stream, slot, Dimension(metadata, formula, unit, purpose))
        add(
            stream,
            None,
            Dimension(
                "presence",
                "1 in current ManoRL adapter",
                "0到1",
                "整路目标的注意力门控；当前两路均存在，缺失流可设0",
            ),
        )
    if len(rows) != 547:
        raise ValueError("goal snapshot width must be 547")
    return rows


def table(rows, columns):
    out = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        out.append(
            "| "
            + " | ".join(str(row[key]).replace("|", "/") for key, _ in columns)
            + " |"
        )
    return "\n".join(out)


def write_guide(output: Path) -> None:
    observations, goals = observation_rows(), goal_rows()
    output.mkdir(parents=True, exist_ok=True)
    overview = []
    offset = 0
    for field, width in RAW_OBSERVATION_FIELDS_VOXMANI453:
        overview.append(
            dict(
                field=field,
                indices=f"{offset}–{offset + width - 1}",
                width=width,
                consumer=observations[offset]["consumer"],
            )
        )
        offset += width
    text = f"""# v8：453维观测与目标输入逐维说明

v8 是原 `v5.voxmani453` 的正式公开名称，核心架构取自 VoxMani
`16a80fc6615c3f82194acd34e45b98d459359325`，运行于 ManoRL。
`--policy-version v8` 和旧名称均可使用。此次命名不改变网络、权重形状、
453维顺序或547维目标序列化格式。持久化ABI与架构ID沿用旧字符串，兼容旧检查点。

## 维数的三个口径

- `observations`: `[B,453]`；包含当前状态、实测历史、终端手目标，以及61维Critic专用信息。
- `states`: `[B,547]`；PPO每步保存的两路目标快照，也会提供给Actor；不是547维Critic特权状态。
- 输入载荷合计每个样本1000个数值，分两个张量传入，并非一个1000维物理观测向量。
- Actor可读取453中的392维；Critic额外读取索引352–364和395–442，共61维。
- 547中有10个`frame_id`只用于记录、不编码。掩码/门控决定有效目标；
  所以1000不能解释成1000个独立物理量，也不是网络内部特征维数。
- 两塔当前侧各70个token，每个128维；Actor低维分支314维，Critic375维。
  314维里含骨架模式下恒0的92维占位，关节/骨骼反力改走骨架分支，信息没有被整体删除。
- 每路目标5个存储槽，加一个网络生成的NULL token，共6个目标token；两个流分别做交叉注意力。

## 读表约定

索引从0开始：索引0是口语中的第1维；索引452是第453维。
表中列出了环境写入值；几何位置在网络中还会展开为点云，不能把各列当作直接拼接的最终特征。
物体与骨骼位置为世界坐标除以{SCENE_SIGMA:g}米，四元数顺序w,x,y,z。
目标快照中的位置直接使用米，没有除以0.2。所有接触力/力矩方向采用世界坐标，
力矩参考物体质心；手作用于物体和物体作用于手的力方向相反。
`weight=max(mass*norm(gravity),1e-6)`，`torque_scale=max(weight*object_radius,1e-6)`。
`now=index*control_timestep`，`end=max(last_index,1)*control_timestep`，`T_MAX={T_MAX:g}`秒。
当前控制频率120Hz，实测历史间隔1/2/4/8/16步约为8.33/16.67/33.33/66.67/133.33毫秒。
四元数与rot6必须成组解释；rot6六个值是矩阵前两列，不是六个欧拉角。

## 字段总表

{table(overview, [('indices', '0基索引'), ('field', '字段'), ('width', '维数'), ('consumer', '读取方')])}

## 两种排列顺序

手骨骼按 `KEYPOINT_NAMES`：掌部、食指3段、中指3段、无名指3段、小指3段、拇指3段。
关节按 `JOINT_NAMES_28`：腕部6自由度、拇指6关节、食指4、中指4、无名指4、小指4。
骨骼顺序中拇指在最后，关节顺序中拇指在最前；不可用同一手指顺序解析二者。
prev_action和servo_error都按28自由度关节顺序；finger_q/finger_qd使用其后22个。
MCP是掌指关节，PIP/DIP是近端/远端指间关节；拇指CMC是腕掌关节，IP是指间关节。

## 几个需要具体理解的通道

- 323质量通道当前恒0；251–310力通道仍按真实质量和尺寸归一化。
- 203–250是物体对手的反力`tanh(-F/10)`；254–301是手对物体的力`F/weight`。
  它们方向、尺度、进入的分支均不同，不能简单当作重复。
- 365支撑值为`tanh(F_hand_z/weight)`。向上力等于重量时为约0.7616，并非100%。
- 366是有接触的骨骼数/16；4块骨骼接触时为0.25。不是四个接触点的计数。
- 367–394伺服误差可表示动作未完全实现时的加载状态，但不是直接测得的电机力矩。
- 352–364和395–442只进入Critic。Actor仍能读取显式目标流与终端手目标；
  “特权隔离”不表示Actor完全不接触参考目标。
- 452标志表示参考时间到达终点，不代表物体操作成功。
- 42–46历史填充标志表示历史不足；几何使用当前姿态替代，标志送进点编码器。
- 全表是程序的数据语义，不是已训练网络的因果解释；不能由某维存在推出模型一定会利用它。

## 453维逐维字典

{table(observations, [('index', '索引'), ('field', '字段'), ('component', '具体分量'), ('formula', '写入公式'), ('unit', '单位/尺度'), ('consumer', '读取方'), ('purpose', '用途')])}

## 目标输入

完整547维逐维说明见 [v8_goal_snapshot_dimensions.md](v8_goal_snapshot_dimensions.md)。
物体每条22维：位置3、旋转6、终点标志1、4个目标几何点12。
手部每条79维：掌位置3、旋转6、手指角22、16块骨骼位置48。
通常取当前、未来6/12/24帧与终点，末端会去重并将有效记录前移、剩余槽屏蔽。
因此存储槽编号不等于固定的未来时间；实际时刻以dt与frame_id为准。

## 权威源码与生成方法

- 字段顺序：`sim/manorl/autonomy_contracts.py::RAW_OBSERVATION_FIELDS_VOXMANI453`
- 写入公式：`sim/manorl/voxmani453/observation.py::build_raw_observation_voxmani453`
- 读取与路由：`sim/manorl/voxmani453/network.py::ObsEncoder.encode`
- 骨骼/关节顺序：`sim/manorl/contracts.py::KEYPOINT_NAMES/JOINT_NAMES_28`
- 目标取样/序列化：`sim/manorl/voxmani453/streams.py`
- 目标编码：`sim/manorl/voxmani453/network.py::StreamEncoder.forward`

重新生成：`python tools/describe_v8_inputs.py --output docs`
机器可读字典：`docs/v8_input_dimensions.json`。字段数量由程序契约校验，解释公式需随源码审阅。
"""
    (output / "v8_observation_dimensions.md").write_text(text, encoding="utf-8")
    goal_text = f"""# v8：547维目标快照逐维说明

这里的索引是`states`张量内部0基索引，不是453维`observations`索引。
两塔都读取目标流。记录先按槽展开raw，再分别存mask、dt、frame_id、type_id，最后presence。
网络输入raw、时间编码、类型嵌入；mask/presence控制注意力；frame_id不编码。
每路5个存储槽，通常为当前/+6/+12/+24/终点，末端有效记录会前移，空槽raw和元数据置零。
“同一个槽”中的目标必须连同当步453维状态一起存入PPO，训练时不可读取已经前进的环境目标。

| states索引 | 内容 |
| --- | --- |
| 0–109 | 5条物体raw，每条22 |
| 110–114 | 物体mask |
| 115–119 | 物体dt |
| 120–124 | 物体frame_id，仅记录 |
| 125–129 | 物体type_id，当前0 |
| 130 | 物体presence，当前1 |
| 131–525 | 5条手部raw，每条79 |
| 526–530 | 手部mask |
| 531–535 | 手部dt |
| 536–540 | 手部frame_id，仅记录 |
| 541–545 | 手部type_id，当前2 |
| 546 | 手部presence，当前1 |

槽编号从0到4；presence不属于某个槽。以下raw公式适用于有效记录，无效记录写零。
位置/几何点为世界坐标米，手指角为弧度，旋转列为无量纲。
注意手部参考骨骼位置由参考物体位姿和物体局部参考骨骼位置还原，并不是实时手骨骼位置。

{table(goals, [('index', '索引'), ('stream', '流'), ('slot', '槽'), ('component', '分量'), ('formula', '写入公式'), ('unit', '单位'), ('purpose', '用途')])}
"""
    (output / "v8_goal_snapshot_dimensions.md").write_text(goal_text, encoding="utf-8")
    (output / "v8_input_dimensions.json").write_text(
        json.dumps(
            dict(
                architecture="v8",
                index_base=0,
                observation_width=453,
                goal_snapshot_width=547,
                payload_width=1000,
                observations=observations,
                goal_snapshot=goals,
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs")
    write_guide(parser.parse_args().output)
