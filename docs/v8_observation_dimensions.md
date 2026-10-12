# v8：453维观测与目标输入逐维说明

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
物体与骨骼位置为世界坐标除以0.2米，四元数顺序w,x,y,z。
目标快照中的位置直接使用米，没有除以0.2。所有接触力/力矩方向采用世界坐标，
力矩参考物体质心；手作用于物体和物体作用于手的力方向相反。
`weight=max(mass*norm(gravity),1e-6)`，`torque_scale=max(weight*object_radius,1e-6)`。
`now=index*control_timestep`，`end=max(last_index,1)*control_timestep`，`T_MAX=5.475`秒。
当前控制频率120Hz，实测历史间隔1/2/4/8/16步约为8.33/16.67/33.33/66.67/133.33毫秒。
四元数与rot6必须成组解释；rot6六个值是矩阵前两列，不是六个欧拉角。

## 字段总表

| 0基索引 | 字段 | 维数 | 读取方 |
| --- | --- | --- | --- |
| 0–6 | obj_part_pose | 7 | Actor + Critic |
| 7–41 | obj_past_pose | 35 | Actor + Critic |
| 42–46 | obj_past_pad | 5 | Actor + Critic |
| 47–158 | hand_body_pose | 112 | Actor + Critic |
| 159–180 | finger_q | 22 | Actor + Critic |
| 181–202 | finger_qd | 22 | Actor + Critic |
| 203–250 | bone_contact | 48 | Actor + Critic |
| 251–310 | obj_force_obs | 60 | Actor + Critic |
| 311–316 | wrist_twist | 6 | Actor + Critic |
| 317–322 | obj_twist | 6 | Actor + Critic |
| 323–323 | mass_obs | 1 | Actor + Critic |
| 324–351 | prev_action | 28 | Actor + Critic |
| 352–364 | priv_13 | 13 | Critic |
| 365–365 | support_ratio_obs | 1 | Actor + Critic |
| 366–366 | contact_bone_count_obs | 1 | Actor + Critic |
| 367–394 | servo_error | 28 | Actor + Critic |
| 395–442 | critic_keypoint_error | 48 | Critic |
| 443–452 | terminal_hand_goal | 10 | Actor + Critic |

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

| 索引 | 字段 | 具体分量 | 写入公式 | 单位/尺度 | 读取方 | 用途 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | obj_part_pose | object.position.x | object.position.x / 0.2 | m / 0.2m | Actor + Critic | 重建当前物体表面点和当前场景特征 |
| 1 | obj_part_pose | object.position.y | object.position.y / 0.2 | m / 0.2m | Actor + Critic | 重建当前物体表面点和当前场景特征 |
| 2 | obj_part_pose | object.position.z | object.position.z / 0.2 | m / 0.2m | Actor + Critic | 重建当前物体表面点和当前场景特征 |
| 3 | obj_part_pose | object.quat.w | object.quat_wxyz[0] | 无量纲 | Actor + Critic | 重建当前物体表面点和当前场景特征；四元数四个分量共同表示朝向 |
| 4 | obj_part_pose | object.quat.x | object.quat_wxyz[1] | 无量纲 | Actor + Critic | 重建当前物体表面点和当前场景特征；四元数四个分量共同表示朝向 |
| 5 | obj_part_pose | object.quat.y | object.quat_wxyz[2] | 无量纲 | Actor + Critic | 重建当前物体表面点和当前场景特征；四元数四个分量共同表示朝向 |
| 6 | obj_part_pose | object.quat.z | object.quat_wxyz[3] | 无量纲 | Actor + Critic | 重建当前物体表面点和当前场景特征；四元数四个分量共同表示朝向 |
| 7 | obj_past_pose | object_at_t_minus_1.position.x | object_at_t_minus_1.position.x / 0.2 | m / 0.2m | Actor + Critic | 重建实测历史物体几何，提供运动趋势 |
| 8 | obj_past_pose | object_at_t_minus_1.position.y | object_at_t_minus_1.position.y / 0.2 | m / 0.2m | Actor + Critic | 重建实测历史物体几何，提供运动趋势 |
| 9 | obj_past_pose | object_at_t_minus_1.position.z | object_at_t_minus_1.position.z / 0.2 | m / 0.2m | Actor + Critic | 重建实测历史物体几何，提供运动趋势 |
| 10 | obj_past_pose | object_at_t_minus_1.quat.w | object_at_t_minus_1.quat_wxyz[0] | 无量纲 | Actor + Critic | 重建实测历史物体几何，提供运动趋势；四元数四个分量共同表示朝向 |
| 11 | obj_past_pose | object_at_t_minus_1.quat.x | object_at_t_minus_1.quat_wxyz[1] | 无量纲 | Actor + Critic | 重建实测历史物体几何，提供运动趋势；四元数四个分量共同表示朝向 |
| 12 | obj_past_pose | object_at_t_minus_1.quat.y | object_at_t_minus_1.quat_wxyz[2] | 无量纲 | Actor + Critic | 重建实测历史物体几何，提供运动趋势；四元数四个分量共同表示朝向 |
| 13 | obj_past_pose | object_at_t_minus_1.quat.z | object_at_t_minus_1.quat_wxyz[3] | 无量纲 | Actor + Critic | 重建实测历史物体几何，提供运动趋势；四元数四个分量共同表示朝向 |
| 14 | obj_past_pose | object_at_t_minus_2.position.x | object_at_t_minus_2.position.x / 0.2 | m / 0.2m | Actor + Critic | 重建实测历史物体几何，提供运动趋势 |
| 15 | obj_past_pose | object_at_t_minus_2.position.y | object_at_t_minus_2.position.y / 0.2 | m / 0.2m | Actor + Critic | 重建实测历史物体几何，提供运动趋势 |
| 16 | obj_past_pose | object_at_t_minus_2.position.z | object_at_t_minus_2.position.z / 0.2 | m / 0.2m | Actor + Critic | 重建实测历史物体几何，提供运动趋势 |
| 17 | obj_past_pose | object_at_t_minus_2.quat.w | object_at_t_minus_2.quat_wxyz[0] | 无量纲 | Actor + Critic | 重建实测历史物体几何，提供运动趋势；四元数四个分量共同表示朝向 |
| 18 | obj_past_pose | object_at_t_minus_2.quat.x | object_at_t_minus_2.quat_wxyz[1] | 无量纲 | Actor + Critic | 重建实测历史物体几何，提供运动趋势；四元数四个分量共同表示朝向 |
| 19 | obj_past_pose | object_at_t_minus_2.quat.y | object_at_t_minus_2.quat_wxyz[2] | 无量纲 | Actor + Critic | 重建实测历史物体几何，提供运动趋势；四元数四个分量共同表示朝向 |
| 20 | obj_past_pose | object_at_t_minus_2.quat.z | object_at_t_minus_2.quat_wxyz[3] | 无量纲 | Actor + Critic | 重建实测历史物体几何，提供运动趋势；四元数四个分量共同表示朝向 |
| 21 | obj_past_pose | object_at_t_minus_4.position.x | object_at_t_minus_4.position.x / 0.2 | m / 0.2m | Actor + Critic | 重建实测历史物体几何，提供运动趋势 |
| 22 | obj_past_pose | object_at_t_minus_4.position.y | object_at_t_minus_4.position.y / 0.2 | m / 0.2m | Actor + Critic | 重建实测历史物体几何，提供运动趋势 |
| 23 | obj_past_pose | object_at_t_minus_4.position.z | object_at_t_minus_4.position.z / 0.2 | m / 0.2m | Actor + Critic | 重建实测历史物体几何，提供运动趋势 |
| 24 | obj_past_pose | object_at_t_minus_4.quat.w | object_at_t_minus_4.quat_wxyz[0] | 无量纲 | Actor + Critic | 重建实测历史物体几何，提供运动趋势；四元数四个分量共同表示朝向 |
| 25 | obj_past_pose | object_at_t_minus_4.quat.x | object_at_t_minus_4.quat_wxyz[1] | 无量纲 | Actor + Critic | 重建实测历史物体几何，提供运动趋势；四元数四个分量共同表示朝向 |
| 26 | obj_past_pose | object_at_t_minus_4.quat.y | object_at_t_minus_4.quat_wxyz[2] | 无量纲 | Actor + Critic | 重建实测历史物体几何，提供运动趋势；四元数四个分量共同表示朝向 |
| 27 | obj_past_pose | object_at_t_minus_4.quat.z | object_at_t_minus_4.quat_wxyz[3] | 无量纲 | Actor + Critic | 重建实测历史物体几何，提供运动趋势；四元数四个分量共同表示朝向 |
| 28 | obj_past_pose | object_at_t_minus_8.position.x | object_at_t_minus_8.position.x / 0.2 | m / 0.2m | Actor + Critic | 重建实测历史物体几何，提供运动趋势 |
| 29 | obj_past_pose | object_at_t_minus_8.position.y | object_at_t_minus_8.position.y / 0.2 | m / 0.2m | Actor + Critic | 重建实测历史物体几何，提供运动趋势 |
| 30 | obj_past_pose | object_at_t_minus_8.position.z | object_at_t_minus_8.position.z / 0.2 | m / 0.2m | Actor + Critic | 重建实测历史物体几何，提供运动趋势 |
| 31 | obj_past_pose | object_at_t_minus_8.quat.w | object_at_t_minus_8.quat_wxyz[0] | 无量纲 | Actor + Critic | 重建实测历史物体几何，提供运动趋势；四元数四个分量共同表示朝向 |
| 32 | obj_past_pose | object_at_t_minus_8.quat.x | object_at_t_minus_8.quat_wxyz[1] | 无量纲 | Actor + Critic | 重建实测历史物体几何，提供运动趋势；四元数四个分量共同表示朝向 |
| 33 | obj_past_pose | object_at_t_minus_8.quat.y | object_at_t_minus_8.quat_wxyz[2] | 无量纲 | Actor + Critic | 重建实测历史物体几何，提供运动趋势；四元数四个分量共同表示朝向 |
| 34 | obj_past_pose | object_at_t_minus_8.quat.z | object_at_t_minus_8.quat_wxyz[3] | 无量纲 | Actor + Critic | 重建实测历史物体几何，提供运动趋势；四元数四个分量共同表示朝向 |
| 35 | obj_past_pose | object_at_t_minus_16.position.x | object_at_t_minus_16.position.x / 0.2 | m / 0.2m | Actor + Critic | 重建实测历史物体几何，提供运动趋势 |
| 36 | obj_past_pose | object_at_t_minus_16.position.y | object_at_t_minus_16.position.y / 0.2 | m / 0.2m | Actor + Critic | 重建实测历史物体几何，提供运动趋势 |
| 37 | obj_past_pose | object_at_t_minus_16.position.z | object_at_t_minus_16.position.z / 0.2 | m / 0.2m | Actor + Critic | 重建实测历史物体几何，提供运动趋势 |
| 38 | obj_past_pose | object_at_t_minus_16.quat.w | object_at_t_minus_16.quat_wxyz[0] | 无量纲 | Actor + Critic | 重建实测历史物体几何，提供运动趋势；四元数四个分量共同表示朝向 |
| 39 | obj_past_pose | object_at_t_minus_16.quat.x | object_at_t_minus_16.quat_wxyz[1] | 无量纲 | Actor + Critic | 重建实测历史物体几何，提供运动趋势；四元数四个分量共同表示朝向 |
| 40 | obj_past_pose | object_at_t_minus_16.quat.y | object_at_t_minus_16.quat_wxyz[2] | 无量纲 | Actor + Critic | 重建实测历史物体几何，提供运动趋势；四元数四个分量共同表示朝向 |
| 41 | obj_past_pose | object_at_t_minus_16.quat.z | object_at_t_minus_16.quat_wxyz[3] | 无量纲 | Actor + Critic | 重建实测历史物体几何，提供运动趋势；四元数四个分量共同表示朝向 |
| 42 | obj_past_pad | history_missing_lag_1 | history_count < 1 | 0或1 | Actor + Critic | 1表示历史不足，姿态由当前姿态填充；作为点特征标志输入，并非硬注意力屏蔽 |
| 43 | obj_past_pad | history_missing_lag_2 | history_count < 2 | 0或1 | Actor + Critic | 1表示历史不足，姿态由当前姿态填充；作为点特征标志输入，并非硬注意力屏蔽 |
| 44 | obj_past_pad | history_missing_lag_4 | history_count < 4 | 0或1 | Actor + Critic | 1表示历史不足，姿态由当前姿态填充；作为点特征标志输入，并非硬注意力屏蔽 |
| 45 | obj_past_pad | history_missing_lag_8 | history_count < 8 | 0或1 | Actor + Critic | 1表示历史不足，姿态由当前姿态填充；作为点特征标志输入，并非硬注意力屏蔽 |
| 46 | obj_past_pad | history_missing_lag_16 | history_count < 16 | 0或1 | Actor + Critic | 1表示历史不足，姿态由当前姿态填充；作为点特征标志输入，并非硬注意力屏蔽 |
| 47 | hand_body_pose | palm.position.x | palm.position.x / 0.2 | m / 0.2m | Actor + Critic | 放置掌部表面点，保留手部局部几何 |
| 48 | hand_body_pose | palm.position.y | palm.position.y / 0.2 | m / 0.2m | Actor + Critic | 放置掌部表面点，保留手部局部几何 |
| 49 | hand_body_pose | palm.position.z | palm.position.z / 0.2 | m / 0.2m | Actor + Critic | 放置掌部表面点，保留手部局部几何 |
| 50 | hand_body_pose | palm.quat.w | palm.quat_wxyz[0] | 无量纲 | Actor + Critic | 放置掌部表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 51 | hand_body_pose | palm.quat.x | palm.quat_wxyz[1] | 无量纲 | Actor + Critic | 放置掌部表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 52 | hand_body_pose | palm.quat.y | palm.quat_wxyz[2] | 无量纲 | Actor + Critic | 放置掌部表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 53 | hand_body_pose | palm.quat.z | palm.quat_wxyz[3] | 无量纲 | Actor + Critic | 放置掌部表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 54 | hand_body_pose | index_mcp.position.x | index_mcp.position.x / 0.2 | m / 0.2m | Actor + Critic | 放置食指近端表面点，保留手部局部几何 |
| 55 | hand_body_pose | index_mcp.position.y | index_mcp.position.y / 0.2 | m / 0.2m | Actor + Critic | 放置食指近端表面点，保留手部局部几何 |
| 56 | hand_body_pose | index_mcp.position.z | index_mcp.position.z / 0.2 | m / 0.2m | Actor + Critic | 放置食指近端表面点，保留手部局部几何 |
| 57 | hand_body_pose | index_mcp.quat.w | index_mcp.quat_wxyz[0] | 无量纲 | Actor + Critic | 放置食指近端表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 58 | hand_body_pose | index_mcp.quat.x | index_mcp.quat_wxyz[1] | 无量纲 | Actor + Critic | 放置食指近端表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 59 | hand_body_pose | index_mcp.quat.y | index_mcp.quat_wxyz[2] | 无量纲 | Actor + Critic | 放置食指近端表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 60 | hand_body_pose | index_mcp.quat.z | index_mcp.quat_wxyz[3] | 无量纲 | Actor + Critic | 放置食指近端表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 61 | hand_body_pose | index_pip.position.x | index_pip.position.x / 0.2 | m / 0.2m | Actor + Critic | 放置食指中段表面点，保留手部局部几何 |
| 62 | hand_body_pose | index_pip.position.y | index_pip.position.y / 0.2 | m / 0.2m | Actor + Critic | 放置食指中段表面点，保留手部局部几何 |
| 63 | hand_body_pose | index_pip.position.z | index_pip.position.z / 0.2 | m / 0.2m | Actor + Critic | 放置食指中段表面点，保留手部局部几何 |
| 64 | hand_body_pose | index_pip.quat.w | index_pip.quat_wxyz[0] | 无量纲 | Actor + Critic | 放置食指中段表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 65 | hand_body_pose | index_pip.quat.x | index_pip.quat_wxyz[1] | 无量纲 | Actor + Critic | 放置食指中段表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 66 | hand_body_pose | index_pip.quat.y | index_pip.quat_wxyz[2] | 无量纲 | Actor + Critic | 放置食指中段表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 67 | hand_body_pose | index_pip.quat.z | index_pip.quat_wxyz[3] | 无量纲 | Actor + Critic | 放置食指中段表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 68 | hand_body_pose | index_dip.position.x | index_dip.position.x / 0.2 | m / 0.2m | Actor + Critic | 放置食指远端表面点，保留手部局部几何 |
| 69 | hand_body_pose | index_dip.position.y | index_dip.position.y / 0.2 | m / 0.2m | Actor + Critic | 放置食指远端表面点，保留手部局部几何 |
| 70 | hand_body_pose | index_dip.position.z | index_dip.position.z / 0.2 | m / 0.2m | Actor + Critic | 放置食指远端表面点，保留手部局部几何 |
| 71 | hand_body_pose | index_dip.quat.w | index_dip.quat_wxyz[0] | 无量纲 | Actor + Critic | 放置食指远端表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 72 | hand_body_pose | index_dip.quat.x | index_dip.quat_wxyz[1] | 无量纲 | Actor + Critic | 放置食指远端表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 73 | hand_body_pose | index_dip.quat.y | index_dip.quat_wxyz[2] | 无量纲 | Actor + Critic | 放置食指远端表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 74 | hand_body_pose | index_dip.quat.z | index_dip.quat_wxyz[3] | 无量纲 | Actor + Critic | 放置食指远端表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 75 | hand_body_pose | middle_mcp.position.x | middle_mcp.position.x / 0.2 | m / 0.2m | Actor + Critic | 放置中指近端表面点，保留手部局部几何 |
| 76 | hand_body_pose | middle_mcp.position.y | middle_mcp.position.y / 0.2 | m / 0.2m | Actor + Critic | 放置中指近端表面点，保留手部局部几何 |
| 77 | hand_body_pose | middle_mcp.position.z | middle_mcp.position.z / 0.2 | m / 0.2m | Actor + Critic | 放置中指近端表面点，保留手部局部几何 |
| 78 | hand_body_pose | middle_mcp.quat.w | middle_mcp.quat_wxyz[0] | 无量纲 | Actor + Critic | 放置中指近端表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 79 | hand_body_pose | middle_mcp.quat.x | middle_mcp.quat_wxyz[1] | 无量纲 | Actor + Critic | 放置中指近端表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 80 | hand_body_pose | middle_mcp.quat.y | middle_mcp.quat_wxyz[2] | 无量纲 | Actor + Critic | 放置中指近端表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 81 | hand_body_pose | middle_mcp.quat.z | middle_mcp.quat_wxyz[3] | 无量纲 | Actor + Critic | 放置中指近端表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 82 | hand_body_pose | middle_pip.position.x | middle_pip.position.x / 0.2 | m / 0.2m | Actor + Critic | 放置中指中段表面点，保留手部局部几何 |
| 83 | hand_body_pose | middle_pip.position.y | middle_pip.position.y / 0.2 | m / 0.2m | Actor + Critic | 放置中指中段表面点，保留手部局部几何 |
| 84 | hand_body_pose | middle_pip.position.z | middle_pip.position.z / 0.2 | m / 0.2m | Actor + Critic | 放置中指中段表面点，保留手部局部几何 |
| 85 | hand_body_pose | middle_pip.quat.w | middle_pip.quat_wxyz[0] | 无量纲 | Actor + Critic | 放置中指中段表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 86 | hand_body_pose | middle_pip.quat.x | middle_pip.quat_wxyz[1] | 无量纲 | Actor + Critic | 放置中指中段表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 87 | hand_body_pose | middle_pip.quat.y | middle_pip.quat_wxyz[2] | 无量纲 | Actor + Critic | 放置中指中段表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 88 | hand_body_pose | middle_pip.quat.z | middle_pip.quat_wxyz[3] | 无量纲 | Actor + Critic | 放置中指中段表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 89 | hand_body_pose | middle_dip.position.x | middle_dip.position.x / 0.2 | m / 0.2m | Actor + Critic | 放置中指远端表面点，保留手部局部几何 |
| 90 | hand_body_pose | middle_dip.position.y | middle_dip.position.y / 0.2 | m / 0.2m | Actor + Critic | 放置中指远端表面点，保留手部局部几何 |
| 91 | hand_body_pose | middle_dip.position.z | middle_dip.position.z / 0.2 | m / 0.2m | Actor + Critic | 放置中指远端表面点，保留手部局部几何 |
| 92 | hand_body_pose | middle_dip.quat.w | middle_dip.quat_wxyz[0] | 无量纲 | Actor + Critic | 放置中指远端表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 93 | hand_body_pose | middle_dip.quat.x | middle_dip.quat_wxyz[1] | 无量纲 | Actor + Critic | 放置中指远端表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 94 | hand_body_pose | middle_dip.quat.y | middle_dip.quat_wxyz[2] | 无量纲 | Actor + Critic | 放置中指远端表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 95 | hand_body_pose | middle_dip.quat.z | middle_dip.quat_wxyz[3] | 无量纲 | Actor + Critic | 放置中指远端表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 96 | hand_body_pose | ring_mcp.position.x | ring_mcp.position.x / 0.2 | m / 0.2m | Actor + Critic | 放置无名指近端表面点，保留手部局部几何 |
| 97 | hand_body_pose | ring_mcp.position.y | ring_mcp.position.y / 0.2 | m / 0.2m | Actor + Critic | 放置无名指近端表面点，保留手部局部几何 |
| 98 | hand_body_pose | ring_mcp.position.z | ring_mcp.position.z / 0.2 | m / 0.2m | Actor + Critic | 放置无名指近端表面点，保留手部局部几何 |
| 99 | hand_body_pose | ring_mcp.quat.w | ring_mcp.quat_wxyz[0] | 无量纲 | Actor + Critic | 放置无名指近端表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 100 | hand_body_pose | ring_mcp.quat.x | ring_mcp.quat_wxyz[1] | 无量纲 | Actor + Critic | 放置无名指近端表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 101 | hand_body_pose | ring_mcp.quat.y | ring_mcp.quat_wxyz[2] | 无量纲 | Actor + Critic | 放置无名指近端表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 102 | hand_body_pose | ring_mcp.quat.z | ring_mcp.quat_wxyz[3] | 无量纲 | Actor + Critic | 放置无名指近端表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 103 | hand_body_pose | ring_pip.position.x | ring_pip.position.x / 0.2 | m / 0.2m | Actor + Critic | 放置无名指中段表面点，保留手部局部几何 |
| 104 | hand_body_pose | ring_pip.position.y | ring_pip.position.y / 0.2 | m / 0.2m | Actor + Critic | 放置无名指中段表面点，保留手部局部几何 |
| 105 | hand_body_pose | ring_pip.position.z | ring_pip.position.z / 0.2 | m / 0.2m | Actor + Critic | 放置无名指中段表面点，保留手部局部几何 |
| 106 | hand_body_pose | ring_pip.quat.w | ring_pip.quat_wxyz[0] | 无量纲 | Actor + Critic | 放置无名指中段表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 107 | hand_body_pose | ring_pip.quat.x | ring_pip.quat_wxyz[1] | 无量纲 | Actor + Critic | 放置无名指中段表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 108 | hand_body_pose | ring_pip.quat.y | ring_pip.quat_wxyz[2] | 无量纲 | Actor + Critic | 放置无名指中段表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 109 | hand_body_pose | ring_pip.quat.z | ring_pip.quat_wxyz[3] | 无量纲 | Actor + Critic | 放置无名指中段表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 110 | hand_body_pose | ring_dip.position.x | ring_dip.position.x / 0.2 | m / 0.2m | Actor + Critic | 放置无名指远端表面点，保留手部局部几何 |
| 111 | hand_body_pose | ring_dip.position.y | ring_dip.position.y / 0.2 | m / 0.2m | Actor + Critic | 放置无名指远端表面点，保留手部局部几何 |
| 112 | hand_body_pose | ring_dip.position.z | ring_dip.position.z / 0.2 | m / 0.2m | Actor + Critic | 放置无名指远端表面点，保留手部局部几何 |
| 113 | hand_body_pose | ring_dip.quat.w | ring_dip.quat_wxyz[0] | 无量纲 | Actor + Critic | 放置无名指远端表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 114 | hand_body_pose | ring_dip.quat.x | ring_dip.quat_wxyz[1] | 无量纲 | Actor + Critic | 放置无名指远端表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 115 | hand_body_pose | ring_dip.quat.y | ring_dip.quat_wxyz[2] | 无量纲 | Actor + Critic | 放置无名指远端表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 116 | hand_body_pose | ring_dip.quat.z | ring_dip.quat_wxyz[3] | 无量纲 | Actor + Critic | 放置无名指远端表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 117 | hand_body_pose | pinky_mcp.position.x | pinky_mcp.position.x / 0.2 | m / 0.2m | Actor + Critic | 放置小指近端表面点，保留手部局部几何 |
| 118 | hand_body_pose | pinky_mcp.position.y | pinky_mcp.position.y / 0.2 | m / 0.2m | Actor + Critic | 放置小指近端表面点，保留手部局部几何 |
| 119 | hand_body_pose | pinky_mcp.position.z | pinky_mcp.position.z / 0.2 | m / 0.2m | Actor + Critic | 放置小指近端表面点，保留手部局部几何 |
| 120 | hand_body_pose | pinky_mcp.quat.w | pinky_mcp.quat_wxyz[0] | 无量纲 | Actor + Critic | 放置小指近端表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 121 | hand_body_pose | pinky_mcp.quat.x | pinky_mcp.quat_wxyz[1] | 无量纲 | Actor + Critic | 放置小指近端表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 122 | hand_body_pose | pinky_mcp.quat.y | pinky_mcp.quat_wxyz[2] | 无量纲 | Actor + Critic | 放置小指近端表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 123 | hand_body_pose | pinky_mcp.quat.z | pinky_mcp.quat_wxyz[3] | 无量纲 | Actor + Critic | 放置小指近端表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 124 | hand_body_pose | pinky_pip.position.x | pinky_pip.position.x / 0.2 | m / 0.2m | Actor + Critic | 放置小指中段表面点，保留手部局部几何 |
| 125 | hand_body_pose | pinky_pip.position.y | pinky_pip.position.y / 0.2 | m / 0.2m | Actor + Critic | 放置小指中段表面点，保留手部局部几何 |
| 126 | hand_body_pose | pinky_pip.position.z | pinky_pip.position.z / 0.2 | m / 0.2m | Actor + Critic | 放置小指中段表面点，保留手部局部几何 |
| 127 | hand_body_pose | pinky_pip.quat.w | pinky_pip.quat_wxyz[0] | 无量纲 | Actor + Critic | 放置小指中段表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 128 | hand_body_pose | pinky_pip.quat.x | pinky_pip.quat_wxyz[1] | 无量纲 | Actor + Critic | 放置小指中段表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 129 | hand_body_pose | pinky_pip.quat.y | pinky_pip.quat_wxyz[2] | 无量纲 | Actor + Critic | 放置小指中段表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 130 | hand_body_pose | pinky_pip.quat.z | pinky_pip.quat_wxyz[3] | 无量纲 | Actor + Critic | 放置小指中段表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 131 | hand_body_pose | pinky_dip.position.x | pinky_dip.position.x / 0.2 | m / 0.2m | Actor + Critic | 放置小指远端表面点，保留手部局部几何 |
| 132 | hand_body_pose | pinky_dip.position.y | pinky_dip.position.y / 0.2 | m / 0.2m | Actor + Critic | 放置小指远端表面点，保留手部局部几何 |
| 133 | hand_body_pose | pinky_dip.position.z | pinky_dip.position.z / 0.2 | m / 0.2m | Actor + Critic | 放置小指远端表面点，保留手部局部几何 |
| 134 | hand_body_pose | pinky_dip.quat.w | pinky_dip.quat_wxyz[0] | 无量纲 | Actor + Critic | 放置小指远端表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 135 | hand_body_pose | pinky_dip.quat.x | pinky_dip.quat_wxyz[1] | 无量纲 | Actor + Critic | 放置小指远端表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 136 | hand_body_pose | pinky_dip.quat.y | pinky_dip.quat_wxyz[2] | 无量纲 | Actor + Critic | 放置小指远端表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 137 | hand_body_pose | pinky_dip.quat.z | pinky_dip.quat_wxyz[3] | 无量纲 | Actor + Critic | 放置小指远端表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 138 | hand_body_pose | thumb_cmc.position.x | thumb_cmc.position.x / 0.2 | m / 0.2m | Actor + Critic | 放置拇指根部表面点，保留手部局部几何 |
| 139 | hand_body_pose | thumb_cmc.position.y | thumb_cmc.position.y / 0.2 | m / 0.2m | Actor + Critic | 放置拇指根部表面点，保留手部局部几何 |
| 140 | hand_body_pose | thumb_cmc.position.z | thumb_cmc.position.z / 0.2 | m / 0.2m | Actor + Critic | 放置拇指根部表面点，保留手部局部几何 |
| 141 | hand_body_pose | thumb_cmc.quat.w | thumb_cmc.quat_wxyz[0] | 无量纲 | Actor + Critic | 放置拇指根部表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 142 | hand_body_pose | thumb_cmc.quat.x | thumb_cmc.quat_wxyz[1] | 无量纲 | Actor + Critic | 放置拇指根部表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 143 | hand_body_pose | thumb_cmc.quat.y | thumb_cmc.quat_wxyz[2] | 无量纲 | Actor + Critic | 放置拇指根部表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 144 | hand_body_pose | thumb_cmc.quat.z | thumb_cmc.quat_wxyz[3] | 无量纲 | Actor + Critic | 放置拇指根部表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 145 | hand_body_pose | thumb_mcp.position.x | thumb_mcp.position.x / 0.2 | m / 0.2m | Actor + Critic | 放置拇指中段表面点，保留手部局部几何 |
| 146 | hand_body_pose | thumb_mcp.position.y | thumb_mcp.position.y / 0.2 | m / 0.2m | Actor + Critic | 放置拇指中段表面点，保留手部局部几何 |
| 147 | hand_body_pose | thumb_mcp.position.z | thumb_mcp.position.z / 0.2 | m / 0.2m | Actor + Critic | 放置拇指中段表面点，保留手部局部几何 |
| 148 | hand_body_pose | thumb_mcp.quat.w | thumb_mcp.quat_wxyz[0] | 无量纲 | Actor + Critic | 放置拇指中段表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 149 | hand_body_pose | thumb_mcp.quat.x | thumb_mcp.quat_wxyz[1] | 无量纲 | Actor + Critic | 放置拇指中段表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 150 | hand_body_pose | thumb_mcp.quat.y | thumb_mcp.quat_wxyz[2] | 无量纲 | Actor + Critic | 放置拇指中段表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 151 | hand_body_pose | thumb_mcp.quat.z | thumb_mcp.quat_wxyz[3] | 无量纲 | Actor + Critic | 放置拇指中段表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 152 | hand_body_pose | thumb_ip.position.x | thumb_ip.position.x / 0.2 | m / 0.2m | Actor + Critic | 放置拇指远端表面点，保留手部局部几何 |
| 153 | hand_body_pose | thumb_ip.position.y | thumb_ip.position.y / 0.2 | m / 0.2m | Actor + Critic | 放置拇指远端表面点，保留手部局部几何 |
| 154 | hand_body_pose | thumb_ip.position.z | thumb_ip.position.z / 0.2 | m / 0.2m | Actor + Critic | 放置拇指远端表面点，保留手部局部几何 |
| 155 | hand_body_pose | thumb_ip.quat.w | thumb_ip.quat_wxyz[0] | 无量纲 | Actor + Critic | 放置拇指远端表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 156 | hand_body_pose | thumb_ip.quat.x | thumb_ip.quat_wxyz[1] | 无量纲 | Actor + Critic | 放置拇指远端表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 157 | hand_body_pose | thumb_ip.quat.y | thumb_ip.quat_wxyz[2] | 无量纲 | Actor + Critic | 放置拇指远端表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 158 | hand_body_pose | thumb_ip.quat.z | thumb_ip.quat_wxyz[3] | 无量纲 | Actor + Critic | 放置拇指远端表面点，保留手部局部几何；四元数四个分量共同表示朝向 |
| 159 | finger_q | j1_thumb_cmc_abd | q_raw[6] | rad | Actor + Critic | 拇指CMC外展实际角度，送入骨架传播 |
| 160 | finger_q | j1_thumb_cmc_flex | q_raw[7] | rad | Actor + Critic | 拇指CMC屈曲实际角度，送入骨架传播 |
| 161 | finger_q | j1_thumb_cmc_twist | q_raw[8] | rad | Actor + Critic | 拇指CMC扭转实际角度，送入骨架传播 |
| 162 | finger_q | j1_thumb_mcp_flex | q_raw[9] | rad | Actor + Critic | 拇指MCP屈曲实际角度，送入骨架传播 |
| 163 | finger_q | j1_thumb_mcp_abd | q_raw[10] | rad | Actor + Critic | 拇指MCP外展实际角度，送入骨架传播 |
| 164 | finger_q | j1_thumb_ip | q_raw[11] | rad | Actor + Critic | 拇指IP屈曲实际角度，送入骨架传播 |
| 165 | finger_q | j2_index_mcp_abd | q_raw[12] | rad | Actor + Critic | 食指MCP外展实际角度，送入骨架传播 |
| 166 | finger_q | j2_index_mcp_flex | q_raw[13] | rad | Actor + Critic | 食指MCP屈曲实际角度，送入骨架传播 |
| 167 | finger_q | j2_index_pip | q_raw[14] | rad | Actor + Critic | 食指PIP屈曲实际角度，送入骨架传播 |
| 168 | finger_q | j2_index_dip | q_raw[15] | rad | Actor + Critic | 食指DIP屈曲实际角度，送入骨架传播 |
| 169 | finger_q | j3_middle_mcp_abd | q_raw[16] | rad | Actor + Critic | 中指MCP外展实际角度，送入骨架传播 |
| 170 | finger_q | j3_middle_mcp_flex | q_raw[17] | rad | Actor + Critic | 中指MCP屈曲实际角度，送入骨架传播 |
| 171 | finger_q | j3_middle_pip | q_raw[18] | rad | Actor + Critic | 中指PIP屈曲实际角度，送入骨架传播 |
| 172 | finger_q | j3_middle_dip | q_raw[19] | rad | Actor + Critic | 中指DIP屈曲实际角度，送入骨架传播 |
| 173 | finger_q | j4_ring_mcp_abd | q_raw[20] | rad | Actor + Critic | 无名指MCP外展实际角度，送入骨架传播 |
| 174 | finger_q | j4_ring_mcp_flex | q_raw[21] | rad | Actor + Critic | 无名指MCP屈曲实际角度，送入骨架传播 |
| 175 | finger_q | j4_ring_pip | q_raw[22] | rad | Actor + Critic | 无名指PIP屈曲实际角度，送入骨架传播 |
| 176 | finger_q | j4_ring_dip | q_raw[23] | rad | Actor + Critic | 无名指DIP屈曲实际角度，送入骨架传播 |
| 177 | finger_q | j5_pinky_mcp_abd | q_raw[24] | rad | Actor + Critic | 小指MCP外展实际角度，送入骨架传播 |
| 178 | finger_q | j5_pinky_mcp_flex | q_raw[25] | rad | Actor + Critic | 小指MCP屈曲实际角度，送入骨架传播 |
| 179 | finger_q | j5_pinky_pip | q_raw[26] | rad | Actor + Critic | 小指PIP屈曲实际角度，送入骨架传播 |
| 180 | finger_q | j5_pinky_dip | q_raw[27] | rad | Actor + Critic | 小指DIP屈曲实际角度，送入骨架传播 |
| 181 | finger_qd | j1_thumb_cmc_abd | qdot_normalized[6] * 2 / 10 | (rad/s) / 10 | Actor + Critic | 拇指CMC外展实际角速度除以10，送入骨架传播 |
| 182 | finger_qd | j1_thumb_cmc_flex | qdot_normalized[7] * 3 / 10 | (rad/s) / 10 | Actor + Critic | 拇指CMC屈曲实际角速度除以10，送入骨架传播 |
| 183 | finger_qd | j1_thumb_cmc_twist | qdot_normalized[8] * 2 / 10 | (rad/s) / 10 | Actor + Critic | 拇指CMC扭转实际角速度除以10，送入骨架传播 |
| 184 | finger_qd | j1_thumb_mcp_flex | qdot_normalized[9] * 4 / 10 | (rad/s) / 10 | Actor + Critic | 拇指MCP屈曲实际角速度除以10，送入骨架传播 |
| 185 | finger_qd | j1_thumb_mcp_abd | qdot_normalized[10] * 2 / 10 | (rad/s) / 10 | Actor + Critic | 拇指MCP外展实际角速度除以10，送入骨架传播 |
| 186 | finger_qd | j1_thumb_ip | qdot_normalized[11] * 4 / 10 | (rad/s) / 10 | Actor + Critic | 拇指IP屈曲实际角速度除以10，送入骨架传播 |
| 187 | finger_qd | j2_index_mcp_abd | qdot_normalized[12] * 2 / 10 | (rad/s) / 10 | Actor + Critic | 食指MCP外展实际角速度除以10，送入骨架传播 |
| 188 | finger_qd | j2_index_mcp_flex | qdot_normalized[13] * 4 / 10 | (rad/s) / 10 | Actor + Critic | 食指MCP屈曲实际角速度除以10，送入骨架传播 |
| 189 | finger_qd | j2_index_pip | qdot_normalized[14] * 4 / 10 | (rad/s) / 10 | Actor + Critic | 食指PIP屈曲实际角速度除以10，送入骨架传播 |
| 190 | finger_qd | j2_index_dip | qdot_normalized[15] * 4 / 10 | (rad/s) / 10 | Actor + Critic | 食指DIP屈曲实际角速度除以10，送入骨架传播 |
| 191 | finger_qd | j3_middle_mcp_abd | qdot_normalized[16] * 2 / 10 | (rad/s) / 10 | Actor + Critic | 中指MCP外展实际角速度除以10，送入骨架传播 |
| 192 | finger_qd | j3_middle_mcp_flex | qdot_normalized[17] * 4 / 10 | (rad/s) / 10 | Actor + Critic | 中指MCP屈曲实际角速度除以10，送入骨架传播 |
| 193 | finger_qd | j3_middle_pip | qdot_normalized[18] * 4 / 10 | (rad/s) / 10 | Actor + Critic | 中指PIP屈曲实际角速度除以10，送入骨架传播 |
| 194 | finger_qd | j3_middle_dip | qdot_normalized[19] * 4 / 10 | (rad/s) / 10 | Actor + Critic | 中指DIP屈曲实际角速度除以10，送入骨架传播 |
| 195 | finger_qd | j4_ring_mcp_abd | qdot_normalized[20] * 2 / 10 | (rad/s) / 10 | Actor + Critic | 无名指MCP外展实际角速度除以10，送入骨架传播 |
| 196 | finger_qd | j4_ring_mcp_flex | qdot_normalized[21] * 4 / 10 | (rad/s) / 10 | Actor + Critic | 无名指MCP屈曲实际角速度除以10，送入骨架传播 |
| 197 | finger_qd | j4_ring_pip | qdot_normalized[22] * 4 / 10 | (rad/s) / 10 | Actor + Critic | 无名指PIP屈曲实际角速度除以10，送入骨架传播 |
| 198 | finger_qd | j4_ring_dip | qdot_normalized[23] * 4 / 10 | (rad/s) / 10 | Actor + Critic | 无名指DIP屈曲实际角速度除以10，送入骨架传播 |
| 199 | finger_qd | j5_pinky_mcp_abd | qdot_normalized[24] * 2 / 10 | (rad/s) / 10 | Actor + Critic | 小指MCP外展实际角速度除以10，送入骨架传播 |
| 200 | finger_qd | j5_pinky_mcp_flex | qdot_normalized[25] * 4 / 10 | (rad/s) / 10 | Actor + Critic | 小指MCP屈曲实际角速度除以10，送入骨架传播 |
| 201 | finger_qd | j5_pinky_pip | qdot_normalized[26] * 4 / 10 | (rad/s) / 10 | Actor + Critic | 小指PIP屈曲实际角速度除以10，送入骨架传播 |
| 202 | finger_qd | j5_pinky_dip | qdot_normalized[27] * 4 / 10 | (rad/s) / 10 | Actor + Critic | 小指DIP屈曲实际角速度除以10，送入骨架传播 |
| 203 | bone_contact | palm.x | tanh(-force_on_object[palm].x / 10) | 无量纲 | Actor + Critic | 物体对掌部的反力，经tanh压缩；送入骨架传播 |
| 204 | bone_contact | palm.y | tanh(-force_on_object[palm].y / 10) | 无量纲 | Actor + Critic | 物体对掌部的反力，经tanh压缩；送入骨架传播 |
| 205 | bone_contact | palm.z | tanh(-force_on_object[palm].z / 10) | 无量纲 | Actor + Critic | 物体对掌部的反力，经tanh压缩；送入骨架传播 |
| 206 | bone_contact | index_mcp.x | tanh(-force_on_object[index_mcp].x / 10) | 无量纲 | Actor + Critic | 物体对食指近端的反力，经tanh压缩；送入骨架传播 |
| 207 | bone_contact | index_mcp.y | tanh(-force_on_object[index_mcp].y / 10) | 无量纲 | Actor + Critic | 物体对食指近端的反力，经tanh压缩；送入骨架传播 |
| 208 | bone_contact | index_mcp.z | tanh(-force_on_object[index_mcp].z / 10) | 无量纲 | Actor + Critic | 物体对食指近端的反力，经tanh压缩；送入骨架传播 |
| 209 | bone_contact | index_pip.x | tanh(-force_on_object[index_pip].x / 10) | 无量纲 | Actor + Critic | 物体对食指中段的反力，经tanh压缩；送入骨架传播 |
| 210 | bone_contact | index_pip.y | tanh(-force_on_object[index_pip].y / 10) | 无量纲 | Actor + Critic | 物体对食指中段的反力，经tanh压缩；送入骨架传播 |
| 211 | bone_contact | index_pip.z | tanh(-force_on_object[index_pip].z / 10) | 无量纲 | Actor + Critic | 物体对食指中段的反力，经tanh压缩；送入骨架传播 |
| 212 | bone_contact | index_dip.x | tanh(-force_on_object[index_dip].x / 10) | 无量纲 | Actor + Critic | 物体对食指远端的反力，经tanh压缩；送入骨架传播 |
| 213 | bone_contact | index_dip.y | tanh(-force_on_object[index_dip].y / 10) | 无量纲 | Actor + Critic | 物体对食指远端的反力，经tanh压缩；送入骨架传播 |
| 214 | bone_contact | index_dip.z | tanh(-force_on_object[index_dip].z / 10) | 无量纲 | Actor + Critic | 物体对食指远端的反力，经tanh压缩；送入骨架传播 |
| 215 | bone_contact | middle_mcp.x | tanh(-force_on_object[middle_mcp].x / 10) | 无量纲 | Actor + Critic | 物体对中指近端的反力，经tanh压缩；送入骨架传播 |
| 216 | bone_contact | middle_mcp.y | tanh(-force_on_object[middle_mcp].y / 10) | 无量纲 | Actor + Critic | 物体对中指近端的反力，经tanh压缩；送入骨架传播 |
| 217 | bone_contact | middle_mcp.z | tanh(-force_on_object[middle_mcp].z / 10) | 无量纲 | Actor + Critic | 物体对中指近端的反力，经tanh压缩；送入骨架传播 |
| 218 | bone_contact | middle_pip.x | tanh(-force_on_object[middle_pip].x / 10) | 无量纲 | Actor + Critic | 物体对中指中段的反力，经tanh压缩；送入骨架传播 |
| 219 | bone_contact | middle_pip.y | tanh(-force_on_object[middle_pip].y / 10) | 无量纲 | Actor + Critic | 物体对中指中段的反力，经tanh压缩；送入骨架传播 |
| 220 | bone_contact | middle_pip.z | tanh(-force_on_object[middle_pip].z / 10) | 无量纲 | Actor + Critic | 物体对中指中段的反力，经tanh压缩；送入骨架传播 |
| 221 | bone_contact | middle_dip.x | tanh(-force_on_object[middle_dip].x / 10) | 无量纲 | Actor + Critic | 物体对中指远端的反力，经tanh压缩；送入骨架传播 |
| 222 | bone_contact | middle_dip.y | tanh(-force_on_object[middle_dip].y / 10) | 无量纲 | Actor + Critic | 物体对中指远端的反力，经tanh压缩；送入骨架传播 |
| 223 | bone_contact | middle_dip.z | tanh(-force_on_object[middle_dip].z / 10) | 无量纲 | Actor + Critic | 物体对中指远端的反力，经tanh压缩；送入骨架传播 |
| 224 | bone_contact | ring_mcp.x | tanh(-force_on_object[ring_mcp].x / 10) | 无量纲 | Actor + Critic | 物体对无名指近端的反力，经tanh压缩；送入骨架传播 |
| 225 | bone_contact | ring_mcp.y | tanh(-force_on_object[ring_mcp].y / 10) | 无量纲 | Actor + Critic | 物体对无名指近端的反力，经tanh压缩；送入骨架传播 |
| 226 | bone_contact | ring_mcp.z | tanh(-force_on_object[ring_mcp].z / 10) | 无量纲 | Actor + Critic | 物体对无名指近端的反力，经tanh压缩；送入骨架传播 |
| 227 | bone_contact | ring_pip.x | tanh(-force_on_object[ring_pip].x / 10) | 无量纲 | Actor + Critic | 物体对无名指中段的反力，经tanh压缩；送入骨架传播 |
| 228 | bone_contact | ring_pip.y | tanh(-force_on_object[ring_pip].y / 10) | 无量纲 | Actor + Critic | 物体对无名指中段的反力，经tanh压缩；送入骨架传播 |
| 229 | bone_contact | ring_pip.z | tanh(-force_on_object[ring_pip].z / 10) | 无量纲 | Actor + Critic | 物体对无名指中段的反力，经tanh压缩；送入骨架传播 |
| 230 | bone_contact | ring_dip.x | tanh(-force_on_object[ring_dip].x / 10) | 无量纲 | Actor + Critic | 物体对无名指远端的反力，经tanh压缩；送入骨架传播 |
| 231 | bone_contact | ring_dip.y | tanh(-force_on_object[ring_dip].y / 10) | 无量纲 | Actor + Critic | 物体对无名指远端的反力，经tanh压缩；送入骨架传播 |
| 232 | bone_contact | ring_dip.z | tanh(-force_on_object[ring_dip].z / 10) | 无量纲 | Actor + Critic | 物体对无名指远端的反力，经tanh压缩；送入骨架传播 |
| 233 | bone_contact | pinky_mcp.x | tanh(-force_on_object[pinky_mcp].x / 10) | 无量纲 | Actor + Critic | 物体对小指近端的反力，经tanh压缩；送入骨架传播 |
| 234 | bone_contact | pinky_mcp.y | tanh(-force_on_object[pinky_mcp].y / 10) | 无量纲 | Actor + Critic | 物体对小指近端的反力，经tanh压缩；送入骨架传播 |
| 235 | bone_contact | pinky_mcp.z | tanh(-force_on_object[pinky_mcp].z / 10) | 无量纲 | Actor + Critic | 物体对小指近端的反力，经tanh压缩；送入骨架传播 |
| 236 | bone_contact | pinky_pip.x | tanh(-force_on_object[pinky_pip].x / 10) | 无量纲 | Actor + Critic | 物体对小指中段的反力，经tanh压缩；送入骨架传播 |
| 237 | bone_contact | pinky_pip.y | tanh(-force_on_object[pinky_pip].y / 10) | 无量纲 | Actor + Critic | 物体对小指中段的反力，经tanh压缩；送入骨架传播 |
| 238 | bone_contact | pinky_pip.z | tanh(-force_on_object[pinky_pip].z / 10) | 无量纲 | Actor + Critic | 物体对小指中段的反力，经tanh压缩；送入骨架传播 |
| 239 | bone_contact | pinky_dip.x | tanh(-force_on_object[pinky_dip].x / 10) | 无量纲 | Actor + Critic | 物体对小指远端的反力，经tanh压缩；送入骨架传播 |
| 240 | bone_contact | pinky_dip.y | tanh(-force_on_object[pinky_dip].y / 10) | 无量纲 | Actor + Critic | 物体对小指远端的反力，经tanh压缩；送入骨架传播 |
| 241 | bone_contact | pinky_dip.z | tanh(-force_on_object[pinky_dip].z / 10) | 无量纲 | Actor + Critic | 物体对小指远端的反力，经tanh压缩；送入骨架传播 |
| 242 | bone_contact | thumb_cmc.x | tanh(-force_on_object[thumb_cmc].x / 10) | 无量纲 | Actor + Critic | 物体对拇指根部的反力，经tanh压缩；送入骨架传播 |
| 243 | bone_contact | thumb_cmc.y | tanh(-force_on_object[thumb_cmc].y / 10) | 无量纲 | Actor + Critic | 物体对拇指根部的反力，经tanh压缩；送入骨架传播 |
| 244 | bone_contact | thumb_cmc.z | tanh(-force_on_object[thumb_cmc].z / 10) | 无量纲 | Actor + Critic | 物体对拇指根部的反力，经tanh压缩；送入骨架传播 |
| 245 | bone_contact | thumb_mcp.x | tanh(-force_on_object[thumb_mcp].x / 10) | 无量纲 | Actor + Critic | 物体对拇指中段的反力，经tanh压缩；送入骨架传播 |
| 246 | bone_contact | thumb_mcp.y | tanh(-force_on_object[thumb_mcp].y / 10) | 无量纲 | Actor + Critic | 物体对拇指中段的反力，经tanh压缩；送入骨架传播 |
| 247 | bone_contact | thumb_mcp.z | tanh(-force_on_object[thumb_mcp].z / 10) | 无量纲 | Actor + Critic | 物体对拇指中段的反力，经tanh压缩；送入骨架传播 |
| 248 | bone_contact | thumb_ip.x | tanh(-force_on_object[thumb_ip].x / 10) | 无量纲 | Actor + Critic | 物体对拇指远端的反力，经tanh压缩；送入骨架传播 |
| 249 | bone_contact | thumb_ip.y | tanh(-force_on_object[thumb_ip].y / 10) | 无量纲 | Actor + Critic | 物体对拇指远端的反力，经tanh压缩；送入骨架传播 |
| 250 | bone_contact | thumb_ip.z | tanh(-force_on_object[thumb_ip].z / 10) | 无量纲 | Actor + Critic | 物体对拇指远端的反力，经tanh压缩；送入骨架传播 |
| 251 | obj_force_obs | gravity.x | mass * gravity_world.x / weight | 无量纲 | Actor + Critic | 重力方向；与接触力分开，普通竖直重力下为[0,0,-1] |
| 252 | obj_force_obs | gravity.y | mass * gravity_world.y / weight | 无量纲 | Actor + Critic | 重力方向；与接触力分开，普通竖直重力下为[0,0,-1] |
| 253 | obj_force_obs | gravity.z | mass * gravity_world.z / weight | 无量纲 | Actor + Critic | 重力方向；与接触力分开，普通竖直重力下为[0,0,-1] |
| 254 | obj_force_obs | force_on_object_from_palm.x | force_on_object[palm].x / weight | N / weight(N) | Actor + Critic | 掌部作用于物体的世界坐标接触力，给两塔低维状态分支 |
| 255 | obj_force_obs | force_on_object_from_palm.y | force_on_object[palm].y / weight | N / weight(N) | Actor + Critic | 掌部作用于物体的世界坐标接触力，给两塔低维状态分支 |
| 256 | obj_force_obs | force_on_object_from_palm.z | force_on_object[palm].z / weight | N / weight(N) | Actor + Critic | 掌部作用于物体的世界坐标接触力，给两塔低维状态分支 |
| 257 | obj_force_obs | force_on_object_from_index_mcp.x | force_on_object[index_mcp].x / weight | N / weight(N) | Actor + Critic | 食指近端作用于物体的世界坐标接触力，给两塔低维状态分支 |
| 258 | obj_force_obs | force_on_object_from_index_mcp.y | force_on_object[index_mcp].y / weight | N / weight(N) | Actor + Critic | 食指近端作用于物体的世界坐标接触力，给两塔低维状态分支 |
| 259 | obj_force_obs | force_on_object_from_index_mcp.z | force_on_object[index_mcp].z / weight | N / weight(N) | Actor + Critic | 食指近端作用于物体的世界坐标接触力，给两塔低维状态分支 |
| 260 | obj_force_obs | force_on_object_from_index_pip.x | force_on_object[index_pip].x / weight | N / weight(N) | Actor + Critic | 食指中段作用于物体的世界坐标接触力，给两塔低维状态分支 |
| 261 | obj_force_obs | force_on_object_from_index_pip.y | force_on_object[index_pip].y / weight | N / weight(N) | Actor + Critic | 食指中段作用于物体的世界坐标接触力，给两塔低维状态分支 |
| 262 | obj_force_obs | force_on_object_from_index_pip.z | force_on_object[index_pip].z / weight | N / weight(N) | Actor + Critic | 食指中段作用于物体的世界坐标接触力，给两塔低维状态分支 |
| 263 | obj_force_obs | force_on_object_from_index_dip.x | force_on_object[index_dip].x / weight | N / weight(N) | Actor + Critic | 食指远端作用于物体的世界坐标接触力，给两塔低维状态分支 |
| 264 | obj_force_obs | force_on_object_from_index_dip.y | force_on_object[index_dip].y / weight | N / weight(N) | Actor + Critic | 食指远端作用于物体的世界坐标接触力，给两塔低维状态分支 |
| 265 | obj_force_obs | force_on_object_from_index_dip.z | force_on_object[index_dip].z / weight | N / weight(N) | Actor + Critic | 食指远端作用于物体的世界坐标接触力，给两塔低维状态分支 |
| 266 | obj_force_obs | force_on_object_from_middle_mcp.x | force_on_object[middle_mcp].x / weight | N / weight(N) | Actor + Critic | 中指近端作用于物体的世界坐标接触力，给两塔低维状态分支 |
| 267 | obj_force_obs | force_on_object_from_middle_mcp.y | force_on_object[middle_mcp].y / weight | N / weight(N) | Actor + Critic | 中指近端作用于物体的世界坐标接触力，给两塔低维状态分支 |
| 268 | obj_force_obs | force_on_object_from_middle_mcp.z | force_on_object[middle_mcp].z / weight | N / weight(N) | Actor + Critic | 中指近端作用于物体的世界坐标接触力，给两塔低维状态分支 |
| 269 | obj_force_obs | force_on_object_from_middle_pip.x | force_on_object[middle_pip].x / weight | N / weight(N) | Actor + Critic | 中指中段作用于物体的世界坐标接触力，给两塔低维状态分支 |
| 270 | obj_force_obs | force_on_object_from_middle_pip.y | force_on_object[middle_pip].y / weight | N / weight(N) | Actor + Critic | 中指中段作用于物体的世界坐标接触力，给两塔低维状态分支 |
| 271 | obj_force_obs | force_on_object_from_middle_pip.z | force_on_object[middle_pip].z / weight | N / weight(N) | Actor + Critic | 中指中段作用于物体的世界坐标接触力，给两塔低维状态分支 |
| 272 | obj_force_obs | force_on_object_from_middle_dip.x | force_on_object[middle_dip].x / weight | N / weight(N) | Actor + Critic | 中指远端作用于物体的世界坐标接触力，给两塔低维状态分支 |
| 273 | obj_force_obs | force_on_object_from_middle_dip.y | force_on_object[middle_dip].y / weight | N / weight(N) | Actor + Critic | 中指远端作用于物体的世界坐标接触力，给两塔低维状态分支 |
| 274 | obj_force_obs | force_on_object_from_middle_dip.z | force_on_object[middle_dip].z / weight | N / weight(N) | Actor + Critic | 中指远端作用于物体的世界坐标接触力，给两塔低维状态分支 |
| 275 | obj_force_obs | force_on_object_from_ring_mcp.x | force_on_object[ring_mcp].x / weight | N / weight(N) | Actor + Critic | 无名指近端作用于物体的世界坐标接触力，给两塔低维状态分支 |
| 276 | obj_force_obs | force_on_object_from_ring_mcp.y | force_on_object[ring_mcp].y / weight | N / weight(N) | Actor + Critic | 无名指近端作用于物体的世界坐标接触力，给两塔低维状态分支 |
| 277 | obj_force_obs | force_on_object_from_ring_mcp.z | force_on_object[ring_mcp].z / weight | N / weight(N) | Actor + Critic | 无名指近端作用于物体的世界坐标接触力，给两塔低维状态分支 |
| 278 | obj_force_obs | force_on_object_from_ring_pip.x | force_on_object[ring_pip].x / weight | N / weight(N) | Actor + Critic | 无名指中段作用于物体的世界坐标接触力，给两塔低维状态分支 |
| 279 | obj_force_obs | force_on_object_from_ring_pip.y | force_on_object[ring_pip].y / weight | N / weight(N) | Actor + Critic | 无名指中段作用于物体的世界坐标接触力，给两塔低维状态分支 |
| 280 | obj_force_obs | force_on_object_from_ring_pip.z | force_on_object[ring_pip].z / weight | N / weight(N) | Actor + Critic | 无名指中段作用于物体的世界坐标接触力，给两塔低维状态分支 |
| 281 | obj_force_obs | force_on_object_from_ring_dip.x | force_on_object[ring_dip].x / weight | N / weight(N) | Actor + Critic | 无名指远端作用于物体的世界坐标接触力，给两塔低维状态分支 |
| 282 | obj_force_obs | force_on_object_from_ring_dip.y | force_on_object[ring_dip].y / weight | N / weight(N) | Actor + Critic | 无名指远端作用于物体的世界坐标接触力，给两塔低维状态分支 |
| 283 | obj_force_obs | force_on_object_from_ring_dip.z | force_on_object[ring_dip].z / weight | N / weight(N) | Actor + Critic | 无名指远端作用于物体的世界坐标接触力，给两塔低维状态分支 |
| 284 | obj_force_obs | force_on_object_from_pinky_mcp.x | force_on_object[pinky_mcp].x / weight | N / weight(N) | Actor + Critic | 小指近端作用于物体的世界坐标接触力，给两塔低维状态分支 |
| 285 | obj_force_obs | force_on_object_from_pinky_mcp.y | force_on_object[pinky_mcp].y / weight | N / weight(N) | Actor + Critic | 小指近端作用于物体的世界坐标接触力，给两塔低维状态分支 |
| 286 | obj_force_obs | force_on_object_from_pinky_mcp.z | force_on_object[pinky_mcp].z / weight | N / weight(N) | Actor + Critic | 小指近端作用于物体的世界坐标接触力，给两塔低维状态分支 |
| 287 | obj_force_obs | force_on_object_from_pinky_pip.x | force_on_object[pinky_pip].x / weight | N / weight(N) | Actor + Critic | 小指中段作用于物体的世界坐标接触力，给两塔低维状态分支 |
| 288 | obj_force_obs | force_on_object_from_pinky_pip.y | force_on_object[pinky_pip].y / weight | N / weight(N) | Actor + Critic | 小指中段作用于物体的世界坐标接触力，给两塔低维状态分支 |
| 289 | obj_force_obs | force_on_object_from_pinky_pip.z | force_on_object[pinky_pip].z / weight | N / weight(N) | Actor + Critic | 小指中段作用于物体的世界坐标接触力，给两塔低维状态分支 |
| 290 | obj_force_obs | force_on_object_from_pinky_dip.x | force_on_object[pinky_dip].x / weight | N / weight(N) | Actor + Critic | 小指远端作用于物体的世界坐标接触力，给两塔低维状态分支 |
| 291 | obj_force_obs | force_on_object_from_pinky_dip.y | force_on_object[pinky_dip].y / weight | N / weight(N) | Actor + Critic | 小指远端作用于物体的世界坐标接触力，给两塔低维状态分支 |
| 292 | obj_force_obs | force_on_object_from_pinky_dip.z | force_on_object[pinky_dip].z / weight | N / weight(N) | Actor + Critic | 小指远端作用于物体的世界坐标接触力，给两塔低维状态分支 |
| 293 | obj_force_obs | force_on_object_from_thumb_cmc.x | force_on_object[thumb_cmc].x / weight | N / weight(N) | Actor + Critic | 拇指根部作用于物体的世界坐标接触力，给两塔低维状态分支 |
| 294 | obj_force_obs | force_on_object_from_thumb_cmc.y | force_on_object[thumb_cmc].y / weight | N / weight(N) | Actor + Critic | 拇指根部作用于物体的世界坐标接触力，给两塔低维状态分支 |
| 295 | obj_force_obs | force_on_object_from_thumb_cmc.z | force_on_object[thumb_cmc].z / weight | N / weight(N) | Actor + Critic | 拇指根部作用于物体的世界坐标接触力，给两塔低维状态分支 |
| 296 | obj_force_obs | force_on_object_from_thumb_mcp.x | force_on_object[thumb_mcp].x / weight | N / weight(N) | Actor + Critic | 拇指中段作用于物体的世界坐标接触力，给两塔低维状态分支 |
| 297 | obj_force_obs | force_on_object_from_thumb_mcp.y | force_on_object[thumb_mcp].y / weight | N / weight(N) | Actor + Critic | 拇指中段作用于物体的世界坐标接触力，给两塔低维状态分支 |
| 298 | obj_force_obs | force_on_object_from_thumb_mcp.z | force_on_object[thumb_mcp].z / weight | N / weight(N) | Actor + Critic | 拇指中段作用于物体的世界坐标接触力，给两塔低维状态分支 |
| 299 | obj_force_obs | force_on_object_from_thumb_ip.x | force_on_object[thumb_ip].x / weight | N / weight(N) | Actor + Critic | 拇指远端作用于物体的世界坐标接触力，给两塔低维状态分支 |
| 300 | obj_force_obs | force_on_object_from_thumb_ip.y | force_on_object[thumb_ip].y / weight | N / weight(N) | Actor + Critic | 拇指远端作用于物体的世界坐标接触力，给两塔低维状态分支 |
| 301 | obj_force_obs | force_on_object_from_thumb_ip.z | force_on_object[thumb_ip].z / weight | N / weight(N) | Actor + Critic | 拇指远端作用于物体的世界坐标接触力，给两塔低维状态分支 |
| 302 | obj_force_obs | hand_torque_on_object.x | sum_bone(torque_about_object_com).x / torque_scale | Nm / torque_scale(Nm) | Actor + Critic | 手对物体质心的净接触力矩 |
| 303 | obj_force_obs | hand_torque_on_object.y | sum_bone(torque_about_object_com).y / torque_scale | Nm / torque_scale(Nm) | Actor + Critic | 手对物体质心的净接触力矩 |
| 304 | obj_force_obs | hand_torque_on_object.z | sum_bone(torque_about_object_com).z / torque_scale | Nm / torque_scale(Nm) | Actor + Critic | 手对物体质心的净接触力矩 |
| 305 | obj_force_obs | non_hand_contact_force.x | (object_all_contact_force-sum_bone(hand_force)).x / weight | N / weight(N) | Actor + Critic | 桌面等非手接触施加于物体的净力；此项不含重力 |
| 306 | obj_force_obs | non_hand_contact_force.y | (object_all_contact_force-sum_bone(hand_force)).y / weight | N / weight(N) | Actor + Critic | 桌面等非手接触施加于物体的净力；此项不含重力 |
| 307 | obj_force_obs | non_hand_contact_force.z | (object_all_contact_force-sum_bone(hand_force)).z / weight | N / weight(N) | Actor + Critic | 桌面等非手接触施加于物体的净力；此项不含重力 |
| 308 | obj_force_obs | non_hand_contact_torque.x | (object_all_contact_torque-sum_bone(hand_torque)).x / torque_scale | Nm / torque_scale(Nm) | Actor + Critic | 桌面等非手接触对物体质心的净力矩 |
| 309 | obj_force_obs | non_hand_contact_torque.y | (object_all_contact_torque-sum_bone(hand_torque)).y / torque_scale | Nm / torque_scale(Nm) | Actor + Critic | 桌面等非手接触对物体质心的净力矩 |
| 310 | obj_force_obs | non_hand_contact_torque.z | (object_all_contact_torque-sum_bone(hand_torque)).z / torque_scale | Nm / torque_scale(Nm) | Actor + Critic | 桌面等非手接触对物体质心的净力矩 |
| 311 | wrist_twist | palm_origin_velocity.x | palm_v_com.x + (palm_w × (palm_origin-palm_com)).x | m/s | Actor + Critic | 掌部原点线速度，用于运动状态编码 |
| 312 | wrist_twist | palm_origin_velocity.y | palm_v_com.y + (palm_w × (palm_origin-palm_com)).y | m/s | Actor + Critic | 掌部原点线速度，用于运动状态编码 |
| 313 | wrist_twist | palm_origin_velocity.z | palm_v_com.z + (palm_w × (palm_origin-palm_com)).z | m/s | Actor + Critic | 掌部原点线速度，用于运动状态编码 |
| 314 | wrist_twist | palm_angular_velocity.x | palm_w.x / 10 | (rad/s) / 10 | Actor + Critic | 掌部角速度，用于运动状态编码 |
| 315 | wrist_twist | palm_angular_velocity.y | palm_w.y / 10 | (rad/s) / 10 | Actor + Critic | 掌部角速度，用于运动状态编码 |
| 316 | wrist_twist | palm_angular_velocity.z | palm_w.z / 10 | (rad/s) / 10 | Actor + Critic | 掌部角速度，用于运动状态编码 |
| 317 | obj_twist | object_com_velocity.x | object_v_com.x | m/s | Actor + Critic | 物体质心线速度，用于运动状态编码 |
| 318 | obj_twist | object_com_velocity.y | object_v_com.y | m/s | Actor + Critic | 物体质心线速度，用于运动状态编码 |
| 319 | obj_twist | object_com_velocity.z | object_v_com.z | m/s | Actor + Critic | 物体质心线速度，用于运动状态编码 |
| 320 | obj_twist | object_angular_velocity.x | object_w.x / 10 | (rad/s) / 10 | Actor + Critic | 物体角速度，用于运动状态编码 |
| 321 | obj_twist | object_angular_velocity.y | object_w.y / 10 | (rad/s) / 10 | Actor + Critic | 物体角速度，用于运动状态编码 |
| 322 | obj_twist | object_angular_velocity.z | object_w.z / 10 | (rad/s) / 10 | Actor + Critic | 物体角速度，用于运动状态编码 |
| 323 | mass_obs | mass_channel | 0 | 无量纲 | Actor + Critic | 当前固定质量下恒0；不提供跨质量差异，力归一化仍读取真实质量 |
| 324 | prev_action | ARTx | clip(previous_raw_action[0], -1, 1) | 无量纲 | Actor + Critic | 腕部x平移上一物理步施加的归一化速率指令；不等于实际位移或原始Normal样本 |
| 325 | prev_action | ARTy | clip(previous_raw_action[1], -1, 1) | 无量纲 | Actor + Critic | 腕部y平移上一物理步施加的归一化速率指令；不等于实际位移或原始Normal样本 |
| 326 | prev_action | ARTz | clip(previous_raw_action[2], -1, 1) | 无量纲 | Actor + Critic | 腕部z平移上一物理步施加的归一化速率指令；不等于实际位移或原始Normal样本 |
| 327 | prev_action | ARRx | clip(previous_raw_action[3], -1, 1) | 无量纲 | Actor + Critic | 腕部x旋转上一物理步施加的归一化速率指令；不等于实际位移或原始Normal样本 |
| 328 | prev_action | ARRy | clip(previous_raw_action[4], -1, 1) | 无量纲 | Actor + Critic | 腕部y旋转上一物理步施加的归一化速率指令；不等于实际位移或原始Normal样本 |
| 329 | prev_action | ARRz | clip(previous_raw_action[5], -1, 1) | 无量纲 | Actor + Critic | 腕部z旋转上一物理步施加的归一化速率指令；不等于实际位移或原始Normal样本 |
| 330 | prev_action | j1_thumb_cmc_abd | clip(previous_raw_action[6], -1, 1) | 无量纲 | Actor + Critic | 拇指CMC外展上一物理步施加的归一化速率指令；不等于实际位移或原始Normal样本 |
| 331 | prev_action | j1_thumb_cmc_flex | clip(previous_raw_action[7], -1, 1) | 无量纲 | Actor + Critic | 拇指CMC屈曲上一物理步施加的归一化速率指令；不等于实际位移或原始Normal样本 |
| 332 | prev_action | j1_thumb_cmc_twist | clip(previous_raw_action[8], -1, 1) | 无量纲 | Actor + Critic | 拇指CMC扭转上一物理步施加的归一化速率指令；不等于实际位移或原始Normal样本 |
| 333 | prev_action | j1_thumb_mcp_flex | clip(previous_raw_action[9], -1, 1) | 无量纲 | Actor + Critic | 拇指MCP屈曲上一物理步施加的归一化速率指令；不等于实际位移或原始Normal样本 |
| 334 | prev_action | j1_thumb_mcp_abd | clip(previous_raw_action[10], -1, 1) | 无量纲 | Actor + Critic | 拇指MCP外展上一物理步施加的归一化速率指令；不等于实际位移或原始Normal样本 |
| 335 | prev_action | j1_thumb_ip | clip(previous_raw_action[11], -1, 1) | 无量纲 | Actor + Critic | 拇指IP屈曲上一物理步施加的归一化速率指令；不等于实际位移或原始Normal样本 |
| 336 | prev_action | j2_index_mcp_abd | clip(previous_raw_action[12], -1, 1) | 无量纲 | Actor + Critic | 食指MCP外展上一物理步施加的归一化速率指令；不等于实际位移或原始Normal样本 |
| 337 | prev_action | j2_index_mcp_flex | clip(previous_raw_action[13], -1, 1) | 无量纲 | Actor + Critic | 食指MCP屈曲上一物理步施加的归一化速率指令；不等于实际位移或原始Normal样本 |
| 338 | prev_action | j2_index_pip | clip(previous_raw_action[14], -1, 1) | 无量纲 | Actor + Critic | 食指PIP屈曲上一物理步施加的归一化速率指令；不等于实际位移或原始Normal样本 |
| 339 | prev_action | j2_index_dip | clip(previous_raw_action[15], -1, 1) | 无量纲 | Actor + Critic | 食指DIP屈曲上一物理步施加的归一化速率指令；不等于实际位移或原始Normal样本 |
| 340 | prev_action | j3_middle_mcp_abd | clip(previous_raw_action[16], -1, 1) | 无量纲 | Actor + Critic | 中指MCP外展上一物理步施加的归一化速率指令；不等于实际位移或原始Normal样本 |
| 341 | prev_action | j3_middle_mcp_flex | clip(previous_raw_action[17], -1, 1) | 无量纲 | Actor + Critic | 中指MCP屈曲上一物理步施加的归一化速率指令；不等于实际位移或原始Normal样本 |
| 342 | prev_action | j3_middle_pip | clip(previous_raw_action[18], -1, 1) | 无量纲 | Actor + Critic | 中指PIP屈曲上一物理步施加的归一化速率指令；不等于实际位移或原始Normal样本 |
| 343 | prev_action | j3_middle_dip | clip(previous_raw_action[19], -1, 1) | 无量纲 | Actor + Critic | 中指DIP屈曲上一物理步施加的归一化速率指令；不等于实际位移或原始Normal样本 |
| 344 | prev_action | j4_ring_mcp_abd | clip(previous_raw_action[20], -1, 1) | 无量纲 | Actor + Critic | 无名指MCP外展上一物理步施加的归一化速率指令；不等于实际位移或原始Normal样本 |
| 345 | prev_action | j4_ring_mcp_flex | clip(previous_raw_action[21], -1, 1) | 无量纲 | Actor + Critic | 无名指MCP屈曲上一物理步施加的归一化速率指令；不等于实际位移或原始Normal样本 |
| 346 | prev_action | j4_ring_pip | clip(previous_raw_action[22], -1, 1) | 无量纲 | Actor + Critic | 无名指PIP屈曲上一物理步施加的归一化速率指令；不等于实际位移或原始Normal样本 |
| 347 | prev_action | j4_ring_dip | clip(previous_raw_action[23], -1, 1) | 无量纲 | Actor + Critic | 无名指DIP屈曲上一物理步施加的归一化速率指令；不等于实际位移或原始Normal样本 |
| 348 | prev_action | j5_pinky_mcp_abd | clip(previous_raw_action[24], -1, 1) | 无量纲 | Actor + Critic | 小指MCP外展上一物理步施加的归一化速率指令；不等于实际位移或原始Normal样本 |
| 349 | prev_action | j5_pinky_mcp_flex | clip(previous_raw_action[25], -1, 1) | 无量纲 | Actor + Critic | 小指MCP屈曲上一物理步施加的归一化速率指令；不等于实际位移或原始Normal样本 |
| 350 | prev_action | j5_pinky_pip | clip(previous_raw_action[26], -1, 1) | 无量纲 | Actor + Critic | 小指PIP屈曲上一物理步施加的归一化速率指令；不等于实际位移或原始Normal样本 |
| 351 | prev_action | j5_pinky_dip | clip(previous_raw_action[27], -1, 1) | 无量纲 | Actor + Critic | 小指DIP屈曲上一物理步施加的归一化速率指令；不等于实际位移或原始Normal样本 |
| 352 | priv_13 | object_reference_position_error.x | (actual_object.x-current_reference_object.x) / 0.2 | m / 0.2m | Critic | 当前物体位置追踪误差；仅Critic |
| 353 | priv_13 | object_reference_position_error.y | (actual_object.y-current_reference_object.y) / 0.2 | m / 0.2m | Critic | 当前物体位置追踪误差；仅Critic |
| 354 | priv_13 | object_reference_position_error.z | (actual_object.z-current_reference_object.z) / 0.2 | m / 0.2m | Critic | 当前物体位置追踪误差；仅Critic |
| 355 | priv_13 | object_reference_rotation_error | shortest_quaternion_angle(actual_object, reference_object) / pi | rad / pi | Critic | 物体朝向误差大小，无方向；仅Critic |
| 356 | priv_13 | object_reference_velocity_error.x | actual_object_v_com.x-reference_object_v_com.x | m/s | Critic | 当前物体质心速度追踪误差；仅Critic |
| 357 | priv_13 | object_reference_velocity_error.y | actual_object_v_com.y-reference_object_v_com.y | m/s | Critic | 当前物体质心速度追踪误差；仅Critic |
| 358 | priv_13 | object_reference_velocity_error.z | actual_object_v_com.z-reference_object_v_com.z | m/s | Critic | 当前物体质心速度追踪误差；仅Critic |
| 359 | priv_13 | hand_position_rms_error | RMS(actual_16_bone_positions-reference_positions) / 0.1 | m / 0.1m | Critic | 16块骨骼位置误差的均方根；仅Critic |
| 360 | priv_13 | finger_angle_rms_error | RMS(actual_22_finger_angles-reference_feasible_angles) | rad | Critic | 22个手指角度误差的均方根；仅Critic |
| 361 | priv_13 | reference_progress | now / end | 无量纲 | Critic | 参考时间进度；仅Critic，不代表任务完成率 |
| 362 | priv_13 | reference_remaining_time | (end-now) / 5.475 | s / T_MAX | Critic | 剩余参考时间；仅Critic |
| 363 | priv_13 | constant_one | 1 | 无量纲 | Critic | 当前实现中的常量占位通道；仅Critic |
| 364 | priv_13 | reference_remaining_short_window | min(end-now, control_timestep) / 5.475 | s / T_MAX | Critic | 最多一个控制步的剩余时间窗口；大部分轨迹中恒定，末端变化；仅Critic |
| 365 | support_ratio_obs | signed_hand_vertical_support | tanh(sum_bone(force_on_object.z) / weight) | 无量纲，范围(-1,1) | Actor + Critic | 手对物体的向上净接触力相对重量；不是支撑百分比，也不含桌面接触力 |
| 366 | contact_bone_count_obs | contact_bone_fraction | count(paired_count > 0) / 16 | 0到1 | Actor + Critic | 有求解手物接触的骨骼占比；不是接触点总数或承载骨骼比例 |
| 367 | servo_error | ARTx | (held_command[0]-q_raw[0]) / 0.04 | m/m | Actor + Critic | 腕部x平移持久伺服目标与实际位置差；提供执行滞后/加载信息，除数为antiwindup尺度 |
| 368 | servo_error | ARTy | (held_command[1]-q_raw[1]) / 0.02 | m/m | Actor + Critic | 腕部y平移持久伺服目标与实际位置差；提供执行滞后/加载信息，除数为antiwindup尺度 |
| 369 | servo_error | ARTz | (held_command[2]-q_raw[2]) / 0.07 | m/m | Actor + Critic | 腕部z平移持久伺服目标与实际位置差；提供执行滞后/加载信息，除数为antiwindup尺度 |
| 370 | servo_error | ARRx | (held_command[3]-q_raw[3]) / 0.3 | rad/rad | Actor + Critic | 腕部x旋转持久伺服目标与实际位置差；提供执行滞后/加载信息，除数为antiwindup尺度 |
| 371 | servo_error | ARRy | (held_command[4]-q_raw[4]) / 0.3 | rad/rad | Actor + Critic | 腕部y旋转持久伺服目标与实际位置差；提供执行滞后/加载信息，除数为antiwindup尺度 |
| 372 | servo_error | ARRz | (held_command[5]-q_raw[5]) / 0.3 | rad/rad | Actor + Critic | 腕部z旋转持久伺服目标与实际位置差；提供执行滞后/加载信息，除数为antiwindup尺度 |
| 373 | servo_error | j1_thumb_cmc_abd | (held_command[6]-q_raw[6]) / 0.2 | rad/rad | Actor + Critic | 拇指CMC外展持久伺服目标与实际位置差；提供执行滞后/加载信息，除数为antiwindup尺度 |
| 374 | servo_error | j1_thumb_cmc_flex | (held_command[7]-q_raw[7]) / 0.35 | rad/rad | Actor + Critic | 拇指CMC屈曲持久伺服目标与实际位置差；提供执行滞后/加载信息，除数为antiwindup尺度 |
| 375 | servo_error | j1_thumb_cmc_twist | (held_command[8]-q_raw[8]) / 0.2 | rad/rad | Actor + Critic | 拇指CMC扭转持久伺服目标与实际位置差；提供执行滞后/加载信息，除数为antiwindup尺度 |
| 376 | servo_error | j1_thumb_mcp_flex | (held_command[9]-q_raw[9]) / 0.35 | rad/rad | Actor + Critic | 拇指MCP屈曲持久伺服目标与实际位置差；提供执行滞后/加载信息，除数为antiwindup尺度 |
| 377 | servo_error | j1_thumb_mcp_abd | (held_command[10]-q_raw[10]) / 0.2 | rad/rad | Actor + Critic | 拇指MCP外展持久伺服目标与实际位置差；提供执行滞后/加载信息，除数为antiwindup尺度 |
| 378 | servo_error | j1_thumb_ip | (held_command[11]-q_raw[11]) / 0.35 | rad/rad | Actor + Critic | 拇指IP屈曲持久伺服目标与实际位置差；提供执行滞后/加载信息，除数为antiwindup尺度 |
| 379 | servo_error | j2_index_mcp_abd | (held_command[12]-q_raw[12]) / 0.15 | rad/rad | Actor + Critic | 食指MCP外展持久伺服目标与实际位置差；提供执行滞后/加载信息，除数为antiwindup尺度 |
| 380 | servo_error | j2_index_mcp_flex | (held_command[13]-q_raw[13]) / 0.35 | rad/rad | Actor + Critic | 食指MCP屈曲持久伺服目标与实际位置差；提供执行滞后/加载信息，除数为antiwindup尺度 |
| 381 | servo_error | j2_index_pip | (held_command[14]-q_raw[14]) / 0.35 | rad/rad | Actor + Critic | 食指PIP屈曲持久伺服目标与实际位置差；提供执行滞后/加载信息，除数为antiwindup尺度 |
| 382 | servo_error | j2_index_dip | (held_command[15]-q_raw[15]) / 0.35 | rad/rad | Actor + Critic | 食指DIP屈曲持久伺服目标与实际位置差；提供执行滞后/加载信息，除数为antiwindup尺度 |
| 383 | servo_error | j3_middle_mcp_abd | (held_command[16]-q_raw[16]) / 0.15 | rad/rad | Actor + Critic | 中指MCP外展持久伺服目标与实际位置差；提供执行滞后/加载信息，除数为antiwindup尺度 |
| 384 | servo_error | j3_middle_mcp_flex | (held_command[17]-q_raw[17]) / 0.35 | rad/rad | Actor + Critic | 中指MCP屈曲持久伺服目标与实际位置差；提供执行滞后/加载信息，除数为antiwindup尺度 |
| 385 | servo_error | j3_middle_pip | (held_command[18]-q_raw[18]) / 0.35 | rad/rad | Actor + Critic | 中指PIP屈曲持久伺服目标与实际位置差；提供执行滞后/加载信息，除数为antiwindup尺度 |
| 386 | servo_error | j3_middle_dip | (held_command[19]-q_raw[19]) / 0.35 | rad/rad | Actor + Critic | 中指DIP屈曲持久伺服目标与实际位置差；提供执行滞后/加载信息，除数为antiwindup尺度 |
| 387 | servo_error | j4_ring_mcp_abd | (held_command[20]-q_raw[20]) / 0.15 | rad/rad | Actor + Critic | 无名指MCP外展持久伺服目标与实际位置差；提供执行滞后/加载信息，除数为antiwindup尺度 |
| 388 | servo_error | j4_ring_mcp_flex | (held_command[21]-q_raw[21]) / 0.35 | rad/rad | Actor + Critic | 无名指MCP屈曲持久伺服目标与实际位置差；提供执行滞后/加载信息，除数为antiwindup尺度 |
| 389 | servo_error | j4_ring_pip | (held_command[22]-q_raw[22]) / 0.35 | rad/rad | Actor + Critic | 无名指PIP屈曲持久伺服目标与实际位置差；提供执行滞后/加载信息，除数为antiwindup尺度 |
| 390 | servo_error | j4_ring_dip | (held_command[23]-q_raw[23]) / 0.35 | rad/rad | Actor + Critic | 无名指DIP屈曲持久伺服目标与实际位置差；提供执行滞后/加载信息，除数为antiwindup尺度 |
| 391 | servo_error | j5_pinky_mcp_abd | (held_command[24]-q_raw[24]) / 0.15 | rad/rad | Actor + Critic | 小指MCP外展持久伺服目标与实际位置差；提供执行滞后/加载信息，除数为antiwindup尺度 |
| 392 | servo_error | j5_pinky_mcp_flex | (held_command[25]-q_raw[25]) / 0.35 | rad/rad | Actor + Critic | 小指MCP屈曲持久伺服目标与实际位置差；提供执行滞后/加载信息，除数为antiwindup尺度 |
| 393 | servo_error | j5_pinky_pip | (held_command[26]-q_raw[26]) / 0.35 | rad/rad | Actor + Critic | 小指PIP屈曲持久伺服目标与实际位置差；提供执行滞后/加载信息，除数为antiwindup尺度 |
| 394 | servo_error | j5_pinky_dip | (held_command[27]-q_raw[27]) / 0.35 | rad/rad | Actor + Critic | 小指DIP屈曲持久伺服目标与实际位置差；提供执行滞后/加载信息，除数为antiwindup尺度 |
| 395 | critic_keypoint_error | palm.x | (actual_origin[palm].x-reference_origin[palm].x) / 0.2 | m / 0.2m | Critic | 掌部实际位置相对当前参考位置的误差；仅用于价值估计 |
| 396 | critic_keypoint_error | palm.y | (actual_origin[palm].y-reference_origin[palm].y) / 0.2 | m / 0.2m | Critic | 掌部实际位置相对当前参考位置的误差；仅用于价值估计 |
| 397 | critic_keypoint_error | palm.z | (actual_origin[palm].z-reference_origin[palm].z) / 0.2 | m / 0.2m | Critic | 掌部实际位置相对当前参考位置的误差；仅用于价值估计 |
| 398 | critic_keypoint_error | index_mcp.x | (actual_origin[index_mcp].x-reference_origin[index_mcp].x) / 0.2 | m / 0.2m | Critic | 食指近端实际位置相对当前参考位置的误差；仅用于价值估计 |
| 399 | critic_keypoint_error | index_mcp.y | (actual_origin[index_mcp].y-reference_origin[index_mcp].y) / 0.2 | m / 0.2m | Critic | 食指近端实际位置相对当前参考位置的误差；仅用于价值估计 |
| 400 | critic_keypoint_error | index_mcp.z | (actual_origin[index_mcp].z-reference_origin[index_mcp].z) / 0.2 | m / 0.2m | Critic | 食指近端实际位置相对当前参考位置的误差；仅用于价值估计 |
| 401 | critic_keypoint_error | index_pip.x | (actual_origin[index_pip].x-reference_origin[index_pip].x) / 0.2 | m / 0.2m | Critic | 食指中段实际位置相对当前参考位置的误差；仅用于价值估计 |
| 402 | critic_keypoint_error | index_pip.y | (actual_origin[index_pip].y-reference_origin[index_pip].y) / 0.2 | m / 0.2m | Critic | 食指中段实际位置相对当前参考位置的误差；仅用于价值估计 |
| 403 | critic_keypoint_error | index_pip.z | (actual_origin[index_pip].z-reference_origin[index_pip].z) / 0.2 | m / 0.2m | Critic | 食指中段实际位置相对当前参考位置的误差；仅用于价值估计 |
| 404 | critic_keypoint_error | index_dip.x | (actual_origin[index_dip].x-reference_origin[index_dip].x) / 0.2 | m / 0.2m | Critic | 食指远端实际位置相对当前参考位置的误差；仅用于价值估计 |
| 405 | critic_keypoint_error | index_dip.y | (actual_origin[index_dip].y-reference_origin[index_dip].y) / 0.2 | m / 0.2m | Critic | 食指远端实际位置相对当前参考位置的误差；仅用于价值估计 |
| 406 | critic_keypoint_error | index_dip.z | (actual_origin[index_dip].z-reference_origin[index_dip].z) / 0.2 | m / 0.2m | Critic | 食指远端实际位置相对当前参考位置的误差；仅用于价值估计 |
| 407 | critic_keypoint_error | middle_mcp.x | (actual_origin[middle_mcp].x-reference_origin[middle_mcp].x) / 0.2 | m / 0.2m | Critic | 中指近端实际位置相对当前参考位置的误差；仅用于价值估计 |
| 408 | critic_keypoint_error | middle_mcp.y | (actual_origin[middle_mcp].y-reference_origin[middle_mcp].y) / 0.2 | m / 0.2m | Critic | 中指近端实际位置相对当前参考位置的误差；仅用于价值估计 |
| 409 | critic_keypoint_error | middle_mcp.z | (actual_origin[middle_mcp].z-reference_origin[middle_mcp].z) / 0.2 | m / 0.2m | Critic | 中指近端实际位置相对当前参考位置的误差；仅用于价值估计 |
| 410 | critic_keypoint_error | middle_pip.x | (actual_origin[middle_pip].x-reference_origin[middle_pip].x) / 0.2 | m / 0.2m | Critic | 中指中段实际位置相对当前参考位置的误差；仅用于价值估计 |
| 411 | critic_keypoint_error | middle_pip.y | (actual_origin[middle_pip].y-reference_origin[middle_pip].y) / 0.2 | m / 0.2m | Critic | 中指中段实际位置相对当前参考位置的误差；仅用于价值估计 |
| 412 | critic_keypoint_error | middle_pip.z | (actual_origin[middle_pip].z-reference_origin[middle_pip].z) / 0.2 | m / 0.2m | Critic | 中指中段实际位置相对当前参考位置的误差；仅用于价值估计 |
| 413 | critic_keypoint_error | middle_dip.x | (actual_origin[middle_dip].x-reference_origin[middle_dip].x) / 0.2 | m / 0.2m | Critic | 中指远端实际位置相对当前参考位置的误差；仅用于价值估计 |
| 414 | critic_keypoint_error | middle_dip.y | (actual_origin[middle_dip].y-reference_origin[middle_dip].y) / 0.2 | m / 0.2m | Critic | 中指远端实际位置相对当前参考位置的误差；仅用于价值估计 |
| 415 | critic_keypoint_error | middle_dip.z | (actual_origin[middle_dip].z-reference_origin[middle_dip].z) / 0.2 | m / 0.2m | Critic | 中指远端实际位置相对当前参考位置的误差；仅用于价值估计 |
| 416 | critic_keypoint_error | ring_mcp.x | (actual_origin[ring_mcp].x-reference_origin[ring_mcp].x) / 0.2 | m / 0.2m | Critic | 无名指近端实际位置相对当前参考位置的误差；仅用于价值估计 |
| 417 | critic_keypoint_error | ring_mcp.y | (actual_origin[ring_mcp].y-reference_origin[ring_mcp].y) / 0.2 | m / 0.2m | Critic | 无名指近端实际位置相对当前参考位置的误差；仅用于价值估计 |
| 418 | critic_keypoint_error | ring_mcp.z | (actual_origin[ring_mcp].z-reference_origin[ring_mcp].z) / 0.2 | m / 0.2m | Critic | 无名指近端实际位置相对当前参考位置的误差；仅用于价值估计 |
| 419 | critic_keypoint_error | ring_pip.x | (actual_origin[ring_pip].x-reference_origin[ring_pip].x) / 0.2 | m / 0.2m | Critic | 无名指中段实际位置相对当前参考位置的误差；仅用于价值估计 |
| 420 | critic_keypoint_error | ring_pip.y | (actual_origin[ring_pip].y-reference_origin[ring_pip].y) / 0.2 | m / 0.2m | Critic | 无名指中段实际位置相对当前参考位置的误差；仅用于价值估计 |
| 421 | critic_keypoint_error | ring_pip.z | (actual_origin[ring_pip].z-reference_origin[ring_pip].z) / 0.2 | m / 0.2m | Critic | 无名指中段实际位置相对当前参考位置的误差；仅用于价值估计 |
| 422 | critic_keypoint_error | ring_dip.x | (actual_origin[ring_dip].x-reference_origin[ring_dip].x) / 0.2 | m / 0.2m | Critic | 无名指远端实际位置相对当前参考位置的误差；仅用于价值估计 |
| 423 | critic_keypoint_error | ring_dip.y | (actual_origin[ring_dip].y-reference_origin[ring_dip].y) / 0.2 | m / 0.2m | Critic | 无名指远端实际位置相对当前参考位置的误差；仅用于价值估计 |
| 424 | critic_keypoint_error | ring_dip.z | (actual_origin[ring_dip].z-reference_origin[ring_dip].z) / 0.2 | m / 0.2m | Critic | 无名指远端实际位置相对当前参考位置的误差；仅用于价值估计 |
| 425 | critic_keypoint_error | pinky_mcp.x | (actual_origin[pinky_mcp].x-reference_origin[pinky_mcp].x) / 0.2 | m / 0.2m | Critic | 小指近端实际位置相对当前参考位置的误差；仅用于价值估计 |
| 426 | critic_keypoint_error | pinky_mcp.y | (actual_origin[pinky_mcp].y-reference_origin[pinky_mcp].y) / 0.2 | m / 0.2m | Critic | 小指近端实际位置相对当前参考位置的误差；仅用于价值估计 |
| 427 | critic_keypoint_error | pinky_mcp.z | (actual_origin[pinky_mcp].z-reference_origin[pinky_mcp].z) / 0.2 | m / 0.2m | Critic | 小指近端实际位置相对当前参考位置的误差；仅用于价值估计 |
| 428 | critic_keypoint_error | pinky_pip.x | (actual_origin[pinky_pip].x-reference_origin[pinky_pip].x) / 0.2 | m / 0.2m | Critic | 小指中段实际位置相对当前参考位置的误差；仅用于价值估计 |
| 429 | critic_keypoint_error | pinky_pip.y | (actual_origin[pinky_pip].y-reference_origin[pinky_pip].y) / 0.2 | m / 0.2m | Critic | 小指中段实际位置相对当前参考位置的误差；仅用于价值估计 |
| 430 | critic_keypoint_error | pinky_pip.z | (actual_origin[pinky_pip].z-reference_origin[pinky_pip].z) / 0.2 | m / 0.2m | Critic | 小指中段实际位置相对当前参考位置的误差；仅用于价值估计 |
| 431 | critic_keypoint_error | pinky_dip.x | (actual_origin[pinky_dip].x-reference_origin[pinky_dip].x) / 0.2 | m / 0.2m | Critic | 小指远端实际位置相对当前参考位置的误差；仅用于价值估计 |
| 432 | critic_keypoint_error | pinky_dip.y | (actual_origin[pinky_dip].y-reference_origin[pinky_dip].y) / 0.2 | m / 0.2m | Critic | 小指远端实际位置相对当前参考位置的误差；仅用于价值估计 |
| 433 | critic_keypoint_error | pinky_dip.z | (actual_origin[pinky_dip].z-reference_origin[pinky_dip].z) / 0.2 | m / 0.2m | Critic | 小指远端实际位置相对当前参考位置的误差；仅用于价值估计 |
| 434 | critic_keypoint_error | thumb_cmc.x | (actual_origin[thumb_cmc].x-reference_origin[thumb_cmc].x) / 0.2 | m / 0.2m | Critic | 拇指根部实际位置相对当前参考位置的误差；仅用于价值估计 |
| 435 | critic_keypoint_error | thumb_cmc.y | (actual_origin[thumb_cmc].y-reference_origin[thumb_cmc].y) / 0.2 | m / 0.2m | Critic | 拇指根部实际位置相对当前参考位置的误差；仅用于价值估计 |
| 436 | critic_keypoint_error | thumb_cmc.z | (actual_origin[thumb_cmc].z-reference_origin[thumb_cmc].z) / 0.2 | m / 0.2m | Critic | 拇指根部实际位置相对当前参考位置的误差；仅用于价值估计 |
| 437 | critic_keypoint_error | thumb_mcp.x | (actual_origin[thumb_mcp].x-reference_origin[thumb_mcp].x) / 0.2 | m / 0.2m | Critic | 拇指中段实际位置相对当前参考位置的误差；仅用于价值估计 |
| 438 | critic_keypoint_error | thumb_mcp.y | (actual_origin[thumb_mcp].y-reference_origin[thumb_mcp].y) / 0.2 | m / 0.2m | Critic | 拇指中段实际位置相对当前参考位置的误差；仅用于价值估计 |
| 439 | critic_keypoint_error | thumb_mcp.z | (actual_origin[thumb_mcp].z-reference_origin[thumb_mcp].z) / 0.2 | m / 0.2m | Critic | 拇指中段实际位置相对当前参考位置的误差；仅用于价值估计 |
| 440 | critic_keypoint_error | thumb_ip.x | (actual_origin[thumb_ip].x-reference_origin[thumb_ip].x) / 0.2 | m / 0.2m | Critic | 拇指远端实际位置相对当前参考位置的误差；仅用于价值估计 |
| 441 | critic_keypoint_error | thumb_ip.y | (actual_origin[thumb_ip].y-reference_origin[thumb_ip].y) / 0.2 | m / 0.2m | Critic | 拇指远端实际位置相对当前参考位置的误差；仅用于价值估计 |
| 442 | critic_keypoint_error | thumb_ip.z | (actual_origin[thumb_ip].z-reference_origin[thumb_ip].z) / 0.2 | m / 0.2m | Critic | 拇指远端实际位置相对当前参考位置的误差；仅用于价值估计 |
| 443 | terminal_hand_goal | final_palm_displacement.x | (final_reference_palm.x-actual_palm.x) / 0.2 | m / 0.2m | Actor + Critic | 给Actor和Critic提供最终掌部目标方向与距离 |
| 444 | terminal_hand_goal | final_palm_displacement.y | (final_reference_palm.y-actual_palm.y) / 0.2 | m / 0.2m | Actor + Critic | 给Actor和Critic提供最终掌部目标方向与距离 |
| 445 | terminal_hand_goal | final_palm_displacement.z | (final_reference_palm.z-actual_palm.z) / 0.2 | m / 0.2m | Actor + Critic | 给Actor和Critic提供最终掌部目标方向与距离 |
| 446 | terminal_hand_goal | terminal_relative_rotation.R00 | R(inverse(current_palm) * final_palm).R00 | 无量纲 | Actor + Critic | 最终掌部相对当前掌部的旋转；前两列按列展开，不是六个角度 |
| 447 | terminal_hand_goal | terminal_relative_rotation.R10 | R(inverse(current_palm) * final_palm).R10 | 无量纲 | Actor + Critic | 最终掌部相对当前掌部的旋转；前两列按列展开，不是六个角度 |
| 448 | terminal_hand_goal | terminal_relative_rotation.R20 | R(inverse(current_palm) * final_palm).R20 | 无量纲 | Actor + Critic | 最终掌部相对当前掌部的旋转；前两列按列展开，不是六个角度 |
| 449 | terminal_hand_goal | terminal_relative_rotation.R01 | R(inverse(current_palm) * final_palm).R01 | 无量纲 | Actor + Critic | 最终掌部相对当前掌部的旋转；前两列按列展开，不是六个角度 |
| 450 | terminal_hand_goal | terminal_relative_rotation.R11 | R(inverse(current_palm) * final_palm).R11 | 无量纲 | Actor + Critic | 最终掌部相对当前掌部的旋转；前两列按列展开，不是六个角度 |
| 451 | terminal_hand_goal | terminal_relative_rotation.R21 | R(inverse(current_palm) * final_palm).R21 | 无量纲 | Actor + Critic | 最终掌部相对当前掌部的旋转；前两列按列展开，不是六个角度 |
| 452 | terminal_hand_goal | reference_end_reached | index >= max(last_index, 1) | 0或1 | Actor + Critic | 参考索引是否达到终点；不是实际任务成功标志 |

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
