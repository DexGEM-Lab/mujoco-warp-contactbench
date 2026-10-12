# v8：547维目标快照逐维说明

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

| 索引 | 流 | 槽 | 分量 | 写入公式 | 单位 | 用途 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | object | 0 | object_goal_position.x | reference_object.position.x | m | 目标物体位置，直接送入目标记录投影 |
| 1 | object | 0 | object_goal_position.y | reference_object.position.y | m | 目标物体位置，直接送入目标记录投影 |
| 2 | object | 0 | object_goal_position.z | reference_object.position.z | m | 目标物体位置，直接送入目标记录投影 |
| 3 | object | 0 | object_goal_rotation.R00 | R(reference_object).R00 | 无量纲 | 目标物体旋转矩阵前两列 |
| 4 | object | 0 | object_goal_rotation.R10 | R(reference_object).R10 | 无量纲 | 目标物体旋转矩阵前两列 |
| 5 | object | 0 | object_goal_rotation.R20 | R(reference_object).R20 | 无量纲 | 目标物体旋转矩阵前两列 |
| 6 | object | 0 | object_goal_rotation.R01 | R(reference_object).R01 | 无量纲 | 目标物体旋转矩阵前两列 |
| 7 | object | 0 | object_goal_rotation.R11 | R(reference_object).R11 | 无量纲 | 目标物体旋转矩阵前两列 |
| 8 | object | 0 | object_goal_rotation.R21 | R(reference_object).R21 | 无量纲 | 目标物体旋转矩阵前两列 |
| 9 | object | 0 | is_endpoint | target_frame == last_frame | 0或1 | 区分最终目标与中间目标 |
| 10 | object | 0 | object_goal_point_0.x | (reference_object.position + R(reference_object) * local_point[0]).x | m | 4个物体目标几何点之一，表明目标几何位置 |
| 11 | object | 0 | object_goal_point_0.y | (reference_object.position + R(reference_object) * local_point[0]).y | m | 4个物体目标几何点之一，表明目标几何位置 |
| 12 | object | 0 | object_goal_point_0.z | (reference_object.position + R(reference_object) * local_point[0]).z | m | 4个物体目标几何点之一，表明目标几何位置 |
| 13 | object | 0 | object_goal_point_16.x | (reference_object.position + R(reference_object) * local_point[16]).x | m | 4个物体目标几何点之一，表明目标几何位置 |
| 14 | object | 0 | object_goal_point_16.y | (reference_object.position + R(reference_object) * local_point[16]).y | m | 4个物体目标几何点之一，表明目标几何位置 |
| 15 | object | 0 | object_goal_point_16.z | (reference_object.position + R(reference_object) * local_point[16]).z | m | 4个物体目标几何点之一，表明目标几何位置 |
| 16 | object | 0 | object_goal_point_32.x | (reference_object.position + R(reference_object) * local_point[32]).x | m | 4个物体目标几何点之一，表明目标几何位置 |
| 17 | object | 0 | object_goal_point_32.y | (reference_object.position + R(reference_object) * local_point[32]).y | m | 4个物体目标几何点之一，表明目标几何位置 |
| 18 | object | 0 | object_goal_point_32.z | (reference_object.position + R(reference_object) * local_point[32]).z | m | 4个物体目标几何点之一，表明目标几何位置 |
| 19 | object | 0 | object_goal_point_48.x | (reference_object.position + R(reference_object) * local_point[48]).x | m | 4个物体目标几何点之一，表明目标几何位置 |
| 20 | object | 0 | object_goal_point_48.y | (reference_object.position + R(reference_object) * local_point[48]).y | m | 4个物体目标几何点之一，表明目标几何位置 |
| 21 | object | 0 | object_goal_point_48.z | (reference_object.position + R(reference_object) * local_point[48]).z | m | 4个物体目标几何点之一，表明目标几何位置 |
| 22 | object | 1 | object_goal_position.x | reference_object.position.x | m | 目标物体位置，直接送入目标记录投影 |
| 23 | object | 1 | object_goal_position.y | reference_object.position.y | m | 目标物体位置，直接送入目标记录投影 |
| 24 | object | 1 | object_goal_position.z | reference_object.position.z | m | 目标物体位置，直接送入目标记录投影 |
| 25 | object | 1 | object_goal_rotation.R00 | R(reference_object).R00 | 无量纲 | 目标物体旋转矩阵前两列 |
| 26 | object | 1 | object_goal_rotation.R10 | R(reference_object).R10 | 无量纲 | 目标物体旋转矩阵前两列 |
| 27 | object | 1 | object_goal_rotation.R20 | R(reference_object).R20 | 无量纲 | 目标物体旋转矩阵前两列 |
| 28 | object | 1 | object_goal_rotation.R01 | R(reference_object).R01 | 无量纲 | 目标物体旋转矩阵前两列 |
| 29 | object | 1 | object_goal_rotation.R11 | R(reference_object).R11 | 无量纲 | 目标物体旋转矩阵前两列 |
| 30 | object | 1 | object_goal_rotation.R21 | R(reference_object).R21 | 无量纲 | 目标物体旋转矩阵前两列 |
| 31 | object | 1 | is_endpoint | target_frame == last_frame | 0或1 | 区分最终目标与中间目标 |
| 32 | object | 1 | object_goal_point_0.x | (reference_object.position + R(reference_object) * local_point[0]).x | m | 4个物体目标几何点之一，表明目标几何位置 |
| 33 | object | 1 | object_goal_point_0.y | (reference_object.position + R(reference_object) * local_point[0]).y | m | 4个物体目标几何点之一，表明目标几何位置 |
| 34 | object | 1 | object_goal_point_0.z | (reference_object.position + R(reference_object) * local_point[0]).z | m | 4个物体目标几何点之一，表明目标几何位置 |
| 35 | object | 1 | object_goal_point_16.x | (reference_object.position + R(reference_object) * local_point[16]).x | m | 4个物体目标几何点之一，表明目标几何位置 |
| 36 | object | 1 | object_goal_point_16.y | (reference_object.position + R(reference_object) * local_point[16]).y | m | 4个物体目标几何点之一，表明目标几何位置 |
| 37 | object | 1 | object_goal_point_16.z | (reference_object.position + R(reference_object) * local_point[16]).z | m | 4个物体目标几何点之一，表明目标几何位置 |
| 38 | object | 1 | object_goal_point_32.x | (reference_object.position + R(reference_object) * local_point[32]).x | m | 4个物体目标几何点之一，表明目标几何位置 |
| 39 | object | 1 | object_goal_point_32.y | (reference_object.position + R(reference_object) * local_point[32]).y | m | 4个物体目标几何点之一，表明目标几何位置 |
| 40 | object | 1 | object_goal_point_32.z | (reference_object.position + R(reference_object) * local_point[32]).z | m | 4个物体目标几何点之一，表明目标几何位置 |
| 41 | object | 1 | object_goal_point_48.x | (reference_object.position + R(reference_object) * local_point[48]).x | m | 4个物体目标几何点之一，表明目标几何位置 |
| 42 | object | 1 | object_goal_point_48.y | (reference_object.position + R(reference_object) * local_point[48]).y | m | 4个物体目标几何点之一，表明目标几何位置 |
| 43 | object | 1 | object_goal_point_48.z | (reference_object.position + R(reference_object) * local_point[48]).z | m | 4个物体目标几何点之一，表明目标几何位置 |
| 44 | object | 2 | object_goal_position.x | reference_object.position.x | m | 目标物体位置，直接送入目标记录投影 |
| 45 | object | 2 | object_goal_position.y | reference_object.position.y | m | 目标物体位置，直接送入目标记录投影 |
| 46 | object | 2 | object_goal_position.z | reference_object.position.z | m | 目标物体位置，直接送入目标记录投影 |
| 47 | object | 2 | object_goal_rotation.R00 | R(reference_object).R00 | 无量纲 | 目标物体旋转矩阵前两列 |
| 48 | object | 2 | object_goal_rotation.R10 | R(reference_object).R10 | 无量纲 | 目标物体旋转矩阵前两列 |
| 49 | object | 2 | object_goal_rotation.R20 | R(reference_object).R20 | 无量纲 | 目标物体旋转矩阵前两列 |
| 50 | object | 2 | object_goal_rotation.R01 | R(reference_object).R01 | 无量纲 | 目标物体旋转矩阵前两列 |
| 51 | object | 2 | object_goal_rotation.R11 | R(reference_object).R11 | 无量纲 | 目标物体旋转矩阵前两列 |
| 52 | object | 2 | object_goal_rotation.R21 | R(reference_object).R21 | 无量纲 | 目标物体旋转矩阵前两列 |
| 53 | object | 2 | is_endpoint | target_frame == last_frame | 0或1 | 区分最终目标与中间目标 |
| 54 | object | 2 | object_goal_point_0.x | (reference_object.position + R(reference_object) * local_point[0]).x | m | 4个物体目标几何点之一，表明目标几何位置 |
| 55 | object | 2 | object_goal_point_0.y | (reference_object.position + R(reference_object) * local_point[0]).y | m | 4个物体目标几何点之一，表明目标几何位置 |
| 56 | object | 2 | object_goal_point_0.z | (reference_object.position + R(reference_object) * local_point[0]).z | m | 4个物体目标几何点之一，表明目标几何位置 |
| 57 | object | 2 | object_goal_point_16.x | (reference_object.position + R(reference_object) * local_point[16]).x | m | 4个物体目标几何点之一，表明目标几何位置 |
| 58 | object | 2 | object_goal_point_16.y | (reference_object.position + R(reference_object) * local_point[16]).y | m | 4个物体目标几何点之一，表明目标几何位置 |
| 59 | object | 2 | object_goal_point_16.z | (reference_object.position + R(reference_object) * local_point[16]).z | m | 4个物体目标几何点之一，表明目标几何位置 |
| 60 | object | 2 | object_goal_point_32.x | (reference_object.position + R(reference_object) * local_point[32]).x | m | 4个物体目标几何点之一，表明目标几何位置 |
| 61 | object | 2 | object_goal_point_32.y | (reference_object.position + R(reference_object) * local_point[32]).y | m | 4个物体目标几何点之一，表明目标几何位置 |
| 62 | object | 2 | object_goal_point_32.z | (reference_object.position + R(reference_object) * local_point[32]).z | m | 4个物体目标几何点之一，表明目标几何位置 |
| 63 | object | 2 | object_goal_point_48.x | (reference_object.position + R(reference_object) * local_point[48]).x | m | 4个物体目标几何点之一，表明目标几何位置 |
| 64 | object | 2 | object_goal_point_48.y | (reference_object.position + R(reference_object) * local_point[48]).y | m | 4个物体目标几何点之一，表明目标几何位置 |
| 65 | object | 2 | object_goal_point_48.z | (reference_object.position + R(reference_object) * local_point[48]).z | m | 4个物体目标几何点之一，表明目标几何位置 |
| 66 | object | 3 | object_goal_position.x | reference_object.position.x | m | 目标物体位置，直接送入目标记录投影 |
| 67 | object | 3 | object_goal_position.y | reference_object.position.y | m | 目标物体位置，直接送入目标记录投影 |
| 68 | object | 3 | object_goal_position.z | reference_object.position.z | m | 目标物体位置，直接送入目标记录投影 |
| 69 | object | 3 | object_goal_rotation.R00 | R(reference_object).R00 | 无量纲 | 目标物体旋转矩阵前两列 |
| 70 | object | 3 | object_goal_rotation.R10 | R(reference_object).R10 | 无量纲 | 目标物体旋转矩阵前两列 |
| 71 | object | 3 | object_goal_rotation.R20 | R(reference_object).R20 | 无量纲 | 目标物体旋转矩阵前两列 |
| 72 | object | 3 | object_goal_rotation.R01 | R(reference_object).R01 | 无量纲 | 目标物体旋转矩阵前两列 |
| 73 | object | 3 | object_goal_rotation.R11 | R(reference_object).R11 | 无量纲 | 目标物体旋转矩阵前两列 |
| 74 | object | 3 | object_goal_rotation.R21 | R(reference_object).R21 | 无量纲 | 目标物体旋转矩阵前两列 |
| 75 | object | 3 | is_endpoint | target_frame == last_frame | 0或1 | 区分最终目标与中间目标 |
| 76 | object | 3 | object_goal_point_0.x | (reference_object.position + R(reference_object) * local_point[0]).x | m | 4个物体目标几何点之一，表明目标几何位置 |
| 77 | object | 3 | object_goal_point_0.y | (reference_object.position + R(reference_object) * local_point[0]).y | m | 4个物体目标几何点之一，表明目标几何位置 |
| 78 | object | 3 | object_goal_point_0.z | (reference_object.position + R(reference_object) * local_point[0]).z | m | 4个物体目标几何点之一，表明目标几何位置 |
| 79 | object | 3 | object_goal_point_16.x | (reference_object.position + R(reference_object) * local_point[16]).x | m | 4个物体目标几何点之一，表明目标几何位置 |
| 80 | object | 3 | object_goal_point_16.y | (reference_object.position + R(reference_object) * local_point[16]).y | m | 4个物体目标几何点之一，表明目标几何位置 |
| 81 | object | 3 | object_goal_point_16.z | (reference_object.position + R(reference_object) * local_point[16]).z | m | 4个物体目标几何点之一，表明目标几何位置 |
| 82 | object | 3 | object_goal_point_32.x | (reference_object.position + R(reference_object) * local_point[32]).x | m | 4个物体目标几何点之一，表明目标几何位置 |
| 83 | object | 3 | object_goal_point_32.y | (reference_object.position + R(reference_object) * local_point[32]).y | m | 4个物体目标几何点之一，表明目标几何位置 |
| 84 | object | 3 | object_goal_point_32.z | (reference_object.position + R(reference_object) * local_point[32]).z | m | 4个物体目标几何点之一，表明目标几何位置 |
| 85 | object | 3 | object_goal_point_48.x | (reference_object.position + R(reference_object) * local_point[48]).x | m | 4个物体目标几何点之一，表明目标几何位置 |
| 86 | object | 3 | object_goal_point_48.y | (reference_object.position + R(reference_object) * local_point[48]).y | m | 4个物体目标几何点之一，表明目标几何位置 |
| 87 | object | 3 | object_goal_point_48.z | (reference_object.position + R(reference_object) * local_point[48]).z | m | 4个物体目标几何点之一，表明目标几何位置 |
| 88 | object | 4 | object_goal_position.x | reference_object.position.x | m | 目标物体位置，直接送入目标记录投影 |
| 89 | object | 4 | object_goal_position.y | reference_object.position.y | m | 目标物体位置，直接送入目标记录投影 |
| 90 | object | 4 | object_goal_position.z | reference_object.position.z | m | 目标物体位置，直接送入目标记录投影 |
| 91 | object | 4 | object_goal_rotation.R00 | R(reference_object).R00 | 无量纲 | 目标物体旋转矩阵前两列 |
| 92 | object | 4 | object_goal_rotation.R10 | R(reference_object).R10 | 无量纲 | 目标物体旋转矩阵前两列 |
| 93 | object | 4 | object_goal_rotation.R20 | R(reference_object).R20 | 无量纲 | 目标物体旋转矩阵前两列 |
| 94 | object | 4 | object_goal_rotation.R01 | R(reference_object).R01 | 无量纲 | 目标物体旋转矩阵前两列 |
| 95 | object | 4 | object_goal_rotation.R11 | R(reference_object).R11 | 无量纲 | 目标物体旋转矩阵前两列 |
| 96 | object | 4 | object_goal_rotation.R21 | R(reference_object).R21 | 无量纲 | 目标物体旋转矩阵前两列 |
| 97 | object | 4 | is_endpoint | target_frame == last_frame | 0或1 | 区分最终目标与中间目标 |
| 98 | object | 4 | object_goal_point_0.x | (reference_object.position + R(reference_object) * local_point[0]).x | m | 4个物体目标几何点之一，表明目标几何位置 |
| 99 | object | 4 | object_goal_point_0.y | (reference_object.position + R(reference_object) * local_point[0]).y | m | 4个物体目标几何点之一，表明目标几何位置 |
| 100 | object | 4 | object_goal_point_0.z | (reference_object.position + R(reference_object) * local_point[0]).z | m | 4个物体目标几何点之一，表明目标几何位置 |
| 101 | object | 4 | object_goal_point_16.x | (reference_object.position + R(reference_object) * local_point[16]).x | m | 4个物体目标几何点之一，表明目标几何位置 |
| 102 | object | 4 | object_goal_point_16.y | (reference_object.position + R(reference_object) * local_point[16]).y | m | 4个物体目标几何点之一，表明目标几何位置 |
| 103 | object | 4 | object_goal_point_16.z | (reference_object.position + R(reference_object) * local_point[16]).z | m | 4个物体目标几何点之一，表明目标几何位置 |
| 104 | object | 4 | object_goal_point_32.x | (reference_object.position + R(reference_object) * local_point[32]).x | m | 4个物体目标几何点之一，表明目标几何位置 |
| 105 | object | 4 | object_goal_point_32.y | (reference_object.position + R(reference_object) * local_point[32]).y | m | 4个物体目标几何点之一，表明目标几何位置 |
| 106 | object | 4 | object_goal_point_32.z | (reference_object.position + R(reference_object) * local_point[32]).z | m | 4个物体目标几何点之一，表明目标几何位置 |
| 107 | object | 4 | object_goal_point_48.x | (reference_object.position + R(reference_object) * local_point[48]).x | m | 4个物体目标几何点之一，表明目标几何位置 |
| 108 | object | 4 | object_goal_point_48.y | (reference_object.position + R(reference_object) * local_point[48]).y | m | 4个物体目标几何点之一，表明目标几何位置 |
| 109 | object | 4 | object_goal_point_48.z | (reference_object.position + R(reference_object) * local_point[48]).z | m | 4个物体目标几何点之一，表明目标几何位置 |
| 110 | object | 0 | mask | target_valid | 0或1 | 有效记录标志，控制目标注意力；无效raw也置零 |
| 111 | object | 1 | mask | target_valid | 0或1 | 有效记录标志，控制目标注意力；无效raw也置零 |
| 112 | object | 2 | mask | target_valid | 0或1 | 有效记录标志，控制目标注意力；无效raw也置零 |
| 113 | object | 3 | mask | target_valid | 0或1 | 有效记录标志，控制目标注意力；无效raw也置零 |
| 114 | object | 4 | mask | target_valid | 0或1 | 有效记录标志，控制目标注意力；无效raw也置零 |
| 115 | object | 0 | dt | (target_frame-current_frame) * control_timestep | s | 目标相对当前时刻；编码为dt/T_MAX与8组sin/cos，共17维 |
| 116 | object | 1 | dt | (target_frame-current_frame) * control_timestep | s | 目标相对当前时刻；编码为dt/T_MAX与8组sin/cos，共17维 |
| 117 | object | 2 | dt | (target_frame-current_frame) * control_timestep | s | 目标相对当前时刻；编码为dt/T_MAX与8组sin/cos，共17维 |
| 118 | object | 3 | dt | (target_frame-current_frame) * control_timestep | s | 目标相对当前时刻；编码为dt/T_MAX与8组sin/cos，共17维 |
| 119 | object | 4 | dt | (target_frame-current_frame) * control_timestep | s | 目标相对当前时刻；编码为dt/T_MAX与8组sin/cos，共17维 |
| 120 | object | 0 | frame_id | target_frame | 帧号 | 保存来源，完全不进入网络编码 |
| 121 | object | 1 | frame_id | target_frame | 帧号 | 保存来源，完全不进入网络编码 |
| 122 | object | 2 | frame_id | target_frame | 帧号 | 保存来源，完全不进入网络编码 |
| 123 | object | 3 | frame_id | target_frame | 帧号 | 保存来源，完全不进入网络编码 |
| 124 | object | 4 | frame_id | target_frame | 帧号 | 保存来源，完全不进入网络编码 |
| 125 | object | 0 | type_id | 0 | 类别编号 | 目标类型嵌入：物体0，手部2 |
| 126 | object | 1 | type_id | 0 | 类别编号 | 目标类型嵌入：物体0，手部2 |
| 127 | object | 2 | type_id | 0 | 类别编号 | 目标类型嵌入：物体0，手部2 |
| 128 | object | 3 | type_id | 0 | 类别编号 | 目标类型嵌入：物体0，手部2 |
| 129 | object | 4 | type_id | 0 | 类别编号 | 目标类型嵌入：物体0，手部2 |
| 130 | object | None | presence | 1 in current ManoRL adapter | 0到1 | 整路目标的注意力门控；当前两路均存在，缺失流可设0 |
| 131 | hand | 0 | palm_goal_position.x | reference_palm.position.x | m | 目标掌部位置 |
| 132 | hand | 0 | palm_goal_position.y | reference_palm.position.y | m | 目标掌部位置 |
| 133 | hand | 0 | palm_goal_position.z | reference_palm.position.z | m | 目标掌部位置 |
| 134 | hand | 0 | palm_goal_rotation.R00 | R(reference_palm).R00 | 无量纲 | 目标掌部旋转矩阵前两列 |
| 135 | hand | 0 | palm_goal_rotation.R10 | R(reference_palm).R10 | 无量纲 | 目标掌部旋转矩阵前两列 |
| 136 | hand | 0 | palm_goal_rotation.R20 | R(reference_palm).R20 | 无量纲 | 目标掌部旋转矩阵前两列 |
| 137 | hand | 0 | palm_goal_rotation.R01 | R(reference_palm).R01 | 无量纲 | 目标掌部旋转矩阵前两列 |
| 138 | hand | 0 | palm_goal_rotation.R11 | R(reference_palm).R11 | 无量纲 | 目标掌部旋转矩阵前两列 |
| 139 | hand | 0 | palm_goal_rotation.R21 | R(reference_palm).R21 | 无量纲 | 目标掌部旋转矩阵前两列 |
| 140 | hand | 0 | j1_thumb_cmc_abd | reference_q_feasible[6] | rad | 拇指CMC外展目标角度 |
| 141 | hand | 0 | j1_thumb_cmc_flex | reference_q_feasible[7] | rad | 拇指CMC屈曲目标角度 |
| 142 | hand | 0 | j1_thumb_cmc_twist | reference_q_feasible[8] | rad | 拇指CMC扭转目标角度 |
| 143 | hand | 0 | j1_thumb_mcp_flex | reference_q_feasible[9] | rad | 拇指MCP屈曲目标角度 |
| 144 | hand | 0 | j1_thumb_mcp_abd | reference_q_feasible[10] | rad | 拇指MCP外展目标角度 |
| 145 | hand | 0 | j1_thumb_ip | reference_q_feasible[11] | rad | 拇指IP屈曲目标角度 |
| 146 | hand | 0 | j2_index_mcp_abd | reference_q_feasible[12] | rad | 食指MCP外展目标角度 |
| 147 | hand | 0 | j2_index_mcp_flex | reference_q_feasible[13] | rad | 食指MCP屈曲目标角度 |
| 148 | hand | 0 | j2_index_pip | reference_q_feasible[14] | rad | 食指PIP屈曲目标角度 |
| 149 | hand | 0 | j2_index_dip | reference_q_feasible[15] | rad | 食指DIP屈曲目标角度 |
| 150 | hand | 0 | j3_middle_mcp_abd | reference_q_feasible[16] | rad | 中指MCP外展目标角度 |
| 151 | hand | 0 | j3_middle_mcp_flex | reference_q_feasible[17] | rad | 中指MCP屈曲目标角度 |
| 152 | hand | 0 | j3_middle_pip | reference_q_feasible[18] | rad | 中指PIP屈曲目标角度 |
| 153 | hand | 0 | j3_middle_dip | reference_q_feasible[19] | rad | 中指DIP屈曲目标角度 |
| 154 | hand | 0 | j4_ring_mcp_abd | reference_q_feasible[20] | rad | 无名指MCP外展目标角度 |
| 155 | hand | 0 | j4_ring_mcp_flex | reference_q_feasible[21] | rad | 无名指MCP屈曲目标角度 |
| 156 | hand | 0 | j4_ring_pip | reference_q_feasible[22] | rad | 无名指PIP屈曲目标角度 |
| 157 | hand | 0 | j4_ring_dip | reference_q_feasible[23] | rad | 无名指DIP屈曲目标角度 |
| 158 | hand | 0 | j5_pinky_mcp_abd | reference_q_feasible[24] | rad | 小指MCP外展目标角度 |
| 159 | hand | 0 | j5_pinky_mcp_flex | reference_q_feasible[25] | rad | 小指MCP屈曲目标角度 |
| 160 | hand | 0 | j5_pinky_pip | reference_q_feasible[26] | rad | 小指PIP屈曲目标角度 |
| 161 | hand | 0 | j5_pinky_dip | reference_q_feasible[27] | rad | 小指DIP屈曲目标角度 |
| 162 | hand | 0 | goal_palm.x | reference_bone_origin_world[palm].x | m | 掌部目标位置 |
| 163 | hand | 0 | goal_palm.y | reference_bone_origin_world[palm].y | m | 掌部目标位置 |
| 164 | hand | 0 | goal_palm.z | reference_bone_origin_world[palm].z | m | 掌部目标位置 |
| 165 | hand | 0 | goal_index_mcp.x | reference_bone_origin_world[index_mcp].x | m | 食指近端目标位置 |
| 166 | hand | 0 | goal_index_mcp.y | reference_bone_origin_world[index_mcp].y | m | 食指近端目标位置 |
| 167 | hand | 0 | goal_index_mcp.z | reference_bone_origin_world[index_mcp].z | m | 食指近端目标位置 |
| 168 | hand | 0 | goal_index_pip.x | reference_bone_origin_world[index_pip].x | m | 食指中段目标位置 |
| 169 | hand | 0 | goal_index_pip.y | reference_bone_origin_world[index_pip].y | m | 食指中段目标位置 |
| 170 | hand | 0 | goal_index_pip.z | reference_bone_origin_world[index_pip].z | m | 食指中段目标位置 |
| 171 | hand | 0 | goal_index_dip.x | reference_bone_origin_world[index_dip].x | m | 食指远端目标位置 |
| 172 | hand | 0 | goal_index_dip.y | reference_bone_origin_world[index_dip].y | m | 食指远端目标位置 |
| 173 | hand | 0 | goal_index_dip.z | reference_bone_origin_world[index_dip].z | m | 食指远端目标位置 |
| 174 | hand | 0 | goal_middle_mcp.x | reference_bone_origin_world[middle_mcp].x | m | 中指近端目标位置 |
| 175 | hand | 0 | goal_middle_mcp.y | reference_bone_origin_world[middle_mcp].y | m | 中指近端目标位置 |
| 176 | hand | 0 | goal_middle_mcp.z | reference_bone_origin_world[middle_mcp].z | m | 中指近端目标位置 |
| 177 | hand | 0 | goal_middle_pip.x | reference_bone_origin_world[middle_pip].x | m | 中指中段目标位置 |
| 178 | hand | 0 | goal_middle_pip.y | reference_bone_origin_world[middle_pip].y | m | 中指中段目标位置 |
| 179 | hand | 0 | goal_middle_pip.z | reference_bone_origin_world[middle_pip].z | m | 中指中段目标位置 |
| 180 | hand | 0 | goal_middle_dip.x | reference_bone_origin_world[middle_dip].x | m | 中指远端目标位置 |
| 181 | hand | 0 | goal_middle_dip.y | reference_bone_origin_world[middle_dip].y | m | 中指远端目标位置 |
| 182 | hand | 0 | goal_middle_dip.z | reference_bone_origin_world[middle_dip].z | m | 中指远端目标位置 |
| 183 | hand | 0 | goal_ring_mcp.x | reference_bone_origin_world[ring_mcp].x | m | 无名指近端目标位置 |
| 184 | hand | 0 | goal_ring_mcp.y | reference_bone_origin_world[ring_mcp].y | m | 无名指近端目标位置 |
| 185 | hand | 0 | goal_ring_mcp.z | reference_bone_origin_world[ring_mcp].z | m | 无名指近端目标位置 |
| 186 | hand | 0 | goal_ring_pip.x | reference_bone_origin_world[ring_pip].x | m | 无名指中段目标位置 |
| 187 | hand | 0 | goal_ring_pip.y | reference_bone_origin_world[ring_pip].y | m | 无名指中段目标位置 |
| 188 | hand | 0 | goal_ring_pip.z | reference_bone_origin_world[ring_pip].z | m | 无名指中段目标位置 |
| 189 | hand | 0 | goal_ring_dip.x | reference_bone_origin_world[ring_dip].x | m | 无名指远端目标位置 |
| 190 | hand | 0 | goal_ring_dip.y | reference_bone_origin_world[ring_dip].y | m | 无名指远端目标位置 |
| 191 | hand | 0 | goal_ring_dip.z | reference_bone_origin_world[ring_dip].z | m | 无名指远端目标位置 |
| 192 | hand | 0 | goal_pinky_mcp.x | reference_bone_origin_world[pinky_mcp].x | m | 小指近端目标位置 |
| 193 | hand | 0 | goal_pinky_mcp.y | reference_bone_origin_world[pinky_mcp].y | m | 小指近端目标位置 |
| 194 | hand | 0 | goal_pinky_mcp.z | reference_bone_origin_world[pinky_mcp].z | m | 小指近端目标位置 |
| 195 | hand | 0 | goal_pinky_pip.x | reference_bone_origin_world[pinky_pip].x | m | 小指中段目标位置 |
| 196 | hand | 0 | goal_pinky_pip.y | reference_bone_origin_world[pinky_pip].y | m | 小指中段目标位置 |
| 197 | hand | 0 | goal_pinky_pip.z | reference_bone_origin_world[pinky_pip].z | m | 小指中段目标位置 |
| 198 | hand | 0 | goal_pinky_dip.x | reference_bone_origin_world[pinky_dip].x | m | 小指远端目标位置 |
| 199 | hand | 0 | goal_pinky_dip.y | reference_bone_origin_world[pinky_dip].y | m | 小指远端目标位置 |
| 200 | hand | 0 | goal_pinky_dip.z | reference_bone_origin_world[pinky_dip].z | m | 小指远端目标位置 |
| 201 | hand | 0 | goal_thumb_cmc.x | reference_bone_origin_world[thumb_cmc].x | m | 拇指根部目标位置 |
| 202 | hand | 0 | goal_thumb_cmc.y | reference_bone_origin_world[thumb_cmc].y | m | 拇指根部目标位置 |
| 203 | hand | 0 | goal_thumb_cmc.z | reference_bone_origin_world[thumb_cmc].z | m | 拇指根部目标位置 |
| 204 | hand | 0 | goal_thumb_mcp.x | reference_bone_origin_world[thumb_mcp].x | m | 拇指中段目标位置 |
| 205 | hand | 0 | goal_thumb_mcp.y | reference_bone_origin_world[thumb_mcp].y | m | 拇指中段目标位置 |
| 206 | hand | 0 | goal_thumb_mcp.z | reference_bone_origin_world[thumb_mcp].z | m | 拇指中段目标位置 |
| 207 | hand | 0 | goal_thumb_ip.x | reference_bone_origin_world[thumb_ip].x | m | 拇指远端目标位置 |
| 208 | hand | 0 | goal_thumb_ip.y | reference_bone_origin_world[thumb_ip].y | m | 拇指远端目标位置 |
| 209 | hand | 0 | goal_thumb_ip.z | reference_bone_origin_world[thumb_ip].z | m | 拇指远端目标位置 |
| 210 | hand | 1 | palm_goal_position.x | reference_palm.position.x | m | 目标掌部位置 |
| 211 | hand | 1 | palm_goal_position.y | reference_palm.position.y | m | 目标掌部位置 |
| 212 | hand | 1 | palm_goal_position.z | reference_palm.position.z | m | 目标掌部位置 |
| 213 | hand | 1 | palm_goal_rotation.R00 | R(reference_palm).R00 | 无量纲 | 目标掌部旋转矩阵前两列 |
| 214 | hand | 1 | palm_goal_rotation.R10 | R(reference_palm).R10 | 无量纲 | 目标掌部旋转矩阵前两列 |
| 215 | hand | 1 | palm_goal_rotation.R20 | R(reference_palm).R20 | 无量纲 | 目标掌部旋转矩阵前两列 |
| 216 | hand | 1 | palm_goal_rotation.R01 | R(reference_palm).R01 | 无量纲 | 目标掌部旋转矩阵前两列 |
| 217 | hand | 1 | palm_goal_rotation.R11 | R(reference_palm).R11 | 无量纲 | 目标掌部旋转矩阵前两列 |
| 218 | hand | 1 | palm_goal_rotation.R21 | R(reference_palm).R21 | 无量纲 | 目标掌部旋转矩阵前两列 |
| 219 | hand | 1 | j1_thumb_cmc_abd | reference_q_feasible[6] | rad | 拇指CMC外展目标角度 |
| 220 | hand | 1 | j1_thumb_cmc_flex | reference_q_feasible[7] | rad | 拇指CMC屈曲目标角度 |
| 221 | hand | 1 | j1_thumb_cmc_twist | reference_q_feasible[8] | rad | 拇指CMC扭转目标角度 |
| 222 | hand | 1 | j1_thumb_mcp_flex | reference_q_feasible[9] | rad | 拇指MCP屈曲目标角度 |
| 223 | hand | 1 | j1_thumb_mcp_abd | reference_q_feasible[10] | rad | 拇指MCP外展目标角度 |
| 224 | hand | 1 | j1_thumb_ip | reference_q_feasible[11] | rad | 拇指IP屈曲目标角度 |
| 225 | hand | 1 | j2_index_mcp_abd | reference_q_feasible[12] | rad | 食指MCP外展目标角度 |
| 226 | hand | 1 | j2_index_mcp_flex | reference_q_feasible[13] | rad | 食指MCP屈曲目标角度 |
| 227 | hand | 1 | j2_index_pip | reference_q_feasible[14] | rad | 食指PIP屈曲目标角度 |
| 228 | hand | 1 | j2_index_dip | reference_q_feasible[15] | rad | 食指DIP屈曲目标角度 |
| 229 | hand | 1 | j3_middle_mcp_abd | reference_q_feasible[16] | rad | 中指MCP外展目标角度 |
| 230 | hand | 1 | j3_middle_mcp_flex | reference_q_feasible[17] | rad | 中指MCP屈曲目标角度 |
| 231 | hand | 1 | j3_middle_pip | reference_q_feasible[18] | rad | 中指PIP屈曲目标角度 |
| 232 | hand | 1 | j3_middle_dip | reference_q_feasible[19] | rad | 中指DIP屈曲目标角度 |
| 233 | hand | 1 | j4_ring_mcp_abd | reference_q_feasible[20] | rad | 无名指MCP外展目标角度 |
| 234 | hand | 1 | j4_ring_mcp_flex | reference_q_feasible[21] | rad | 无名指MCP屈曲目标角度 |
| 235 | hand | 1 | j4_ring_pip | reference_q_feasible[22] | rad | 无名指PIP屈曲目标角度 |
| 236 | hand | 1 | j4_ring_dip | reference_q_feasible[23] | rad | 无名指DIP屈曲目标角度 |
| 237 | hand | 1 | j5_pinky_mcp_abd | reference_q_feasible[24] | rad | 小指MCP外展目标角度 |
| 238 | hand | 1 | j5_pinky_mcp_flex | reference_q_feasible[25] | rad | 小指MCP屈曲目标角度 |
| 239 | hand | 1 | j5_pinky_pip | reference_q_feasible[26] | rad | 小指PIP屈曲目标角度 |
| 240 | hand | 1 | j5_pinky_dip | reference_q_feasible[27] | rad | 小指DIP屈曲目标角度 |
| 241 | hand | 1 | goal_palm.x | reference_bone_origin_world[palm].x | m | 掌部目标位置 |
| 242 | hand | 1 | goal_palm.y | reference_bone_origin_world[palm].y | m | 掌部目标位置 |
| 243 | hand | 1 | goal_palm.z | reference_bone_origin_world[palm].z | m | 掌部目标位置 |
| 244 | hand | 1 | goal_index_mcp.x | reference_bone_origin_world[index_mcp].x | m | 食指近端目标位置 |
| 245 | hand | 1 | goal_index_mcp.y | reference_bone_origin_world[index_mcp].y | m | 食指近端目标位置 |
| 246 | hand | 1 | goal_index_mcp.z | reference_bone_origin_world[index_mcp].z | m | 食指近端目标位置 |
| 247 | hand | 1 | goal_index_pip.x | reference_bone_origin_world[index_pip].x | m | 食指中段目标位置 |
| 248 | hand | 1 | goal_index_pip.y | reference_bone_origin_world[index_pip].y | m | 食指中段目标位置 |
| 249 | hand | 1 | goal_index_pip.z | reference_bone_origin_world[index_pip].z | m | 食指中段目标位置 |
| 250 | hand | 1 | goal_index_dip.x | reference_bone_origin_world[index_dip].x | m | 食指远端目标位置 |
| 251 | hand | 1 | goal_index_dip.y | reference_bone_origin_world[index_dip].y | m | 食指远端目标位置 |
| 252 | hand | 1 | goal_index_dip.z | reference_bone_origin_world[index_dip].z | m | 食指远端目标位置 |
| 253 | hand | 1 | goal_middle_mcp.x | reference_bone_origin_world[middle_mcp].x | m | 中指近端目标位置 |
| 254 | hand | 1 | goal_middle_mcp.y | reference_bone_origin_world[middle_mcp].y | m | 中指近端目标位置 |
| 255 | hand | 1 | goal_middle_mcp.z | reference_bone_origin_world[middle_mcp].z | m | 中指近端目标位置 |
| 256 | hand | 1 | goal_middle_pip.x | reference_bone_origin_world[middle_pip].x | m | 中指中段目标位置 |
| 257 | hand | 1 | goal_middle_pip.y | reference_bone_origin_world[middle_pip].y | m | 中指中段目标位置 |
| 258 | hand | 1 | goal_middle_pip.z | reference_bone_origin_world[middle_pip].z | m | 中指中段目标位置 |
| 259 | hand | 1 | goal_middle_dip.x | reference_bone_origin_world[middle_dip].x | m | 中指远端目标位置 |
| 260 | hand | 1 | goal_middle_dip.y | reference_bone_origin_world[middle_dip].y | m | 中指远端目标位置 |
| 261 | hand | 1 | goal_middle_dip.z | reference_bone_origin_world[middle_dip].z | m | 中指远端目标位置 |
| 262 | hand | 1 | goal_ring_mcp.x | reference_bone_origin_world[ring_mcp].x | m | 无名指近端目标位置 |
| 263 | hand | 1 | goal_ring_mcp.y | reference_bone_origin_world[ring_mcp].y | m | 无名指近端目标位置 |
| 264 | hand | 1 | goal_ring_mcp.z | reference_bone_origin_world[ring_mcp].z | m | 无名指近端目标位置 |
| 265 | hand | 1 | goal_ring_pip.x | reference_bone_origin_world[ring_pip].x | m | 无名指中段目标位置 |
| 266 | hand | 1 | goal_ring_pip.y | reference_bone_origin_world[ring_pip].y | m | 无名指中段目标位置 |
| 267 | hand | 1 | goal_ring_pip.z | reference_bone_origin_world[ring_pip].z | m | 无名指中段目标位置 |
| 268 | hand | 1 | goal_ring_dip.x | reference_bone_origin_world[ring_dip].x | m | 无名指远端目标位置 |
| 269 | hand | 1 | goal_ring_dip.y | reference_bone_origin_world[ring_dip].y | m | 无名指远端目标位置 |
| 270 | hand | 1 | goal_ring_dip.z | reference_bone_origin_world[ring_dip].z | m | 无名指远端目标位置 |
| 271 | hand | 1 | goal_pinky_mcp.x | reference_bone_origin_world[pinky_mcp].x | m | 小指近端目标位置 |
| 272 | hand | 1 | goal_pinky_mcp.y | reference_bone_origin_world[pinky_mcp].y | m | 小指近端目标位置 |
| 273 | hand | 1 | goal_pinky_mcp.z | reference_bone_origin_world[pinky_mcp].z | m | 小指近端目标位置 |
| 274 | hand | 1 | goal_pinky_pip.x | reference_bone_origin_world[pinky_pip].x | m | 小指中段目标位置 |
| 275 | hand | 1 | goal_pinky_pip.y | reference_bone_origin_world[pinky_pip].y | m | 小指中段目标位置 |
| 276 | hand | 1 | goal_pinky_pip.z | reference_bone_origin_world[pinky_pip].z | m | 小指中段目标位置 |
| 277 | hand | 1 | goal_pinky_dip.x | reference_bone_origin_world[pinky_dip].x | m | 小指远端目标位置 |
| 278 | hand | 1 | goal_pinky_dip.y | reference_bone_origin_world[pinky_dip].y | m | 小指远端目标位置 |
| 279 | hand | 1 | goal_pinky_dip.z | reference_bone_origin_world[pinky_dip].z | m | 小指远端目标位置 |
| 280 | hand | 1 | goal_thumb_cmc.x | reference_bone_origin_world[thumb_cmc].x | m | 拇指根部目标位置 |
| 281 | hand | 1 | goal_thumb_cmc.y | reference_bone_origin_world[thumb_cmc].y | m | 拇指根部目标位置 |
| 282 | hand | 1 | goal_thumb_cmc.z | reference_bone_origin_world[thumb_cmc].z | m | 拇指根部目标位置 |
| 283 | hand | 1 | goal_thumb_mcp.x | reference_bone_origin_world[thumb_mcp].x | m | 拇指中段目标位置 |
| 284 | hand | 1 | goal_thumb_mcp.y | reference_bone_origin_world[thumb_mcp].y | m | 拇指中段目标位置 |
| 285 | hand | 1 | goal_thumb_mcp.z | reference_bone_origin_world[thumb_mcp].z | m | 拇指中段目标位置 |
| 286 | hand | 1 | goal_thumb_ip.x | reference_bone_origin_world[thumb_ip].x | m | 拇指远端目标位置 |
| 287 | hand | 1 | goal_thumb_ip.y | reference_bone_origin_world[thumb_ip].y | m | 拇指远端目标位置 |
| 288 | hand | 1 | goal_thumb_ip.z | reference_bone_origin_world[thumb_ip].z | m | 拇指远端目标位置 |
| 289 | hand | 2 | palm_goal_position.x | reference_palm.position.x | m | 目标掌部位置 |
| 290 | hand | 2 | palm_goal_position.y | reference_palm.position.y | m | 目标掌部位置 |
| 291 | hand | 2 | palm_goal_position.z | reference_palm.position.z | m | 目标掌部位置 |
| 292 | hand | 2 | palm_goal_rotation.R00 | R(reference_palm).R00 | 无量纲 | 目标掌部旋转矩阵前两列 |
| 293 | hand | 2 | palm_goal_rotation.R10 | R(reference_palm).R10 | 无量纲 | 目标掌部旋转矩阵前两列 |
| 294 | hand | 2 | palm_goal_rotation.R20 | R(reference_palm).R20 | 无量纲 | 目标掌部旋转矩阵前两列 |
| 295 | hand | 2 | palm_goal_rotation.R01 | R(reference_palm).R01 | 无量纲 | 目标掌部旋转矩阵前两列 |
| 296 | hand | 2 | palm_goal_rotation.R11 | R(reference_palm).R11 | 无量纲 | 目标掌部旋转矩阵前两列 |
| 297 | hand | 2 | palm_goal_rotation.R21 | R(reference_palm).R21 | 无量纲 | 目标掌部旋转矩阵前两列 |
| 298 | hand | 2 | j1_thumb_cmc_abd | reference_q_feasible[6] | rad | 拇指CMC外展目标角度 |
| 299 | hand | 2 | j1_thumb_cmc_flex | reference_q_feasible[7] | rad | 拇指CMC屈曲目标角度 |
| 300 | hand | 2 | j1_thumb_cmc_twist | reference_q_feasible[8] | rad | 拇指CMC扭转目标角度 |
| 301 | hand | 2 | j1_thumb_mcp_flex | reference_q_feasible[9] | rad | 拇指MCP屈曲目标角度 |
| 302 | hand | 2 | j1_thumb_mcp_abd | reference_q_feasible[10] | rad | 拇指MCP外展目标角度 |
| 303 | hand | 2 | j1_thumb_ip | reference_q_feasible[11] | rad | 拇指IP屈曲目标角度 |
| 304 | hand | 2 | j2_index_mcp_abd | reference_q_feasible[12] | rad | 食指MCP外展目标角度 |
| 305 | hand | 2 | j2_index_mcp_flex | reference_q_feasible[13] | rad | 食指MCP屈曲目标角度 |
| 306 | hand | 2 | j2_index_pip | reference_q_feasible[14] | rad | 食指PIP屈曲目标角度 |
| 307 | hand | 2 | j2_index_dip | reference_q_feasible[15] | rad | 食指DIP屈曲目标角度 |
| 308 | hand | 2 | j3_middle_mcp_abd | reference_q_feasible[16] | rad | 中指MCP外展目标角度 |
| 309 | hand | 2 | j3_middle_mcp_flex | reference_q_feasible[17] | rad | 中指MCP屈曲目标角度 |
| 310 | hand | 2 | j3_middle_pip | reference_q_feasible[18] | rad | 中指PIP屈曲目标角度 |
| 311 | hand | 2 | j3_middle_dip | reference_q_feasible[19] | rad | 中指DIP屈曲目标角度 |
| 312 | hand | 2 | j4_ring_mcp_abd | reference_q_feasible[20] | rad | 无名指MCP外展目标角度 |
| 313 | hand | 2 | j4_ring_mcp_flex | reference_q_feasible[21] | rad | 无名指MCP屈曲目标角度 |
| 314 | hand | 2 | j4_ring_pip | reference_q_feasible[22] | rad | 无名指PIP屈曲目标角度 |
| 315 | hand | 2 | j4_ring_dip | reference_q_feasible[23] | rad | 无名指DIP屈曲目标角度 |
| 316 | hand | 2 | j5_pinky_mcp_abd | reference_q_feasible[24] | rad | 小指MCP外展目标角度 |
| 317 | hand | 2 | j5_pinky_mcp_flex | reference_q_feasible[25] | rad | 小指MCP屈曲目标角度 |
| 318 | hand | 2 | j5_pinky_pip | reference_q_feasible[26] | rad | 小指PIP屈曲目标角度 |
| 319 | hand | 2 | j5_pinky_dip | reference_q_feasible[27] | rad | 小指DIP屈曲目标角度 |
| 320 | hand | 2 | goal_palm.x | reference_bone_origin_world[palm].x | m | 掌部目标位置 |
| 321 | hand | 2 | goal_palm.y | reference_bone_origin_world[palm].y | m | 掌部目标位置 |
| 322 | hand | 2 | goal_palm.z | reference_bone_origin_world[palm].z | m | 掌部目标位置 |
| 323 | hand | 2 | goal_index_mcp.x | reference_bone_origin_world[index_mcp].x | m | 食指近端目标位置 |
| 324 | hand | 2 | goal_index_mcp.y | reference_bone_origin_world[index_mcp].y | m | 食指近端目标位置 |
| 325 | hand | 2 | goal_index_mcp.z | reference_bone_origin_world[index_mcp].z | m | 食指近端目标位置 |
| 326 | hand | 2 | goal_index_pip.x | reference_bone_origin_world[index_pip].x | m | 食指中段目标位置 |
| 327 | hand | 2 | goal_index_pip.y | reference_bone_origin_world[index_pip].y | m | 食指中段目标位置 |
| 328 | hand | 2 | goal_index_pip.z | reference_bone_origin_world[index_pip].z | m | 食指中段目标位置 |
| 329 | hand | 2 | goal_index_dip.x | reference_bone_origin_world[index_dip].x | m | 食指远端目标位置 |
| 330 | hand | 2 | goal_index_dip.y | reference_bone_origin_world[index_dip].y | m | 食指远端目标位置 |
| 331 | hand | 2 | goal_index_dip.z | reference_bone_origin_world[index_dip].z | m | 食指远端目标位置 |
| 332 | hand | 2 | goal_middle_mcp.x | reference_bone_origin_world[middle_mcp].x | m | 中指近端目标位置 |
| 333 | hand | 2 | goal_middle_mcp.y | reference_bone_origin_world[middle_mcp].y | m | 中指近端目标位置 |
| 334 | hand | 2 | goal_middle_mcp.z | reference_bone_origin_world[middle_mcp].z | m | 中指近端目标位置 |
| 335 | hand | 2 | goal_middle_pip.x | reference_bone_origin_world[middle_pip].x | m | 中指中段目标位置 |
| 336 | hand | 2 | goal_middle_pip.y | reference_bone_origin_world[middle_pip].y | m | 中指中段目标位置 |
| 337 | hand | 2 | goal_middle_pip.z | reference_bone_origin_world[middle_pip].z | m | 中指中段目标位置 |
| 338 | hand | 2 | goal_middle_dip.x | reference_bone_origin_world[middle_dip].x | m | 中指远端目标位置 |
| 339 | hand | 2 | goal_middle_dip.y | reference_bone_origin_world[middle_dip].y | m | 中指远端目标位置 |
| 340 | hand | 2 | goal_middle_dip.z | reference_bone_origin_world[middle_dip].z | m | 中指远端目标位置 |
| 341 | hand | 2 | goal_ring_mcp.x | reference_bone_origin_world[ring_mcp].x | m | 无名指近端目标位置 |
| 342 | hand | 2 | goal_ring_mcp.y | reference_bone_origin_world[ring_mcp].y | m | 无名指近端目标位置 |
| 343 | hand | 2 | goal_ring_mcp.z | reference_bone_origin_world[ring_mcp].z | m | 无名指近端目标位置 |
| 344 | hand | 2 | goal_ring_pip.x | reference_bone_origin_world[ring_pip].x | m | 无名指中段目标位置 |
| 345 | hand | 2 | goal_ring_pip.y | reference_bone_origin_world[ring_pip].y | m | 无名指中段目标位置 |
| 346 | hand | 2 | goal_ring_pip.z | reference_bone_origin_world[ring_pip].z | m | 无名指中段目标位置 |
| 347 | hand | 2 | goal_ring_dip.x | reference_bone_origin_world[ring_dip].x | m | 无名指远端目标位置 |
| 348 | hand | 2 | goal_ring_dip.y | reference_bone_origin_world[ring_dip].y | m | 无名指远端目标位置 |
| 349 | hand | 2 | goal_ring_dip.z | reference_bone_origin_world[ring_dip].z | m | 无名指远端目标位置 |
| 350 | hand | 2 | goal_pinky_mcp.x | reference_bone_origin_world[pinky_mcp].x | m | 小指近端目标位置 |
| 351 | hand | 2 | goal_pinky_mcp.y | reference_bone_origin_world[pinky_mcp].y | m | 小指近端目标位置 |
| 352 | hand | 2 | goal_pinky_mcp.z | reference_bone_origin_world[pinky_mcp].z | m | 小指近端目标位置 |
| 353 | hand | 2 | goal_pinky_pip.x | reference_bone_origin_world[pinky_pip].x | m | 小指中段目标位置 |
| 354 | hand | 2 | goal_pinky_pip.y | reference_bone_origin_world[pinky_pip].y | m | 小指中段目标位置 |
| 355 | hand | 2 | goal_pinky_pip.z | reference_bone_origin_world[pinky_pip].z | m | 小指中段目标位置 |
| 356 | hand | 2 | goal_pinky_dip.x | reference_bone_origin_world[pinky_dip].x | m | 小指远端目标位置 |
| 357 | hand | 2 | goal_pinky_dip.y | reference_bone_origin_world[pinky_dip].y | m | 小指远端目标位置 |
| 358 | hand | 2 | goal_pinky_dip.z | reference_bone_origin_world[pinky_dip].z | m | 小指远端目标位置 |
| 359 | hand | 2 | goal_thumb_cmc.x | reference_bone_origin_world[thumb_cmc].x | m | 拇指根部目标位置 |
| 360 | hand | 2 | goal_thumb_cmc.y | reference_bone_origin_world[thumb_cmc].y | m | 拇指根部目标位置 |
| 361 | hand | 2 | goal_thumb_cmc.z | reference_bone_origin_world[thumb_cmc].z | m | 拇指根部目标位置 |
| 362 | hand | 2 | goal_thumb_mcp.x | reference_bone_origin_world[thumb_mcp].x | m | 拇指中段目标位置 |
| 363 | hand | 2 | goal_thumb_mcp.y | reference_bone_origin_world[thumb_mcp].y | m | 拇指中段目标位置 |
| 364 | hand | 2 | goal_thumb_mcp.z | reference_bone_origin_world[thumb_mcp].z | m | 拇指中段目标位置 |
| 365 | hand | 2 | goal_thumb_ip.x | reference_bone_origin_world[thumb_ip].x | m | 拇指远端目标位置 |
| 366 | hand | 2 | goal_thumb_ip.y | reference_bone_origin_world[thumb_ip].y | m | 拇指远端目标位置 |
| 367 | hand | 2 | goal_thumb_ip.z | reference_bone_origin_world[thumb_ip].z | m | 拇指远端目标位置 |
| 368 | hand | 3 | palm_goal_position.x | reference_palm.position.x | m | 目标掌部位置 |
| 369 | hand | 3 | palm_goal_position.y | reference_palm.position.y | m | 目标掌部位置 |
| 370 | hand | 3 | palm_goal_position.z | reference_palm.position.z | m | 目标掌部位置 |
| 371 | hand | 3 | palm_goal_rotation.R00 | R(reference_palm).R00 | 无量纲 | 目标掌部旋转矩阵前两列 |
| 372 | hand | 3 | palm_goal_rotation.R10 | R(reference_palm).R10 | 无量纲 | 目标掌部旋转矩阵前两列 |
| 373 | hand | 3 | palm_goal_rotation.R20 | R(reference_palm).R20 | 无量纲 | 目标掌部旋转矩阵前两列 |
| 374 | hand | 3 | palm_goal_rotation.R01 | R(reference_palm).R01 | 无量纲 | 目标掌部旋转矩阵前两列 |
| 375 | hand | 3 | palm_goal_rotation.R11 | R(reference_palm).R11 | 无量纲 | 目标掌部旋转矩阵前两列 |
| 376 | hand | 3 | palm_goal_rotation.R21 | R(reference_palm).R21 | 无量纲 | 目标掌部旋转矩阵前两列 |
| 377 | hand | 3 | j1_thumb_cmc_abd | reference_q_feasible[6] | rad | 拇指CMC外展目标角度 |
| 378 | hand | 3 | j1_thumb_cmc_flex | reference_q_feasible[7] | rad | 拇指CMC屈曲目标角度 |
| 379 | hand | 3 | j1_thumb_cmc_twist | reference_q_feasible[8] | rad | 拇指CMC扭转目标角度 |
| 380 | hand | 3 | j1_thumb_mcp_flex | reference_q_feasible[9] | rad | 拇指MCP屈曲目标角度 |
| 381 | hand | 3 | j1_thumb_mcp_abd | reference_q_feasible[10] | rad | 拇指MCP外展目标角度 |
| 382 | hand | 3 | j1_thumb_ip | reference_q_feasible[11] | rad | 拇指IP屈曲目标角度 |
| 383 | hand | 3 | j2_index_mcp_abd | reference_q_feasible[12] | rad | 食指MCP外展目标角度 |
| 384 | hand | 3 | j2_index_mcp_flex | reference_q_feasible[13] | rad | 食指MCP屈曲目标角度 |
| 385 | hand | 3 | j2_index_pip | reference_q_feasible[14] | rad | 食指PIP屈曲目标角度 |
| 386 | hand | 3 | j2_index_dip | reference_q_feasible[15] | rad | 食指DIP屈曲目标角度 |
| 387 | hand | 3 | j3_middle_mcp_abd | reference_q_feasible[16] | rad | 中指MCP外展目标角度 |
| 388 | hand | 3 | j3_middle_mcp_flex | reference_q_feasible[17] | rad | 中指MCP屈曲目标角度 |
| 389 | hand | 3 | j3_middle_pip | reference_q_feasible[18] | rad | 中指PIP屈曲目标角度 |
| 390 | hand | 3 | j3_middle_dip | reference_q_feasible[19] | rad | 中指DIP屈曲目标角度 |
| 391 | hand | 3 | j4_ring_mcp_abd | reference_q_feasible[20] | rad | 无名指MCP外展目标角度 |
| 392 | hand | 3 | j4_ring_mcp_flex | reference_q_feasible[21] | rad | 无名指MCP屈曲目标角度 |
| 393 | hand | 3 | j4_ring_pip | reference_q_feasible[22] | rad | 无名指PIP屈曲目标角度 |
| 394 | hand | 3 | j4_ring_dip | reference_q_feasible[23] | rad | 无名指DIP屈曲目标角度 |
| 395 | hand | 3 | j5_pinky_mcp_abd | reference_q_feasible[24] | rad | 小指MCP外展目标角度 |
| 396 | hand | 3 | j5_pinky_mcp_flex | reference_q_feasible[25] | rad | 小指MCP屈曲目标角度 |
| 397 | hand | 3 | j5_pinky_pip | reference_q_feasible[26] | rad | 小指PIP屈曲目标角度 |
| 398 | hand | 3 | j5_pinky_dip | reference_q_feasible[27] | rad | 小指DIP屈曲目标角度 |
| 399 | hand | 3 | goal_palm.x | reference_bone_origin_world[palm].x | m | 掌部目标位置 |
| 400 | hand | 3 | goal_palm.y | reference_bone_origin_world[palm].y | m | 掌部目标位置 |
| 401 | hand | 3 | goal_palm.z | reference_bone_origin_world[palm].z | m | 掌部目标位置 |
| 402 | hand | 3 | goal_index_mcp.x | reference_bone_origin_world[index_mcp].x | m | 食指近端目标位置 |
| 403 | hand | 3 | goal_index_mcp.y | reference_bone_origin_world[index_mcp].y | m | 食指近端目标位置 |
| 404 | hand | 3 | goal_index_mcp.z | reference_bone_origin_world[index_mcp].z | m | 食指近端目标位置 |
| 405 | hand | 3 | goal_index_pip.x | reference_bone_origin_world[index_pip].x | m | 食指中段目标位置 |
| 406 | hand | 3 | goal_index_pip.y | reference_bone_origin_world[index_pip].y | m | 食指中段目标位置 |
| 407 | hand | 3 | goal_index_pip.z | reference_bone_origin_world[index_pip].z | m | 食指中段目标位置 |
| 408 | hand | 3 | goal_index_dip.x | reference_bone_origin_world[index_dip].x | m | 食指远端目标位置 |
| 409 | hand | 3 | goal_index_dip.y | reference_bone_origin_world[index_dip].y | m | 食指远端目标位置 |
| 410 | hand | 3 | goal_index_dip.z | reference_bone_origin_world[index_dip].z | m | 食指远端目标位置 |
| 411 | hand | 3 | goal_middle_mcp.x | reference_bone_origin_world[middle_mcp].x | m | 中指近端目标位置 |
| 412 | hand | 3 | goal_middle_mcp.y | reference_bone_origin_world[middle_mcp].y | m | 中指近端目标位置 |
| 413 | hand | 3 | goal_middle_mcp.z | reference_bone_origin_world[middle_mcp].z | m | 中指近端目标位置 |
| 414 | hand | 3 | goal_middle_pip.x | reference_bone_origin_world[middle_pip].x | m | 中指中段目标位置 |
| 415 | hand | 3 | goal_middle_pip.y | reference_bone_origin_world[middle_pip].y | m | 中指中段目标位置 |
| 416 | hand | 3 | goal_middle_pip.z | reference_bone_origin_world[middle_pip].z | m | 中指中段目标位置 |
| 417 | hand | 3 | goal_middle_dip.x | reference_bone_origin_world[middle_dip].x | m | 中指远端目标位置 |
| 418 | hand | 3 | goal_middle_dip.y | reference_bone_origin_world[middle_dip].y | m | 中指远端目标位置 |
| 419 | hand | 3 | goal_middle_dip.z | reference_bone_origin_world[middle_dip].z | m | 中指远端目标位置 |
| 420 | hand | 3 | goal_ring_mcp.x | reference_bone_origin_world[ring_mcp].x | m | 无名指近端目标位置 |
| 421 | hand | 3 | goal_ring_mcp.y | reference_bone_origin_world[ring_mcp].y | m | 无名指近端目标位置 |
| 422 | hand | 3 | goal_ring_mcp.z | reference_bone_origin_world[ring_mcp].z | m | 无名指近端目标位置 |
| 423 | hand | 3 | goal_ring_pip.x | reference_bone_origin_world[ring_pip].x | m | 无名指中段目标位置 |
| 424 | hand | 3 | goal_ring_pip.y | reference_bone_origin_world[ring_pip].y | m | 无名指中段目标位置 |
| 425 | hand | 3 | goal_ring_pip.z | reference_bone_origin_world[ring_pip].z | m | 无名指中段目标位置 |
| 426 | hand | 3 | goal_ring_dip.x | reference_bone_origin_world[ring_dip].x | m | 无名指远端目标位置 |
| 427 | hand | 3 | goal_ring_dip.y | reference_bone_origin_world[ring_dip].y | m | 无名指远端目标位置 |
| 428 | hand | 3 | goal_ring_dip.z | reference_bone_origin_world[ring_dip].z | m | 无名指远端目标位置 |
| 429 | hand | 3 | goal_pinky_mcp.x | reference_bone_origin_world[pinky_mcp].x | m | 小指近端目标位置 |
| 430 | hand | 3 | goal_pinky_mcp.y | reference_bone_origin_world[pinky_mcp].y | m | 小指近端目标位置 |
| 431 | hand | 3 | goal_pinky_mcp.z | reference_bone_origin_world[pinky_mcp].z | m | 小指近端目标位置 |
| 432 | hand | 3 | goal_pinky_pip.x | reference_bone_origin_world[pinky_pip].x | m | 小指中段目标位置 |
| 433 | hand | 3 | goal_pinky_pip.y | reference_bone_origin_world[pinky_pip].y | m | 小指中段目标位置 |
| 434 | hand | 3 | goal_pinky_pip.z | reference_bone_origin_world[pinky_pip].z | m | 小指中段目标位置 |
| 435 | hand | 3 | goal_pinky_dip.x | reference_bone_origin_world[pinky_dip].x | m | 小指远端目标位置 |
| 436 | hand | 3 | goal_pinky_dip.y | reference_bone_origin_world[pinky_dip].y | m | 小指远端目标位置 |
| 437 | hand | 3 | goal_pinky_dip.z | reference_bone_origin_world[pinky_dip].z | m | 小指远端目标位置 |
| 438 | hand | 3 | goal_thumb_cmc.x | reference_bone_origin_world[thumb_cmc].x | m | 拇指根部目标位置 |
| 439 | hand | 3 | goal_thumb_cmc.y | reference_bone_origin_world[thumb_cmc].y | m | 拇指根部目标位置 |
| 440 | hand | 3 | goal_thumb_cmc.z | reference_bone_origin_world[thumb_cmc].z | m | 拇指根部目标位置 |
| 441 | hand | 3 | goal_thumb_mcp.x | reference_bone_origin_world[thumb_mcp].x | m | 拇指中段目标位置 |
| 442 | hand | 3 | goal_thumb_mcp.y | reference_bone_origin_world[thumb_mcp].y | m | 拇指中段目标位置 |
| 443 | hand | 3 | goal_thumb_mcp.z | reference_bone_origin_world[thumb_mcp].z | m | 拇指中段目标位置 |
| 444 | hand | 3 | goal_thumb_ip.x | reference_bone_origin_world[thumb_ip].x | m | 拇指远端目标位置 |
| 445 | hand | 3 | goal_thumb_ip.y | reference_bone_origin_world[thumb_ip].y | m | 拇指远端目标位置 |
| 446 | hand | 3 | goal_thumb_ip.z | reference_bone_origin_world[thumb_ip].z | m | 拇指远端目标位置 |
| 447 | hand | 4 | palm_goal_position.x | reference_palm.position.x | m | 目标掌部位置 |
| 448 | hand | 4 | palm_goal_position.y | reference_palm.position.y | m | 目标掌部位置 |
| 449 | hand | 4 | palm_goal_position.z | reference_palm.position.z | m | 目标掌部位置 |
| 450 | hand | 4 | palm_goal_rotation.R00 | R(reference_palm).R00 | 无量纲 | 目标掌部旋转矩阵前两列 |
| 451 | hand | 4 | palm_goal_rotation.R10 | R(reference_palm).R10 | 无量纲 | 目标掌部旋转矩阵前两列 |
| 452 | hand | 4 | palm_goal_rotation.R20 | R(reference_palm).R20 | 无量纲 | 目标掌部旋转矩阵前两列 |
| 453 | hand | 4 | palm_goal_rotation.R01 | R(reference_palm).R01 | 无量纲 | 目标掌部旋转矩阵前两列 |
| 454 | hand | 4 | palm_goal_rotation.R11 | R(reference_palm).R11 | 无量纲 | 目标掌部旋转矩阵前两列 |
| 455 | hand | 4 | palm_goal_rotation.R21 | R(reference_palm).R21 | 无量纲 | 目标掌部旋转矩阵前两列 |
| 456 | hand | 4 | j1_thumb_cmc_abd | reference_q_feasible[6] | rad | 拇指CMC外展目标角度 |
| 457 | hand | 4 | j1_thumb_cmc_flex | reference_q_feasible[7] | rad | 拇指CMC屈曲目标角度 |
| 458 | hand | 4 | j1_thumb_cmc_twist | reference_q_feasible[8] | rad | 拇指CMC扭转目标角度 |
| 459 | hand | 4 | j1_thumb_mcp_flex | reference_q_feasible[9] | rad | 拇指MCP屈曲目标角度 |
| 460 | hand | 4 | j1_thumb_mcp_abd | reference_q_feasible[10] | rad | 拇指MCP外展目标角度 |
| 461 | hand | 4 | j1_thumb_ip | reference_q_feasible[11] | rad | 拇指IP屈曲目标角度 |
| 462 | hand | 4 | j2_index_mcp_abd | reference_q_feasible[12] | rad | 食指MCP外展目标角度 |
| 463 | hand | 4 | j2_index_mcp_flex | reference_q_feasible[13] | rad | 食指MCP屈曲目标角度 |
| 464 | hand | 4 | j2_index_pip | reference_q_feasible[14] | rad | 食指PIP屈曲目标角度 |
| 465 | hand | 4 | j2_index_dip | reference_q_feasible[15] | rad | 食指DIP屈曲目标角度 |
| 466 | hand | 4 | j3_middle_mcp_abd | reference_q_feasible[16] | rad | 中指MCP外展目标角度 |
| 467 | hand | 4 | j3_middle_mcp_flex | reference_q_feasible[17] | rad | 中指MCP屈曲目标角度 |
| 468 | hand | 4 | j3_middle_pip | reference_q_feasible[18] | rad | 中指PIP屈曲目标角度 |
| 469 | hand | 4 | j3_middle_dip | reference_q_feasible[19] | rad | 中指DIP屈曲目标角度 |
| 470 | hand | 4 | j4_ring_mcp_abd | reference_q_feasible[20] | rad | 无名指MCP外展目标角度 |
| 471 | hand | 4 | j4_ring_mcp_flex | reference_q_feasible[21] | rad | 无名指MCP屈曲目标角度 |
| 472 | hand | 4 | j4_ring_pip | reference_q_feasible[22] | rad | 无名指PIP屈曲目标角度 |
| 473 | hand | 4 | j4_ring_dip | reference_q_feasible[23] | rad | 无名指DIP屈曲目标角度 |
| 474 | hand | 4 | j5_pinky_mcp_abd | reference_q_feasible[24] | rad | 小指MCP外展目标角度 |
| 475 | hand | 4 | j5_pinky_mcp_flex | reference_q_feasible[25] | rad | 小指MCP屈曲目标角度 |
| 476 | hand | 4 | j5_pinky_pip | reference_q_feasible[26] | rad | 小指PIP屈曲目标角度 |
| 477 | hand | 4 | j5_pinky_dip | reference_q_feasible[27] | rad | 小指DIP屈曲目标角度 |
| 478 | hand | 4 | goal_palm.x | reference_bone_origin_world[palm].x | m | 掌部目标位置 |
| 479 | hand | 4 | goal_palm.y | reference_bone_origin_world[palm].y | m | 掌部目标位置 |
| 480 | hand | 4 | goal_palm.z | reference_bone_origin_world[palm].z | m | 掌部目标位置 |
| 481 | hand | 4 | goal_index_mcp.x | reference_bone_origin_world[index_mcp].x | m | 食指近端目标位置 |
| 482 | hand | 4 | goal_index_mcp.y | reference_bone_origin_world[index_mcp].y | m | 食指近端目标位置 |
| 483 | hand | 4 | goal_index_mcp.z | reference_bone_origin_world[index_mcp].z | m | 食指近端目标位置 |
| 484 | hand | 4 | goal_index_pip.x | reference_bone_origin_world[index_pip].x | m | 食指中段目标位置 |
| 485 | hand | 4 | goal_index_pip.y | reference_bone_origin_world[index_pip].y | m | 食指中段目标位置 |
| 486 | hand | 4 | goal_index_pip.z | reference_bone_origin_world[index_pip].z | m | 食指中段目标位置 |
| 487 | hand | 4 | goal_index_dip.x | reference_bone_origin_world[index_dip].x | m | 食指远端目标位置 |
| 488 | hand | 4 | goal_index_dip.y | reference_bone_origin_world[index_dip].y | m | 食指远端目标位置 |
| 489 | hand | 4 | goal_index_dip.z | reference_bone_origin_world[index_dip].z | m | 食指远端目标位置 |
| 490 | hand | 4 | goal_middle_mcp.x | reference_bone_origin_world[middle_mcp].x | m | 中指近端目标位置 |
| 491 | hand | 4 | goal_middle_mcp.y | reference_bone_origin_world[middle_mcp].y | m | 中指近端目标位置 |
| 492 | hand | 4 | goal_middle_mcp.z | reference_bone_origin_world[middle_mcp].z | m | 中指近端目标位置 |
| 493 | hand | 4 | goal_middle_pip.x | reference_bone_origin_world[middle_pip].x | m | 中指中段目标位置 |
| 494 | hand | 4 | goal_middle_pip.y | reference_bone_origin_world[middle_pip].y | m | 中指中段目标位置 |
| 495 | hand | 4 | goal_middle_pip.z | reference_bone_origin_world[middle_pip].z | m | 中指中段目标位置 |
| 496 | hand | 4 | goal_middle_dip.x | reference_bone_origin_world[middle_dip].x | m | 中指远端目标位置 |
| 497 | hand | 4 | goal_middle_dip.y | reference_bone_origin_world[middle_dip].y | m | 中指远端目标位置 |
| 498 | hand | 4 | goal_middle_dip.z | reference_bone_origin_world[middle_dip].z | m | 中指远端目标位置 |
| 499 | hand | 4 | goal_ring_mcp.x | reference_bone_origin_world[ring_mcp].x | m | 无名指近端目标位置 |
| 500 | hand | 4 | goal_ring_mcp.y | reference_bone_origin_world[ring_mcp].y | m | 无名指近端目标位置 |
| 501 | hand | 4 | goal_ring_mcp.z | reference_bone_origin_world[ring_mcp].z | m | 无名指近端目标位置 |
| 502 | hand | 4 | goal_ring_pip.x | reference_bone_origin_world[ring_pip].x | m | 无名指中段目标位置 |
| 503 | hand | 4 | goal_ring_pip.y | reference_bone_origin_world[ring_pip].y | m | 无名指中段目标位置 |
| 504 | hand | 4 | goal_ring_pip.z | reference_bone_origin_world[ring_pip].z | m | 无名指中段目标位置 |
| 505 | hand | 4 | goal_ring_dip.x | reference_bone_origin_world[ring_dip].x | m | 无名指远端目标位置 |
| 506 | hand | 4 | goal_ring_dip.y | reference_bone_origin_world[ring_dip].y | m | 无名指远端目标位置 |
| 507 | hand | 4 | goal_ring_dip.z | reference_bone_origin_world[ring_dip].z | m | 无名指远端目标位置 |
| 508 | hand | 4 | goal_pinky_mcp.x | reference_bone_origin_world[pinky_mcp].x | m | 小指近端目标位置 |
| 509 | hand | 4 | goal_pinky_mcp.y | reference_bone_origin_world[pinky_mcp].y | m | 小指近端目标位置 |
| 510 | hand | 4 | goal_pinky_mcp.z | reference_bone_origin_world[pinky_mcp].z | m | 小指近端目标位置 |
| 511 | hand | 4 | goal_pinky_pip.x | reference_bone_origin_world[pinky_pip].x | m | 小指中段目标位置 |
| 512 | hand | 4 | goal_pinky_pip.y | reference_bone_origin_world[pinky_pip].y | m | 小指中段目标位置 |
| 513 | hand | 4 | goal_pinky_pip.z | reference_bone_origin_world[pinky_pip].z | m | 小指中段目标位置 |
| 514 | hand | 4 | goal_pinky_dip.x | reference_bone_origin_world[pinky_dip].x | m | 小指远端目标位置 |
| 515 | hand | 4 | goal_pinky_dip.y | reference_bone_origin_world[pinky_dip].y | m | 小指远端目标位置 |
| 516 | hand | 4 | goal_pinky_dip.z | reference_bone_origin_world[pinky_dip].z | m | 小指远端目标位置 |
| 517 | hand | 4 | goal_thumb_cmc.x | reference_bone_origin_world[thumb_cmc].x | m | 拇指根部目标位置 |
| 518 | hand | 4 | goal_thumb_cmc.y | reference_bone_origin_world[thumb_cmc].y | m | 拇指根部目标位置 |
| 519 | hand | 4 | goal_thumb_cmc.z | reference_bone_origin_world[thumb_cmc].z | m | 拇指根部目标位置 |
| 520 | hand | 4 | goal_thumb_mcp.x | reference_bone_origin_world[thumb_mcp].x | m | 拇指中段目标位置 |
| 521 | hand | 4 | goal_thumb_mcp.y | reference_bone_origin_world[thumb_mcp].y | m | 拇指中段目标位置 |
| 522 | hand | 4 | goal_thumb_mcp.z | reference_bone_origin_world[thumb_mcp].z | m | 拇指中段目标位置 |
| 523 | hand | 4 | goal_thumb_ip.x | reference_bone_origin_world[thumb_ip].x | m | 拇指远端目标位置 |
| 524 | hand | 4 | goal_thumb_ip.y | reference_bone_origin_world[thumb_ip].y | m | 拇指远端目标位置 |
| 525 | hand | 4 | goal_thumb_ip.z | reference_bone_origin_world[thumb_ip].z | m | 拇指远端目标位置 |
| 526 | hand | 0 | mask | target_valid | 0或1 | 有效记录标志，控制目标注意力；无效raw也置零 |
| 527 | hand | 1 | mask | target_valid | 0或1 | 有效记录标志，控制目标注意力；无效raw也置零 |
| 528 | hand | 2 | mask | target_valid | 0或1 | 有效记录标志，控制目标注意力；无效raw也置零 |
| 529 | hand | 3 | mask | target_valid | 0或1 | 有效记录标志，控制目标注意力；无效raw也置零 |
| 530 | hand | 4 | mask | target_valid | 0或1 | 有效记录标志，控制目标注意力；无效raw也置零 |
| 531 | hand | 0 | dt | (target_frame-current_frame) * control_timestep | s | 目标相对当前时刻；编码为dt/T_MAX与8组sin/cos，共17维 |
| 532 | hand | 1 | dt | (target_frame-current_frame) * control_timestep | s | 目标相对当前时刻；编码为dt/T_MAX与8组sin/cos，共17维 |
| 533 | hand | 2 | dt | (target_frame-current_frame) * control_timestep | s | 目标相对当前时刻；编码为dt/T_MAX与8组sin/cos，共17维 |
| 534 | hand | 3 | dt | (target_frame-current_frame) * control_timestep | s | 目标相对当前时刻；编码为dt/T_MAX与8组sin/cos，共17维 |
| 535 | hand | 4 | dt | (target_frame-current_frame) * control_timestep | s | 目标相对当前时刻；编码为dt/T_MAX与8组sin/cos，共17维 |
| 536 | hand | 0 | frame_id | target_frame | 帧号 | 保存来源，完全不进入网络编码 |
| 537 | hand | 1 | frame_id | target_frame | 帧号 | 保存来源，完全不进入网络编码 |
| 538 | hand | 2 | frame_id | target_frame | 帧号 | 保存来源，完全不进入网络编码 |
| 539 | hand | 3 | frame_id | target_frame | 帧号 | 保存来源，完全不进入网络编码 |
| 540 | hand | 4 | frame_id | target_frame | 帧号 | 保存来源，完全不进入网络编码 |
| 541 | hand | 0 | type_id | 2 | 类别编号 | 目标类型嵌入：物体0，手部2 |
| 542 | hand | 1 | type_id | 2 | 类别编号 | 目标类型嵌入：物体0，手部2 |
| 543 | hand | 2 | type_id | 2 | 类别编号 | 目标类型嵌入：物体0，手部2 |
| 544 | hand | 3 | type_id | 2 | 类别编号 | 目标类型嵌入：物体0，手部2 |
| 545 | hand | 4 | type_id | 2 | 类别编号 | 目标类型嵌入：物体0，手部2 |
| 546 | hand | None | presence | 1 in current ManoRL adapter | 0到1 | 整路目标的注意力门控；当前两路均存在，缺失流可设0 |
