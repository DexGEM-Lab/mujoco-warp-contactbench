# 任务：三个物体动作对（cube2:02 / banana:09 / largeclamp:04）prefix-only 合成 + 离线扩增

## 目标
每个动作对生成 720 行（240 base × 3 增强）：
- prefix-only 合成（3 稳健源 × 80 slots：Far 50 + Near 30，每 slot 12 fallback）
- 三条件 gate + prefix 碰撞 gate
- 离线 retreat（真实最后接触帧+15，替换尾部）
- 离线 translate ±20cm + rotate ±30°（LHS 均匀）
- 输出独立 Lance 发布 NAS

## 关键约束
- 合同与正式 1035 一致：prefix 必须有（缺 prefix 的版本被用户拒绝并删除）
- ckp = checkpoint-001000（daa1d2e7…，pre180 训练、pre60 合成合同）
- 3 动作全部在 ckp 训练集内（cylinder6 不在，不可用）
- 成功率基线见 local SOP 第 0 节
