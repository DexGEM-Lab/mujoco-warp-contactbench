# ManoRL v5 到 v6 五版本训练记录

状态：2026-09-19 正式训练执行中。本文只记录实际运行证据；没有完成训练或自然终止
评估的版本不得填写“成功”。

## 1. 实验目标

按相同环境、数据、奖励、PPO和动作分布，对下列五个策略进行可比训练：

```text
v5 → v5.25 → v5.5 → v5.75 → v6
```

每个训练进程只绑定一张RTX 4090。两张GPU作为并行队列使用，不允许一个任务
跨卡，也不终止其他用户的进程。

最终每个版本需要交付：

- 最佳checkpoint及其SHA256；
- 训练更新数、环境转换数和墙钟时间；
- 收敛或未收敛的判据；
- 自然首次终止的完整抓取评估；
- 抓取成功率、拇指接触、对向接触、载荷空中帧和掉落率；
- 最佳实际物理轨迹视频。

## 2. 固定比较协议

除模型/观测版本外，正式实验固定：

| 项目 | 设置 |
|---|---|
| 服务器 | `cty@192.168.9.220` |
| GPU | 两张独立RTX 4090；每任务仅一张 |
| 数据包 | `cube2_02_v295_f120_pre180_post180` |
| 训练引用 | 数据包内全部参考轨迹，固定round-robin |
| 环境数 | 2048 |
| rollout长度 | 32 |
| PPO epochs | 4 |
| minibatches | 16 |
| 学习率 | `3e-5` |
| Critic | 独立Critic |
| teacher anchor | beta=1，passes=2，仅训练监督 |
| 奖励 | 当前ManoRL Reward v10路线 |
| 动作分布 | Raw Gaussian；物理执行前裁剪 |
| 初始预算 | 每版本1000 updates，即65,536,000环境转换 |
| checkpoint间隔 | 50 updates |
| 训练输出 | NAS目录，不写满远端Home |

1000 updates是第一阶段预算，不自动等同于收敛。若最后200次更新的关键成功指标
仍持续改善，则从最后checkpoint继续；若指标长期为零或退化，则记录为未收敛并
先诊断机制，不用总Reward代替成功率。

## 3. 最佳checkpoint选择

候选checkpoint统一进行自然首次终止评估，不允许用
`full-horizon diagnostic continuation`冒充成功。排序优先级为：

1. `natural full-horizon success`；
2. `loaded airborne frames`；
3. `opposing contact`和`thumb contact`；
4. 更低的`contact loss`和`drop rate`；
5. 更低的物体路径误差。

Reward只作为训练诊断和同版本趋势指标，不是跨版本的最终成功判据。

## 4. 训练队列

| 顺序 | 版本 | 实现状态 | GPU | 正式训练状态 |
|---:|---|---|---|---|
| 1 | v5 | 已实现并接通CLI | GPU0 | 正式训练中 |
| 2 | v5.25 | 已实现；全参考有限性复验通过 | GPU1 | 待立即启动 |
| 3 | v5.5 | 已实现；聚焦测试通过 | 待分配 | 未启动 |
| 4 | v5.75 | 已实现；聚焦测试通过 | 待分配 | 未启动 |
| 5 | v6 | 已实现 | 待分配 | 未启动 |

## 5. 已完成的运行证据

### 5.1 v5 单元与依赖环境测试

远端环境：

```text
/home/cty/miniconda3/envs/contact-conditioned-autonomy
```

聚焦测试：

```text
tests/manorl/test_autonomy_v5_pointcloud.py
tests/manorl/test_autonomy_v6.py
```

结果：

```text
10 passed in 10.21s
```

### 5.2 v5真实GPU检查

```text
GPU: RTX 4090，CUDA_VISIBLE_DEVICES=0
identity: cube2_02_2833
raw observation: (1, 1342)
control timestep: 1/120 s
physics timestep: 1/480 s
physics substeps: 4
constraint capacity: 512/world
```

结果：资产、轨迹包、Warp接触运行时和v5观测均成功初始化。

### 5.3 v5 PPO冒烟

此运行只验证训练链路，不计入正式抓取结果：

```text
num_envs = 64
updates = 2
rollouts = 8
learning_epochs = 1
mini_batches = 4
transitions = 1024
learning_rate = 3e-5
teacher_anchor_beta = 1
teacher_anchor_passes = 1
```

结果：

```text
update 1: valid=1, exact_kl_mean=0.0060705
update 2: valid=1, exact_kl_mean=0.0042162
checkpoint写入成功
```

前16个控制步尚未进入接触阶段，因此：

```text
paired_loaded_contact_fraction = 0
airborne_5mm_fraction = 0
```

这不能解释为训练失败或抓取失败，只能说明完整的采样、PPO更新、teacher更新、
遥测和checkpoint链路已通过。

### 5.4 五版本实现与参数量检查

远端正式Conda环境对五个独立Critic模型逐项统计可训练参数：

```text
v5       285,753
v5.25    232,377
v5.5     288,569
v5.75    293,241
v6     1,780,025
```

聚焦测试覆盖v5.25/v5.5/v5.75观测宽度、区域绑定、模型前后向、CLI和checkpoint
识别，并连同v5/v6测试在远端环境得到：

```text
19 passed in 36.19s
```

### 5.5 v5正式训练当前证据

正式配置为2048环境、32步rollout、4个PPO epochs和16个minibatches。GPU0在PPO
阶段实测约20.7GB显存，瞬时计算利用率达到99%。截至当前记录点：

```text
update = 25
transitions = 1,638,400
reward_mean = 0.20269765
paired_loaded_contact_fraction = 0.00618172
airborne_5mm_fraction = 0
```

这表示策略已出现少量载荷接触，但还没有离桌5mm证据，不能判定抓取成功或收敛。

### 5.6 v5.25首次PPO冒烟

第一次64环境、2 updates冒烟在物理/contact有效性门控处终止：

```text
RuntimeError: invalid v4 physics/contact reduction; checkpoint withheld
```

GPU未发生OOM，且正式checkpoint未被写出。该失败保留为启动前诊断项，不得写成
架构训练成功。

同一实现随后在单参考轨迹、相同64环境/2 updates设置下通过：

```text
update 1: valid=1
update 2: valid=1
transitions=1024
checkpoint写入成功
```

因此模型前后向和单参考物理链路可运行；剩余问题收敛到“全参考初始随机策略下，
至少一个环境触发有效性门控”，正式全参考训练仍未启动。

逐层诊断发现，非有限值只位于第48–63环境的109维参考块中，具体为相对掌姿
旋转6D字段；手部区域Token、点云和接触输入本身均有限。修复方式是在相对四元数
相乘前使用防溢出归一化，正常单位四元数保持原数值路径。修复后的全参考64环境
复验结果为：

```text
obs_finite=True
mean_finite=True
action_finite=True
physical_valid=True
contact_valid=True
reward_valid=True
combined_valid=True
```

v5.25因此具备启动正式全参考训练的运行条件。

### 5.7 当前评估视频链路

当前frozen evaluation产物已补充实际/参考物体四元数和逐区域接触计数。新增NPZ
渲染器直接读取真实策略产生的物理轨迹，并生成“Policy actual / Reference”并排
MP4；不会使用参考手位姿冒充实际策略状态。

## 6. 正式结果表

| 版本 | 参数量 | updates | 转换数 | 墙钟时间 | 自然成功率 | 最佳抓取表现 | 收敛判断 | 视频 |
|---|---:|---:|---:|---:|---:|---|---|---|
| v5 | 285,753 | 25（训练中） | 1,638,400 | 训练中 | 未评估 | 少量载荷接触，尚未离桌 | 未收敛 | — |
| v5.25 | 232,377 | — | — | — | — | 全参考有限性复验通过 | 待启动 | — |
| v5.5 | 288,569 | — | — | — | — | 未训练 | 待定 | — |
| v5.75 | 293,241 | — | — | — | — | 未训练 | 待定 | — |
| v6 | 1,780,025 | — | — | — | — | 未训练 | 待定 | — |

后续只在评估文件、checkpoint和视频均实际存在时更新本表。
