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
| GPU | 两张独立RTX 4090；每任务仅一张；单任务约20GB即可，不再为追求90%占用率做额外显存档位测试 |
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
| 1 | v5 | 已实现并接通CLI | GPU0 | 正式训练中（update 354） |
| 2 | v5.25 | 已实现；全参考有限性复验通过 | GPU1 | 正式训练中（update 112） |
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
阶段实测约20.7GB显存，瞬时计算利用率达到100%。截至2026-09-19 18:41 UTC：

```text
update = 354
transitions = 23,199,744
reward_mean = 0.08780221
paired_loaded_contact_fraction = 0.01644611
loaded_airborne_fraction = 0.00048828
termination/deviation_rate = 1.0
termination/reference_complete_rate = 0.0
```

这表示训练窗口中已经出现少量载荷接触和离桌5mm事件，但同一窗口所有已完成
episode仍因偏差终止；窗口统计不能代替自然首次终止评估，因此当前仍不能判定
完整抓取成功或收敛。

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

v5.25因此具备启动正式全参考训练的运行条件。正式进程已于
2026-09-19 18:12 UTC绑定GPU1启动；首次全参考设备缓存完成后已进入PPO。截至
18:41 UTC的正式证据为：

```text
update = 112
transitions = 7,340,032
valid = 1
reward_mean = 0.13641311
paired_loaded_contact_fraction = 0.00882626
loaded_airborne_fraction = 0.00009155
termination/deviation_rate = 1.0
termination/reference_complete_rate = 0.0
```

GPU1训练期当前约占13.8GB。当前更新持续有限有效；训练窗口中已有极少量载荷
离桌帧，但所有已完成episode仍因偏差终止，因此不能填写自然成功率或收敛结论。

### 5.7 当前评估视频链路

当前frozen evaluation产物已补充实际/参考物体四元数和逐区域接触计数。新增NPZ
渲染器直接读取真实策略产生的物理轨迹，并生成“Policy actual / Reference”并排
MP4；不会使用参考手位姿冒充实际策略状态。

使用v5 `update 200` checkpoint进行了自然首次终止评估，作为早期诊断和录像链路
验证，不作为最佳结果：

```text
frames = 228
natural termination reason = 2（偏差）
loaded contact frames = 3
opposing loaded frames = 0
airborne frames = 0
loaded airborne frames = 0
peak bottom clearance = -0.00010781 m
```

该checkpoint没有完成抓取。对应并排视频已成功生成并读回：

```text
resolution = 1280×480
fps = 30
frames = 57
duration = 1.9 s
```

这只证明评估与录像产物链路有效，视频内容本身是一次失败轨迹。

### 5.8 顺序训练监督器

为避免长训练结束后GPU空转，已在NAS部署只负责顺序启动的监督器：

```text
tmux: manorl-v5-v6-queue
script: /mnt/nas-222-projects/cty/manorl-autonomy/ops/
        v5-v6-ablation-20260919/supervise_manorl_v5_v6_ablation.py
status: /mnt/nas-222-projects/cty/manorl-autonomy/outputs/
        v5-v6-ablation-20260919/queue.status.json
events: /mnt/nas-222-projects/cty/manorl-autonomy/outputs/
        v5-v6-ablation-20260919/queue.events.jsonl
```

它执行显式依赖状态机：

```text
v5 exit.status=0     → GPU0空闲 → v5.5
v5.25 exit.status=0  → GPU1空闲 → v5.75
v5.5和v5.75均成功   → GPU0或GPU1空闲 → v6
```

监督器不终止任何进程；目标GPU必须没有compute PID且显存低于1GB才允许启动。
前置任务非零退出或tmux消失但没有`exit.status`时，队列快速失败并停止下游启动，
保留现场供诊断。当前状态已读回验证为v5/v5.25运行中，其余三版等待。

### 5.9 checkpoint物理摘要与严格成功判据

新增`tools/summarize_manorl_autonomy_evaluation.py`，只读取自然首次终止的
`eval.json`和真实物理`eval.npz`。诊断续跑产物会被拒绝。固定判据为：

```text
loaded contact: 任一区域配对力 > 0.02 N
airborne: 物体最低碰撞顶点高于桌面 > 5 mm
thumb contact: thumb_cmc/thumb_mcp/thumb_ip任一区域loaded
opposing contact: thumb loaded且任一index/middle/ring/pinky区域loaded
strict grasp success:
  自然终止reason=1
  且连续loaded-airborne至少0.25 s（120 Hz下为30帧）
```

`reason=1`单独只表示参考时域走完，不会被摘要器冒充抓取成功。checkpoint排序首先
比较严格成功和自然完整时域，其次比较载荷空中帧、持续长度、对向/拇指接触、
接触丢失、掉落和物体路径误差；Reward只保留为旁证。

对v5 update 200已有失败轨迹做回归，摘要器得到：

```text
strict_grasp_success = false
terminal_reason_code = 2
loaded_contact_frames = 3
thumb_loaded_frames = 0
opposing_loaded_frames = 0
loaded_airborne_frames = 0
contact_loss_fraction_after_first_loaded = 0.92857143
final_object_path_error_m = 0.10134772
```

这与先前人工检查一致，并补充证明它接触后很快丢失、末端路径偏差超过10cm。
摘要已保存为该评估目录下的`summary.json`。

### 5.10 自动评估与最终视频队列

CPU评估监督器已部署并在独立tmux中运行：

```text
tmux: manorl-v5-v6-eval
script: /mnt/nas-222-projects/cty/manorl-autonomy/ops/
        v5-v6-ablation-20260919/supervise_manorl_v5_v6_evaluation.py
status: /mnt/nas-222-projects/cty/manorl-autonomy/outputs/
        v5-v6-ablation-20260919/evaluation-queue.status.json
```

每个版本只有在正式训练`exit.status=0`后才进入评估，执行顺序为：

1. 验证从update 50到1000的20个周期checkpoint完整存在；
2. 对20个checkpoint逐个执行CPU自然首次终止评估；
3. 使用5.9节物理排序选最佳checkpoint，不按Reward选取；
4. 从checkpoint provenance读取固定5条验证和5条测试身份；
5. 对最佳checkpoint计算验证/测试严格抓取成功率；
6. 从标准训练身份与10条held-out轨迹中选择物理表现最好的一条；
7. 渲染`Policy actual / Reference`并排MP4，并通过`ffprobe`或完整
   `imageio`解码读回验证分辨率、帧数、帧率、时长和文件大小。

评估使用CPU，不占用两张训练GPU。监督器目前处于等待状态；v5和v5.25均未结束，
因此尚未启动任何checkpoint扫描，也没有提前生成“最佳”视频。

### 5.11 正式运行来源与墙钟起点

已从tmux session创建时间、活跃`/proc/<pid>/cmdline`和可加载checkpoint
provenance固化两个首发任务的运行元数据：

| 版本 | 启动时间（UTC） | 来源提交 | 已验证checkpoint | SHA256前缀 |
|---|---|---|---|---|
| v5 | `2026-09-19T17:10:00Z` | `f906030aaf86` | update 300 | `e5bc10865207` |
| v5.25 | `2026-09-19T18:12:14Z` | `c6fb1da11e5f` | update 50 | `419ebafef8b8` |

两个checkpoint均通过CPU反序列化，记录50条参考、独立Critic、正确版本ABI和有限
模型参数；训练日志中未发现Traceback、CUDA OOM或RuntimeError。完整元数据位于：

```text
v5-formal/run-metadata.json
v525-formal/run-metadata.json
```

最终“收敛时间”将使用上述启动时间到被选最佳checkpoint文件落盘时间的差值，
同时单独记录1000 updates总训练墙钟时间，避免用估算时间代替实测。

## 6. 正式结果表

| 版本 | 参数量 | updates | 转换数 | 墙钟时间 | 自然成功率 | 最佳抓取表现 | 收敛判断 | 视频 |
|---|---:|---:|---:|---:|---:|---|---|---|
| v5 | 285,753 | 354（训练中） | 23,199,744 | 约1小时32分（训练中） | 未评估 | 训练窗口出现少量载荷离桌事件，但完成episode仍100%偏差终止 | 未收敛 | update 200失败诊断视频 |
| v5.25 | 232,377 | 112（训练中） | 7,340,032 | 约29分（含初始化，训练中） | 未评估 | 少量载荷离桌事件，但完成episode仍100%偏差终止 | 尚无收敛证据 | — |
| v5.5 | 288,569 | — | — | — | — | 未训练 | 待定 | — |
| v5.75 | 293,241 | — | — | — | — | 未训练 | 待定 | — |
| v6 | 1,780,025 | — | — | — | — | 未训练 | 待定 | — |

后续只在评估文件、checkpoint和视频均实际存在时更新本表。
