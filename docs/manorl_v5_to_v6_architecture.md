# 两个 v5 架构与 ManoRL v6 架构记录

状态：2026-09-19 五个阶段版本均已完成代码实现；正式训练与自然终止评估仍以
训练记录文档为准。本文中的
“两个 v5”分别指：

1. **ManoRL v5-pointcloud**：`mujoco-warp-contactbench` 的
   `case/contact-conditioned-autonomy/v5-pointcloud`；
2. **VoxMani v0.5**：`voxMani` 的
   `case/auton-policy/v05-force-obs`，下文简称 **VoxMani v5**。

在此前开发中暂称为 `v5.1-pointcloud` 的混合架构，从本版本起正式命名为：

> **ManoRL v6 Region-Token Cross-Attention Policy**

对应版本标识为：

```text
CLI policy version:   v6
Observation contract: manorl.autonomy.observation.v6.region-token-cross-attention
Checkpoint format:    manorl.autonomy.ppo.v6
Model architecture:   manorl.autonomy.actor_critic.v6.region-token-cross-attention
```

实现状态边界：

- **已实现并完成聚焦依赖环境测试**：ManoRL v5、v5.25、v5.5、v5.75、v6；
- 下文参数量来自当前代码的精确可训练参数统计；
- 实现通过不等同于训练收敛或抓取成功，后者只记录在正式实验结果中；
- 本文中的 **ManoRL v5.5** 是从ManoRL v5走向v6的过渡版本，不是
  `VoxMani v0.5` 或本文简称的“VoxMani v5”。

## 1. 三个版本的定位

| 项目 | ManoRL v5-pointcloud | VoxMani v5 | ManoRL v6 |
|---|---|---|---|
| 主要目标 | 用点云替换手工接触几何 | 完整时空Token策略 | 在现有ManoRL训练管线中引入区域Token和目标注意力 |
| 原始观测宽度 | 1342 | 796 | 2205 |
| 参数量 | 285,753 | 4,517,173 | 1,780,025 |
| 物体表达 | 64点全局PointNet | 16细粒度Patch + 4粗粒度Patch | 16个物体Patch Token |
| 手部表达 | 256点全局PointNet | 16个骨骼皮肤Token + Rodrigues骨架流 | 16个手部区域Token |
| 目标表达 | 数值参考；参考手云未进入网络 | Actor 64个、Critic 128个Goal Token | 16个同帧参考手Goal Token |
| 当前Token数量 | 无显式Token集合 | 70 | 34 |
| Attention | 无 | 4层Self + Cross Attention | Actor/Critic各2层Self + Cross Attention |
| Actor/Critic | 可共享或分离MLP | 完全独立双塔 | 共享观测Token编码器，融合塔独立 |
| 动作头 | 统一28维线性头 | 腕部6维 + 逐关节22维 | 统一28维线性头 |
| 策略分布 | Raw Gaussian + Clip | Tanh-squashed Gaussian | Raw Gaussian + Clip |
| PPO | ManoRL/skrl PPO | VoxMani自定义PPO | 保留ManoRL/skrl PPO |
| 奖励 | ManoRL Reward v10路线 | VoxMani奖励/MDP | 保留ManoRL Reward v10路线 |

## 2. ManoRL v5-pointcloud

### 2.1 观测

ManoRL v5-pointcloud 的1342维观测为：

| 观测块 | 维度 |
|---|---:|
| 当前状态 | 119 |
| 当前参考状态 | 109 |
| 未来数值参考 | 12 |
| 接触意图与逐区域力 | 80 |
| 物体点云 `64×3` | 192 |
| 当前手点云 `256×3` | 768 |
| 动作类别 | 50 |
| 物体几何 | 12 |
| 合计 | **1342** |

其中接触块为：

```text
reference confidence  16
reference valid       16
paired force xyz      48
```

### 2.2 网络

```mermaid
flowchart LR
    O["物体点云 64×3"] --> OP["Object PointNet"] --> OE["64维全局特征"]
    H["当前手点云 256×3"] --> HP["Hand PointNet"] --> HE["64维全局特征"]
    N["数值状态 382维"] --> C["拼接为510维"]
    OE --> C
    HE --> C
    C --> A["Actor MLP 510→128→128→28"]
    C --> V["Critic MLP 510→128→128→1"]
```

### 2.3 局限

- 256个手点被一次全局池化，16个手部区域身份丢失；
- 接触力无法与具体手指/骨骼Token绑定；
- `hand_cloud_reference` 已经由缓存编译，但没有进入观测和网络；
- 当前手型与参考手型之间没有Cross-Attention；
- 接触力使用固定 `tanh(force / 5N)`，大力区域容易饱和；
- slip通道被删除；
- 原始提交只增加了观测构造器和模型类，正式训练管线当时仍主要走v4。

## 3. VoxMani v5

### 3.1 观测与Token

VoxMani v5 使用796维紧凑ABI，在Torch侧展开为三个主要Head：

1. **Head 1：场景Token**
   - 当前物体16个细粒度Patch；
   - 桌面16个Patch；
   - 过去5帧，每帧4个物体粗粒度Patch；
   - 合计52个Scene Token。
2. **Head 2：手部Token**
   - 16个骨骼区域；
   - 每个骨骼16个表面点；
   - 皮肤点云Token与Neural Rodrigues骨架流融合。
3. **Head 3：目标Token**
   - Actor：16个目标时刻 × 4个粗粒度Patch = 64个Token；
   - Critic：32个目标时刻 × 4个粗粒度Patch = 128个Token。

当前集合为：

```text
52 Scene + 16 Hand + 1 State + 1 Readout = 70 Tokens
```

### 3.2 网络

```mermaid
flowchart LR
    S["52 Scene Tokens"] --> CUR["70 Current Tokens"]
    H["16 Hand Tokens\n皮肤点云 + Rodrigues骨架"] --> CUR
    ST["State Token"] --> CUR
    R["Readout Token"] --> CUR

    GA["Actor Goal Memory\n64 Tokens"] --> AT["Actor 4层\nSelf + Cross Attention"]
    GC["Critic Goal Memory\n128 Tokens"] --> CT["Critic 4层\nSelf + Cross Attention"]
    CUR --> AT
    CUR --> CT

    AT --> W["腕部6维头"]
    AT --> F["逐关节22维头"]
    CT --> V["Value"]
```

主要网络参数为：

```text
model_dim = 128
attention_heads = 4
FFN_dim = 512
trunk_layers = 4
actor/critic = 完全独立双塔
```

### 3.3 力观测

VoxMani v5增加60维物体侧受力分解：

```text
gravity                  3
per-bone hand force     48
hand torque              3
other force              3
other torque             3
```

力使用 `F / (m|g|)`，力矩使用 `τ / (m|g|r)` 归一化，不使用固定5N裁剪。

## 4. ManoRL v6

### 4.1 设计边界

v6不是把VoxMani整套环境和PPO直接复制过来，而是在保持当前ManoRL
MDP、奖励、控制和训练基础设施不变的情况下，迁移其更适合接触抓取的
观测组织与Token融合思想。

保持不变：

- MuJoCo/MJX-Warp环境和终止语义；
- Reward v10混合奖励；
- 当前PPO/GAE和teacher anchor；
- 28维动作空间；
- rate-limited measured-state命令；
- anti-windup；
- Raw Gaussian概率模型和物理动作裁剪；
- 自然首次终止评估边界。

### 4.2 2205维观测

| 观测块 | 维度 | 来源 |
|---|---:|---|
| 当前状态 | 119 | ManoRL v5 |
| 当前参考状态 | 109 | ManoRL v5 |
| 未来数值参考 | 12 | ManoRL v5 |
| 16区域接触观测 | 160 | VoxMani区域力思想 + ManoRL contact reducer |
| 物体Wrench | 15 | VoxMani物体侧受力分解 |
| 物体点云 | 192 | ManoRL v5 |
| 当前手点云 | 768 | ManoRL v5 |
| 参考手点云 | 768 | 激活ManoRL v5缓存中未使用的数据 |
| 动作类别 | 50 | ManoRL v5 |
| 物体几何 | 12 | ManoRL v5 |
| 合计 | **2205** | |

每个手部区域有10维绑定特征：

```text
reference confidence      1
reference valid           1
measured contact active   1
log1p(contact count)      1
paired force xyz          3
tangential slip xyz       3
```

物体Wrench为15维：

```text
gravity                  3
summed hand force        3
summed hand torque       3
non-hand force           3
non-hand torque          3
```

力在当前物体坐标系表达，并按 `m|g|` 归一化；力矩按 `m|g|r` 归一化。
动态力和力矩使用 `asinh` 压缩，以保留大接触冲量的相对差异。

### 4.3 Token编码

```text
物体：64点 → 16组×4点 → Point Stem → 16 Object Tokens
当前手：16区域×16点 → Hand Stem → 16 Current Hand Tokens
参考手：16区域×16点 → 同一Hand Stem → 16 Goal Hand Tokens
数值状态：317维 → 256 → 128 → 1 State Token
另有1个可学习Readout Token
```

当前Token集合：

```text
16 Object + 16 Current Hand + 1 State + 1 Readout = 34 Tokens
```

目标Memory：

```text
16 Reference Hand Tokens
```

区域接触特征经过 `10→128` 投影后，只加入同编号的当前手Token。例如第5区域
的接触力只改变第5个手部Token，不再在全局池化中丢失身份。

### 4.4 Actor/Critic

```mermaid
flowchart LR
    O["16 Object Tokens"] --> C["34 Current Tokens"]
    H["16 Current Hand Tokens\n+ 对应区域接触"] --> C
    S["1 State Token"] --> C
    R["1 Readout Token"] --> C
    G["16 Reference Hand Goal Tokens"] --> A
    G --> V
    C --> A["Actor独立融合塔\n2× Self-Attention + Cross-Attention"]
    C --> V["Critic独立融合塔\n2× Self-Attention + Cross-Attention"]
    A --> M["28维动作Mean"]
    V --> Q["1维Value"]
```

网络参数：

```text
raw observation = 2205
model_dim = 128
attention_heads = 4
FFN_dim = 512
fusion_layers = 2
current_tokens = 34
goal_tokens = 16
trainable parameters = 1,780,025
```

Actor和Critic共享观测Token编码器，包括Point Stem、区域/patch embedding、
接触投影和State MLP；进入Attention融合阶段后使用相互独立的两座塔。

## 5. v6分别借鉴了什么

### 5.1 从ManoRL v5-pointcloud继承

- 当前119维物理状态和109维参考状态；
- 6/12/24帧未来物体数值参考；
- 64点物体点云；
- 16区域×16点当前手点云；
- 动作类别和物体几何编码；
- 已编译的参考手点云缓存；
- MuJoCo/MJX-Warp环境、Reward v10、PPO和动作控制管线；
- checkpoint严格ABI和provenance检查。

### 5.2 从VoxMani v5借鉴

- 将64个物体点组织为16个局部Patch Token；
- 将256个手点保留为16个骨骼/区域Token；
- 为区域和Patch加入身份embedding；
- 将接触力绑定到对应手部区域，而不是全局拼接；
- 使用目标Memory和Cross-Attention表达“当前状态应向哪里变化”；
- Actor/Critic使用独立的融合塔；
- 使用物体侧手力、其他力以及关于COM的力矩分解；
- 使用 `m|g|` 和 `m|g|r` 进行物理尺度归一化；
- 采用128维、4头、512维FFN这一经过验证的Attention尺度。

### 5.3 v6对两个v5的再设计

- ManoRL v5的全局手PointNet改为16个区域Token；
- 原先未使用的参考手云正式成为Cross-Attention目标；
- ManoRL v5删除的slip重新作为逐区域观测加入；
- 固定 `tanh(force/5N)` 改为质量/重力/半径归一化与 `asinh`；
- 不复制VoxMani庞大的时空场景和多时刻Goal集合，只使用当前任务最直接的
  同帧参考手目标；
- Attention层数从4层减到2层，将参数量控制在VoxMani v5的约39%；
- 第一阶段保留ManoRL统一28维动作头，避免同时改变观测、注意力和动作头；
- 第一阶段保留Raw Gaussian + Clip，Tanh-squashed Gaussian作为后续独立消融。

## 6. ManoRL v5到v6的分阶段迭代路线

直接从28.6万参数、双全局PointNet的v5跳到178万参数、双Attention融合塔的v6，
会同时改变手部表达、目标表达、接触表达、物体表达和Actor/Critic融合方式。即使
最终性能发生变化，也很难判断是哪一项改动造成的。因此建议将完整路线拆为：

```text
ManoRL v5 → v5.25 → v5.5 → v5.75 → v6
```

其中v5和v6是已实现端点；v5.25、v5.5和v5.75是用于消融和归因的计划版本。

### 6.1 版本总表

| 版本 | 状态 | 原始观测 | 核心改动 | Attention | 参数量 |
|---|---|---:|---|---|---:|
| v5 | 已实现 | 1342 | 双全局PointNet | 无 | 285,753 |
| v5.25 | 已实现 | 1342 | 当前手点云改为16个区域Token | 无 | 232,377 |
| v5.5 | 已实现 | 1486 | 参考手区域位姿 + 1层Cross-Attention | 1层手部Cross | 288,569 |
| v5.75 | 已实现 | 1581 | 区域动态接触、slip和物体Wrench | 1层手部Cross | 293,241 |
| v6 | 已实现 | 2205 | 物体Patch、完整参考手云、独立双融合塔 | Actor/Critic各2层 | 1,780,025 |

参数量说明：五个数字均使用独立Critic配置，对当前代码中所有
`requires_grad=True` 参数逐项求和得到。

### 6.2 v5：双全局PointNet基线（已实现）

v5保持最紧凑的点云策略：

```text
1342维原始观测
├── 64个物体点 → Object PointNet → 64维全局特征
├── 256个当前手点 → Hand PointNet → 64维全局特征
└── 382维数值观测

64 + 64 + 382 = 510维融合特征
→ Actor MLP / Critic MLP
```

优点是结构简单、参数少、容易训练和定位PPO问题；主要不足是256个手点被一次
全局池化，拇指、指尖和掌部等16个区域的身份无法被显式保留。

### 6.3 v5.25：区域化当前手编码（已实现）

v5.25只改变当前手点云编码器，不改变观测ABI：

```text
当前手点云 256×3
→ 16个区域 × 每区域16点
→ 共享区域PointNet
→ 16个64维Current Hand Token
→ 加入区域身份Embedding
→ 区域池化
→ 64维手部特征
→ 继续进入原510维融合和Actor/Critic MLP
```

保持不变：

- 原始观测仍为1342维；
- 物体仍使用全局Object PointNet；
- 不引入参考手Token和Cross-Attention；
- 不修改80维接触块、PPO、奖励、teacher、控制器和28维动作头。

该版本只回答一个问题：**在不增加目标条件和Attention的情况下，保留16个手部
区域身份，是否能提高拇指接触率和对向接触率？**

### 6.4 v5.5：参考手区域目标与单层Cross-Attention（已实现）

建议正式名称：

> **ManoRL v5.5 Region-Aware Hand Goal**

在v5.25基础上，为每个参考手区域加入9维位姿：

```text
position xyz = 3
rotation 6D  = 6
每区域       = 9
16个区域     = 16 × 9 = 144维
```

因此原始观测宽度变为：

```text
1342 + 144 = 1486维
```

计划网络：

```mermaid
flowchart LR
    HC["当前手点云\n16区域×16点"] --> HE["共享区域PointNet\n16×64 Current Tokens"]
    HR["参考手区域位姿\n16×9"] --> PE["Pose MLP\n16×64 Goal Tokens"]
    HE --> CA["1层Cross-Attention\nQ=Current, K/V=Goal"]
    PE --> CA
    CA --> HP["区域池化\n64维手特征"]
    O["物体点云"] --> OP["原Object PointNet\n64维"]
    N["原数值特征\n382维"] --> F["510维融合"]
    HP --> F
    OP --> F
    F --> A["原Actor MLP\n→ 28维Mean"]
    F --> V["原Critic MLP\n→ Value"]
```

v5.5只增加一层手部目标Cross-Attention，仍保留：

- 原物体全局PointNet；
- 原Actor/Critic MLP；
- 原28维统一动作头；
- Raw Gaussian + Clip策略分布；
- 当前ManoRL PPO、Reward v10、teacher和控制基础设施。

该版本用于回答：**当前手区域显式查询参考手区域目标，是否能降低手型误差和
接触建立时间？**

### 6.5 v5.75：区域接触与Wrench增强（已实现）

v5.75在v5.5上增加动态接触表达，但暂不升级为完整v6融合塔。

原v5的80维接触块继续保留：

```text
reference confidence  16
reference valid       16
paired force xyz      48
合计                  80
```

新增80维逐区域动态接触：

```text
measured active       16
log1p(contact count)  16
tangential slip xyz   48
合计                  80
```

再新增15维物体Wrench：

```text
gravity                  3
summed hand force        3
summed hand torque       3
non-hand force           3
non-hand torque          3
合计                    15
```

因此原始观测宽度为：

```text
1486 + 80 + 15 = 1581维
```

每个当前手区域Token绑定同编号区域的：

```text
confidence
valid
active
log1p(contact count)
paired force xyz
tangential slip xyz
```

力按 `F/(m|g|)`、力矩按 `τ/(m|g|r)` 归一化，动态力和力矩使用 `asinh`
压缩，避免固定5N裁剪过早饱和。

该阶段仍不加入：

- 物体Patch Token；
- 完整参考手点云；
- 双层Attention；
- Actor/Critic独立Attention融合塔。

该版本用于回答：**区域力、接触状态、slip和物体Wrench是否能帮助策略识别
载荷转移与接触丢失，从而降低抓取后的掉落率？**

### 6.6 v6：完整区域Token Cross-Attention（已实现）

v6在v5.75的接触表达基础上完成两项主要升级。

第一项是将参考手的144维区域位姿升级为完整参考手点云：

```text
16区域 × 16点 × xyz = 768维
768 - 144 = 新增624维
1581 + 624 = 2205维
```

第二项是把特征融合从“全局PointNet + 单层手部Cross-Attention + MLP”
升级为：

```text
16 Object Patch Tokens
+ 16 Current Hand Tokens
+ 1 State Token
+ 1 Readout Token
= 34 Current Tokens

16 Reference Hand Cloud Tokens
= Goal Memory

Actor：2层Self-Attention + Cross-Attention
Critic：2层Self-Attention + Cross-Attention
```

与v5.75相比，v6同时引入物体局部Patch身份、完整参考手表面几何，以及
Actor/Critic相互独立的目标条件融合塔。它的表达能力更强，但归因和训练成本
也更高，因此应在中间版本消融后判断完整升级是否必要。

### 6.7 完整路线图

```mermaid
flowchart LR
    V5["v5 已实现\n双全局PointNet\n1342维"]
    V525["v5.25 已实现\n16区域手Token\n1342维"]
    V55["v5.5 已实现\n参考手区域Token\n1层Cross-Attention\n1486维"]
    V575["v5.75 已实现\n区域接触+Slip+Wrench\n1581维"]
    V6["v6 已实现\n物体Patch+完整Goal云\n独立双融合塔\n2205维"]

    V5 -->|"验证区域身份"| V525
    V525 -->|"验证目标手条件"| V55
    V55 -->|"验证接触表达"| V575
    V575 -->|"验证完整Token融合"| V6
```

### 6.8 分阶段实验与判定标准

每次升级只改变该阶段声明的模型/观测因素，并固定：

- 相同环境、物理资产、控制器和终止条件；
- 相同Reward v10、PPO、teacher设置和动作分布；
- 相同训练/验证划分、随机种子和环境转换总数；
- 相同完整起点、自然首次终止的评估协议。

各阶段的核心假设：

| 对比 | 只验证的核心问题 | 重点指标 |
|---|---|---|
| v5 → v5.25 | 区域身份是否有效 | 拇指接触率、对向接触率 |
| v5.25 → v5.5 | 参考手目标条件是否有效 | 手型误差、首次接触时间 |
| v5.5 → v5.75 | 动态接触与Wrench是否有效 | 接触丢失率、下落速度、掉落率 |
| v5.75 → v6 | 完整Token融合是否值得额外复杂度 | 完整成功率、复杂物体/未见轨迹泛化 |

所有版本至少统一报告：

- natural full-horizon success；
- peak lift与loaded airborne frames；
- opposing contact与thumb contact；
- contact loss与drop rate；
- 推理延迟、训练吞吐和精确参数量。

总Reward和启用`full-horizon diagnostic continuation`后的轨迹不能代替自然首次
终止成功率；近接触初始化或teacher追踪结果也不能表述为完整自主抓取成功。

## 7. 明确没有从VoxMani v5迁移的部分

- 桌面16个Patch Token；
- 过去5帧物体粗粒度Token；
- Actor 64 / Critic 128个多时刻Goal Token；
- Neural Rodrigues骨架流；
- Critic专用13维privileged observation；
- 腕部6维和逐关节22维分离动作头；
- Tanh-squashed Gaussian；
- VoxMani自定义双优化器PPO；
- BF16、goal reuse、CUDA graph等训练优化。

这些内容如果后续需要引入，应分别做消融实验，不能与v6首轮观测改动同时加入，
否则无法判断性能变化来自哪一部分。

## 8. 代码位置与使用

核心代码：

```text
sim/manorl/autonomy_v6_model.py
sim/manorl/autonomy_v5_intermediate_model.py
sim/manorl/autonomy_v4.py::build_raw_observation_v6
sim/manorl/autonomy_contracts.py
sim/manorl/autonomy_batch.py
sim/manorl/autonomy_training.py
tools/train_manorl_autonomy.py
tests/manorl/test_autonomy_v6.py
tests/manorl/test_autonomy_v525.py
tests/manorl/test_autonomy_v55_v575.py
```

显式选择v6：

```bash
python tools/train_manorl_autonomy.py inspect \
  --policy-version v6 \
  --device cpu \
  --num-envs 1 \
  --package /path/to/package \
  --identity cube2_02_2833
```

v4仍是兼容默认值。v4、ManoRL v5和v6具有不同的观测与checkpoint ABI，
不得互相强制加载权重。
