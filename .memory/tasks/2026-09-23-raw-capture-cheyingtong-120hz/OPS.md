## 2026-09-23T21:15:00+08:00 — pre60/j2.5x/early120 训练启动

Branch feat/raw-capture-transfer-120hz @ bccf83c (smoke clone Server1 同步）。

配置确认：
- cube1 全动作（remake 01/02/03/04/09/10/18 + guangguan 01/02/03/04/09/10，合计 2392 条）→ GPU0
- cube2 全动作（remake 01/02/03/04/09/10/11/18 + guangguan 01/02/03/04/10/11/18，排除单轨迹的 guangguan cube2:11，合计 1562 条）→ GPU1
- pre_padding=60（运动起点前 0.5s）、post_padding=0、early_phase_steps=120、joint 2.5x/2.5x、raw_gesture、residual_joint_mode=all、action_penalty_scale=1.0、512 envs、5000 updates。

启动前在分支上做的三个修复（均已提交）：
- 525ee4f 放宽 TrainingBudget raw_transfer pre_padding 门 + 新增 early_phase_steps 旋钮（默认30→可配）
- 2e33b9d 放宽 TrajectorySelection raw_transfer pre_padding 门
- bccf83c 包选择校验豁免 pre_padding（运行时裁剪参数，不是包内容参数）

W&B:
- cube1: run qceeopst https://wandb.ai/sunjay45711-dexerto/mujoco-mano/runs/qceeopst
- cube2: run cfxlv5a6 https://wandb.ai/sunjay45711-dexerto/mujoco-mano/runs/cfxlv5a6
- group: raw120-pre60

状态（21:05 观察）: cube1 update≥17、cube2 update≥18，GPU 22221MiB 各占、85%/40% 利用率，无错误。
日志: Server1 /tmp/smoke120hz/train-cube{1,2}.log；输出 /tmp/smoke120hz/train-cube{1,2}-pre60-j25-early120/。

## 2026-09-23T23:55:00+08:00 — 4096 修复 + 2.16x 加速 + Server2 启动

4096 OOM 根因（用户质疑后复查）：之前"4096 跑不了"的结论是 probe 配置错误——probe 误用 constraint_capacity=24576（真实默认 512）且没传 CCD 旋钮。真实机制：raw-transfer 默认 warp_ccd_contacts_per_world=None → 每世界全尺寸 CCD scratch 随 envs 线性爆；加 --warp-ccd-contacts-per-world 8 后 4096 物理步进吞吐 23857 world-steps/s（512 的 3.1 倍）。训练侧还需 XLA 池与 Torch 分显存：XLA_PYTHON_CLIENT_MEM_FRACTION=0.70 + PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True。修复 9b732ae：评估环境补齐 warp_ccd 传参（否则初始 checkpoint 签名不匹配）。

迭代加速 A/B（GPU2，4096 envs，15 updates）：基线 10.2-10.4k transitions/s vs --device-resident-controls true --device-transition true --capture-transition-diagnostics false 22.4k/s（2.16x）；reward/contact/episode 分布一致。已应用到生产：cube1/cube2 以 23k/s 跑（wandb 名 …-n4096-devtr-u5000），5000 updates 预计 ~12h。代价：无 transition 快照诊断；评估走独立 runtime。

Server2（ubuntu@192.168.10.22，2×4090）：仓库/资产 pin 778614d/cheyingtong manifest/两个 Lance（v978/v530）全部验证可达。scissor 无 DexStream 资产（365 条轨迹排除）。编译：remake-others 7072 条 83 对已出包；guangguan-others 4354 条 91 对编译中（compound bottle,cap/cap,bottle 55 行按文档以 bottle:18 override 归并）。Server1↔Server2 ssh 不通，资产/代码经本机中继 tar|ssh 传输。裸仓库：server2 ~/dexrobot/FromSSH/_raw120.git。


## 2026-09-24T00:45:00+08:00 — Server2 两路启动完成

Server2 混合对象训练（unified-object-batch，pre60/j2.5x/early120/raw_gesture）：
- remake-others（7072 条、83 对）GPU0 @3072 envs，~7.5k transitions/s，update 64+，W&B run y4ljfe1m
- guangguan-others（4312 条、91 对，bottle:18 override）GPU1 @2048 envs，~6.3k/s，update 10+，W&B run ue4hmrwn

容量教训（Server2 unified 混合对象比单对象重得多）：
- persistent CCD workspace 是进程级单例；训练 env(4096) 与评估 env(<=128) 形状不同会冲突 -> unified 训练不能开 --warp-persistent-ccd-workspace。
- unified 4096 训练态 Warp 图外部分配 OOM；3072 对 remake-others 可以，guangguan-others（含 bottle+cap 复合场景）仍 OOM -> 2048 稳定。
- 杀进程后 GPU 显存释放有延迟，立即重启会让新进程 CUDA init 报 no supported devices——等显存归零再启。

四路现役训练（全部 pre60/j2.5x/early120/120Hz raw-transfer）：
- server1 GPU0 cube1 4096 devtr 23k/s；server1 GPU1 cube2 4096 devtr 22.8k/s
- server2 GPU0 remake-others 3072 7.5k/s；server2 GPU1 guangguan-others 2048 6.3k/s


## 2026-09-24T02:00:00+08:00 — 值守体系 + RL 分析 v0

监控/自愈（ scheduled subagent 被平台 scheduledRuns.enabled=false 挡住；settings 已改但需 Pi 重载生效）：
- NAS 共享监控：两台服务器各跑 tmux raw120-monitor，每 15min 写 /mnt/nas-222-projects/mocap_v2/manorl_raw120_monitor/status.log（本机同挂载可读）。
- 崩溃自愈 watchdog（/tmp/manorl_watchdog.sh）：空闲等待，进程死亡且有 last.pt 时自动 --resume-checkpoint 续训（新输出后缀 -rN、wandb 名加 -rN，最多 3 次）；检测到最终 metrics json（正常完成）则停止监视。cube1/cube2/remake 已挂 watchdog；guangguan 用阻塞式 supervisor（/tmp/manorl_supervise.sh）管理。

RL 分析 v0（episode jsonl 逐集记录，长度=survival/0.001）：
- "121 悬崖"证伪：early120 后集中死亡只出现在 update 3（回放时机）；真实 episode 中位长度全程稳定 ~220-230 步。
- cube1/cube2 学习健康：成功率 0%→13-17%（u750）、回报中位 0.14→42、episode 均分持续上升（cube1 90.7@u762）。p90 长度 ~500 超过中位 → 一部分 episode 学会接触后保持。
- 隐忧 1：u300 后成功率进入平台（10-13%）——当前 raw_gesture 力阈值接触奖励可能封顶了"接近但不稳"的 grasp 修正信号；正是 residual-log-geometry 分支要解决的问题。
- 隐忧 2：server2 混合对象成功率低（remake u238 succ 2.6%；guangguan 重启后 u36 succ 0.6%）。
- guangguan 两次崩溃：(1) njmax=512 约束容量饱和（混合对象约束多）→ 修：--constraint-capacity 2048；(2) 罕见物理 NaN（3.5M transitions 一次）。判别实验：纯参考 700 步 + 全幅随机动作 4×400 步均干净 → NaN 是策略-物理交互的稀有尾部事件，不是参考或场景本身坏。对策就是 watchdog 续训。

当前状态：cube1 u762+/cube2 u749+（23k/s），remake-others u238+（8.3k/s），guangguan-others 重启中（4.6k/s）。


## 2026-09-24T02:30:00+08:00 — u1000 里程碑分析（episode jsonl 全量）

状态：cube1 u1121（succ 17.8% epMean 112.6）、cube2 u1104（succ 27.4% epMean 119.7）；remake-others u340+；guangguan-others 第二次崩溃（非有限接触力，~3.5M transitions 处，与第一次同位置）后带诊断+checkpoint间隔50 重启（u7）。分支新增 2ae588b（环境崩溃前打印非有限 env/identity/object，fail-closed 语义不变）。

cube 系列窗口对比（succ / 关键项均值）：
- u<=100: succ 3-4%；z 位置 15、rotation 68、contact 6.6-7.2、stability 2、action_penalty -10
- u300-400: succ 11-14%；z 28-33、contact 8.6-14、stability 6-7.5
- 近 200 update: succ 14.7%(cube1)/22.6%(cube2)；z 37-48、contact 13.7-20.5、stability 9-12、penalty -17~-20
- 学习仍在继续，无平台期。rotation 大但平（跟踪保底分）；增长来自 z 位置、contact、stability。

按动作（近期窗口，cube1 / cube2 成功率）：
- 09 Large-Diameter 32.9% / 22.4%；10 Extension-Type 21.1% / 53.2%；01-04 pinch/prismatic 12-24%
- **18 Lateral 0.6% / 7.5% 系统性失败**：len_med 195、p25 172（早于其他动作的 222-254 但远晚于 121 早死线），ret_med 35.4 中位不低——能拿到跟踪/接触分但几乎完不成。判断：不是参考物理崩，是横向捏抓在自由物体+120Hz 下本就难稳定；候选改进：几何接触奖励（我们 residual 分支的逐帧锚点 intent 恰好在接触位置精度上加信号）、或 action-18 专项增广。
- 动作 10 高成功低回报（完成多但每集分低）——episode 结构短，正常。

监控：NAS status.log 每 15min 双服务器快照正常累积。


## 2026-09-24T04:10:00+08:00 — 巡检 + remake-others 对象分解

fleet: cube1 u1736 succ 19.4% epMean 119.9；cube2 u1709 succ 30.4% epMean 131.6（仍在涨）；remake-others u683 succ 2.9%；guangguan-others u126 排除 cylinder7:18 后无崩溃。

guangguan 三次崩溃定位链： cylinder7_18_4443 在轨迹 1776 步物体 NaN（诊断输出生效）；参考在该帧静止且速度正常 -> 策略-物理交互发散；seed 固定导致完全可复现。排除 cylinder7:18（48/4312）后通过原崩溃点。

remake-others 近 150 update 按对象成功率（n, len_med, ret_med）：
- 零成功: bowl 0/2676、cuboid1 0/6344、cylinder4 0.1%/7488
- 近零: largeclamp 0.7%、cylinder2 0.8%、cylinder7 1.1%（len_med 138 极短，ret_med 3.9 最低——疑似早死或参考本身就短）
- 较好: cuboid3 9.8%、mayonnaisebottle 9.2%、cylinder3 5.8%
- 长episode但不完成: pitcherbase(len 323, ret 72, succ 2.6%)、powerdrill(len 281, ret 57.7, succ 3.7%)
解读：统一批次混训按平均学习，难对象被系统性放弃。改进候选：按对象课程/采样权重、或拆分对象组训练。


## 2026-09-24T05:05:00+08:00 — u2000 里程碑

状态：cube1 u2007（窗口 succ 17.7%，累计 epMean 110.7）、cube2 u1978（窗口 succ 36.9%，epMean 164.8，仍快速上涨）；remake-others u751 succ 4.8% epMean 58.4；guangguan-others u198 稳定无崩溃。

action 18（Lateral）时间序列判别证据（200 update 桶成功率）：
- cube1: other 4.7%→25.1% 单调爬升；a18 全程 0.0-0.8% 平坦——完全不学。
- cube2: other 5.9%→32.1%；a18 1.1%→8.5% 后停滞在 7.5-8.5%。
- 交叉证据：NaN 发散凶手 cylinder7_18_4443 也是 action 18。
- 机制判断：Lateral 横向捏抓的接触面小（拇指侧+食指侧），raw_gesture 的期望接触区域/力阈值信号对它可能不匹配；策略能跟踪拿分（ret_med 35）但完不成。改进候选（按优先级）：几何锚点接触奖励对照实验、action18 接触区域审查、动作18 专项增广。


## 2026-09-24T08:30:00+08:00 — 物理证据评估（确定性 rollout，u3400 checkpoints）

方法：/tmp/eval_raw120_physical.py（server1 GPU2），load_skrl_checkpoint_for_inference + 确定性策略，全程 rollout，指标=接触帧/离桌帧(>5mm)/接触+离桌帧/峰值间隙/最终偏差/完成率。64 envs 全动作覆盖。

cube1 u3400（n, completed, 接触+离桌帧中位, 峰值间隙, 最终距离中位）：
- 04: 9/9, 126, 18.2cm, 1.1cm —— 近乎完美
- 01: 7/10, 133, 15.9cm, 2.9cm；03: 7/9, 120, 16.1cm, 2.3cm；10: 7/9, 147, 16.6cm, 2.1cm
- 02 (Prismatic-3-Finger): 3/9, 0, 0.07cm（一个离群 19.8cm）—— 基本不抬
- 09: 3/9, 167, 17.1cm（最高 23.7cm）, 最终 10.2cm —— 抓起但终点偏出
- 18 (Lateral): 0/9, 但 144 接触+离桌帧、峰值 15.9cm（max 27.4cm）—— **能抓能抬，后期丢掉**（final 10.3cm 超 10cm 偏差线）

cube2 u3400: action 18 2/8 completed、峰值 17.7cm——好于 cube1。

关键结论修正：action 18 不是学不会抓，而是抓起来后保持不住（真实抬升 15-16cm 后终点出界）。与日志成功率一致但物理含义完全不同——改进目标从"学会抓"变为"学会在 Lateral 姿态下保持/放置"。

remake-others 两次崩溃原因：njmax=512 约束容量饱和（与 guangguan 首崩同类，非 NaN）。watchdog 两次从 checkpoint 续训成功（机制验证）。已修：--constraint-capacity 2048 + checkpoint 间隔 50，r3 从 r1/last.pt 恢复（损失 ~54 updates）。
guangguan-others u960 succ 7.4% epMean 98.5 健康爬升。cube1 u3464 succ 26.7%；cube2 u3414 succ 30.5% epMax 2057。


## 2026-09-24T09:05:00+08:00 — u3700 曲线形态

200-update 桶成功率（u1200 起）：
- cube1: 16.2→17.2→19.5→20.7→21.3→23.1→24.2→25.8→26.4→26.7→27.8→29.3→29.9%（u3704），回报中位 40.9→44.3。仍在以 ~1%/200update 爬升，轻微减速无平台。
- cube2: 23.2→24.6→27.0→29.0→31.5→32.6→33.5→34.0→33.8→34.4→34.9→35.4→36.0%（u3652），回报中位 34.8→38.5。~0.5%/200update。
- 判断：继续跑满 5000 是正确的（曲线下无证据支持早停换配置）。

fleet: cube1 u3699/cube2 u3646（23k/s）；guangguan u1073 succ 6.9% epMean 97.1；remake r3 u75 恢复中（constraint 2048 + ckpt50）。
