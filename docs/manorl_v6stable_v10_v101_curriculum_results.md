# ManoRL v6-stable (Reward v10 / v10.1 / v10.1+curriculum) 消融训练结果分析

状态:最终版(2026-09-21 06:30;三组训练与全部评估已完成,下一轮已启动)

## 运行信息

- 机器:ylang-U22(192.168.9.220,4× RTX 4090 24GB),conda `contact-conditioned-autonomy`
- 代码快照:`/mnt/user-home/cty/workplace/manorl-v6-stable-v101-4090-20260920`(commit 2acef6f,feat/v6-stable-v101-curriculum)
- 输出:`/mnt/nas-222-projects/cty/manorl-autonomy/outputs/v6-stable-v101-4090-20260920/`
  - `v6-stable-v10-gpu0-r2`(GPU0,reward v10)
  - `v6-stable-v101-gpu2-r2`(GPU2,reward v10.1)
  - `v6-stable-v101-curriculum-gpu3-r2`(GPU3,reward v10.1 + 三阶段课程)
- 公共配置:v6 架构、1024 env、1000 updates、rollouts 32、epochs 2、minibatches 16、lr 1e-5、target KL 0.05、teacher anchor β 1.0→0.2(2 passes)、severe -100、checkpoint 间隔 16、数据包 cube2_02_v295_f120_pre180_post180、基准身份 cube2_02_2833(--all-references)

## 事故与修复记录

- 2026-09-21 02:00 首轮(-r1,无后缀目录)三组全部 OOM 崩溃于首个 PPO update:启动命令绕过 train.sh,缺 `XLA_PYTHON_CLIENT_PREALLOCATE=false`,JAX 预占 ~75% 显存(18.8GB),torch 分配 256MB 失败。
- 02:27 以 `-r2` 目录重启,补齐 `XLA_PYTHON_CLIENT_PREALLOCATE=false`、`PYTHONPATH`、`PYTHONNOUSERSITE=1`、`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`;训练正常,显存 ~19.4GB/24GB。

## 训练曲线要点

- update ~105 时:v10 airborne 0.009 / v10.1 airborne 0.048(早期对向 8 帧门控未拖慢学习)/ curriculum 已晋升 stage 3,airborne 峰值 0.775 后回落 ~0.39(stage 3 起点更难),severe 均值 -100→-79.7。
- 三组 KL 全程 mean ~0.012、max ≤0.037,远低于 target 0.05,稳定化 PPO(lr 1e-5、epochs 2、target KL 0.05、anchor 衰减)起效——对比此前 v6(A800)的 KL 0.45–0.59 失控。**这是本轮最重要的方法论结论:上一轮"最佳行为被后续更新覆盖"的问题是 PPO 稳定性问题,不是架构问题。**
- **训练聚合指标与冻结评测严重背离**:v10/v10.1 训练曲线 return_mean 全程为负(~-55~-95)、airborne 率仅 0.002–0.008,但基准身份冻结评测自 u208 起稳定 strict 成功——聚合指标是 1024 env × 全部 50 条参考上随机策略的均值(-100 severe 主导),不能作为收敛判据;curriculum 组则相反,聚合指标好看全是重置点红利。**这条任务线上必须按身份做冻结 checkpoint 评估。**
- v10.1 组 u720 checkpoint 出现一次性全程评测完全失败(0 帧有载离桌,reason 2),u784 起恢复——中期坍塌后自愈,记录为稳定性观察点。

## 结果(最终)

### 全程冻结评测(基准身份 cube2_02_2833,17 个 checkpoint/组)

| 组 | strict 成功 | 最佳 checkpoint | 有载离桌(连续) | 峰值间隙 | 路径 RMSE |
|---|---|---|---|---|---|
| v10 (GPU0) | **14/17**(u208 起全成功,含 final) | update000656 | 130 帧 | 171.2mm | 0.0240 |
| v10.1 (GPU2) | **13/17**(u208 起成功,u720 坍塌一次) | update000208 | **143 帧** | **176.0mm** | 0.0292 |
| v10.1+curriculum (GPU3) | **0/17** | —(全部 0 帧有载离桌) | 0 | ~0 | 0.0069→0.0092(随训练单调变差) |

- 2026-09-21 04:05:v10 组 update000272 达成首个 strict_grasp_success(连续 128 帧有载离桌、峰值 170.4mm、RMSE 0.0189)。
- curriculum 组确凿失败:17/17 checkpoint(含最终)全程评测有载离桌全为 0,且路径 RMSE 随训练单调恶化——接近段遗忘持续加剧。训练曲线 airborne 0.4+ 全部为 stage 重置点红利。

### 跨轨迹泛化(held-out:5 validation + 5 test,最佳 checkpoint)

| checkpoint | validation | test |
|---|---|---|
| v10 update000656 | 0/5 | 0/5 |
| v10.1 update000208 | 0/5 | 0/5 |

**跨轨迹泛化仍为 0%,与此前五版架构消融结论一致**——策略只解出了基准轨迹,这是当前最核心的未解问题。

### 成功视频

- v10 update000272(首个成功):NAS `v6-stable-v10-gpu0-r2/evaluations/update000272/actual-vs-reference.mp4`
- v10 update000656(组内最佳):NAS `v6-stable-v10-gpu0-r2/evaluations/best/actual-vs-reference.mp4`
- v10.1 update000208(全场最佳,143 帧):NAS `v6-stable-v101-gpu2-r2/evaluations/best/actual-vs-reference.mp4`
- 本地副本:`/home/cty/workplace/v6stable-evaluation-20260921/`(三段视频均已抽帧验证:接近→抓取→高抬→稳定空中保持→跟随下降→全程自然结束)

## 迭代决定与下一轮(已启动)

1. **v10.1 略优于 v10 且机制更合理**(时序门控防奖励投机),作为主线奖励;v10 不再单独迭代。
2. **原课程机制废弃**:已实证其阶段替换式重置导致接近段遗忘。已实现防遗忘混合重置(`mixed_stage_frame`,autonomy_curriculum.py):stage≥2 时按比例确定性混入早期 stage 重置点(`--curriculum-mix-previous 0.25`)和第 0 帧全程重置(`--curriculum-full-horizon 0.15`);新增 3 项单测,curriculum+reward v10 共 15 项测试通过(test_autonomy_v4.py 有 1 项数据路径导致的预存失败,与本次改动无关,未修改快照上同样失败)。
3. **下一轮训练(06:24 启动,GPU3)**:`v6-stable-v102-mixed-curriculum-gpu3`,v10.1 + 混合重置课程,其余配置与本轮完全一致(可比性),部署快照 `/mnt/user-home/cty/workplace/manorl-v6-stable-v102-4090-20260921`。验收标准:stage 3 时期 checkpoint 的全程冻结评测 strict 成功(本轮为 0/17),且 held-out 成功率 > 0。
4. **泛化是下一阶段主战场**:若混合课程仍 0/10,方向改为按身份多样性课程(分时段切换参考轨迹的子集训练+held-out 早停)或更大参考子集的轮换训练;12N 扰动恢复测试待单轨迹全程成功稳定后再启动。
5. 观察项:curriculum 组 c_loss 曾达 2.1e4(return 量级大),如下一轮 critic 不稳则引入 value normalization;v10.1 u720 型中期坍塌需在下一轮巡检中关注。

## 第二轮(v102 混合课程)结果(最终,2026-09-21 09:10)

- **2026-09-21 08:05:首要验收点通过**——v102 u208(stage 3 时期)全程冻结评测 strict=True,连续有载离桌 131 帧、峰值 173.6mm;上一轮原课程 stage 3 时期为 0/17。防遗忘混合重置(mix-previous 0.25 + full-horizon 0.15)解决了接近段遗忘。
- 训练进展:u19 升 stage 2、u35 升 stage 3;stage 3 时期 airborne ~0.30(含 15% 全程环境的更难分布,与第一轮 curriculum 组的 0.42 不可直接比)。训练 08:48 完成(exit=0),KL 全程 ≤0.037,无中期坍塌。
- **全程冻结评测(基准身份,17 个 checkpoint):14/17 strict 成功**(u208 起全部成功,含 final)——追平 v10(14/17)、超过 v10.1(13/17),且这是带课程机制的结果。最佳 update000272:连续 135 帧有载离桌、135 帧对向承载、峰值间隙 ~17cm、路径 RMSE 0.0257。
- **跨轨迹泛化(u272,held-out 5 val + 5 test):0/10**——第二验收点未通过,与此前所有版本一致;混合课程解决的是遗忘而非泛化。
- 最佳视频:NAS `v6-stable-v102-mixed-curriculum-gpu3/evaluations/best/actual-vs-reference.mp4`;本地副本 `/home/cty/workplace/v6stable-evaluation-20260921/v6-stable-v102-mixed-curriculum-gpu3-update000272.actual-vs-reference.mp4`(已抽帧验证:接近→抓取→高抬→空中保持→跟随下降→全程完成)。

### 两轮结论汇总

| 组 | 全程 strict 成功 | 最佳有载离桌 | held-out |
|---|---|---|---|
| v10 | 14/17 | 130 帧 (u656) | 0/10 |
| v10.1 | 13/17 | 143 帧 (u208) | 0/10 |
| v10.1+原课程 | 0/17 | 0 | — |
| **v10.1+混合课程 (v102)** | **14/17** | 135 帧 (u272) | 0/10 |

**下一步主攻泛化**(按优先级):1) 分时段参考子集课程——按 identity 分片轮换训练子集并以 held-out strict 成功率早停/选 checkpoint;2) warm-start 自 v102-u272 在全 50 条参考上继续训练并加强 anchor 衰减;3) 单轨迹成功稳定后再进 12N 扰动恢复测试。

## 第三轮:held-out 失败归因(2026-09-21 09:50,v102-u272 全部 10 条 held-out trace + 视频)

**结论:接近段已泛化,失败在"抓握成形→抬升转化"段。** 10/10 轨迹策略都成功接近并建立有载接触(42–93 帧),但全部在 ~215–249 帧(路径误差触 10cm 阈值)偏差终止、峰值间隙 ~0mm。按抓握质量分两类:

- **A 类(3/10:2844、2870、2872)——抓握方向错误**:接近方位与参考不一致,接触瞬间把物体碰斜(2872 视频 t=1.1s 可见方块被碰歪),拇指对向从未形成(thumb=0、opp=0);
- **B 类(7/10:2846/2851/2853/2888/2854/2868/2882)——抓握成形但不抬升**:拇指对向接触已形成(12–39 帧),但手指包络不足(2888 视频:抓握角度偏差、指面未充分包裹),无法把接触转化为抬升。

**根因推断**:环境-参考分配是固定轮转同参考重置(fixed_round_robin_same_reference_reset),每个 env 绑定一条参考;聚合 return_mean 全程 ≈-100 说明绝大多数非基准 env 一直早期失败、几乎不产生成功抓握信号,策略只在基准身份上学到了完整抓抬——是优化难度问题而非数据缺失。据此:GPU2 warm-start 续训把 teacher anchor 抬高(2.0→0.5)以逐 env 重新锚定各自参考的抓握动作;GPU3 做分配轮换让每个 env 接触多条身份。

- 证据:NAS `v6-stable-v102-mixed-curriculum-gpu3/evaluations-heldout/update000272/{validation,test}/*/summary.json` 与 2872/2888 两段视频。

### 第三轮中期结果

- **warm-start 全参考续训(v103-gpu2,v102-u272 起点,anchor 2.0→0.5,1000 updates)**:基准身份全程成功保持(538 步 reason=1,return 4125.5,演示视频 `/home/cty/workplace/v6-stable-v103-warmstart-allref-gpu2-final.actual-vs-reference.mp4`);anchor MSE 0.033→0.007 单调下降;**但 held-out 仍 0/10(2026-09-21 14:15,最终 checkpoint)**——第四个 0/10,warm-start + 高 anchor 也未攻破泛化。补测中间 checkpoint(u272,anchor 仍强时)进行中。
- **分片轮换课程(v103-gpu3)**:14:05 启动,phase-a(20 条身份,340 updates)训练中。
- 初步推论:单任务策略对已见轨迹的拟合不受 anchor 强度限制(anchor MSE 降得很快),瓶颈更可能在**物理抓握的可迁移性**——残差位置控制(±1cm/步)下,策略学会的是"基准轨迹特定的抓握几何",而不是"按物体点云泛化的抓握策略"。若分片轮换也 0/10,下一步应转向观测/动作空间层面的改动(如增大接触相关观测权重、按物体几何归一化抓握目标,或直接提高非基准身份的采样权重)。

## 第三轮:warm-start 全参考续训结果(2026-09-21 14:10)

- **GPU2(v103-warmstart-allref,自 v102-u272 warm-start,teacher anchor 2.0→0.5,全 50 参考,1000 updates)**:训练健康(KL ≤0.034,anchor MSE 0.033→0.007 单调下降);基准身份最终 checkpoint 全程 538 步自然完成、return 4125.5(技能未破坏,演示视频 `/home/cty/workplace/v6-stable-v103-warmstart-allref-gpu2-final.actual-vs-reference.mp4`)。
- **held-out 泛化:0/10**——warm-start + 抬高 anchor 未攻破泛化;失败模式与 v102-u272 一致(全部 ~230 帧偏差终止、0 帧有载离桌)。结论:在已经会抓一条轨迹的策略上"摊"到 50 条参考,单靠 anchor 监督不足以形成可迁移的抓握;分片轮换课程(GPU3,进行中)是下一张牌。
- GPU3 分片轮换:14:05 启动,phase-a(20 条身份,340 updates)进行中。

### 分片轮换课程进度(2026-09-21 15:45)

- phase-a 于 14:53 完成(340/340,exit=0);其 checkpoint 的 held-out 评估 **0/10**——符合预期(只训了 shard A),决定性结果在 phase-c。
- phase-b resume 续训修复两处校验:teacher_anchor 衰减日程的 `schedule.updates` 跨度(340→680)在 config 与 provenance 两侧的 resume 一致性比较中豁免(autonomy_batch_training.py;衰减配方本身仍强校验,phase-b 会从 β≈0.6 处继续平滑衰减);supervisor 增加已完成阶段跳过。
- phase-b 于 15:35 重启并确认从 update 340 续训(当前 ~350/680,预计 ~16:30 完成;phase-c 预计 ~17:20)。

### 分片轮换课程最终结果(2026-09-21 17:38,决定性)

- 三阶段全部完成(phase-a 340 / phase-b 680 / phase-c 1000 累计 updates,exit 均 0);resume 链路的 teacher_anchor 日程豁免有效,phase-c 从 β≈0.38 处平滑续训。
- **基准身份(phase-c 最终):strict=True**——132 帧连续有载离桌、峰值 170.8mm、RMSE 0.0247,单轨迹技能未被轮换破坏(视频 `phase-c/benchmark-eval/actual-vs-reference.mp4`)。
- **held-out(phase-c 最终):0/10**(phase-a/b 同样 0/10)——**分片轮换路线证伪**。两条泛化路线(warm-start、分片轮换)在只改数据/课程、不改奖励的情况下全部失败。
- **第三轮总结论:泛化瓶颈不在数据暴露量,而在学习信号本身**——奖励的接触/对向/保持项锚定参考轨迹的接触意图,跟踪三连锚定参考路径,策略只能逐条记忆。第四轮主线 = Reward v10.2:接触/抬升/放置改纯物理打分、跟踪降权去门控、post_grasp_drop 参考下降豁免、新增放置姿态奖励。

## 第四轮:Reward v11(忠实移植 v5-log-reward,2026-09-22 01:30 完成)

- **实现**:`sim/manorl/autonomy_reward_v11.py` 忠实移植 `case/contact-conditioned-autonomy/v5-log-reward` 分支——**纯 9 项**(跟踪三连用 log 曲线 + 静态旋转覆盖 45°/×0.75;fingers/hand_relative/geometry;severe −75),刻意不含接触/抬升/掉落层(分支文档 §8)。契约 `manorl.autonomy.reward.v11.log-curve-9term`,16 项测试通过。
- **离线打分**(reward-scoring-v11-faithful-20260921,实验台 `manorl-reward-playground-v11-faithful-cube2_02_2833.html`):三条 v10.1 时代成功轨迹得 **91.7~94.0 分**(对照该分支官方 checkpoint 校准值 214.7);object_rotation −104(静态覆盖如实惩罚了 89° 放置偏差)、fingers +98.7、position +45.0。
- **训练**(v6 + v11 忠实版 + 混合课程 + 稳定化 PPO,GPU2,22:51–01:11,1000 updates):KL ≤0.037,ret_mean 从 ~−20 改善到末期更高(聚合口径),课程 stage 3 正常。
- **结果(最终 checkpoint)**:
  - **基准轨迹 strict=True**:538 帧全程、131 帧连续有载离桌、峰值 169.3mm、RMSE 0.0244——纯 9 项模仿奖励也能训出完整抓取(视频 `/home/cty/workplace/v6-v11-faithful-mixed-gpu2-final.actual-vs-reference.mp4`,已验证)。
  - **落地姿态:tilt 末段仍 89.2°**——静态旋转覆盖在打分上惩罚了 −104,但 1000 updates 内没能让策略改掉翻转行为(覆盖触发条件 v_eff≤0.01 且误差>45°,惩罚时长/强度不足以压过 fingers/geometry 的梯度)。
  - **held-out:0/10**——与 v10.1 时代完全一致的失败模式(接近正常、~230 帧偏差终止、0 有载离桌)。
- **第四轮总结论**:在 v6 架构 + 稳定化 PPO + 混合课程下,**三种本质不同的奖励族(v10 接触驱动 20 项 / v10.1 时序门控 20 项 / v11 纯模仿 9 项)都能收敛到基准轨迹 strict 成功,但跨轨迹泛化全部 0/10 且失败模式相同**。这排除了"奖励塑形"作为泛化瓶颈的唯一解释——更深的候选:① 每 env 绑定单条参考 + 优化预算内多数 env 未能精通自己的参考(优化难度);② 观测虽含点云,但策略收敛路径仍逐条拟合;③ 需要机制层面的泛化手段(域随机化初始位姿、课程式参考切换、或更多 updates)。
- 下一步候选(按优先级):A) 初始位姿/物体位置域随机化(env reset 加扰动);B) 每 env 定期换绑参考(轮换 assignment 需要重建 runtime,工作量中等);C) 训练预算翻倍(2000 updates)看聚合指标是否继续上行。

## 第五轮:v11.1 / v11.2 / 域随机化 三路对照(2026-09-22 17:15 完成)

| 路 | 基准 strict | 落地 tilt | held-out | 判定 |
|---|---|---|---|---|
| v11.1(+opposition_align 对捏对齐) | ✅ 130 帧有载离桌 | **89.2°(分毫未改)** | 0/10 | 对捏对齐证伪(修不了翻面) |
| v11.2(tracking+opposition 压缩双项) | ✅ 131 帧 | **89.2°** | 0/10 | 压缩证伪 |
| v11.1+域随机化(±2cm/±15° 与 ±1cm/±10° 两档) | 无法自举(ret_max 全程钉死 −75) | — | — | 域随机化与关键帧课程不兼容,证伪 |

- **翻面问题定性完成**:v11 / v11.1 / v11.2 三种配方训出的策略,tilt 轨迹逐位相同(67°→89°)——绝对值惩罚类奖励(旋转覆盖/对捏对齐/重加权)全部无法让 PPO 跨出"斜抓"局部最优。惩罚改变的是打分位置,改变不了优化地形。
- **下一轮(v11.3)必选项**:A) 势函数差分奖励(奖"每步姿态误差缩小量",policy-invariant,削平山谷);B) 在手姿态守恒项(惩罚物体相对手掌的朝向变化,轨迹无关,机械时刻直接约束)。两者均 ~10 行改动,可先离线打分验证量级再训。

## 可靠度战役:10 条 Lance 轨迹 v6+v11 实测(2026-09-23 03:25 完成)

任务:对 9 个 guangguan 数据集(11 个版本行)逐一编译轨迹包、以 v6 架构 + v11 奖励 + 稳定化 PPO(混合课程)训练 1000 updates,逐条严格评测 + 成功对比视频。

| # | 轨迹 | 物体 | 参考数 | strict | 有载离桌 | 峰值 |
|---|---|---|---|---|---|---|
| 1 | clean_v295(cube2:02) | cube2 | 50 | ✅ | 130 帧 | 169.1mm |
| 2 | clean_v407(cube2:02) | cube2 | 50 | ✅ | 130 帧 | 169.0mm |
| 3 | clean_v530(cube2:02) | cube2 | 50 | ✅ | 130 帧 | 169.1mm |
| 4 | daily_20260817_v14 | cube1 | 50 | ❌ | 0 | 0 |
| 5 | daily_20260818_v8 | cube2_01 | 17 | ❌ | 0 | 0.3mm |
| 6 | daily_20260819_v15 | cuboid1 | 6 | ❌ | 0 | 0 |
| 7 | daily_20260820_v4 | cylinder2(复合场景) | 34 | ❌ | 0 | 0.1mm |
| 8 | daily_20260821_v3 | cylinder3(复合场景) | 7 | ❌ | 0 | 0 |
| 9 | daily_20260824_v6 | cylinder3_10(复合场景) | 49 | ✅ | **167 帧** | **212.7mm** |
| — | 20260811_20260813_v7 | banana 等 18 种 | 921 | 不适用(未指定 pair) | — | — |
| — | 20260814_v3 | largeclamp 等 | 428 | 不适用(未指定 pair) | — | — |

**可靠度:可训练任务 9 个中 strict 成功 4 个(44%);按数据类型:clean 聚合数据 3/3(100%),daily 单次采集 1/6(17%)。**

核心结论:
1. **配方的可靠性与数据规模/质量强相关,与物体类型弱相关**——cube2/cube2_01/cube1/cuboid1/cylinder2/cylinder3 都试过,参考数 ≤34 的 4 个任务(17/18/19/21)全败;50 条量级的任务里 clean 清洗数据 3/3 全成、daily_24(49 条)也成,只有 daily_17(50 条但含坏示范)失败——**数量和质量缺一不可**。(更正:此前表格中 daily_19/20/21/24 参考数误记为 7/1/1/1,实际训练通过 --reference-identities 传入同前缀组的全部轨迹:6/34/7/49 条。)
2. 复合场景(带干扰物体)在 256 envs 下可训练(GPU 显存 21.1GB,1024 envs OOM),其成功例(daily_24)抬升 212.7mm、167 帧有载离桌,是**全程最高峰值**——复合场景不是障碍。
3. 工程记录:轨迹包须本地盘编译(NAS 不支持原子重命名);单版本含数百条多物体轨迹的包需显式 --pairs;单 env 复合场景成本约为 cube2 的 4 倍。
4. 产物:9 段 actual-vs-reference 视频在 /home/cty/workplace/reliability-videos-20260922/;全部 checkpoint/日志在 NAS outputs/reliability-v11-20260922/。
