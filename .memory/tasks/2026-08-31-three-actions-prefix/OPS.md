# Operations（追加式，不可变）

- 2026-08-31T09:00+08 — 编译 3 动作 pre60 bundle（cube2:02=50 源、banana:09=36 源、largeclamp:04=50 源，全 0 拒绝）+ predecode。
- 2026-08-31T09:30+08 — 探针筛选（10 seed × 全部源，GPU0/1/2 并行）：cube2:02 91.4%（45/50 稳健）、banana:09 75.6%（26/36）、largeclamp:04 40.2%（18/50）。选 3 条稳健源/动作。
- 2026-08-31T10:00+08 — **歧义教训 1**：我直接用"3 源 × 80 seed"（direct 起始、无 prefix）生成了 720，用户发现缺 prefix 拒绝。正确合同 = prefix-only 合成（Far/Near slots + fallback）。旧版已删。
- 2026-08-31T11:00+08 — 构建 parent descriptors（从探针 Lance）+ coverage plan（cube2/banana 用 6000 池；largeclamp cell 支持不足，改用 12000 池解决）。
- 2026-08-31T13:00+08 — 修复 vectorized runner `_new_batch_runtime` 缺 `required_pre_padding` 参数（movement_pre_padding 硬编码 60），同步后 3 动作并行正式合成（GPU0/1/2）。
- 2026-08-31T14:42+08 — banana:09 完成 240/240（0 exhausted），先行 retreat+扩增发布：NAS banana_09_augmented_720_prefix_20260831.lance（SHA 317996da…）。
- 2026-08-31T15:07+08 — cube2:02 210/240（14 exhausted）、largeclamp:04 201/240（9 exhausted）按用户指示停止（收益递减），retreat+扩增发布：cube2_02_augmented_prefix_20260831.lance（630 行，SHA 58924fe9…）、largeclamp_04_augmented_prefix_20260831.lance（603 行，SHA 6684dabf…）。
- 2026-08-31T15:30+08 — 删除缺 prefix 旧版（banana_09/cube2_02/largeclamp_04_augmented_720_20260831.lance），保留 _prefix 版。
- 2026-08-31T16:00+08 — **歧义教训 2**：用户反馈需求歧义。根因 = 产品合同（prefix/retreat/扩增语义、成功率两阶段）没有固化为不可变标准。已将"产品合同总纲"写入 local SOP 第 0 节。
- 2026-08-31T17:00+08 — 收尾完成。NAS 最终回读验证 3 数据集全部通过（banana 720=240×3、cube2 630=210×3、largeclamp 603=201×3，uuid 唯一，manifest SHA OK）。清理：4 个 tmux 会话（formal/probe2/largeclamp-all/bench-parallel）已杀、无孤儿进程、GPU0/1/2 空闲、server1 /tmp 临时脚本已清、`-pre60.pending`（9 文件部分构建，无引用）已删、`-pre60.pending.compile-evidence`（审计证据 40K×3）保留。pre60 bundle 完整性复核：cube2 50/50、banana 36/36、largeclamp 50/50 pkl 且 pickle SHA 匹配。数据根 271M（probes 230M + pre60 42M + descriptors/plans/selection）保留为 provenance。
- 2026-08-31T18:00+08 — 吞吐测试结论（无真实合成）：同 source 多 env 并行单卡不线性（3 env 21 step/s、96 env 30、384 env 20、1023 env 11.6），device_resident_controls 边际提升（384 env 20→25），capture_transition_diagnostics 无影响。瓶颈 = MJX 单 step 固定开销非 env 数。训练 4096/8192 能跑但吞吐不随 env 线性。生产提速 = 多卡 × 每卡 ~96 env（约 4.5×），非单卡大 batch。
- 2026-09-01T10:00+08 — 资产对比（server1 repo vs main.pi05Mint exp-state50-contact-rate5 manorl-native-28d）：40 物体清单一致；37/40 文件级 SHA 相同；仅 faucet_base（coacd 分解版本不同）、bottlewithcap（URDF 不同）、sphere3（coder 多一 urdf）有差异——均非生产物体。生产相关（banana/bowl/mayonnaisebottle/cube2/largeclamp）完全一致。
- 2026-09-01T11:00+08 — 文档沉淀：docs/manorl_synthesis_methodology_prompt.md（可移植 4 阶段方法论 prompt）；local SOP 第 0 节产品合同总纲（术语消歧 + 成功率基线 + ckp 训练物体清单）；全部并入 dev（af86df4）。
