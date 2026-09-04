# 可移植 Prompt：机械手操作合成数据的 Prefix + Retreat + 场景增强方法论

> 本文档是**跨项目可复用的方法论 prompt**。核心思想：用仿真（solver）生成"动作怎么去"，用离线构造生成"动作怎么回 + 摆放多样性"。已在 ManoRL（MJX-Warp + PPO，右手 28-DoF，120Hz）验证。

---

## 使用方法

把下面 `=== PROMPT ===` 之间的内容作为一个整体 prompt 交给另一个项目/agent；末尾 `=== 参数速查 ===` 是 ManoRL 的具体数值，供对照翻译，不属于 prompt 本体。

---

=== PROMPT ===

你在为机械手操作（grasp/place）生成**训练用合成轨迹数据**。目标不是最大真实感，而是**分布完整 + 语义自洽 + 可验证**。请按以下四阶段执行，不要跳过、不要合并、不要凭记忆修改合同。

## 阶段 0 — 先定合同（禁止跳过）

开工前把以下条目写成一个 `CONTRACT.md` 并由需求方确认，之后每次任务先读它：

1. **参考/控制/物理时钟**（如 120/120/480，固定 4 物理子步）
2. **参考轨迹的 pre-padding 合同**（轨迹裁剪到 pre-N 起点）与**模型训练时的观测 pre-padding**（二者可以不同：观测对齐模型，轨迹按合成合同裁剪）
3. **可接受的末端质量门**（数值，见阶段 2）
4. **每个"数据行"的完整语义**：它 = 前缀 + 完整参考 + 尾部替换后的回撤段。缺任何一段都不算完成
5. **成功率报告口径**：筛选阶段 vs 正式生产阶段分开报

## 阶段 1 — 仿真合成 base（prefix 必须由 solver 生成）

对每条被选中的参考轨迹（source）：

1. **选源**：先用小规模探针（10 seed × 全部源）量化每个源的通过率，只选**多次全过**的稳健源（通常每个动作 3 条）
2. **Prefix = 确定性空间分布采样**：不要用"同一个起点换随机种子"冒充 prefix。
   - Far：起手位置在物体周边远环（半径/方位角/高度按格子分层采样）
   - Near：起手位置在"抓取前近处"按经验分位数分格子采样
   - 每格配 N 个 fallback 种子，同格失败只在本格重试（bounded shortfall，禁止跨格补）
3. **solver 跑完整参考**：从 prefix 终点以零位误差 splice 进参考起点，之后由 checkpoint 策略/控制器执行到参考结束
4. 逐帧记录：状态、物体位姿、**接触力**、参考映射、命令映射

**为什么 prefix 必须仿真**：approach 阶段决定接触建立的质量，只有 solver 能给出真实接触动力学；离线几何"看起来接近"的起点往往导致策略失稳（物体被推出偏差阈值）。

## 阶段 2 — 三条件原子门（gate 决定保存，不与 reward 混）

一条 base 行**只有全部满足**才写盘：

1. 策略跑完参考末帧（termination reason = 完整完成）
2. 末帧物体姿态误差（intrinsic XYZ 平均）≤ 阈值（如 35°）
3. 全程有 ≥ 阈值的**解算接触帧数**（如 ≥101 帧、单帧法向力 >0.2N，仅统计受控手 vs 目标物体）

prefix 期间任何受控手—桌面或受控手—物体接触（>0.2N）直接拒绝。reward 数值不参与是否写盘。

## 阶段 3 — 离线 retreat（为什么它可以离线）

**核心洞察**：runtime（solver 里）你不知道"接触在哪一帧结束"，所以历史上 retreat 只能做在运动窗口结束附近、经常错位；**但离线时 contact 数组逐帧都在**，你可以精确知道：

1. 从持久化 contact 找**真实最后接触帧** `last_contact`
2. **锚点 = last_contact + 15 帧**
3. **替换锚点之后整个尾部**（不是加 30 帧短尾）：长度 = 原轨迹总长 − 锚点 − 1
4. 只重写**受控手手腕 XYZ**：smoothstep 形变，前 2 帧与源一致（splice 速度/加速度误差=0），后 3 帧到达"更远、更高"的终点（方向 = 锚点→原末尾方向 + 额外水平/角度/高度扰动）
5. 其余全部保留原值：长度、时间戳、手指关节、物体、contact、reference

**为什么可以离线**：回撤段发生在接触已结束后（锚点在最后接触帧之后），物体已静止/脱离，此时手腕的 kinematic 构造不改变任何已发生物理事件的正确性；contact 数组锚点之后本来就是空的（真实记录），不会引入伪造接触。

## 阶段 4 — 离线场景增强（摆放多样性）

对带 retreat 的完整行做**整场景刚性变换**，用于 VLA 泛化：

- **XY 平移**（如 ±20cm）：手/物体/参考/接触世界系位置一起平移；局部系位置、力、手指不变
- **绕物体中心、绕桌面法向 Z 旋转**（如 ±30°）：手绕物体转、物体原位转朝向、力向量随世界转

**变换必须施加到"状态"和"目标"双份**（urdf_dof 与 urdf_dof_target 等价的"实际+参考目标"同构字段）——只改一个会造成状态与目标错位。

采样用 **Latin Hypercube**（分层均匀覆盖），保持 base : translate : rotate = 1 : 1 : 1，每条 base 恰好 1 个平移变体 + 1 个旋转变体（可追溯）。

**旋转表示要先实证再写代码**：同一个仓库里，"物体朝向"可能是轴角(rotvec)，"手朝向"可能是外欧拉/内欧拉+rotvec 镜像，"目标"字段可能又是另一套。用真实数据反向验证（如 rotvec↔欧拉互转残差），把表示写进合同再动笔。旋转组合用 `R_world @ R_self`，别用 `R_self @ R_world`（方向会反）。

## 收尾 — 每个动作的交付清单

- base：N 行（稳健源 × 每源 slots）
- retreat：N/N 行全带回撤（0 plain；若某行接触持续到末帧无法锚定，如实保留为 plain 或剔除并报告）
- 增强：N translate + N rotate，共 3N 行
- UUID：base 保留原 id；变体行用 `uuid5(base_id, augmentation_identity)`（可反解追溯）
- 发布：stage → 逐文件 SHA 回读 → rename；**不要覆盖**仍在使用的旧数据集，用独立后缀
- shortfall：每 cell fallback 用尽即为 bounded shortfall，如实报告，禁止跨 cell 补齐

## 常见失败模式（先看这个再动手）

1. "加 30 帧短尾"当 retreat —— 错，retreat = 替换整个尾部
2. "同起点多 seed"当 prefix —— 错，prefix 必须是空间分布采样
3. 变换只改状态不改目标 —— 错，双份都要
4. 旋转方向写反（mats @ R.T vs R @ mats）—— 用数值验证
5. 用了模型没训练过的物体 —— 通过率≈0，先查训练覆盖
6. 合成快慢不归因 —— 并行先测吞吐再定 batch，别盲目上大 batch

=== PROMPT ===

---

## 参数速查（ManoRL 已用数值，供翻译参考，不属于 prompt）

| 参数 | ManoRL 值 |
|---|---|
| 时钟 | 120 / 120 / 480，4 子步 |
| 合成 pre-padding 合同 | pre60 基座 + prefix（总 pre-padding 100–360） |
| 模型训练观测 pre-padding | 180（movement_pre_padding） |
| Far prefix | 半径 0.30–1.00m、±30°、Z +0.08–0.30m |
| Near prefix | 5(距离)×3(方位)×2(高度)，经验分位数 |
| 每格 fallback | 12（候选池 6000，分布差时 12000） |
| 门 | reason=完成 且 末帧 XYZ mean ≤35° 且 ≥101 帧接触 >0.2N，prefix 无接触 |
| retreat 锚点 | 真实最后接触帧 + 15，替换整尾（155–251 帧） |
| retreat 方向 | 锚点→原末尾 + 3–15cm 水平、±30°、Z +4–10cm |
| 增强 | 平移 ±20cm、旋转 ±30°（LHS，1:1:1） |
| 源通过率参考 | 训练集内物体：20–91%；训练集外（如 cylinder6）：≈0 |
| checkpoint | 单一 native checkpoint（SHA 固定） |
| 每动作产量 | 3 源 × 80 slots → 240 base → 720 行（含 shortfall 时按实报） |

## 已验证数据（NAS 发布，作参考样例）

- `for_vla_manorl_prefix_near_far_4pairs_20260826_with_retreat.lance`（1035 行，4 动作）
- `banana_09_augmented_720_prefix_20260831.lance` / `cube2_02_...` / `largeclamp_04_...`（720/630/603 行）

## 工具映射（ManoRL 仓库，供找对应物）

| 阶段 | ManoRL 工具 |
|---|---|
| 编译/predecode | `compile_manorl_trajectory_package.py`、`predecode_package_for_synthesis.py` |
| 探针选源 | `export_manorl_synthetic_lance.py`（episodes=10, num_envs=全部源） |
| coverage plan | `build_manorl_prefix_only_coverage_plan.py` |
| prefix 合成 | `run_manorl_prefix_only_coverage_plan_vectorized.py` |
| 离线 retreat | `build_manorl_full_retreat.py` |
| 离线增强 | `build_manorl_xy_translate.py`、`build_manorl_xy_rotate.py` |
