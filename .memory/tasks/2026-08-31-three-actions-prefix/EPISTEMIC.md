# Epistemic（可变模型）

## 当前状态
3 动作已交付：banana:09 720（满额）、cube2:02 630（shortfall 30）、largeclamp:04 603（shortfall 39）。全部 prefix+retreat+增强，NAS 已发布。

## 关键机制
- ckp-001000（pre180 训练）在训练集内物体的 prefix 合成可行；成功率差异大（cube2 91% → largeclamp 40%）
- 选定 10/10 稳健源后正式合成率显著高于探针均值（banana 100%、cube2 87.5%、largeclamp 83.8%）
- largeclamp Near cell 支持不足 → 候选池 12000 解决（不跨 cell 补）
- shortfall 是 bounded 设计（每 cell 12 fallback 用尽即停）

## 歧义教训（已固化到 SOP 第 0 节）
1. "扩增"必须完整 = retreat + translate + rotate，缺 prefix 的错误被用户捕获
2. 成功率必须分探针/正式两阶段，不可混用
3. pre60/pre180 分 ckp 训练合同 vs 合成合同
4. 新任务开始先读 SOP 第 0 节，不凭记忆重建合同
