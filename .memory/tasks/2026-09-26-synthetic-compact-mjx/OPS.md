# synthetic-compact-mjx task

## 2026-09-26 — 合成 compact 默认 + 容量原则

- 合成导出默认翻转为 compact-replay-visual（316d9e6）：今后合成直接产 VLA 格式，不走投影；full 仍可显式选。
- 文档新增"容量是合成的正确性地板"（docs/generated_physical_references.md）。
- 设定核对：用户清单与代码逐项一致。
- 训练容量上调计划：warp_ccd_contacts_per_world 8→16，应用于下一轮与补跑；运行中队列脚本不可热改，本轮保持 8。
- 补跑脚本 /tmp/retry_bottle_pitcherbase.sh（server2）：等 GPU 排空后跑；bottle 关 device-transition（复合场景 device 路径非有限值），pitcherbase 开；均为 ccd16/njmax2048。
