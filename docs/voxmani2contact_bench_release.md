# case/voxmani2contact_bench 分支交付

本分支发布 `/home/cty/workplace/voxmani2contact_bench` 的 v8 迁移版。
原始源码提交为 `c17000d7041d48c79fc0ae8c5c0dd31eed47591a`。
网络依据 VoxMani 提交 `16a80fc6615c3f82194acd34e45b98d459359325`，
采用453维观测、547维目标快照、独立 Actor/Critic 和两路目标融合。
训练、控制、奖励、终止和重置沿用 ManoRL v11 的适配语义。
这次发布仅补齐交付材料，没有修改模型、奖励或 PPO 实现。

## 包含内容

- 当前提交的全部源码、配置、测试、架构及逐维说明。
- `data/cube2_all_v295_f120_pre180_post180`：真实参考包及原始 manifest，
  全部数组已按其 SHA-256 校验。训练不需要访问原始 NAS。
- 资产、Lance exporter、schema 和 analyzer 的原始固定版本 submodule 指针。
- `vendor/voxmani453`：四个依赖的独立 Git bundles、已物化资产的 LFS
  对象归档以及校验 manifest，可从本分支恢复依赖而无需访问内网 Git。
- `docs/validation/voxmani453`：发布前已有的 CPU 训练、续训和评估记录。
  它们属于既有结果；本次发布没有重新运行仿真或训练。

生成的训练输出、缓存、虚拟环境和本机 Git 配置仍由各机器自行管理。
旧 `checkpoints/manorl_autonomy_v4_cube2` 是旧模型检查点，不能加载到 v8。
本分支不宣称策略已收敛或具备抓取成功率。

## 获取分支

```bash
git clone --branch case/voxmani2contact_bench \
  git@github.com:DexGEM-Lab/mujoco-warp-contactbench.git
cd mujoco-warp-contactbench
```

不要在上述克隆命令添加 `--recurse-submodules`，可直接用下节的随附 bundles
恢复完整依赖。原 `.gitmodules` URL 保留来源信息，其中 Lance 依赖指向内网。

## 从随附 bundles 恢复固定依赖

以下命令用于新克隆且尚未初始化 submodule 的工作区。每个 bundle 保留其固定
提交的历史；先克隆外层仓库，再克隆其嵌套依赖。

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone vendor/voxmani453/dexstream_digital_assets.bundle \
  assets/dexstream_digital_assets
git -C assets/dexstream_digital_assets checkout --detach f98da997f316c8a6b4bc2931cabed19e831ef163

git clone vendor/voxmani453/lance_manager.bundle 3rd_party/lance_manager
git -C 3rd_party/lance_manager checkout --detach 72240d77acc38aba5041f8788c4a0506dc80055d

git clone vendor/voxmani453/lance_schema.bundle 3rd_party/lance_manager/schema
git -C 3rd_party/lance_manager/schema checkout --detach bf3d7bb8d23cd4662a9c507a5f7f6dbd42899c84

git clone vendor/voxmani453/data_structure_analyzer.bundle \
  3rd_party/lance_manager/external/data_structure_analyzer
git -C 3rd_party/lance_manager/external/data_structure_analyzer \
  checkout --detach 5e662fc1a376ad5cb3781fbbe2bf8ff7d75cafd3

mkdir -p assets/dexstream_digital_assets/.git/lfs/objects
tar -xzf vendor/voxmani453/dexstream_lfs_objects.tar.gz \
  -C assets/dexstream_digital_assets/.git/lfs/objects
git -C assets/dexstream_digital_assets lfs checkout
```

资产 bundle 克隆时暂时跳过 LFS 自动下载，然后执行上面的 LFS 对象恢复和
`lfs checkout`。归档覆盖原交付目录已下载的 LFS 对象，恢复后的依赖文件
与原交付目录逐文件相同。资产仓库中其他未下载的模型仍保留其原始 LFS 指针；
如果以后使用这些模型，需要从资产远端下载对应对象。

依赖提交与 bundle/LFS 归档校验值见
`vendor/voxmani453/manifest.json`。参考数组校验值见参考包的 `manifest.json`。

## Python 环境与运行入口

使用 Python 3.11 和 `pyproject.toml` 中的依赖版本；例如在具备 CUDA 的机器上：

```bash
uv sync
uv run python tools/run_voxmani453_experiment.py \
  --output outputs/voxmani453_gpu_run001 --wait-seconds 3600
```

或者在已安装依赖的环境中使用
`python tools/train_manorl_autonomy.py train --policy-version v8 --reward-version v11`，
完整训练、续训和评估参数见 `docs/voxmani453_v11.md`。
文档中的远端 Conda 路径是当时那台机器的路径，克隆后需要使用本机环境。

CPU 既有证据见：

- `docs/validation/voxmani453/cpu_train_resume_evaluate.json`
- `docs/validation/voxmani453/cpu_delivery_check.json`

这些短程执行记录不能替代 GPU 完整训练、收敛和任务成功率验证。
