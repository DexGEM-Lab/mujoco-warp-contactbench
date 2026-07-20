# MuJoCo Warp ContactBench

Standalone MJX-Warp sim-to-Lance export repo.

This repository migrates [DexHandRL](http://192.168.10.116/dexrobot-oss/dexrobot_isaac/-/tree/dev/change_source?ref_type=heads) from the Isaac Gym simulator to MuJoCo.

This repo is intentionally decoupled from the larger `contactbench` workspace. The MANO hand asset and `lance_manager` are git submodules. The repo contains MuJoCo scene builders, MJX-Warp contact extraction, direct Lance export, and Docker/runtime helpers.

## Layout

```text
assets/mano_hand_s02/        submodule: MANO MJCF/URDF/STL assets used by simulation
3rd_party/lance_manager/     submodule: generated_data Lance writer/schema stack
sim/                         MJX-Warp simulation, scenarios, schema helpers, and export code
sim/benchmarks/ball_pit/     deterministic ball-pit scenario and camera helpers
sim/common/                  contact schema helpers and validators
scripts/                     build, sim-to-Lance, smoke test, and env helper scripts
outputs/                     generated outputs; ignored by git
Dockerfile                   Debian + Miniforge + CUDA + uv runtime image
```

## Clone / Submodules

After cloning this repo elsewhere, initialize submodules first:

```bash
git submodule update --init --recursive
```

Current submodules:

```text
assets/mano_hand_s02 -> git@192.168.10.116:ai/group-ai-public/group-sim-assets/mano_hand_s02.git
3rd_party/lance_manager -> git@192.168.10.116:ai/group-dexcanvas/lance_manager.git
```

## Local uv Environment

```bash
scripts/setup_local_env.sh
JAX_PLATFORMS=cpu .venv/bin/python sim/smoke_test.py --strict --device cpu
```

The local uv environment is useful for CPU checks and development. GPU execution is supported through the Docker image by default. The setup script also installs the optional `dexhandrl` extra for the skrl migration entry points.

## Build Docker Image

```bash
scripts/build_docker.sh
```

This creates:

```text
mujoco-warp-contactbench:latest
```

The image does not copy repo source code. Runtime scripts mount the workspace at `/workspace/mujoco-warp-contactbench`, so code changes do not require image rebuilds unless dependencies or the Dockerfile change.

## Docker X11 Viewer

`docker-compose.yml` forwards the host X11 socket and `${HOME}/.Xauthority` into the container so GUI tools such as `mujoco.viewer` can open on the host display. From an active X11 desktop/session:

```bash
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose --profile gpu run --rm contactbench-gpu bash
```

Inside the container, run GUI tools directly, for example:

```bash
xclock
python -m mujoco.viewer path/to/model.xml
```

The compose service uses host networking so SSH X11 forwarding values such as `DISPLAY=localhost:16.0` continue to work inside the container.

## Sim-to-Lance Export

Run MJX-Warp simulation and write Lance directly, without contact JSON intermediates. GPU is the default device:

```bash
CUDA_VISIBLE_DEVICES=0 scripts/run_sim_lance.sh
```

Use CPU execution for the same MJX-Warp backend with:

```bash
DEVICE=cpu scripts/run_sim_lance.sh
```

Default output:

```text
outputs/mjx_warp_contactbench_generated.lance
```

Extra simulation arguments are forwarded to `sim/sim_to_lance.py`, for example:

```bash
DEVICE=cpu OUTPUT=outputs/test.lance scripts/run_sim_lance.sh --duration-seconds 1 --ball-count 16
```

The exporter uses `3rd_party/lance_manager/schema/schemas/generated_data_schema.jsonc` and writes with the submodule's Lance writer.

## Smoke Test

```bash
scripts/run_smoke_test.sh
```

This validates MuJoCo, JAX/MJX availability for the requested device, and a short MJX-Warp sim-to-Lance export. Set `DEVICE=cpu` to smoke test CPU execution.

## DexHandRL Migration

The `sim/dexhandrl/` package contains the local migration surface for moving the
DexHandRL IsaacGym reconstruction task onto this repo's MuJoCo MJX-Warp backend.
The first acceptance path is reference PD replay with `use_residual=false` on the
same Lance dataset used by the IsaacGym task:

```bash
PYTHON=/home/zrg/miniconda3/envs/mujoco_contactbench/bin/python \
  DEVICE=cpu scripts/run_dexhandrl_reference_pd.sh --object cube1 --action 01 --max-frames 200
```

Default Lance input:

```text
/mnt/nas-222-project/mocap/for_retargeting/dexhand021pro/results_all_mano/all/lance_new_all_generated_mano/lance_result_archive_full/dexhand021pro_lance_new_all_generated_mano_retarget_result_archive.lance
```

The replay writes metrics under `outputs/` by default and keeps
the generated scene XML there for PD/contact tuning. Override
`DEXHANDRL_OUTPUT_ROOT` if you want a different location.

The skrl-facing environment entry point is
`sim.dexhandrl.skrl_env.DexHandRLMJXEnv`. It exposes the IsaacGym-compatible
22D action space and 461D policy observation layout:

```bash
PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu \
  /home/zrg/miniconda3/envs/mujoco_contactbench/bin/python - <<'PY'
import numpy as np
from sim.dexhandrl import DexHandRLMJXEnv, DexHandRLMJXEnvConfig

env = DexHandRLMJXEnv(DexHandRLMJXEnvConfig(device="cpu", use_residual=False))
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(np.zeros(env.num_actions, dtype=np.float32))
print(obs.shape, reward, terminated, truncated, info["current_frame"])
env.close()
PY
```

The skrl launcher is:

```bash
bash scripts/run_dexhandrl_skrl.sh --mode inspect --device cpu --num-envs 1
```

The recommended local setup is `scripts/setup_local_env.sh`, which installs the
`dexhandrl` extra containing `torch`, `gymnasium`, and `skrl`. If you manage the
environment yourself, install those packages before using the skrl launcher.

For training, switch to `--mode train` and point `--checkpoint` at a saved skrl
checkpoint if you want to resume a MuJoCo run. Output defaults also live under
`outputs/`.

Current 461D IsaacGym rl-games checkpoints can be loaded directly. The loader
maps PointNet, FiLM, actor/critic, and output heads and restores rl-games input
and value running statistics:

```bash
PYTHON=/home/zrg/miniconda3/envs/mujoco_contactbench/bin/python \
  bash scripts/run_dexhandrl_skrl.sh \
  --mode rollout --device cpu --num-envs 1 \
  --object-name cube1 --action 01 --sequence-index 1 \
  --rlgames-checkpoint /path/to/last_Dexhand021proReconstruction_ep_20000.pth
```

The deterministic rollout uses the policy mean and writes
`rollout_metrics.json` below the selected output directory. Checkpoints with an
older observation layout, such as 429D weights, are rejected rather than padded
because their observation semantics are not equivalent.

For a seq1 parity run with the current simulator-specific finger settings and
full observation/action diagnostics:

```bash
PYTHON=/home/zrg/miniconda3/envs/mujoco_contactbench/bin/python \
  bash scripts/run_dexhandrl_skrl.sh \
  --mode rollout --device cpu --num-envs 1 \
  --object-name cube1 --action 01 --sequence-index 1 \
  --finger-kp 15 --finger-kv 0 --finger-force 50 \
  --hand-self-collision-mode disable_within_finger \
  --rollout-keep-observation \
  --rlgames-checkpoint /path/to/last_Dexhand021proReconstruction_ep_20000.pth
```

`sequence-index 1` is the second discovered `(cube1, 01)` trajectory (Lance
row/sequence id 22 in the current archive). For point-by-point checkpoint
diagnostics, run IsaacGym with `task.pointCloudEncoding.isDynamic=false`; the
MuJoCo irregular-mesh sampler mirrors IsaacGym's 10x surface oversampling,
seed-42 canonical sample, and farthest-point downsampling.

Current seq1 acceptance status:

- Reference PD (`use_residual=false`) is at the same object-position-error
  scale: about 0.009 m mean in IsaacGym and 0.011 m in MuJoCo over the shared
  pre-lift window (Lance frames 62–307). Over the full shared 632-frame
  sequence, the means are about 0.0856 m and 0.0754 m respectively; this is
  an error-scale comparison, not a claim that either pure-PD run completes the
  lift.
- The converted 461D rl-games checkpoint reproduces recorded policy actions
  from either backend to below `1e-6`, including RunningMeanStd, PointNet, FiLM,
  and actor weights.
- The residual-policy physics result is not yet accepted. IsaacGym completes
  632 frames with about 0.022 m mean and 0.074 m max object error. The final
  MuJoCo mapping still crosses the default 0.1 m threshold near frame 303. A
  relaxed-threshold diagnostic run completed 632 frames, but reached about
  0.282 m during the lift because thumb/index contact was lost.

The diagnostic overrides `--hand-self-collision-mode`,
`--object-position-terminal-threshold`, and `--object-friction` are explicit.
Their defaults preserve the current task semantics; relaxing the terminal
threshold is not an acceptance result.

The DexHand021Pro hand and default `cube1` object assets are included in the
repository under `assets/isaac_source_root/`. To use an external asset bundle
or add more objects, set `DEXHANDRL_ISAAC_SOURCE_ROOT` or run the import helper.

```text
assets/isaac_source_root/assets/all_assets/Assets/...
```

Set `DEXHANDRL_ISAAC_SOURCE_ROOT` or pass `--isaac-source-root` to override the
asset root if needed. `assets/README.md` documents the source commit and the
minimal bundle/import workflow.

To refresh the local cache from the IsaacGym checkout, run:

```bash
bash scripts/import_isaac_assets.sh
```

## Notes

- MJX-Warp contact points are raw `_impl.contact__pos`, not projected onto the ball or hand mesh.
- MJX-Warp internal `_impl` fields are useful but private/unstable compared with stable MuJoCo CPU APIs.
- `outputs/` is intentionally ignored and contains local generated artifacts.
- Default runtime artifacts are written to `outputs/`; set `DEXHANDRL_OUTPUT_ROOT`
  or `OUTPUT_ROOT` when you intentionally want a separate runtime directory.
