# MuJoCo Warp ContactBench

Standalone MJX-Warp sim-to-Lance export repo.

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

The local uv environment is useful for CPU checks and development. GPU execution is supported through the Docker image by default.

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

`docker-compose.yml` forwards the host X11 socket and an xauth cookie into the container so GUI tools such as `mujoco.viewer` can open on the host display. From an active X11 desktop/session:

```bash
scripts/run_x11_viewer.sh
```

By default this opens the 3x3x3 puzzle cube asset at `.tmp/mujoco_cube/cube_3x3x3.xml`. To open another MJCF model:

```bash
MODEL_XML=path/to/model.xml scripts/run_x11_viewer.sh
```

The host needs `xauth` installed. The script writes the temporary cookie file to `.tmp/docker.xauth` and runs the compose service as the current UID/GID.

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

## Notes

- MJX-Warp contact points are raw `_impl.contact__pos`, not projected onto the ball or hand mesh.
- MJX-Warp internal `_impl` fields are useful but private/unstable compared with stable MuJoCo CPU APIs.
- `outputs/` is intentionally ignored and contains local generated artifacts.
