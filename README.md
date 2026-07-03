# MuJoCo Warp ContactBench

Standalone MuJoCo CPU / MJX-Warp GPU sim-to-Lance export and Rerun visualization repo.

This repo is intentionally decoupled from the larger `contactbench` workspace. The MANO hand asset and `lance_manager` are git submodules. The repo contains MuJoCo scene builders, direct CPU and MJX-Warp GPU Lance export, and a Rerun notebook.

## Layout

```text
assets/mano_hand_s02/        submodule: MANO MJCF/URDF/STL assets used by simulation and replay
benchmarks/ball_pit/         deterministic ball-pit scenario and camera helpers
common/                      contact JSON schema helpers and validators
3rd_party/lance_manager/     submodule: full generated_data Lance writer/schema stack
docker/mujoco_mjx/           Dockerfile and MuJoCo/MJX-Warp simulation scripts
notebooks/                   Rerun SDK visualization notebook
scripts/                     build, sim-to-Lance, smoke test, and notebook helper scripts
logs/                        generated outputs; ignored by git
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

## Quick Start: Visualization Only

```bash
cd /path/to/mujoco-warp-contactbench
git submodule update --init --recursive
scripts/setup_viz_env.sh
scripts/open_rerun_notebook.sh
```

In Jupyter/VS Code, open:

```text
notebooks/rerun_cpu_gpu_contacts.ipynb
```

Use kernel:

```text
MuJoCo Warp ContactBench (.venv)
```

The notebook reads generated outputs from `logs/`. Run the JSON debug export scripts first if you want to visualize CPU/GPU contact JSON outputs in the notebook.

Default Rerun behavior:

```text
- non-contacting balls are hidden
- current-frame contacting balls are highlighted
- current-frame contact points are highlighted
- MANO hand URDF meshes are animated
- no Rerun summary pane is logged
```

## Build Docker Image

```bash
scripts/build_docker.sh
```

This creates:

```text
mujoco-warp-contactbench:latest
```

## Sim-to-Lance Export

Run simulation and write Lance directly, without contact JSON intermediates:

```bash
CUDA_VISIBLE_DEVICES=0 scripts/run_sim_lance.sh
```

Output:

```text
logs/mujoco_cpu_mjx_warp_gpu_ball_pit_generated.lance
```

Use `BACKEND=cpu` or `BACKEND=gpu` to export only one backend. Extra simulation arguments are forwarded to `docker/mujoco_mjx/sim_to_lance.py`, for example:

```bash
BACKEND=cpu scripts/run_sim_lance.sh --duration-seconds 1 --ball-count 16
```

The exporter uses `3rd_party/lance_manager/schema/schemas/generated_data_schema.jsonc` from the submodule and writes with the submodule's configured Lance writer.

## Smoke Test

```bash
scripts/run_smoke_test.sh
```

This validates MuJoCo availability and writes a small contact fixture to `logs/mujoco_mjx_live_contact.json`.

## Notes

- CPU contact points are raw MuJoCo solver contact positions, not projected onto ball surfaces.
- GPU contact points are raw MJX-Warp `_impl.contact__pos`, not projected onto the ball or hand mesh.
- MJX-Warp internal `_impl` fields are useful but private/unstable compared with the stable MuJoCo CPU `data.contact` API.
- `logs/` is intentionally ignored and contains local generated artifacts.
