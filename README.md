# MuJoCo Warp ContactBench

Standalone MuJoCo CPU / MJX-Warp GPU contact export and Rerun visualization repo.

This repo is intentionally decoupled from the larger `contactbench` workspace. The MANO hand asset and `lance_manager` are git submodules. The repo contains MuJoCo scene builders, CPU and MJX-Warp GPU export scripts, sample outputs, Lance export glue, and a Rerun notebook.

## Layout

```text
assets/mano_hand_s02/        submodule: MANO MJCF/URDF/STL assets used by simulation and replay
benchmarks/ball_pit/         deterministic ball-pit scenario and camera helpers
common/                      contact JSON schema helpers and validators
lance_manager/               submodule: full generated_data Lance writer/schema stack
docker/mujoco_mjx/           Dockerfile and MuJoCo/MJX-Warp simulation scripts
notebooks/                   Rerun SDK visualization notebook
scripts/                     build, export, and notebook helper scripts
data/sample_logs/            checked-in sample CPU/GPU contact outputs for immediate visualization
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
lance_manager       -> git@192.168.10.116:ai/group-dexcanvas/lance_manager.git
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

The notebook first looks in `logs/` for freshly generated outputs. If those files do not exist, it uses `data/sample_logs/`, so the visualization works before rerunning simulation.

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

## CPU MuJoCo Contact Export

```bash
scripts/run_cpu_export.sh
```

Outputs:

```text
logs/mujoco_cpu_ball_pit_contact_10s.json
logs/mujoco_cpu_ball_pit_contact_10s.svg
```

The CPU path exports raw MuJoCo `data.contact[i].pos` contact points and CPU `mj_contactForce` force values.

## MJX-Warp GPU Contact Export

Use a visible GPU device. Example for host GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 scripts/run_mjx_warp_export.sh
```

Outputs:

```text
logs/mjx_warp_gpu_ball_pit_contact_10s.json
logs/mjx_warp_gpu_ball_pit_contact_native_10s.json
logs/mjx_warp_gpu_contact_to_hand_mesh_error_raw.json
logs/mjx_warp_gpu_contact_error_scene.xml
```

The GPU path reads MJX-Warp native contact buffers from `Data._impl`, including contact position, geom ids, contact distance, contact frame, `efc_address`, and `efc__force` rows. The ContactBench-compatible GPU JSON stores a normal-force proxy, not a CPU `mj_contactForce` 6D wrench.

## Lance Export

After CPU/GPU JSON is generated, or using the included sample JSON fallback:

```bash
scripts/setup_viz_env.sh
scripts/export_lance.sh
```

Output:

```text
logs/mujoco_cpu_mjx_warp_gpu_ball_pit_generated.lance
```

The exporter uses `lance_manager/schema/schemas/generated_data_schema.jsonc` from the submodule and writes with the submodule's configured Lance writer.

## Optional MJX-Warp Video Rollout

```bash
CUDA_VISIBLE_DEVICES=0 scripts/run_mjx_warp_video.sh
```

Outputs:

```text
logs/mjx_warp_ball_pit_rollout.json
logs/mjx_warp_ball_pit.mp4
logs/mjx_warp_ball_pit_scene.xml
```

## Smoke Test

```bash
scripts/run_smoke_test.sh
```

This validates MuJoCo availability and writes a small contact fixture to `logs/mujoco_mjx_live_contact.json`.

## Notes

- CPU contact points are raw MuJoCo solver contact positions, not projected onto ball surfaces.
- GPU contact points are raw MJX-Warp `_impl.contact__pos`, not projected onto the ball or hand mesh.
- MJX-Warp internal `_impl` fields are useful but private/unstable compared with the stable MuJoCo CPU `data.contact` API.
- `logs/` is intentionally ignored; use `data/sample_logs/` for portable sample artifacts.
