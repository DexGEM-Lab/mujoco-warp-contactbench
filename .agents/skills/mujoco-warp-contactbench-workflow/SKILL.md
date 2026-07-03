# MuJoCo Warp ContactBench Workflow

Use this skill when working in this repository, especially for simulation export, Docker/local environment changes, generated data handling, or repository restructuring.

## Project Direction

- The supported simulation/export workflow is direct MJX-Warp simulation to Lance export.
- MJX-Warp is the only backend. Do not reintroduce the old MuJoCo CPU backend or the old JSON intermediate export chain.
- CPU support still matters: CPU and GPU are device choices under the same MJX-Warp/Warp path.
- Preserve `--device cpu|gpu` semantics in Python CLIs and `DEVICE=cpu|gpu` semantics in shell scripts.
- Generated artifacts belong under `outputs/`, not `logs/`.
- `lance_manager` lives under `3rd_party/lance_manager`.
- Source code should live in normal repo packages such as `sim/`; do not recreate `docker/mujoco_mjx/` as a source package.

## Protect Generated Lance Data

- Do not casually delete `.lance` datasets or other generated Lance outputs.
- Treat `outputs/*.lance/` as potentially useful user artifacts even when ignored by git.
- Before removing, replacing, or regenerating Lance data, ask the user or make the destructive behavior explicit.
- It is acceptable for scripts to replace an output only when the user explicitly passes `--replace` or sets an output path for that run.
- Do not clean `outputs/` as a routine final step. If test artifacts were created, report their paths instead of deleting them unless the user asked for cleanup.

## Docker Rules

- The Dockerfile belongs at repository root as `Dockerfile`.
- The image should be based on plain Debian, not `nvidia/cuda`.
- Install Miniforge first.
- Install CUDA packages through Miniforge/conda.
- Install apt system packages after conda CUDA.
- Use `uv` from Miniforge to build the Python environment.
- Use BuildKit cache mounts on heavyweight `RUN` layers.
- Keep stable base `ARG`/`ENV` values near the top; keep frequently changed runtime env vars late to avoid invalidating heavy layers.
- Do not `COPY .` into the image. Runtime scripts should mount the source workspace.
- Keep concise comments explaining Dockerfile layers.
- Preserve the JAX CUDA library precedence fix: uv-installed NVIDIA wheel library directories should appear before `/opt/conda/lib` in `LD_LIBRARY_PATH`.

## Local uv Environment

- Keep local `uv` usable for development and CPU smoke checks.
- Keep `pyproject.toml` and `uv.lock` synchronized when dependencies change.
- Local GPU execution is not the default supported path; Docker is the supported GPU path unless the user asks otherwise.

## Verification Expectations

For changes touching simulation/export/environment behavior, prefer these checks when feasible:

```bash
.venv/bin/python -m py_compile sim/scene.py sim/warp_contact_export.py sim/sim_to_lance.py sim/smoke_test.py tools/export_contactbench_lance.py
scripts/build_docker.sh
DEVICE=cpu scripts/run_smoke_test.sh
DEVICE=gpu scripts/run_smoke_test.sh
```

For a short direct export check:

```bash
DEVICE=cpu OUTPUT=outputs/test_cpu.lance scripts/run_sim_lance.sh --scenario sparse_debug --ball-count 8 --duration-seconds 0.2 --settle-frames 2
DEVICE=gpu OUTPUT=outputs/test_gpu.lance scripts/run_sim_lance.sh --scenario sparse_debug --ball-count 8 --duration-seconds 0.2 --settle-frames 2
```

Do not delete the generated Lance outputs after verification unless the user asks for cleanup.

## Stale Reference Checks

After restructuring or workflow edits, check for stale references such as:

```bash
rg -n "docker/mujoco_mjx|docker\\.mujoco_mjx|logs/|\\blogs\\b|mujoco_cpu|run_cpu_export|run_mjx_warp|export_contacts|steady_state|mjx_warp_ball_pit_video|--backend|BACKEND=|mjx_warp_gpu|_export_gpu_contacts|GPU ContactBench|setup_viz" . -g '!outputs/**' -g '!*.lance' -g '!uv.lock'
```

Expected result is no stale hits, except when intentionally discussing historical context.
