# syntax=docker/dockerfile:1.7
FROM debian:bookworm-slim

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Stable build parameters only. Keep frequently changed runtime knobs later so
# Docker can reuse the heavy Miniforge/CUDA/apt/uv dependency layers.
ARG MINIFORGE_VERSION=25.3.1-0
ARG CUDA_VERSION=12.4
ARG PYTHON_VERSION=3.11

# Base install paths used by early layers.
ENV DEBIAN_FRONTEND=noninteractive \
    CONDA_DIR=/opt/conda \
    CONDA_PKGS_DIRS=/tmp/conda-pkgs

# Fetch Miniforge with BuildKit ADD so no apt download tools are needed before
# the conda bootstrap layer.
ADD https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh /tmp/miniforge.sh

# Install Miniforge and configure conda-forge once. This should be one of the
# most stable layers in the image.
RUN --mount=type=cache,target=/root/.cache/miniforge,sharing=locked \
    bash /tmp/miniforge.sh -b -p "${CONDA_DIR}" && \
    rm -f /tmp/miniforge.sh && \
    "${CONDA_DIR}/bin/conda" config --system --set auto_update_conda false && \
    "${CONDA_DIR}/bin/conda" config --system --set channel_priority strict && \
    "${CONDA_DIR}/bin/conda" config --system --add channels conda-forge

# Install CUDA from conda before system packages. The conda package cache keeps
# large CUDA artifacts reusable across rebuilds.
RUN --mount=type=cache,target=/tmp/conda-pkgs,sharing=locked \
    "${CONDA_DIR}/bin/conda" install --copy -y -n base \
      "cuda-version=${CUDA_VERSION}" \
      "cuda-toolkit=${CUDA_VERSION}"

# Install Debian runtime libraries needed by MuJoCo/EGL/OpenGL and git for
# submodule-aware workflows. Apt indexes and packages use BuildKit caches.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      git \
      libegl1 \
      libgl1 \
      libglew2.2 \
      libglfw3 \
      libglvnd0 \
      libglx0 \
      libice6 \
      libosmesa6 \
      libsm6 \
      libx11-6 \
      libxcursor1 \
      libxext6 \
      libxfixes3 \
      libxi6 \
      libxinerama1 \
      libxkbcommon0 \
      libxrandr2 \
      libxrender1 \
      libxxf86vm1 \
      x11-apps

# Install only Python tooling in conda. Project dependencies go into the uv
# virtualenv below, so changing Python deps does not require re-solving CUDA.
RUN --mount=type=cache,target=/tmp/conda-pkgs,sharing=locked \
    "${CONDA_DIR}/bin/conda" install --copy -y -n base \
      "python=${PYTHON_VERSION}" \
      "uv" && \
    "${CONDA_DIR}/bin/python" --version && \
    "${CONDA_DIR}/bin/uv" --version

WORKDIR /opt/mujoco-warp-contactbench

# Copy only dependency manifests before uv sync. Source changes later should not
# invalidate the dependency installation layer.
COPY pyproject.toml uv.lock ./

ENV UV_CACHE_DIR=/root/.cache/uv \
    UV_LINK_MODE=copy

# Build the project Python environment with uv. The second install adds GPU sim
# packages that are intentionally kept out of the local visualization pyproject.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    "${CONDA_DIR}/bin/uv" sync --frozen --extra dexhandrl --python "${CONDA_DIR}/bin/python" --no-install-project && \
    "${CONDA_DIR}/bin/uv" pip install --python .venv/bin/python \
      "imageio" \
      "imageio-ffmpeg" \
      "mujoco>=3.10" \
      "mujoco-mjx[warp]>=3.10" \
      "warp-lang" \
      "jax[cuda12]>=0.4.35"

# Runtime environment goes late so tweaks to rendering/GPU env vars do not
# invalidate the heavy dependency layers above. Put uv-installed NVIDIA wheel
# libraries before conda CUDA libs so JAX uses the CUDA runtime it was packaged with.
ENV VENV_DIR=/opt/mujoco-warp-contactbench/.venv \
    PY_SITE=/opt/mujoco-warp-contactbench/.venv/lib/python${PYTHON_VERSION}/site-packages

ENV PATH=${VENV_DIR}/bin:${CONDA_DIR}/bin:${PATH} \
    LD_LIBRARY_PATH=${PY_SITE}/nvidia/cublas/lib:${PY_SITE}/nvidia/cuda_cupti/lib:${PY_SITE}/nvidia/cuda_nvcc/lib:${PY_SITE}/nvidia/cuda_nvrtc/lib:${PY_SITE}/nvidia/cuda_runtime/lib:${PY_SITE}/nvidia/cudnn/lib:${PY_SITE}/nvidia/cufft/lib:${PY_SITE}/nvidia/cusolver/lib:${PY_SITE}/nvidia/cusparse/lib:${PY_SITE}/nvidia/nccl/lib:${PY_SITE}/nvidia/nvjitlink/lib:${PY_SITE}/nvidia/nvshmem/lib:${CONDA_DIR}/lib:${LD_LIBRARY_PATH} \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# Source code is not copied into the image. Runtime scripts mount the workspace at
# /workspace/mujoco-warp-contactbench so code edits do not require image rebuilds.
CMD ["python", "sim/smoke_test.py", "--strict"]
