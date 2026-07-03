FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    libegl1 \
    libgl1 \
    libglew2.2 \
    libglvnd0 \
    libglx0 \
    libosmesa6 \
    libx11-6 \
    libxext6 \
    python-is-python3 \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel
RUN python -m pip install --no-cache-dir \
    "numpy" \
    "imageio" \
    "imageio-ffmpeg" \
    "mujoco>=3.10" \
    "mujoco-mjx[warp]>=3.10" \
    "warp-lang" \
    "jax[cuda12]>=0.4.35"

WORKDIR /workspace/mujoco-warp-contactbench
COPY . /workspace/mujoco-warp-contactbench

CMD ["python", "docker/mujoco_mjx/smoke_test.py", "--strict"]
