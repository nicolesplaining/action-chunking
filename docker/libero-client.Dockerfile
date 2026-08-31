# Derived from the pinned OpenPI examples/libero/Dockerfile at commit 215abfb.
# The only dependency-resolution change is an explicit build constraint for
# Python 3.8 source packages; runtime requirements remain upstream-identical.
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04@sha256:2d913b09e6be8387e1a10976933642c73c840c0b735f0bf3c28d97fc9bc422e0
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

RUN apt-get update && \
    apt-get install -y \
    make \
    g++ \
    clang \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libegl1 \
    libglew-dev \
    libglfw3-dev \
    libgles2-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6

WORKDIR /app/third_party/openpi
ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/.venv

COPY third_party/openpi/examples/libero/requirements.txt /tmp/requirements.txt
COPY third_party/openpi/third_party/libero/requirements.txt /tmp/requirements-libero.txt
COPY third_party/openpi/packages/openpi-client/pyproject.toml /tmp/openpi-client/pyproject.toml
COPY constraints/libero-build.txt /tmp/libero-build.txt

RUN uv venv --python 3.8 $UV_PROJECT_ENVIRONMENT
RUN uv pip sync \
    /tmp/requirements.txt \
    /tmp/requirements-libero.txt \
    /tmp/openpi-client/pyproject.toml \
    --build-constraint /tmp/libero-build.txt \
    --extra-index-url https://download.pytorch.org/whl/cu113 \
    --index-strategy=unsafe-best-match

ENV PYTHONPATH=/app/third_party/openpi:/app/third_party/openpi/packages/openpi-client/src:/app/third_party/openpi/third_party/libero
ENV LIBERO_CONFIG_PATH=/tmp/libero
RUN mkdir -p /tmp/libero && cat <<'EOF' > /tmp/libero/config.yaml
benchmark_root: /app/third_party/openpi/third_party/libero/libero/libero
bddl_files: /app/third_party/openpi/third_party/libero/libero/libero/bddl_files
init_states: /app/third_party/openpi/third_party/libero/libero/libero/init_files
datasets: /app/third_party/openpi/third_party/libero/libero/datasets
assets: /app/third_party/openpi/third_party/libero/libero/libero/assets
EOF

RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    echo '{"file_format_version" : "1.0.0", "ICD" : { "library_path" : "libEGL_nvidia.so.0" }}' \
    > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

CMD ["/bin/bash", "-c", "source /.venv/bin/activate && python examples/libero/main.py $CLIENT_ARGS"]
