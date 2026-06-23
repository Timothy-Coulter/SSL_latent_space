FROM nvcr.io/nvidia/pytorch:25.08-py3

# Avoid interactive prompts during package installs
ARG DEBIAN_FRONTEND=noninteractive

# Non-root user for devcontainers
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000

# System setup: essentials and common tools only (use Python/PyTorch from base image)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       sudo \
       ca-certificates \
       curl \
       git \
       bash-completion \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user matching common devcontainer defaults
RUN set -eux; \
    target_user=${USERNAME}; target_uid=${USER_UID}; target_gid=${USER_GID}; \
    if getent group ${target_gid} >/dev/null; then \
        existing_group=$(getent group ${target_gid} | cut -d: -f1); \
        if [ "$existing_group" != "$target_user" ]; then groupmod -n "$target_user" "$existing_group"; fi; \
    else \
        groupadd --gid ${target_gid} "$target_user"; \
    fi; \
    if getent passwd ${target_uid} >/dev/null; then \
        existing_user=$(getent passwd ${target_uid} | cut -d: -f1); \
        if [ "$existing_user" != "$target_user" ]; then \
            usermod -l "$target_user" "$existing_user"; \
            usermod -d "/home/$target_user" -m "$target_user" || true; \
            usermod -g ${target_gid} "$target_user"; \
        fi; \
    elif id -u "$target_user" >/dev/null 2>&1; then \
        usermod -u ${target_uid} -g ${target_gid} "$target_user"; \
    else \
        useradd --uid ${target_uid} --gid ${target_gid} -m "$target_user" -s /bin/bash; \
    fi; \
    echo "$target_user ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$target_user; \
    chmod 0440 /etc/sudoers.d/$target_user

# Use Python and PyTorch that ship with the base image (likely via /opt/conda)

# Install uv (fast Python package/deps manager) — pinned version for reproducibility
ENV UV_INSTALL_DIR=/usr/local/bin
# See https://docs.astral.sh/uv/getting-started/installation/ for install flags
ENV UV_VERSION=v0.4.27
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_VERSION=${UV_VERSION} sh

# Performance: centralize caches and set common tool env vars
ENV HOME=/home/${USERNAME} \
    PIP_CACHE_DIR=/home/${USERNAME}/.cache/pip \
    UV_CACHE_DIR=/home/${USERNAME}/.cache/uv \
    HF_HOME=/home/${USERNAME}/.cache/huggingface \
    TORCH_HOME=/home/${USERNAME}/.cache/torch \
    PYTHONUNBUFFERED=1 \
    # Prefer the system interpreter from PATH (Conda python)
    UV_PYTHON=python

# Pre-create cache dirs and hand over ownership to the dev user
RUN mkdir -p "$PIP_CACHE_DIR" "$UV_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" \
    && chown -R ${USERNAME}:${USERNAME} /home/${USERNAME}/.cache

# Working directory
WORKDIR /workspaces

# Set default user and workdir for devcontainer usage
USER ${USERNAME}
WORKDIR /workspaces

# Helpful defaults inside the container
ENV PATH=/home/${USERNAME}/.local/bin:$PATH

# Pre-warm Python environment with uv at the end of the build
# This leverages Docker layer caching to speed up first container start.
# Note: The devcontainer mounts caches as named volumes at runtime; this
# primarily benefits image build caching and may still reduce sync time.
WORKDIR /tmp/project-bootstrap
COPY --chown=${USERNAME}:${USERNAME} pyproject.toml uv.lock ./
RUN uv venv --system-site-packages \
    && uv sync --extra dev

# Return to the default workspace
WORKDIR /workspaces

# nvidia-container-toolkit will inject GPUs when run with `--gpus all`.
CMD ["bash"]
