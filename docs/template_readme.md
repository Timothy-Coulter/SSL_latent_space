PyTorch DevContainer (CUDA) Template — Quickstart
================================================

This template provides a ready-to-use VS Code Dev Container for PyTorch on NVIDIA GPUs, based on the NGC image `nvcr.io/nvidia/pytorch:25.08-py3`. It installs development tooling with `uv` while using the system Python and preinstalled PyTorch/CUDA from the base image — PyTorch is never re-downloaded.

Prerequisites
-------------
- Docker 24+
- NVIDIA GPU drivers + NVIDIA Container Toolkit (for GPU access)
- Access to NGC: `docker login nvcr.io` with your NGC API key
- VS Code + Dev Containers extension (or GitHub Codespaces)
 - Optional: host `~/.env` file with API keys and env vars (see below)

- The following folders are created:
  - If running in WSL (use powershell):
    ```
    wsl -d docker-desktop sh -c '
    set -e
    mkdir -p /mnt/wsl/Ubuntu/data/caches/{torch,huggingface} \
              /mnt/wsl/Ubuntu/data/projects/pytorch-devcontainer-cuda-12.8/{data,datasets}
    ln -sfn /mnt/wsl/Ubuntu/data /data
    chown -R 1000:1000 /mnt/wsl/Ubuntu/data || true'
    ```

  - If running in Ubuntu:
    ```
    sudo mkdir -p /data/caches/{torch,huggingface} /data/projects/pytorch-devcontainer-cuda-12.8/{data,datasets}
    sudo chown -R "$USER:$USER" /data/caches /data/projects
    ```

  - Note: replace the repo name if required.

What you get
------------
- Base image: `nvcr.io/nvidia/pytorch:25.08-py3` (Python 3.12, Torch from NGC 25.08, CUDA as provided by the image)
- `uv` package manager, strict typing (mypy), ruff/black/isort
- High-performance bind mounts for torch/huggingface caches and project data; pip/uv caches stay in container volumes (works in WSL via /mnt/wsl/<distro>/data or native Linux via /data)
- GPU-enabled run args: `--gpus all --ipc host`
- JupyterLab auto-starts on port 8888 (no token)
- TensorBoard auto-starts on port 6006
- VizTracer installed and VS Code extension preloaded
- Ruff pydocstyle configured for Google docstrings

- Repo Layout
-------------
- `src/`, `tests/`, `examples/`, `docs/`, `misc/`
- `.devcontainer/` with Dockerfile + devcontainer.json
- `dev.sh` with commands: `format`, `lint`, `lint-fix`, `typecheck`, `test`, `all-checks`, `versions`

Open in Dev Container (recommended)
-----------------------------------
Option A — Open Local Folder in Container
1) Clone the repo locally.
2) Open the folder in VS Code.
3) Use: Command Palette → “Dev Containers: Reopen in Container”.

Option B — Clone Repo in a Container Volume
1) VS Code → Command Palette → “Dev Containers: Clone Repository in Container Volume…”.
2) Paste this repo URL.
3) VS Code opens it directly inside the container.

When the container starts, it automatically runs:
- `uv venv --system-site-packages` (so the venv inherits system packages)
- `uv sync --extra dev` (installs project + dev tooling, but never PyTorch)
- JupyterLab on `http://localhost:8888` (no token)
- TensorBoard on `http://localhost:6006` (logdir: `./runs`)

Build-time optimization:
- The Dockerfile now runs `uv venv --system-site-packages && uv sync --extra dev` at the end of the image build to pre-warm dependency resolution and downloads. Post-create still runs the same commands to ensure the workspace venv is ready on first boot.

Check versions:
- `./dev.sh versions`

GPU Access
----------
Run the dev container on a host with NVIDIA drivers and NVIDIA Container Toolkit. The devcontainer config includes `--gpus all --ipc host`. Inside the container, verify:

```
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

Expect `True` and CUDA version output if GPU is available.

Installing additional packages
------------------------------
- Preferred: `uv add <pkg>` then `uv sync` (already uses a venv inheriting system site-packages). If `<pkg>` depends on torch, it will detect the preinstalled torch — no re-download.
- System environment: `uv pip install --system <pkg>` (installs into the base image Python).

Note: Do not add `torch`, `torchvision`, or `torchaudio` to the project’s default dependencies — they come from the base image. This avoids reinstalling PyTorch.

Security
--------
- JupyterLab and TensorBoard start without auth tokens inside the container for convenience. They are only exposed via local port forwarding by VS Code. If you expose the container beyond localhost, enable tokens/passwords.

Data and Caches
---------------
- Host layout (WSL/Ubuntu): `/data/caches/{torch,huggingface}` and `/data/projects/<repo>/{data,datasets}`. For WSL, the Docker daemon sees these under `/mnt/wsl/$WSL_DISTRO_NAME/data/...`, so the binds point there. The `initializeCommand` creates both `/mnt/wsl/$WSL_DISTRO_NAME/data/...` and `/data/...` best-effort.
- Container mounts: pip/uv caches stay in named volumes; torch/huggingface caches bind to `/home/vscode/.cache/{torch,huggingface}` from the host; datasets bind to `/datasets`; project data binds to `/data`.
- Works without host binds: if the host paths are missing, you can still use repo-local `./datasets` and `./data` under `/workspaces/<repo>`. In code, use `os.path.isdir("/datasets")` to choose between `/datasets` and `./datasets`.
- WSL host prep (one-time): `sudo mkdir -p /data/caches/{torch,huggingface} /data/projects/<repo>/{data,datasets} && sudo chown -R "$USER:$USER" /data/caches /data/projects` (the Docker daemon consumes them via `/mnt/wsl/$WSL_DISTRO_NAME/data/...`).
- Native Linux host prep (one-time): ensure `/data/caches/{torch,huggingface}` and `/data/projects/<repo>/{data,datasets}` exist and are writable by your user.


Environment Variables (.env)
----------------------------
- If you have a `~/.env` on your host, the devcontainer will detect it and pass all variables into the container at start using Docker’s `--env-file`.
- Nothing to configure: it is copied (or an empty file is created) via `initializeCommand` to `.devcontainer/.env.devcontainer` and automatically applied on `docker run`.
- The file `.devcontainer/.env.devcontainer` is git-ignored.

Format and usage
- File path: `~/.env` on your host machine.
- Format: simple KEY=VALUE lines; blank lines and `#` comments allowed.
  - Example:
    - `OPENAI_API_KEY=sk-...`
    - `WANDB_API_KEY=...`
    - `HF_TOKEN=...`
- No `export` statements; quotes are preserved as part of the value; no shell expansion.
- In code and notebooks, read values from `os.environ["KEY"]` (already in the process env).

Updating variables
- If you change `~/.env`, run “Dev Containers: Rebuild Container” (or `devcontainer up --workspace-folder .`) to re-sync and apply changes.
- Alternatively, edit `.devcontainer/.env.devcontainer` directly in the workspace and rebuild to apply.

Notes and security
- `.devcontainer/.env.devcontainer` exists on disk inside the repo folder and is ignored by Git; avoid committing secrets elsewhere.
- For shared/team use, prefer per-user `~/.env` rather than committing any env files to the repo.

Developer Commands
------------------
- `./dev.sh format` — ruff format + black + isort
- `./dev.sh lint` — ruff check
- `./dev.sh lint-fix` — ruff check --fix
- `./dev.sh typecheck` — mypy (strict)
- `./dev.sh test` — pytest -n auto --reruns 2
- `./dev.sh all-checks` — format + lint-fix + typecheck + test
- `./dev.sh versions` — print Python / Torch / CUDA info

Using the Dev Container CLI (optional)
--------------------------------------
If you prefer the Dev Container CLI instead of VS Code UI:

- One-off via npx (no install):
  - `npx -y @devcontainers/cli build --workspace-folder .`
  - `npx -y @devcontainers/cli up --workspace-folder .`

The CLI requires Docker and access to `nvcr.io` (ensure `docker login nvcr.io`).

Profiling with VizTracer
------------------------
- VizTracer is included in the `dev` extra and the VS Code extension is preinstalled.
- From VS Code: use the VizTracer commands (e.g., “VizTracer: Start Tracing”).
- From terminal: `viztracer your_script.py` then open the generated report.

Docstrings Style
----------------
- Ruff’s pydocstyle rules are enabled with the Google convention. Write docstrings in Google style and run `./dev.sh lint` to check.

Notes
-----
- If you ever change the Python in the base image, update `requires-python` and tool target versions in `pyproject.toml`.
- If an added package hard-pins a different torch version, the resolver may try to fetch it. Keep torch out of project deps and rely on the preinstalled version.
- Compatibility pins: the NGC image bundles Torch built against NumPy 1.26.x. The project pins keep the scientific stack aligned: `numpy<2`, `scipy<1.16`, `pandas<2.3`, `scikit-learn<1.7`. This avoids ABI/runtime errors when importing Torch. When you upgrade to an image where Torch supports NumPy 2.x, relax these pins and run `uv lock && uv sync`.
- Tests live under `tests/` (configured via `pyproject.toml`).

Happy hacking!
