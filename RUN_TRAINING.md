# Running SSL Training

This repo includes a minimal self-supervised learning (SSL) training loop with selectable algorithms:
SimCLR, MoCo, BYOL, SwAV, and VICReg.

Training is driven by a TOML config file in `configs/`, writes artifacts to `results/`, and logs to
TensorBoard.

## 1) Prerequisites

- Use the devcontainer (recommended) or an environment that matches `pyproject.toml`.
- Ensure datasets exist under `datasets/data`:
  - CIFAR-10: `datasets/data/cifar-10-batches-py/`
  - CIFAR-100: `datasets/data/cifar-100-python/`
  - STL-10: `datasets/data/stl10_binary/`

The default training configs set `download = false` to avoid network access. If you want torchvision
to download missing data, set `download = true` in your config.

## 2) Pick a config

Smoke configs (tiny CPU runs):

- `configs/simclr_fake_smoke.toml`
- `configs/moco_fake_smoke.toml`
- `configs/byol_fake_smoke.toml`
- `configs/swav_fake_smoke.toml`
- `configs/vicreg_fake_smoke.toml`

Training configs (intended for longer runs):

- `configs/simclr_cifar10_train.toml`
- `configs/moco_cifar10_train.toml`
- `configs/byol_cifar10_train.toml`
- `configs/vicreg_cifar10_train.toml`
- `configs/swav_stl10_train.toml`

## 3) Run training

From the repo root:

```bash
python -m src.training.cli --config configs/simclr_cifar10_train.toml
```

### Verbose console logging

Set:

```toml
[loop]
verbose = 1
log_every_n_steps = 50
```

With `verbose = 1`, the trainer prints `epoch`, `step`, and `loss` every `log_every_n_steps`.

## 4) TensorBoard

Logs are written under:

- `runs/<experiment>/<algorithm>-<timestamp>/`
  
You can disable TensorBoard logging by setting:

```toml
[logging]
enabled = false
```

Start TensorBoard:

```bash
tensorboard --logdir runs
```

Then open the shown URL (usually `http://localhost:6006`).

## 5) Output files

Each run directory contains:

- `config.toml`: a copy of the exact config used for the run
- `config.json`: the parsed config (useful for programmatic inspection)
TensorBoard logs are written separately to `runs/` (see above).

## 6) Common adjustments

- Reduce CPU dataloader overhead (e.g. for debugging):
  - set `num_workers = 0`
- Force CPU:
  - set `[loop] device = "cpu"`
- Limit runtime without changing epochs:
  - set `[loop] steps_per_epoch = <N>`
