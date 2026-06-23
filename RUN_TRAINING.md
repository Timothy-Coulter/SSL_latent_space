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
If checkpointing is enabled, checkpoints are saved under:

- `results/<experiment>/<algorithm>-<timestamp>/checkpoints/`

The best checkpoint is `best.pt` (when `[checkpoint] save_best = true`).
TensorBoard logs are written separately to `runs/` (see above).

## 6) Common adjustments

- Enable early stopping + best checkpoint:
  - set `[early_stopping] enabled = true`
  - set `[early_stopping] patience = <N>`
  - set `[checkpoint] save_best = true`
- Reduce CPU dataloader overhead (e.g. for debugging):
  - set `num_workers = 0`
- Force CPU:
  - set `[loop] device = "cpu"`
- Limit runtime without changing epochs:
  - set `[loop] steps_per_epoch = <N>`

## 7) Fine-tune a pretrained SSL backbone (downstream classification)

Fine-tuning uses `src.downstream` and expects a pretrained SSL checkpoint from `results/.../checkpoints/`.

### Step 1: Train SSL and locate a checkpoint

After an SSL run, use one of:

- `results/<experiment>/<algorithm>-<timestamp>/checkpoints/best.pt` (if `[checkpoint] save_best = true`)
- `results/<experiment>/<algorithm>-<timestamp>/checkpoints/final.pt` (always written when `[checkpoint] save_final = true`)

### Step 2: Create a fine-tuning config

Example `configs/finetune_cifar10.toml`:

```toml
pretrained_checkpoint = "results/simclr_cifar10/simclr-YYYYMMDD-HHMMSS/checkpoints/best.pt"
num_classes = 10

[data]
dataset = "cifar10"
root = "datasets/data"
download = false
batch_size = 128
num_workers = 4
image_size = 32

[loop]
epochs = 5
seed = 0
device = "cuda"
log_every_n_steps = 50
verbose = 1
early_stopping_enabled = true
early_stopping_patience = 5
early_stopping_min_delta = 0.0

[optim]
lr = 0.001
weight_decay = 0.0001

[logging]
enabled = true
experiment = "finetune_cifar10"
results_dir = "results"
tensorboard_dir = "runs"
save_best_checkpoint = true
```

### Step 3: Run fine-tuning

```bash
python -m src.downstream.cli --config configs/finetune_cifar10_simclr_best.toml
```

### Fine-tuning outputs

Each fine-tune run directory contains:

- `config.toml` / `config.json`: the exact config used
- `metrics.json`: includes `best_acc` and `final_acc`
- `final_finetuned.pt`: always written
- `best_finetuned.pt`: written when `[logging] save_best_checkpoint = true`
