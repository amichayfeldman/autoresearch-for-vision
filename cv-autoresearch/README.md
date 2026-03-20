# cv-autoresearch

AI-directed automated hyperparameter and augmentation search for computer vision models. Claude (`claude -p`) acts as the search director at every iteration, deciding whether to EXPLORE the full search space or EXPLOIT a promising direction.

## How It Works

```
task_description
       │
       ▼
Claude generates torchmetrics metric config (once at startup)
       │
       ▼
┌─────────────────────────────────────────────────┐
│  PHASE 1: Hyperparameter Search (hp_trials)     │
│                                                 │
│  for each trial:                                │
│    Claude → EXPLORE or EXPLOIT?                 │
│    Optuna samples hyperparams                   │
│    PyTorch Lightning trains model               │
│    Evaluator computes primary metric            │
│    Baseline updated if improved                 │
└─────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│  PHASE 2: Augmentation Search (aug_trials)      │
│  (same flow, hyperparams fixed to best found)   │
└─────────────────────────────────────────────────┘
       │
       ▼
  best config + summary dict
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Claude CLI (`claude -p`) installed and authenticated

## Installation

```bash
pip install -e .
```

For development (tests, linting):
```bash
pip install -e ".[dev]"
```

## Quick Start — Python API

```python
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision

from cv_autoresearch import run_autoresearch, SearchConfig


class CIFAR10Classifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.resnet18(weights=None)
        self.backbone.fc = torch.nn.Linear(512, 10)

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        return F.cross_entropy(self(x), y)

    def configure_optimizers(self):
        pass  # Overridden by AutoResearchModule per trial


import torchvision.transforms as T
base_transform = T.Compose([T.ToTensor()])
train_ds = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=base_transform)
val_ds   = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=base_transform)

config = SearchConfig(
    task_description="10-class image classification on CIFAR-10. Optimize top-1 accuracy.",
    primary_metric="accuracy",
    higher_is_better=True,
    hp_trials=30,
    aug_trials=20,
    epochs_per_trial=5,
)

result = run_autoresearch(CIFAR10Classifier(), train_ds, val_ds, config)
print(f"Best accuracy: {result['best_metric']['value']:.4f}")
```

See `examples/cifar10_classification.py` for the full runnable example.

## Quick Start — CLI

```bash
cv-autoresearch run \
  --trainer-module examples.cifar10_classification.CIFAR10Classifier \
  --train-dataset examples.cifar10_classification.get_datasets \
  --task "10-class image classification on CIFAR-10. Optimize top-1 accuracy." \
  --metric accuracy \
  --higher-is-better \
  --hp-trials 30 --aug-trials 20 --epochs 5 \
  --output results.json
```

## CLI Reference

### `cv-autoresearch run`

| Flag | Default | Description |
|------|---------|-------------|
| `--trainer-module` | required | Dotted path to LightningModule class |
| `--train-dataset` | required | Dotted path to factory function `() -> (train_ds, val_ds)` |
| `--task` | required | Free-text task description for Claude |
| `--metric` | required | Primary metric name (e.g. `accuracy`) |
| `--higher-is-better` / `--lower-is-better` | `--higher-is-better` | Optimization direction |
| `--hp-trials` | 50 | Total hyperparameter phase trials |
| `--aug-trials` | 30 | Total augmentation phase trials |
| `--epochs` | 10 | Training epochs per trial |
| `--device` | `cuda` | Compute device (`cuda`, `cpu`, `mps`) |
| `--output` | `./autoresearch_results.json` | Path for JSON results |
| `--storage` | `sqlite:///autoresearch.db` | Optuna storage URL |

### `cv-autoresearch history`

| Flag | Default | Description |
|------|---------|-------------|
| `--storage` | `sqlite:///autoresearch.db` | Optuna storage URL |
| `--top` | 10 | Number of top improvements to show |

### `cv-autoresearch resume`

| Flag | Default | Description |
|------|---------|-------------|
| `--storage` | `sqlite:///autoresearch.db` | Optuna storage URL |
| `--trainer-module` | required | Same as `run` |
| `--train-dataset` | required | Same as `run` |
| (other flags) | same as `run` | Passed through to SearchConfig |

## User Model Interface

Your `LightningModule` must implement:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Used by the system for validation inference."""
    ...

def training_step(self, batch, batch_idx) -> torch.Tensor:
    """Return a loss tensor."""
    ...

def configure_optimizers(self):
    pass  # Leave empty — AutoResearchModule overrides this per trial
```

## SearchConfig Fields

| Field | Default | Description |
|-------|---------|-------------|
| `task_description` | required | Free-text CV task description for Claude |
| `primary_metric` | required | Metric name to optimize |
| `higher_is_better` | required | True for accuracy/F1, False for loss/error |
| `hp_trials` | 50 | Total hyperparameter trials budget |
| `aug_trials` | 30 | Total augmentation trials budget |
| `epochs_per_trial` | 10 | Training epochs per trial |
| `exploit_trials_per_directive` | 5 | Trials per EXPLOIT directive |
| `optuna_storage` | `sqlite:///autoresearch.db` | Optuna DB URL |
| `optuna_seed` | 42 | Random seed for TPE sampler |
| `hp_overrides` | `{}` | Fixed overrides for specific HP params |
| `aug_overrides` | `{}` | Fixed overrides for specific aug params |
| `device` | `cuda` | Compute device (`cuda`, `cpu`, `mps`) |
| `num_workers` | 4 | DataLoader worker count |
| `checkpoint_dir` | `./checkpoints` | Model checkpoint directory |
| `metric_config_path` | `config/metrics/generated.yaml` | Where to write Claude-generated metric |

## Hyperparameter Search Space

| Parameter | Type | Range / Choices |
|-----------|------|-----------------|
| `learning_rate` | float (log) | 1e-5 → 1e-1 |
| `weight_decay` | float (log) | 1e-6 → 1e-2 |
| `batch_size` | categorical | 8, 16, 32, 64, 128, 256 |
| `optimizer_type` | categorical | adam, adamw, sgd |
| `lr_scheduler` | categorical | cosine, step, onecycle, cosine_with_restarts |
| `lr_scheduler_gamma` | float | 0.1 → 0.9 |
| `lr_scheduler_step_size` | int | 1 → 20 |
| `warmup_epochs` | int | 0 → 10 |
| `warmup_momentum` | float | 0.0 → 0.95 |
| `gradient_clip_val` | float | 0.1 → 10.0 |
| `label_smoothing` | float | 0.0 → 0.2 |
| `dropout_rate` | float | 0.0 → 0.7 |
| `mixed_precision` | categorical | True, False |
| `ema_decay` | float | 0.99 → 0.9999 |
| `momentum` *(SGD only)* | float | 0.8 → 0.99 |
| `beta1` *(Adam/AdamW only)* | float | 0.85 → 0.95 |
| `beta2` *(Adam/AdamW only)* | float | 0.99 → 0.9999 |

## Augmentation Search Space

15 Albumentations transforms, each independently enabled/disabled and parameterized:
HorizontalFlip, VerticalFlip, RandomBrightnessContrast, ColorJitter, GaussianBlur, GaussNoise, RandomResizedCrop, RandomScale, Rotate, Perspective, CoarseDropout, Sharpen, CLAHE, RandomGamma, Normalize.

## Error Handling

When a trial raises a `RuntimeError` (OOM, NaN loss, shape mismatch, etc.):
- The error is caught and recorded as a `FAILED` entry in history
- Optuna receives the worst possible objective value, deprioritizing that region
- The baseline is **not** updated
- Claude sees the error message in subsequent prompts and can avoid similar configs

```
trial=7 | phase=hyperparameter | mode=explore | status=failed | error=CUDA out of memory
```

## Resuming Runs

Optuna's `load_if_exists=True` means studies automatically continue from where they left off:

```bash
cv-autoresearch resume \
  --storage sqlite:///autoresearch.db \
  --trainer-module myproject.models.MyModel \
  --train-dataset myproject.data.get_datasets
```

## VLM Integration (Future)

Subclass `VLMHooks` to add vision-language model analysis at three hook points:

```python
from cv_autoresearch.vlm.hooks import VLMHooks

class MyVLMHooks(VLMHooks):
    def analyze_iteration(self, trial_result, history, baseline):
        # Called after every trial
        ...

    def analyze_phase(self, phase, study, baseline):
        # Called after each search phase
        ...

    def analyze_experiment(self, history, baseline):
        # Called at the end of the full experiment
        ...

result = run_autoresearch(model, train_ds, val_ds, config, vlm_hooks=MyVLMHooks())
```
