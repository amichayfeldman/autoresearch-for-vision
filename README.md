# cv-autoresearch

Agent-managed computer-vision experimentation.

`cv-autoresearch` runs a baseline, gives an external agent the latest metrics and
bounded edit surface, lets the agent make exactly one intended training change,
then trains and promotes the result only when the primary metric improves.

There is no Optuna loop and no explore/exploit mode. The agent owns the next
iteration decision: it may change a config value, a training hyperparameter, data
handling, evaluation code, or task wiring, as long as the changed files are in
the configured editable path list.

## How It Works

```text
task prompt + optional model path
        |
        v
task-wiring agent edits allowed wiring/evaluation files
        |
        v
pre-train verification checks precision and recall are available
        |
        v
baseline training run
        |
        v
for each iteration:
  agent receives history, metrics, baseline, editable paths, forbidden paths
  agent makes one intended change in allowed files
  manager rejects forbidden or unrelated changes
  training runs with the changed code/config
  result is promoted only if the primary metric improves
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Claude CLI or another configured agent command

The default agent command is configured in `configs/agent/default.yaml`.

## Installation

```bash
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

## Repository Layout

```text
src/cv_autoresearch/engine/  Current agent-managed training engine
configs/                     Hydra configs for data, model, training, agent, history
agents/skills/               Agent instructions and edit-boundary definitions
tests/                       Pytest suite
examples/                    Notes for running example configs
```

## Quick Start

Run the managed iteration loop with the default prompt-task config:

```bash
cv-autoresearch run \
  --task-prompt "Classify images and improve macro F1." \
  iteration.max_iterations=3
```

Use a prompt file when the task description is longer:

```bash
cv-autoresearch run \
  --task-prompt-file ./task.md \
  --model-path ./model.pt \
  iteration.max_iterations=5 \
  history.output_dir=outputs/my_run
```

Trailing arguments are passed through as Hydra overrides.

You can also call the engine directly:

```python
from hydra import compose, initialize_config_dir

from cv_autoresearch import manage_iterations

with initialize_config_dir(version_base=None, config_dir="/path/to/autoresearch-for-vision/configs"):
    config = compose(
        config_name="prompt_task",
        overrides=["iteration.max_iterations=3"],
    )

records = manage_iterations(config, repo_root="/path/to/autoresearch-for-vision")
```

## Edit Boundaries

Allowed and forbidden paths are part of the agent config:

```yaml
agent:
  editable_paths:
    - src/cv_autoresearch/engine/training/
    - src/cv_autoresearch/engine/data/
    - src/cv_autoresearch/engine/models/
    - src/cv_autoresearch/engine/losses/
    - src/cv_autoresearch/engine/evaluation/
    - src/cv_autoresearch/engine/task_wiring/
    - configs/model/
    - configs/data/
    - configs/evaluation/
    - configs/augmentations/
    - configs/optimizer/
    - configs/scheduler/
    - configs/trainer/
  forbidden_paths:
    - src/cv_autoresearch/engine/manager/
    - src/cv_autoresearch/engine/history/
    - agents/skills/
    - tests/
```

The manager records each iteration under `history.output_dir`, including the
changed files, patch, metrics, checkpoint path, promotion status, and summary.

## Useful Commands

```bash
pytest
cv-autoresearch-train iteration.max_epochs=1
cv-autoresearch-iterations iteration.max_iterations=3
```
