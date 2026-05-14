# cv-autoresearch

Karpathy-style autoresearch adapted to computer vision.

There is no manager package, Hydra config tree, hidden CLI, or internal agent
orchestrator. The workflow is intentionally small:

```text
human edits program.md
        |
        v
human opens Codex/Claude/Cursor in this repo
        |
        v
agent follows program.md and edits train.py
        |
        v
python train.py writes metrics and artifacts
```

## Files

```text
program.md  The human-readable research protocol and task spec.
train.py    The complete CV setup, training, evaluation, and logging surface.
agents/     Optional prompt files for coding agents. Not runtime code.
skills/     Optional setup skills for CV task and metric wiring. Not runtime code.
tests/      Small contract tests for the Markdown protocol and train script.
```

## Install

```bash
pip install -e .
```

## Run

```bash
python train.py
```

The default task is a tiny deterministic CV fallback: classify whether an image
contains a bright square. During Phase 1, the agent replaces the dataset/model or
feature hooks in `train.py` for the real CV task. After precision and recall are
numeric, normal research iterations keep editing `train.py` one lever at a time.

`train.py` is intentionally built on the usual CV research stack: PyTorch
Lightning owns the training loop, Albumentations owns image augmentation, and the
model hook assumes the real task will usually import a preconfigured model or
load a checkpoint from a library.

When the model comes from an installed Python package, `train.py` can import the
model class or factory directly through `MODEL_IMPORT`. When the model and
training code live in another repo, `train.py` should become a thin adapter
around that repo: add the external checkout to the import path, instantiate the
model, and either wrap it in Lightning or set `TRAINING_BACKEND = "external"` to
call the source repo's own training and prediction entrypoints. Metrics still
flow back through this repo as numeric precision and recall.

During Phase 1, the human may point the coding agent at
`agents/cv-wiring-agent.md` and the setup skills under `skills/`. These files are
plain prompt guidance for bridging the user task definition to concrete CV
inputs, outputs, and metric calculation. They may instruct the agent to search
the internet for task-specific metric conventions, dataset schemas, or model
output formats, but they do not invoke agents or run orchestration themselves.

Outputs are written under `outputs/baseline/` by default:

- `pretrain_metrics.json`
- `metrics.json`
- `run.json`
- `checkpoint.json`

## Develop

```bash
python -m unittest discover -s tests
```
