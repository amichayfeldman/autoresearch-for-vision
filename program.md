# Program.md-First CV Autoresearch

This repo is intentionally small. The human edits this file, opens Codex,
Claude, Cursor, or another coding agent in the repo root, and says: "Follow
`program.md`."

There is no separate manager system. `train.py` is the program. The agent edits
`train.py`, runs it, reads the metrics it writes, and iterates.

## Task

Replace this section with the concrete computer-vision task:

- Data:
- Labels:
- Model or checkpoint:
- Model library/source:
- External repo path, if any:
- Model import path, if any:
- External training entrypoint, if any:
- Primary metric:
- Minimum acceptable precision:
- Minimum acceptable recall:
- Training budget:

## Phase 1: CV Setup

Wire the task inside `train.py` before running autonomous experiments.

Optional setup prompt assets:

- `agents/cv-wiring-agent.md`
- `skills/cv-task-bridge/SKILL.md`
- `skills/cv-metric-bridge/SKILL.md`

Allowed setup work:

- Replace the fallback synthetic dataset with the real dataset loader.
- Import, configure, or load the task model in `build_model()`. In CV this is
  usually a library model, pretrained checkpoint, detector, segmenter, embedding
  model, or other preconfigured artifact, not a from-scratch model.
- If the model comes from an installed Python package, wire `MODEL_IMPORT` to
  the package factory or class, such as `some_package.models:create_model`.
- If the model/training source is another local repo, inspect that repo and wire
  `EXTERNAL_REPO_PATH` plus any needed `MODEL_IMPORT`, `TRAINING_BACKEND`,
  `EXTERNAL_TRAIN_IMPORT`, and `EXTERNAL_PREDICT_IMPORT` in `train.py`.
- Use PyTorch Lightning for training when this repo owns the loop. If the
  connected repo already owns training, set `TRAINING_BACKEND = "external"` and
  adapt to its entrypoint, such as `model.train(...)`, `trainer.fit(...)`, or a
  project-specific train function.
- Use Albumentations for image preprocessing and augmentation.
- Define how predictions and labels map to task-specific precision and recall.
- Keep result logging in `outputs/<run_name>/`.

During setup, the agent may search the internet, inspect installed package docs,
and inspect connected repos to bridge the user task definition to concrete data
schemas, model output formats, training entrypoints, and metric calculation.
Prefer primary sources such as dataset docs, official model docs, benchmark
papers, evaluation scripts, maintained library docs, and the source code of the
connected repo. Record any source that changes the metric or training contract.

Setup verification command:

```bash
pip install -e .
python train.py
```

The setup phase is accepted only when `pretrain_metrics.json` and `metrics.json`
contain numeric `precision` and `recall`.

## Phase 2: Baseline

Run the first accepted baseline:

```bash
python train.py
```

Record:

- command
- `outputs/<run_name>/metrics.json`
- `outputs/<run_name>/checkpoint.json`
- one-sentence baseline note

## Phase 3: Experiment Loop

For each autonomous research iteration:

1. Read `program.md` and the latest `outputs/<run_name>/run.json`.
2. Choose exactly one intended change.
3. Edit only `train.py`.
4. Run `python train.py`.
5. Compare precision, recall, and the primary metric against the current best.
6. Keep the change only if it satisfies the acceptance rules.
7. Record the result in this file or in a short note next to the output.

Good one-change levers include:

- augmentation choice or strength
- learning rate
- epochs
- threshold, if the metric contract exposes one
- model source, checkpoint, head, external train entrypoint, or capacity
- regularization
- loss weighting

## Edit Rules

- The agent may edit `program.md` only to append human-readable result notes.
- The agent may edit `train.py` for setup and experiments.
- The agent may read `agents/` and `skills/` as prompt guidance, but these files
  must remain plain Markdown guidance and not runtime orchestration.
- Do not create a hidden manager, CLI, Hydra config tree, package under `src`,
  or internal agent orchestration system.
- Do not edit generated outputs to make a result look better.

## Result Note Format

```text
Run:
Command:
Change:
Precision:
Recall:
F1:
Decision:
Next:
```

## Acceptance Rules

- Precision and recall must be numeric for every accepted run.
- A baseline is accepted only after the CV setup path is wired in `train.py`.
- An experiment is accepted only if it changes one intended lever.
- Promote a result only when the primary metric improves and any minimum
  precision/recall constraints still pass.
