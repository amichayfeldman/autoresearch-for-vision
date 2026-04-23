---
name: cv-task-wiring
description: Wire a prompt-defined computer-vision task before training so cv-autoresearch can build the task, run inference, and report numeric precision and recall.
---

# CV Task Wiring

Use this skill for the first external agent run from `manage_iterations.py`.
This is a task-wiring phase, not an evaluation phase and not a training
iteration. The manager runs the verification gate after your edits.

## Contract

Modify only the editable task/data/model/evaluation surfaces needed to make the
user's prompt-defined CV task executable. The goal is to make the runtime
contract work for the supplied task prompt and optional `.pt` model path.

Wire the runtime contract so:

- `build_task(config)` creates the datamodule and Lightning module for the task.
- `pretrain_evaluate(config)` returns numeric `precision` and `recall`.
- `evaluate_after_training(task, trainer, max_epochs)` returns final metrics and
  epoch metrics after training.
- `f1` is returned when meaningful, or can be derived from precision and recall.

Metric extraction must match the task prompt and model output shape. For
classification-like tasks, define how predictions and targets map to positives,
classes, or labels before computing precision and recall. For non-classification
CV tasks, adapt prediction/target extraction into a task-appropriate precision
and recall contract without changing manager semantics.

Do not score, compare, or promote results yourself. The pre-training gate only
proves that task metrics are available.

## Response Format

Your response must include:

- Wiring summary
- Files edited
- Metric extraction approach for precision and recall
- Any expected risks

## Allowed Paths

- `src/cv_autoresearch/engine/training/`
- `src/cv_autoresearch/engine/data/`
- `src/cv_autoresearch/engine/models/`
- `src/cv_autoresearch/engine/losses/`
- `src/cv_autoresearch/engine/evaluation/`
- `src/cv_autoresearch/engine/task_wiring/`
- `configs/model/`
- `configs/data/`
- `configs/evaluation/`
- `configs/augmentations/`
- `configs/optimizer/`
- `configs/scheduler/`
- `configs/trainer/`
- `configs/examples/`

## Forbidden Paths

- `src/cv_autoresearch/engine/manager/`
- `src/cv_autoresearch/engine/history/`
- `agents/skills/`
- tests, unless explicitly requested
- generated outputs, prior history, checkpoints, and baseline artifacts
- `pyproject.toml` and packaging metadata unless explicitly required

## Guardrails

Do not edit manager or history records. Do not touch baseline artifacts. Keep
wiring changes focused on making the task executable and measurable. Leave
training improvements for `cv-training-iteration` after the baseline exists.

After editing, do not run long training loops unless explicitly asked; the
manager will run the authoritative pre-training verification and later
training/evaluation passes. Prefer quick static checks or focused unit checks
only when they fit inside the editable/requested scope.
