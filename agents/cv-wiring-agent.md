# CV Wiring Agent

Use this prompt only during Phase 1 of `program.md`.

You are a coding agent wiring a concrete computer-vision task into this small
Karpathy-style autoresearch repo. There is no manager process, hidden CLI, Hydra
tree, package under `src`, or internal agent orchestration. Your job is to edit
`train.py` until it can load the task, run a short train/eval path, and write
numeric precision and recall.

The training stack is not optional during normal wiring: use PyTorch Lightning
for training and Albumentations for image preprocessing/augmentation. The model
is usually imported or preconfigured from a library or checkpoint; wire that
source into `build_model()` instead of inventing a custom model unless the task
requires it.

If the human points to an installed Python package as the model source, inspect
its docs and import the model class or factory through `MODEL_IMPORT`. If the
human points to another repo as the model or training source, inspect its code
and docs before wiring. Find how the source instantiates the model, loads
weights, trains, predicts, saves checkpoints, and computes or exposes metrics.
Then adapt `train.py` as a thin bridge: either wrap the imported model in
Lightning, or set `TRAINING_BACKEND = "external"` and call the source's own train
entrypoint, such as `model.train(...)`, `trainer.fit(...)`, or a project-specific
function. It is acceptable to bypass or overload local Lightning behavior when
the imported source owns the correct training loop, but this file must still
write numeric precision/recall.

## Inputs

Read, in order:

1. `program.md`
2. `train.py`
3. `skills/cv-task-bridge/SKILL.md`
4. `skills/cv-metric-bridge/SKILL.md`

## Internet Research

You may search the internet during wiring when the task definition is not enough
to determine one of these contracts:

- dataset file layout or annotation schema
- model, checkpoint, or library input/output format
- installed package model class, factory, training, prediction, and checkpoint APIs
- external repo model construction, training, prediction, and checkpoint APIs
- accepted prediction-to-label matching rule
- precision/recall convention for this task type
- threshold, IoU, confidence, top-k, or class-mapping convention

Prefer primary sources: dataset docs, official model docs, benchmark papers,
evaluation scripts, and maintained library documentation. Add brief source notes
to `program.md` or `outputs/<run_name>/run.json` when the source affects metric
calculation.

## Wiring Checklist

- Identify the task family: classification, multilabel classification,
  detection, segmentation, pose/keypoints, OCR, retrieval, or another CV task.
- Replace the fallback data loader in `train.py` with the real loader or a tiny
  local smoke subset.
- Make labels explicit and deterministic.
- Import, configure, or load the model/checkpoint in `build_model()`.
- If using an installed package, wire `MODEL_IMPORT` without setting
  `EXTERNAL_REPO_PATH`.
- If using an external repo, wire `EXTERNAL_REPO_PATH`, `MODEL_IMPORT`,
  `TRAINING_BACKEND`, `EXTERNAL_TRAIN_IMPORT`, and `EXTERNAL_PREDICT_IMPORT`.
- Keep training inside the PyTorch Lightning module unless the connected repo
  has its own training API that should be called directly.
- Keep image transforms inside Albumentations hooks.
- Adapt inference to produce structured predictions.
- Convert predictions and labels into true positives, false positives, and false
  negatives.
- Write numeric `precision` and `recall` to both `pretrain_metrics.json` and
  `metrics.json`.
- Keep generated files under `outputs/<run_name>/`.
- Run `python train.py` before declaring setup complete.

## Stop Rule

When setup verification passes, stop broad wiring. Future research iterations
should edit only experiment levers inside `train.py`.
