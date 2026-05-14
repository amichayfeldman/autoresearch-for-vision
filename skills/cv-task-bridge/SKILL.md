# CV Task Bridge

Use this skill during Phase 1 setup when the task section in `program.md` must be
translated into concrete code inside `train.py`.

## Goal

Turn a human CV task description into four explicit contracts:

- data contract: where examples come from and how each example is read
- label contract: what counts as ground truth
- model contract: which installed library, local repo, or checkpoint supplies
  the model and what it needs
- training contract: whether this file uses Lightning or delegates to another
  repo's training entrypoint
- prediction contract: what the imported or preconfigured model returns
- run contract: what files are written under `outputs/<run_name>/`

## Procedure

1. Classify the task family.
2. Find the smallest runnable dataset path first: a small split, a few local
   images, a fixture, or a synthetic adapter that matches the real schema.
3. Make labels boring and inspectable: class ids, booleans, boxes, masks,
   keypoints, or text strings.
4. Keep image preprocessing and augmentation in the Albumentations hooks.
5. Put the imported library model or checkpoint loader in `build_model()`.
6. If an installed package owns the model, wire its import path or factory in
   `train.py`.
7. If another repo owns the model or training loop, inspect that repo and wire
   the repo path, import path, model factory, train function, and predict
   function in `train.py`.
8. Keep training inside the PyTorch Lightning module unless the connected source
   has the correct training API; then use `TRAINING_BACKEND = "external"` and
   bridge to that API.
9. Keep inference output structured enough that metrics do not parse display
   strings.
10. Preserve `python train.py` as the only required command.

## When To Search

Search the internet if the task references a public dataset, benchmark, model,
checkpoint, installed package, or external repo and the local files do not define
the schema or entrypoint.

Prefer official dataset documentation, benchmark evaluation code, papers, or
official model/library docs before blog posts.

Record any source that changes code behavior in a short note, including what it
settled, such as "COCO boxes are xywh in annotation files" or "model logits map
to class ids from labels.txt" or "the external repo trains through
package.runner:train".
