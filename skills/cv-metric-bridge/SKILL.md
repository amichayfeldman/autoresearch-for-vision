# CV Metric Bridge

Use this skill during Phase 1 setup when model outputs must be converted into
numeric precision and recall.

## Goal

Every accepted setup run must write numeric `precision` and `recall` in:

- `outputs/<run_name>/pretrain_metrics.json`
- `outputs/<run_name>/metrics.json`

## Metric Contract

Start from the structured outputs produced by the imported model, checkpoint, or
external repo prediction API, not from trainer logs. PyTorch Lightning often
owns optimization and logging, but the task metric bridge should still be
explicit code that maps predictions and labels into counts. If training is
delegated to another repo, write an adapter that calls its prediction API and
returns the same metric inputs.

Implement metrics from counts, not from formatted logs:

- true positives: predictions that correctly match ground truth
- false positives: predictions that do not match ground truth
- false negatives: ground-truth items the model missed

Then compute:

```text
precision = tp / (tp + fp) when tp + fp > 0 else 0.0
recall = tp / (tp + fn) when tp + fn > 0 else 0.0
```

If the task convention defines a different empty-set behavior, document it in
`program.md` and keep the values numeric.

## Task Families

- Classification: match predicted class to label; expose threshold only for
  binary or multilabel tasks.
- Detection: match predicted boxes to ground-truth boxes by class and IoU
  threshold; each ground-truth item may match at most one prediction.
- Segmentation: choose pixel, mask, or instance matching before coding; document
  the selected convention.
- Pose/keypoints: choose visibility and distance/OKS-style matching before
  coding; document the selected convention.
- OCR or text-in-image: define exact, normalized, or edit-distance matching
  before coding.

## When To Search

Search the internet when the benchmark, dataset, or model family has a standard
metric bridge. Prefer official evaluation scripts and benchmark docs. Keep the
implementation minimal enough for a smoke run, but align the setup metric with
the task's accepted convention whenever possible.
