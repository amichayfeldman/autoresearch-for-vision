---
name: cv-training-iteration
description: Choose, sample, and generate exactly one evidence-based next training change for cv-autoresearch managed iterations after task wiring and baseline training exist.
---

# CV Training Iteration

Use this skill when acting as the external agent for a post-baseline
`manage_iterations.py` training iteration. The task is already wired. The
manager owns orchestration, history, baseline promotion, and verification. The
agent owns one scoped train-time improvement at a time.

## Contract

Treat the prompt as a request for the next single training change. Use the
current baseline, latest metrics, previous insight, and history text as evidence.
Do not edit task wiring again unless the latest failure clearly shows the
existing wiring blocks training or metric extraction.

Pick one lever, make the smallest coherent code/config edit for that lever, and
leave other promising ideas for future iterations. A change that touches multiple
files can still be one change only when the files are required for the same lever
(for example, adding one augmentation option plus its config).

Every change must be wired end-to-end through the training pipeline. If the
lever changes code behavior, update any required config/registry/initialization
so the manager run actually uses the new behavior. Example: if adding a model
layer, also update the model build/compile path so training instantiates the
updated architecture.

Allowed change levers:

- data processing
- model architecture
- losses
- augmentations
- hyperparameters
- scheduler behavior
- optimizer settings
- train-time parameters

## Choosing The Next Change

Use this metric-driven policy:

- Optimize explore-exploit tradeoff with an exploration bias by default.
- Prefer exploration (new high-signal lever not yet tried) over exploitation
  (small local refinement of the current best idea), unless recent evidence
  strongly supports a clear exploit step.
- Practical target: spend most iterations exploring (about 70%) and fewer
  iterations exploiting (about 30%), then rebalance only when history shows a
  stable winning direction.

- Precision below recall: sample changes that reduce false positives, such as
  stricter thresholds, stronger regularization, class weighting for negative
  errors, or cleaner augmentations.
- Recall below precision: sample changes that reduce false negatives, such as
  class weighting for positives, milder augmentations, longer training, or lower
  decision thresholds when task wiring supports thresholds.
- Both precision and recall low: sample representation or optimization changes,
  such as a modest model-capacity increase, learning-rate adjustment, optimizer
  change, scheduler choice, or data normalization fix.
- Training failure, NaN, OOM, or shape error: generate the minimal
  training-surface fix that addresses the failure, and describe it as the one
  intended change.
- Flat metrics across history: sample one high-signal search dimension that has
  not just been tried; do not repeat a reverted or non-improving idea without a
  new reason from history.

Prefer changes with a clear expected metric direction and low blast radius. If
two ideas look equally promising, choose the one that edits fewer files and is
easier for the manager to attribute to the next metric result.

Never bundle an architecture change with optimizer/scheduler/data changes in the
same iteration. Never edit generated outputs, checkpoints, baseline state, or
history to make a result look better.

## Example Single-Change Iterations

Use this short format:

- `Change:` one lever only.
- `Files:` exact paths.
- `Why:` one sentence.
- `Expected:` metric direction.
- `Insight:` what to try next if it works/fails.

Examples:

- `Change:` Increase head dropout `0.20 -> 0.35`. `Files:` `configs/model/tiny_cnn.yaml`. `Why:` reduce overconfident false positives. `Expected:` precision up, recall may dip. `Insight:` if recall drops too much, tune threshold next.
- `Change:` Increase positive loss weight `1.0 -> 1.4`. `Files:` `configs/loss/cross_entropy.yaml`. `Why:` penalize false negatives more. `Expected:` recall up, precision may dip. `Insight:` if precision collapses, reduce the weight change.
- `Change:` Add `RandomErasing(p=0.15)`. `Files:` `configs/augmentations/train.yaml`. `Why:` improve occlusion robustness. `Expected:` recall/PR-AUC up on cluttered cases. `Insight:` if clean validation drops, lower `p`.
- `Change:` Add `GaussianBlur(blur_limit=(3,5), p=0.20)`. `Files:` `configs/augmentations/blur.yaml`. `Why:` robustness to blur/focus noise. `Expected:` blurry-sample recall up. `Insight:` if sharp-image metrics drop, reduce blur strength/probability.
- `Change:` Add `HorizontalFlip(p=0.5)`. `Files:` `configs/augmentations/hflip.yaml`. `Why:` left-right invariance. `Expected:` recall/F1 up on mirrored views. `Insight:` if gains are narrow, keep and tune sampling later.
- `Change:` Add `VerticalFlip(p=0.2)`. `Files:` `configs/augmentations/vflip.yaml`. `Why:` top-bottom invariance (when valid). `Expected:` recall up on inverted views. `Insight:` if global metrics fall, revert.
- `Change:` Add `Affine(scale=(0.95,1.05), rotate=(-10,10), translate_percent=(-0.03,0.03), p=0.35)`. `Files:` `configs/augmentations/affine.yaml`. `Why:` geometric robustness. `Expected:` recall/PR-AUC up on shifted/rotated samples. `Insight:` if masks/bboxes degrade, tighten ranges.
- `Change:` Add one extra head layer and wire model compile/build to use it. `Files:` `src/cv_autoresearch/engine/models/simple.py`, `configs/model/tiny_cnn.yaml`. `Why:` increase head capacity. `Expected:` precision/recall up if underfitting. `Insight:` if train up but val flat, revert and prefer regularization.
- `Change:` Reduce LR `3e-4 -> 1e-4`. `Files:` `configs/optimizer/adamw.yaml`. `Why:` stabilize noisy updates. `Expected:` smoother validation, possible F1 gain. `Insight:` if too slow without gain, restore LR.
- `Change:` Switch `AdamW -> SGD(momentum=0.9, nesterov=true)`. `Files:` `configs/config.yaml`. `Why:` test a stronger generalization regime. `Expected:` possible precision/generalization gain after enough epochs. `Insight:` if persistently worse, keep optimizer fixed and tune LR/scheduler.

## Albumentations Explore List (Snapshot)

Source (accessed April 23, 2026): `https://explore.albumentations.ai/`

Image-only transforms (extracted list):

- `AdditiveNoise`, `AdvancedBlur`, `AtmosphericFog`, `AutoContrast`, `Blur`,
  `CLAHE`, `ChannelDropout`, `ChannelShuffle`, `ChannelSwap`,
  `ChromaticAberration`, `ColorJitter`, `Colorize`, `Defocus`, `Dithering`,
  `Downscale`, `Emboss`, `Enhance`, `Equalize`, `FDA`, `FancyPCA`,
  `FilmGrain`, `FromFloat`, `GaussNoise`, `GaussianBlur`, `GlassBlur`,
  `HEStain`, `Halftone`, `HistogramMatching`, `HueSaturationValue`,
  `ISONoise`, `Illumination`, `ImageCompression`, `InvertImg`, `LensFlare`,
  `MedianBlur`, `ModeFilter`, `MotionBlur`, `MultiplicativeNoise`,
  `Normalize`, `PhotoMetricDistort`, `PixelDistributionAdaptation`,
  `PlanckianJitter`, `PlasmaBrightnessContrast`, `PlasmaShadow`, `Posterize`,
  `RGBShift`, `RandomBrightnessContrast`, `RandomFog`, `RandomGamma`,
  `RandomGravel`, `RandomRain`, `RandomShadow`, `RandomSnow`,
  `RandomSunFlare`, `RandomToneCurve`, `RingingOvershoot`,
  `SaltAndPepper`, `Sharpen`, `ShotNoise`, `Solarize`, `Spatter`,
  `Superpixels`, `TextImage`, `ToFloat`, `ToGray`, `ToRGB`, `ToSepia`,
  `UnsharpMask`, `Vignetting`, `ZoomBlur`.

Dual transforms (extracted list):

- `Affine`, `AtLeastOneBBoxRandomCrop`, `BBoxSafeRandomCrop`, `CenterCrop`,
  `CoarseDropout`, `ConstrainedCoarseDropout`, `CopyAndPaste`, `Crop`,
  `CropAndPad`, `CropNonEmptyMaskIfExists`, `D4`, `ElasticTransform`,
  `Erasing`, `FrequencyMasking`, `GridDistortion`, `GridDropout`,
  `GridElasticDeform`, `GridMask`, `HorizontalFlip`, `LetterBox`,
  `LongestMaxSize`, `MaskDropout`, `Morphological`, `Mosaic`, `NoOp`,
  `OpticalDistortion`, `OverlayElements`, `Pad`, `PadIfNeeded`, `Perspective`,
  `PiecewiseAffine`, `PixelDropout`, `PixelSpread`, `RandomCrop`,
  `RandomCropFromBorders`, `RandomCropNearBBox`, `RandomGridShuffle`,
  `RandomResizedCrop`, `RandomRotate90`, `RandomScale`,
  `RandomSizedBBoxSafeCrop`, `RandomSizedCrop`, `Resize`, `Rotate`,
  `SafeRotate`, `ShiftScaleRotate`, `SmallestMaxSize`, `SquareSymmetry`,
  `ThinPlateSpline`, `TimeMasking`, `TimeReverse`, `Transpose`,
  `VerticalFlip`, `WaterRefraction`, `XYMasking`.

3D transforms (extracted list):

- `CenterCrop3D`, `CoarseDropout3D`, `CubicSymmetry`, `GridShuffle3D`,
  `Pad3D`, `PadIfNeeded3D`, `RandomCrop3D`.

## Response Format

Your response must include:

- Exactly one intended change
- Files edited
- Reason it should help
- Expected metric effect
- Post-result insight

## Allowed Paths

- `src/cv_autoresearch/engine/training/`
- `src/cv_autoresearch/engine/data/`
- `src/cv_autoresearch/engine/models/`
- `src/cv_autoresearch/engine/losses/`
- `src/cv_autoresearch/engine/evaluation/`
- `src/cv_autoresearch/engine/task_wiring/`
- `configs/model/`
- `configs/loss/`
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

Do not edit manager or history records. Do not touch baseline artifacts. If an
iteration seems to need multiple unrelated edits, choose the single most likely
training improvement and leave the rest for later iterations.

Before editing, inspect the latest prompt, history, and baseline metrics. After
editing, do not run long training loops unless explicitly asked; the manager will
run the authoritative training/evaluation pass. Prefer quick static checks or
focused unit checks only when they fit inside the editable/requested scope. At
minimum, verify the edited pipeline path is reachable (imports/build/compile
path resolves) so the manager run cannot silently skip the intended change.
