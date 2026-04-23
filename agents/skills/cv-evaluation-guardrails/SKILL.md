---
name: cv-evaluation-guardrails
description: Preserve immutable task metrics and evaluator boundaries in cv-autoresearch.
---

# CV Evaluation Guardrails

Use this skill when wiring or reviewing `cv-autoresearch` evaluation behavior.

The first agent run may modify task-specific evaluation wiring. That run is not
itself the evaluation phase: it prepares code so the manager can run a
pre-training verification gate. Promotion uses `evaluation.primary_metric`.

## Required Metrics

- Every task must report numeric `precision` and `recall`.
- Report `f1` when meaningful. If omitted, the runtime may derive it from precision and recall.
- Metric extraction must match the user's task prompt and model output shape.

## Forbidden to Agents

Agents must not edit:

- `src/cv_autoresearch/engine/manager/`
- `src/cv_autoresearch/engine/history/`
- `agents/skills/`
- tests, unless explicitly requested
- generated outputs, prior history, checkpoints, and baseline artifacts
- `pyproject.toml` and packaging metadata unless explicitly required

If evaluation behavior changes, add focused synthetic fixtures for the prompt-wired metric contract.
