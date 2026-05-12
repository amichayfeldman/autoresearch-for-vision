# Task Prompt Examples

The old `TaskDef` script examples were removed with the Optuna search stack.
Current runs are driven by Hydra configs and a task prompt consumed by the
task-wiring agent.

Use this directory for task prompt snippets or notes that can be passed to:

```bash
cv-autoresearch run --task-prompt-file examples/task_definitions/my_task.md
```

The agent may edit only the paths configured in `configs/agent/default.yaml`.
Per-iteration changes are recorded under `history.output_dir`.
