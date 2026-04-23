# cv-autoresearch Prompt-First Example

Run a single configured training execution after task wiring is available:

```bash
python run_training.py --config-name examples/prompt_task --task-prompt-file task.md
```

Run the full wiring plus baseline/iteration manager:

```bash
python manage_iterations.py --config-name examples/prompt_task --task-prompt "Classify scratched vs clean parts" --model-path model.pt
```

The first agent call is a wiring phase, not an evaluation or training iteration.
It adapts the task-specific code so the manager can verify numeric precision and
recall before training starts. That verification gate is only a contract check;
baseline scoring starts with iteration 0. Hydra remains responsible for runtime
controls such as output directory, agent command, trainer limits, and iteration
budget.
