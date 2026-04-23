"""Agent command integration."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AgentResult:
    """Result from invoking the external agent."""

    returncode: int
    stdout: str
    stderr: str

    @property
    def text(self) -> str:
        return "\n".join(part for part in [self.stdout, self.stderr] if part).strip()


def build_agent_prompt(
    *,
    history_text: str,
    baseline: dict[str, Any],
    previous_insight: str,
    metrics: dict[str, float],
    editable_paths: list[str],
    forbidden_paths: list[str],
) -> str:
    """Create the prompt for a one-change training iteration."""
    return (
        "You are running one CV training iteration. Make exactly one intended training change.\n"
        "Use the cv-training-iteration skill.\n"
        f"Current baseline: {baseline}\n"
        f"Latest metrics: {metrics}\n"
        f"Previous insight: {previous_insight}\n"
        f"History:\n{history_text}\n"
        f"Editable paths: {editable_paths}\n"
        f"Forbidden paths: {forbidden_paths}\n"
        "Respond with: intended change, files edited, reason, expected metric effect, post-result insight."
    )


def build_task_wiring_prompt(
    *,
    task_prompt: str,
    model_path: str | None,
    editable_paths: list[str],
    forbidden_paths: list[str],
) -> str:
    """Create the initial prompt that wires the user task before training."""
    model_text = model_path or "No model path supplied."
    return (
        "You are wiring a computer-vision task for cv-autoresearch before training starts.\n"
        "This agent run is a code-wiring phase, not an evaluation phase and not a training phase.\n"
        "Use the cv-task-wiring and cv-evaluation-guardrails skills.\n"
        "Modify the code as needed to load data, load or define the model, run inference, "
        "extract predictions/targets, and compute task-specific metrics.\n"
        "Do not score or compare the task yourself. After your wiring, the manager will run a "
        "verification gate that only checks numeric precision and recall are available. "
        "Return f1 too when meaningful, or make it derivable from precision and recall.\n"
        f"Task description:\n{task_prompt}\n"
        f"Optional .pt model path: {model_text}\n"
        f"Editable paths: {editable_paths}\n"
        f"Forbidden paths: {forbidden_paths}\n"
        "Respond with: wiring summary, files edited, metric extraction approach, expected risks."
    )


def run_agent(command: str, prompt: str, timeout_seconds: int) -> AgentResult:
    """Run the configured agent command with the prompt on stdin."""
    if not command.strip():
        return AgentResult(returncode=0, stdout="No agent command configured.", stderr="")
    completed = subprocess.run(
        command.split(),
        input=prompt,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    return AgentResult(completed.returncode, completed.stdout, completed.stderr)
