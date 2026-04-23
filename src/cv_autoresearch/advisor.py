"""Lean Claude-directed search advisor.

Replaces search_director.py — signature updated to accept TaskDef directly,
prompt updated to reference macro F1 as the fixed metric.
"""

from __future__ import annotations

import subprocess
from typing import Any

from cv_autoresearch.search.space import PARAM_REGISTRY
from cv_autoresearch.task import TaskDef
from cv_autoresearch.types import Directive, SearchMode

# First parameter in the registry — used as the default fallback target.
_FALLBACK_PARAM: str = next(iter(PARAM_REGISTRY))


def get_next_directive(
    task: TaskDef,
    history: Any,  # SearchHistory — imported lazily to avoid circular deps
    best_f1: float,
) -> Directive:
    """Ask claude -p what to do next in the search.

    Args:
        task: TaskDef with description and num_classes.
        history: SearchHistory with to_text() method.
        best_f1: Current best macro F1 score.

    Returns:
        Directive with EXPLORE or EXPLOIT mode and a required target_param.
    """
    prompt = _build_prompt(task, history, best_f1)
    try:
        return _parse_directive(_call_claude(prompt))
    except Exception:
        return _explore_fallback()


def _build_prompt(task: TaskDef, history: Any, best_f1: float) -> str:
    """Build the directive prompt for claude -p.

    Args:
        task: TaskDef with description and num_classes.
        history: SearchHistory with to_text() method.
        best_f1: Current best macro F1 score.

    Returns:
        Formatted prompt string.
    """
    history_text = history.to_text(max_entries=20) if hasattr(history, "to_text") else str(history)
    available_params = ", ".join(PARAM_REGISTRY.keys())
    return f"""You are directing a CV hyperparameter/augmentation search.

Task: {task.description} ({task.num_classes} classes)
Metric: macro F1 (fixed). Best so far = {best_f1:.4f}
Available params: {available_params}

History (most recent first):
{history_text}

Choose ONE action for the next experiment:

1. EXPLORE: Sample a parameter from its full default range (good when history is short or diverse exploration needed)
2. EXPLOIT: Target a parameter with a narrower custom range (good when a promising direction is identified)

You MUST always choose a parameter. Reply in EXACTLY this format:
MODE: EXPLORE|EXPLOIT
PARAM: <parameter_name>
RANGE: <low,high or NONE>
REASON: <one sentence>
"""


def _parse_directive(response: str) -> Directive:
    """Parse Claude's structured response into a Directive.

    Args:
        response: Raw string response from claude.

    Returns:
        Parsed Directive, or EXPLORE fallback on any parse error.
    """
    try:
        lines = {
            line.split(":", 1)[0].strip().upper(): line.split(":", 1)[1].strip()
            for line in response.strip().splitlines()
            if ":" in line
        }
        if not lines:
            return _explore_fallback()

        mode_str = lines.get("MODE", "EXPLORE").upper()
        mode = SearchMode.EXPLORE if mode_str == "EXPLORE" else SearchMode.EXPLOIT

        param_str = lines.get("PARAM", "").strip()
        if not param_str or param_str.upper() == "NONE":
            return _explore_fallback()
        target_param = param_str

        range_str = lines.get("RANGE", "NONE").strip()
        target_range: list[float] | None = None
        if range_str.upper() != "NONE":
            parts = [p.strip() for p in range_str.split(",")]
            try:
                target_range = [float(p) for p in parts if p]
            except ValueError:
                target_range = None

        reason = lines.get("REASON", "No reason provided.")

        return Directive(
            mode=mode,
            target_param=target_param,
            target_range=target_range,
            reason=reason,
        )
    except Exception:
        return _explore_fallback()


def _explore_fallback() -> Directive:
    """Return a safe EXPLORE directive with the first PARAM_REGISTRY param.

    Returns:
        EXPLORE Directive targeting the default fallback parameter.
    """
    return Directive(
        mode=SearchMode.EXPLORE,
        target_param=_FALLBACK_PARAM,
        target_range=None,
        reason="Fallback to EXPLORE due to parse failure.",
    )


def _call_claude(prompt: str) -> str:
    """Run `claude -p <prompt>` and return stdout.

    Args:
        prompt: The prompt to pass to claude.

    Returns:
        stdout from claude as a string.

    Raises:
        RuntimeError: If claude -p returns a non-zero exit code.
    """
    result = subprocess.run(
        ["claude", "-p", prompt],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"claude -p failed: {result.stderr}")
    return result.stdout.strip()
