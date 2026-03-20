"""Claude-directed search strategy for EXPLORE/EXPLOIT decisions."""

from __future__ import annotations

import subprocess
from typing import Any

from cv_autoresearch.search.space import PARAM_REGISTRY
from cv_autoresearch.types import Baseline, Directive, SearchMode

# First parameter in the registry — used as the default fallback target.
_FALLBACK_PARAM: str = next(iter(PARAM_REGISTRY))


def get_next_directive(
    task_description: str,
    history: Any,  # SearchHistory — imported lazily to avoid circular deps
    baseline: Baseline,
    config: Any,  # SearchConfig
) -> Directive:
    """Ask claude -p what to do next in the search.

    Sends the task description, full experiment history, and current
    baseline to claude -p and parses its structured response.

    Falls back to EXPLORE with the first PARAM_REGISTRY param if claude
    fails or response cannot be parsed.

    Args:
        task_description: User's task description.
        history: All experiment history so far (SearchHistory).
        baseline: Current best baseline.
        config: Experiment config (SearchConfig).

    Returns:
        Directive with EXPLORE or EXPLOIT mode and a required target_param.
    """
    prompt = _build_directive_prompt(task_description, history, baseline)
    try:
        response = _call_claude(prompt)
        return _parse_directive(response)
    except Exception:
        return _explore_fallback()


def _build_directive_prompt(
    task_description: str,
    history: Any,
    baseline: Baseline,
) -> str:
    """Build the directive prompt for claude -p.

    Args:
        task_description: User's task description.
        history: SearchHistory with to_text() method.
        baseline: Current best baseline.

    Returns:
        Formatted prompt string.
    """
    history_text = history.to_text(max_entries=20) if hasattr(history, "to_text") else str(history)
    available_params = ", ".join(PARAM_REGISTRY.keys())
    return f"""You are directing an automated hyperparameter/augmentation search for a CV model.

Task: {task_description}
Primary metric: best so far = {baseline.primary_metric_value:.6f}

Available parameters: {available_params}

Experiment history (most recent first):
{history_text}

Choose ONE action for the next experiment:

1. EXPLORE: Sample a parameter from its full default range (good when history is short or diverse exploration needed)
2. EXPLOIT: Target a parameter with a narrower custom range (good when a promising direction is identified)

You MUST always choose a parameter. Reply in EXACTLY this format:
MODE: EXPLORE|EXPLOIT
PARAM: <parameter_name>
RANGE: <low,high or choice1,choice2 or NONE>
REASON: <one sentence>
"""


def _parse_directive(response: str) -> Directive:
    """Parse Claude's structured response into a Directive.

    Parses the fixed-format response:
        MODE: EXPLORE|EXPLOIT
        PARAM: <name>
        RANGE: <range or NONE>
        REASON: <sentence>

    Returns EXPLORE directive with fallback param if parsing fails.

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
