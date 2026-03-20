"""Claude-directed search strategy for EXPLORE/EXPLOIT decisions."""

from __future__ import annotations

import subprocess
from typing import Any

from cv_autoresearch.types import Baseline, Directive, SearchMode, SearchPhase


def get_next_directive(
    task_description: str,
    history: Any,  # SearchHistory — imported lazily to avoid circular deps
    baseline: Baseline,
    current_phase: SearchPhase,
    config: Any,  # SearchConfig
) -> Directive:
    """Ask claude -p what to do next in the search.

    Sends the task description, full experiment history, and current
    baseline to claude -p and parses its structured response.

    Falls back to EXPLORE if claude fails or response cannot be parsed.

    Args:
        task_description: User's task description.
        history: All experiment history so far (SearchHistory).
        baseline: Current best baseline.
        current_phase: HYPERPARAMETER or AUGMENTATION.
        config: Experiment config (SearchConfig).

    Returns:
        Directive with EXPLORE or EXPLOIT mode.
    """
    prompt = _build_directive_prompt(task_description, history, baseline, current_phase)
    try:
        response = _call_claude(prompt)
        return _parse_directive(response, current_phase)
    except Exception:
        # Graceful fallback: always EXPLORE if claude fails or parse errors
        return _explore_fallback(current_phase)


def _build_directive_prompt(
    task_description: str,
    history: Any,
    baseline: Baseline,
    phase: SearchPhase,
) -> str:
    """Build the directive prompt for claude -p.

    Args:
        task_description: User's task description.
        history: SearchHistory with to_text() method.
        baseline: Current best baseline.
        phase: Current search phase.

    Returns:
        Formatted prompt string.
    """
    history_text = history.to_text(max_entries=20) if hasattr(history, "to_text") else str(history)
    return f"""You are directing an automated hyperparameter/augmentation search for a CV model.

Task: {task_description}
Current phase: {phase.value}
Primary metric: best so far = {baseline.primary_metric_value:.6f}

Experiment history (most recent first):
{history_text}

Choose ONE action for the next experiment:

1. EXPLORE: Sample from the full {phase.value} search space (good when history is short or diverse exploration needed)
2. EXPLOIT: Target a specific parameter with a narrower range (good when a promising direction is identified)

Reply in EXACTLY this format:
MODE: EXPLORE|EXPLOIT
PARAM: <parameter_name or NONE>
RANGE: <low,high or choice1,choice2 or NONE>
PHASE: hyperparameter|augmentation
REASON: <one sentence>
"""


def _parse_directive(response: str, phase: SearchPhase) -> Directive:
    """Parse Claude's structured response into a Directive.

    Parses the fixed-format response:
        MODE: EXPLORE|EXPLOIT
        PARAM: <name or NONE>
        RANGE: <range or NONE>
        PHASE: hyperparameter|augmentation
        REASON: <sentence>

    Returns EXPLORE directive if parsing fails (graceful fallback).

    Args:
        response: Raw string response from claude.
        phase: Current search phase (used as fallback).

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
            return _explore_fallback(phase)

        mode_str = lines.get("MODE", "EXPLORE").upper()
        mode = SearchMode.EXPLORE if mode_str == "EXPLORE" else SearchMode.EXPLOIT

        param_str = lines.get("PARAM", "NONE").strip()
        target_param = None if param_str.upper() == "NONE" else param_str

        range_str = lines.get("RANGE", "NONE").strip()
        target_range: list[float] | None = None
        if range_str.upper() != "NONE" and target_param is not None:
            parts = [p.strip() for p in range_str.split(",")]
            target_range = [float(p) for p in parts if p]

        phase_str = lines.get("PHASE", phase.value).strip().lower()
        parsed_phase = (
            SearchPhase(phase_str) if phase_str in {p.value for p in SearchPhase} else phase
        )

        reason = lines.get("REASON", "No reason provided.")

        return Directive(
            mode=mode,
            target_param=target_param,
            target_range=target_range,
            phase=parsed_phase,
            reason=reason,
        )
    except Exception:
        return _explore_fallback(phase)


def _explore_fallback(phase: SearchPhase) -> Directive:
    """Return a safe EXPLORE directive as fallback.

    Args:
        phase: Current search phase.

    Returns:
        EXPLORE Directive with no target param.
    """
    return Directive(
        mode=SearchMode.EXPLORE,
        target_param=None,
        target_range=None,
        phase=phase,
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
