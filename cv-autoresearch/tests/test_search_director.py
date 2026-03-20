"""Tests for cv_autoresearch.advisor.search_director."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from cv_autoresearch.advisor.search_director import (
    _build_directive_prompt,
    _call_claude,
    _explore_fallback,
    _parse_directive,
    get_next_directive,
)
from cv_autoresearch.types import Baseline, Directive, SearchMode, SearchPhase, TrialId


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def baseline() -> Baseline:
    return Baseline(
        primary_metric_value=0.876543,
        hyperparams={"lr": 1e-3, "batch_size": 32},
        augmentation_config={"flip": True},
        trial_id=TrialId(5),
    )


@pytest.fixture()
def mock_history() -> MagicMock:
    history = MagicMock()
    history.to_text.return_value = "trial 1: val_acc=0.85\ntrial 2: val_acc=0.87"
    return history


# ---------------------------------------------------------------------------
# _build_directive_prompt
# ---------------------------------------------------------------------------


def test_build_directive_prompt_contains_task_description(baseline: Baseline, mock_history: MagicMock) -> None:
    """Prompt must include the user's task description."""
    prompt = _build_directive_prompt(
        "classify cats vs dogs",
        mock_history,
        baseline,
        SearchPhase.HYPERPARAMETER,
    )
    assert "classify cats vs dogs" in prompt


def test_build_directive_prompt_contains_phase_value(baseline: Baseline, mock_history: MagicMock) -> None:
    """Prompt must include the current phase value string."""
    prompt = _build_directive_prompt(
        "any task",
        mock_history,
        baseline,
        SearchPhase.AUGMENTATION,
    )
    assert SearchPhase.AUGMENTATION.value in prompt


def test_build_directive_prompt_contains_metric_as_float(baseline: Baseline, mock_history: MagicMock) -> None:
    """Prompt must include baseline metric formatted as a float."""
    prompt = _build_directive_prompt(
        "any task",
        mock_history,
        baseline,
        SearchPhase.HYPERPARAMETER,
    )
    # Formatted as float with 6 decimal places
    assert f"{baseline.primary_metric_value:.6f}" in prompt


def test_build_directive_prompt_calls_history_to_text(baseline: Baseline, mock_history: MagicMock) -> None:
    """history.to_text(max_entries=20) must be called to render history."""
    _build_directive_prompt("task", mock_history, baseline, SearchPhase.HYPERPARAMETER)
    mock_history.to_text.assert_called_once_with(max_entries=20)


def test_build_directive_prompt_includes_history_text(baseline: Baseline, mock_history: MagicMock) -> None:
    """Rendered history text must appear in the prompt."""
    prompt = _build_directive_prompt("task", mock_history, baseline, SearchPhase.HYPERPARAMETER)
    assert "trial 1: val_acc=0.85" in prompt


# ---------------------------------------------------------------------------
# _parse_directive
# ---------------------------------------------------------------------------


EXPLORE_RESPONSE = """\
MODE: EXPLORE
PARAM: NONE
RANGE: NONE
PHASE: hyperparameter
REASON: History is short so broad exploration is best.
"""

EXPLOIT_RESPONSE = """\
MODE: EXPLOIT
PARAM: lr
RANGE: 0.001,0.01
PHASE: hyperparameter
REASON: Learning rate shows promise in recent trials.
"""

AUGMENTATION_EXPLOIT_RESPONSE = """\
MODE: EXPLOIT
PARAM: flip_prob
RANGE: 0.3,0.7
PHASE: augmentation
REASON: Flip probability is a key augmentation parameter.
"""


@pytest.mark.parametrize(
    "response, expected_mode, expected_param, expected_range, expected_phase",
    [
        (
            EXPLORE_RESPONSE,
            SearchMode.EXPLORE,
            None,
            None,
            SearchPhase.HYPERPARAMETER,
        ),
        (
            EXPLOIT_RESPONSE,
            SearchMode.EXPLOIT,
            "lr",
            [0.001, 0.01],
            SearchPhase.HYPERPARAMETER,
        ),
        (
            AUGMENTATION_EXPLOIT_RESPONSE,
            SearchMode.EXPLOIT,
            "flip_prob",
            [0.3, 0.7],
            SearchPhase.AUGMENTATION,
        ),
    ],
)
def test_parse_directive_valid_responses(
    response: str,
    expected_mode: SearchMode,
    expected_param: str | None,
    expected_range: list[float] | None,
    expected_phase: SearchPhase,
) -> None:
    """Parse valid structured Claude responses into correct Directives."""
    directive = _parse_directive(response, SearchPhase.HYPERPARAMETER)
    assert directive.mode == expected_mode
    assert directive.target_param == expected_param
    assert directive.target_range == expected_range
    assert directive.phase == expected_phase


@pytest.mark.parametrize(
    "bad_response",
    [
        "",
        "this is not structured at all",
        "NO COLONS HERE\nAND HERE",
        "MODE EXPLORE\nPARAM NONE",  # missing colons
    ],
)
def test_parse_directive_garbage_falls_back_to_explore(bad_response: str) -> None:
    """Garbage or empty response must fall back to EXPLORE Directive."""
    directive = _parse_directive(bad_response, SearchPhase.HYPERPARAMETER)
    assert directive.mode == SearchMode.EXPLORE
    assert directive.target_param is None
    assert directive.target_range is None


def test_parse_directive_missing_mode_line_falls_back_to_explore() -> None:
    """Response without MODE line must fall back to EXPLORE."""
    response = """\
PARAM: lr
RANGE: 0.001,0.01
PHASE: hyperparameter
REASON: Some reason.
"""
    directive = _parse_directive(response, SearchPhase.HYPERPARAMETER)
    # Missing MODE defaults to EXPLORE per the implementation
    assert directive.mode == SearchMode.EXPLORE


# ---------------------------------------------------------------------------
# _explore_fallback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("phase", list(SearchPhase))
def test_explore_fallback_returns_explore_directive(phase: SearchPhase) -> None:
    """_explore_fallback must always return an EXPLORE Directive for any phase."""
    directive = _explore_fallback(phase)
    assert directive == Directive(
        mode=SearchMode.EXPLORE,
        target_param=None,
        target_range=None,
        phase=phase,
        reason="Fallback to EXPLORE due to parse failure.",
    )


# ---------------------------------------------------------------------------
# get_next_directive
# ---------------------------------------------------------------------------


def test_get_next_directive_valid_exploit_response(baseline: Baseline, mock_history: MagicMock) -> None:
    """get_next_directive returns correct Directive when claude returns valid EXPLOIT response."""
    valid_response = EXPLOIT_RESPONSE

    with patch("cv_autoresearch.advisor.search_director._call_claude", return_value=valid_response):
        directive = get_next_directive(
            task_description="classify cats vs dogs",
            history=mock_history,
            baseline=baseline,
            current_phase=SearchPhase.HYPERPARAMETER,
            config=MagicMock(),
        )

    assert directive.mode == SearchMode.EXPLOIT
    assert directive.target_param == "lr"
    assert directive.target_range == [0.001, 0.01]
    assert directive.phase == SearchPhase.HYPERPARAMETER


def test_get_next_directive_claude_raises_falls_back_to_explore(
    baseline: Baseline, mock_history: MagicMock
) -> None:
    """get_next_directive must not propagate exceptions from _call_claude; falls back to EXPLORE."""
    with patch(
        "cv_autoresearch.advisor.search_director._call_claude",
        side_effect=RuntimeError("claude -p failed: timeout"),
    ):
        directive = get_next_directive(
            task_description="any task",
            history=mock_history,
            baseline=baseline,
            current_phase=SearchPhase.AUGMENTATION,
            config=MagicMock(),
        )

    assert directive.mode == SearchMode.EXPLORE
    assert directive.target_param is None
    assert directive.phase == SearchPhase.AUGMENTATION


# ---------------------------------------------------------------------------
# _call_claude (basic contract test — does not actually invoke claude binary)
# ---------------------------------------------------------------------------


def test_call_claude_raises_on_nonzero_exit() -> None:
    """_call_claude must raise RuntimeError when the subprocess exits non-zero."""
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stderr = "some error"
    mock_result.stdout = ""

    with patch("subprocess.run", return_value=mock_result):
        with pytest.raises(RuntimeError, match="claude -p failed"):
            _call_claude("some prompt")


def test_call_claude_returns_stdout_on_success() -> None:
    """_call_claude must return stripped stdout on success."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "  MODE: EXPLORE\n"
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result):
        result = _call_claude("some prompt")

    assert result == "MODE: EXPLORE"
