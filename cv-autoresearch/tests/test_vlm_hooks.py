"""Tests for cv_autoresearch.vlm.hooks."""

from __future__ import annotations

from unittest.mock import MagicMock

from cv_autoresearch.vlm.hooks import AsyncVLMHooks, VLMHooks


def test_vlm_hooks_can_be_instantiated() -> None:
    hooks = VLMHooks()
    assert isinstance(hooks, VLMHooks)


def test_vlm_hooks_analyze_iteration_returns_none() -> None:
    hooks = VLMHooks()
    result = hooks.analyze_iteration(MagicMock(), MagicMock(), MagicMock())
    assert result is None


def test_vlm_hooks_analyze_phase_returns_none() -> None:
    hooks = VLMHooks()
    result = hooks.analyze_phase(MagicMock(), MagicMock(), MagicMock())
    assert result is None


def test_vlm_hooks_analyze_experiment_returns_none() -> None:
    hooks = VLMHooks()
    result = hooks.analyze_experiment(MagicMock(), MagicMock())
    assert result is None


def test_async_vlm_hooks_is_subclass_of_vlm_hooks() -> None:
    assert issubclass(AsyncVLMHooks, VLMHooks)


def test_async_vlm_hooks_can_be_instantiated() -> None:
    hooks = AsyncVLMHooks()
    assert isinstance(hooks, AsyncVLMHooks)
